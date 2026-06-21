"""
Prompt工程与模板管理（阶段59）
对应第一性原理：Prompt = 任务的认知约束；模板化 = 复用+版本化+可观测

提供能力:
    - PromptTemplate: Prompt模板（变量插值、版本管理、Few-shot支持）
    - PromptVariable: 模板变量（类型、必填、默认值、描述）
    - PromptSection: 模板片段（Role/Context/Examples/Task/Output）
    - FewShotExample: Few-shot示例
    - PromptChain: Prompt链（多步模板串联）
    - PromptRegistry: 模板注册表（单例，集中管理所有模板）
    - RenderedPrompt: 渲染后的Prompt（含元数据）
    - PromptFormatter: 格式化器（chatml/alpaca/raw）
    - 便捷函数: register_prompt / render_prompt / get_prompt 等

使用示例:
    from seva import PromptTemplate, register_prompt, render_prompt

    # 定义模板
    template = PromptTemplate(
        name="classify_intent",
        sections=[
            PromptSection.role("你是意图分类器"),
            PromptSection.task("分类用户输入的意图"),
            PromptSection.examples([...]),
            PromptSection.output_format("JSON: {intent, confidence}"),
        ],
    )
    register_prompt(template)

    # 渲染
    rendered = render_prompt("classify_intent", text="你好")
    print(rendered.to_messages())
"""
from __future__ import annotations

import os
import json
import re
import time
import uuid
import threading
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
PROMPTS_DIR = SEVA_HOME / "prompts"


# ============================================================================
# SectionType - 模板段类型
# ============================================================================

class SectionType(Enum):
    """模板段类型"""
    ROLE = "role"                # 角色定义
    CONTEXT = "context"          # 上下文
    TASK = "task"                # 任务说明
    EXAMPLES = "examples"        # 示例
    INPUT = "input"              # 输入数据
    CONSTRAINTS = "constraints"  # 约束
    OUTPUT = "output"            # 输出格式
    CUSTOM = "custom"            # 自定义


# ============================================================================
# PromptVariable - 模板变量
# ============================================================================

@dataclass
class PromptVariable:
    """模板变量定义"""
    name: str
    description: str = ""
    required: bool = True
    default: Any = None
    var_type: str = "string"  # string/number/bool/list/dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "var_type": self.var_type,
        }


# ============================================================================
# FewShotExample - Few-shot示例
# ============================================================================

@dataclass
class FewShotExample:
    """Few-shot示例"""
    user_input: str
    assistant_output: str
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "user": self.user_input,
            "assistant": self.assistant_output,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"FewShotExample(user={self.user_input[:30]!r}...)"


# ============================================================================
# PromptSection - 模板段
# ============================================================================

@dataclass
class PromptSection:
    """模板段"""
    section_type: SectionType
    content: str
    variables: List[PromptVariable] = field(default_factory=list)
    section_examples: List[FewShotExample] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.section_type.value,
            "content": self.content,
            "variables": [v.to_dict() for v in self.variables],
            "examples": [e.to_dict() for e in self.section_examples],
            "enabled": self.enabled,
        }

    @classmethod
    def role(cls, content: str, **kwargs) -> "PromptSection":
        """角色段"""
        return cls(section_type=SectionType.ROLE, content=content, **kwargs)

    @classmethod
    def context(cls, content: str, **kwargs) -> "PromptSection":
        """上下文段"""
        return cls(section_type=SectionType.CONTEXT, content=content, **kwargs)

    @classmethod
    def task(cls, content: str, **kwargs) -> "PromptSection":
        """任务段"""
        return cls(section_type=SectionType.TASK, content=content, **kwargs)

    @classmethod
    def examples(cls, examples_list: List[FewShotExample], **kwargs) -> "PromptSection":
        """示例段"""
        content = "以下是一些示例："
        return cls(section_type=SectionType.EXAMPLES, content=content,
                  section_examples=examples_list, **kwargs)

    @classmethod
    def input_data(cls, content: str, **kwargs) -> "PromptSection":
        """输入段"""
        return cls(section_type=SectionType.INPUT, content=content, **kwargs)

    @classmethod
    def constraints(cls, content: str, **kwargs) -> "PromptSection":
        """约束段"""
        return cls(section_type=SectionType.CONSTRAINTS, content=content, **kwargs)

    @classmethod
    def output_format(cls, content: str, **kwargs) -> "PromptSection":
        """输出格式段"""
        return cls(section_type=SectionType.OUTPUT, content=content, **kwargs)

    @classmethod
    def custom(cls, content: str, **kwargs) -> "PromptSection":
        """自定义段"""
        return cls(section_type=SectionType.CUSTOM, content=content, **kwargs)


# ============================================================================
# PromptFormat - 输出格式
# ============================================================================

class PromptFormat(Enum):
    """Prompt输出格式"""
    RAW = "raw"            # 原始字符串
    CHATML = "chatml"      # ChatML格式（多消息）
    ALPACA = "alpaca"      # Alpaca格式
    MESSAGES = "messages"  # 消息列表（兼容OpenAI）


# ============================================================================
# PromptTemplate - Prompt模板
# ============================================================================

@dataclass
class PromptTemplate:
    """
    Prompt模板

    使用示例:
        template = PromptTemplate(
            name="classify_intent",
            description="意图分类",
            sections=[
                PromptSection.role("你是意图分类器"),
                PromptSection.task("分类用户输入的意图: 体育/科技/娱乐"),
                PromptSection.output_format('JSON: {"intent": "...", "confidence": 0.0}'),
            ],
            variables=[
                PromptVariable(name="text", description="用户输入"),
            ],
        )
    """
    name: str
    sections: List[PromptSection] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    variables: List[PromptVariable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    output_format: PromptFormat = PromptFormat.MESSAGES
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, str]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "sections": [s.to_dict() for s in self.sections],
            "variables": [v.to_dict() for v in self.variables],
            "tags": list(self.tags),
            "output_format": self.output_format.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "response_format": self.response_format,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        sections_data = data.get("sections", [])
        sections = []
        for s in sections_data:
            variables = [PromptVariable(**v) for v in s.get("variables", [])]
            examples = [
                FewShotExample(user_input=e["user"], assistant_output=e["assistant"],
                              description=e.get("description", ""))
                for e in s.get("examples", [])
            ]
            sections.append(PromptSection(
                section_type=SectionType(s["type"]),
                content=s["content"],
                variables=variables,
                section_examples=examples,
                enabled=s.get("enabled", True),
            ))

        variables = [PromptVariable(**v) for v in data.get("variables", [])]
        output_format = data.get("output_format", "messages")
        if isinstance(output_format, str):
            output_format = PromptFormat(output_format)

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            sections=sections,
            variables=variables,
            tags=data.get("tags", []),
            output_format=output_format,
            model=data.get("model"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            stop=data.get("stop"),
            response_format=data.get("response_format"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            author=data.get("author", ""),
        )

    def get_required_variables(self) -> List[PromptVariable]:
        """获取所有必填变量"""
        result = []
        for var in self.variables:
            if var.required:
                result.append(var)
        for section in self.sections:
            for var in section.variables:
                if var.required and var not in result:
                    result.append(var)
        return result

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """验证输入，返回缺失变量名列表"""
        missing = []
        for var in self.get_required_variables():
            if var.name not in inputs and var.default is None:
                missing.append(var.name)
        return missing

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, version={self.version!r})"


# ============================================================================
# RenderedPrompt - 渲染后的Prompt
# ============================================================================

@dataclass
class RenderedPrompt:
    """渲染后的Prompt"""
    template_name: str
    template_version: str
    raw_content: str
    messages: List[Dict[str, str]]  # 兼容OpenAI消息格式
    system_content: Optional[str] = None
    user_content: Optional[str] = None
    variables_used: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if not self.hash:
            content = self.raw_content or self.user_content or ""
            self.hash = hashlib.sha256(
                (content + self.template_version).encode("utf-8")
            ).hexdigest()[:16]

    def to_messages(self) -> List[Dict[str, str]]:
        """转为OpenAI消息格式"""
        if self.messages:
            return list(self.messages)
        result = []
        if self.system_content:
            result.append({"role": "system", "content": self.system_content})
        if self.user_content:
            result.append({"role": "user", "content": self.user_content})
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "template_version": self.template_version,
            "raw_content": self.raw_content,
            "messages": self.messages,
            "system_content": self.system_content,
            "user_content": self.user_content,
            "variables_used": self.variables_used,
            "hash": self.hash,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# ============================================================================
# PromptFormatter - Prompt格式化器
# ============================================================================

class PromptFormatter:
    """Prompt格式化器"""

    @staticmethod
    def format(template: PromptTemplate, inputs: Dict[str, Any],
               format_type: PromptFormat = PromptFormat.MESSAGES) -> RenderedPrompt:
        """
        渲染模板

        Args:
            template: Prompt模板
            inputs: 变量输入
            format_type: 输出格式

        Returns:
            渲染后的Prompt
        """
        # 合并默认值
        merged_inputs = dict(inputs)
        for var in template.variables:
            if var.name not in merged_inputs and var.default is not None:
                merged_inputs[var.name] = var.default

        # 按段类型组织内容
        sections_by_type: Dict[SectionType, List[PromptSection]] = {}
        for section in template.sections:
            if not section.enabled:
                continue
            if section.section_type not in sections_by_type:
                sections_by_type[section.section_type] = []
            sections_by_type[section.section_type].append(section)

        # 渲染各段
        rendered_sections: Dict[SectionType, str] = {}
        for section_type, sections in sections_by_type.items():
            parts = []
            for section in sections:
                # 变量插值
                content = _interpolate(section.content, merged_inputs)
                parts.append(content)
                # 处理examples段
                if section_type == SectionType.EXAMPLES and section.section_examples:
                    examples_text = _format_examples(section.section_examples, merged_inputs)
                    parts.append(examples_text)
            rendered_sections[section_type] = "\n".join(parts)

        # 按固定顺序组装
        system_parts: List[str] = []
        user_parts: List[str] = []
        raw_parts: List[str] = []

        # system部分：role + context + constraints
        for st in [SectionType.ROLE, SectionType.CONTEXT, SectionType.CONSTRAINTS]:
            if st in rendered_sections:
                system_parts.append(rendered_sections[st])
                raw_parts.append(rendered_sections[st])

        # user部分：task + examples + input + output
        for st in [SectionType.TASK, SectionType.EXAMPLES, SectionType.INPUT,
                   SectionType.OUTPUT, SectionType.CUSTOM]:
            if st in rendered_sections:
                user_parts.append(rendered_sections[st])
                raw_parts.append(rendered_sections[st])

        system_content = "\n\n".join(system_parts) if system_parts else None
        user_content = "\n\n".join(user_parts) if user_parts else None
        raw_content = "\n\n".join(raw_parts)

        # 消息列表
        messages: List[Dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})

        return RenderedPrompt(
            template_name=template.name,
            template_version=template.version,
            raw_content=raw_content,
            messages=messages,
            system_content=system_content,
            user_content=user_content,
            variables_used=merged_inputs,
            model=template.model,
            temperature=template.temperature,
            max_tokens=template.max_tokens,
            response_format=template.response_format,
        )


# ============================================================================
# PromptChain - Prompt链
# ============================================================================

class PromptChain:
    """Prompt链：将多个模板串联执行，前一个的输出作为后一个的输入"""

    def __init__(self, name: str):
        self.name = name
        self._steps: List[Tuple[PromptTemplate, Dict[str, Any]]] = []
        self._lock = threading.RLock()

    def add_step(self, template: PromptTemplate, input_mapper: Optional[Callable] = None
                 ) -> "PromptChain":
        """添加步骤"""
        with self._lock:
            self._steps.append((template, input_mapper))
        return self

    def get_steps(self) -> List[PromptTemplate]:
        """获取所有步骤的模板"""
        with self._lock:
            return [t for t, _ in self._steps]

    def clear(self) -> None:
        """清空"""
        with self._lock:
            self._steps.clear()

    def step_count(self) -> int:
        """步骤数"""
        with self._lock:
            return len(self._steps)

    def __repr__(self) -> str:
        return f"PromptChain(name={self.name!r}, steps={self.step_count()})"


# ============================================================================
# PromptRegistry - 模板注册表（单例）
# ============================================================================

class PromptRegistry:
    """Prompt模板注册表（单例）"""

    _instance: Optional["PromptRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PromptRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._templates: Dict[str, PromptTemplate] = {}
        self._data_lock = threading.RLock()

        # 加载持久化的模板
        self._load()

    def register(self, template: PromptTemplate, overwrite: bool = True) -> bool:
        """注册模板"""
        with self._data_lock:
            if template.name in self._templates and not overwrite:
                return False
            template.updated_at = time.time()
            self._templates[template.name] = template
            self._save()
            return True

    def unregister(self, name: str) -> bool:
        """注销模板"""
        with self._data_lock:
            if name in self._templates:
                del self._templates[name]
                self._save()
                return True
            return False

    def get(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        with self._data_lock:
            return self._templates.get(name)

    def list(self) -> List[str]:
        """列出所有模板名"""
        with self._data_lock:
            return list(self._templates.keys())

    def find_by_tag(self, tag: str) -> List[PromptTemplate]:
        """按标签查找"""
        with self._data_lock:
            return [t for t in self._templates.values() if tag in t.tags]

    def count(self) -> int:
        """模板数量"""
        with self._data_lock:
            return len(self._templates)

    def clear(self) -> None:
        """清空所有模板"""
        with self._data_lock:
            self._templates.clear()
            self._save()

    def _save(self) -> None:
        """持久化到磁盘"""
        try:
            PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = PROMPTS_DIR / "registry.json"
            data = {name: t.to_dict() for name, t in self._templates.items()}
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        """从磁盘加载"""
        file_path = PROMPTS_DIR / "registry.json"
        if not file_path.exists():
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._data_lock:
                for name, t_data in data.items():
                    self._templates[name] = PromptTemplate.from_dict(t_data)
        except Exception:
            pass

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（清空内存，但保留磁盘文件）"""
        with cls._lock:
            cls._instance = None

    @classmethod
    def reset_all(cls) -> None:
        """完全重置：清空内存并删除磁盘文件"""
        with cls._lock:
            if cls._instance is not None:
                # 直接清空内存，不触发 _save()
                cls._instance._templates.clear()
            cls._instance = None
            # 尝试删除持久化文件
            try:
                file_path = PROMPTS_DIR / "registry.json"
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass


# ============================================================================
# 内部工具函数
# ============================================================================

_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}|\$(\w+)")


def _interpolate(text: str, inputs: Dict[str, Any]) -> str:
    """变量插值：支持{{var}}和$var两种语法"""

    def replacer(match):
        var_name = match.group(1) or match.group(2)
        if var_name in inputs:
            value = inputs[var_name]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        return match.group(0)  # 保留原占位符

    return _VAR_PATTERN.sub(replacer, text)


def _format_examples(examples: List[FewShotExample], inputs: Dict[str, Any]) -> str:
    """格式化示例"""
    parts = []
    for i, ex in enumerate(examples, 1):
        user = _interpolate(ex.user_input, inputs)
        assistant = _interpolate(ex.assistant_output, inputs)
        desc = f" ({ex.description})" if ex.description else ""
        parts.append(f"示例 {i}{desc}:\n用户: {user}\n助手: {assistant}")
    return "\n\n".join(parts)


# ============================================================================
# 便捷函数
# ============================================================================

def get_prompt_registry() -> PromptRegistry:
    """获取Prompt注册表"""
    return PromptRegistry()


def register_prompt(template: PromptTemplate, overwrite: bool = True) -> bool:
    """注册模板"""
    return PromptRegistry().register(template, overwrite=overwrite)


def get_prompt(name: str) -> Optional[PromptTemplate]:
    """获取模板"""
    return PromptRegistry().get(name)


def list_prompts() -> List[str]:
    """列出所有模板"""
    return PromptRegistry().list()


def unregister_prompt(name: str) -> bool:
    """注销模板"""
    return PromptRegistry().unregister(name)


def render_prompt(name: str, inputs: Optional[Dict[str, Any]] = None,
                  format_type: PromptFormat = PromptFormat.MESSAGES) -> Optional[RenderedPrompt]:
    """
    渲染Prompt

    Args:
        name: 模板名
        inputs: 变量输入
        format_type: 输出格式
    """
    template = get_prompt(name)
    if template is None:
        return None
    return PromptFormatter.format(template, inputs or {}, format_type)


def delete_prompt(name: str) -> bool:
    """删除模板（unregister_prompt的别名）"""
    return unregister_prompt(name)


def reset_prompts() -> None:
    """重置所有模板（清空内存并删除磁盘文件）"""
    PromptRegistry.reset_all()
