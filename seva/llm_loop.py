"""
16阶段循环的LLM增强（阶段60）
对应第一性原理：LLM = 认知引擎；SEVA = 控制骨架。
原则：
    1. LLM作为可选项，不破坏现有SEVA控制流
    2. 每个LLM增强点都有降级路径（无LLM / 失败 → 规则引擎）
    3. LLM调用必须：超时保护、错误降级、缓存
    4. JSON结构化输出 → 自动解析 → 融合到阶段结果
    5. 观测每个阶段的LLM成本（token、延迟、命中率）

涉及阶段映射（16个认知阶段）:
    1. perception (LLM: 语义解析)
    2. drive_engine (LLM: 情境化驱动调整)
    3. goal_generator (LLM: 目标生成)
    4. predictor (LLM: 多步预测)
    5. decision_maker (LLM: 决策解释)
    6. safety_ethics (LLM: 语义安全审查)
    7. executor (LLM: 动作参数化)
    8. error_evaluator (LLM: 错误归因)
    9. task_update (LLM: 任务状态总结)
    10. skill_update (LLM: 技能元信息)
    11. coordination (LLM: 协作分配)
    12. evolution (LLM: 进化策略)
    13. optimizer (LLM: 优化器诊断)
    14. memory (LLM: 记忆摘要)
    15. metacognition (LLM: 自我反思)
    16. timing_control (LLM: 时序策略)
"""
from __future__ import annotations

import json
import re
import time
import logging
import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .llm import (
    Message, LLMError, get_llm_client, get_llm_stats,
)
from .prompts import (
    PromptTemplate, PromptSection, PromptVariable, FewShotExample,
    PromptFormat, register_prompt, render_prompt, get_prompt, list_prompts,
)


logger = logging.getLogger("seva.llm_loop")


# ============================================================================
# 常量与状态枚举
# ============================================================================

class LLMStageStatus(Enum):
    """LLM阶段调用状态"""
    SUCCESS = "success"               # LLM成功
    SKIPPED = "skipped"               # 未启用/无输入
    CACHED = "cached"                 # 缓存命中
    DEGRADED = "degraded"             # 失败但已降级
    DISABLED = "disabled"             # 全局禁用
    ERROR = "error"                   # 错误


class LLMRole(Enum):
    """LLM在循环中的角色"""
    ENHANCER = "enhancer"        # 增强原结果
    REPLACER = "replacer"        # 替换原结果
    EXPLICATOR = "explicator"    # 解释原结果


@dataclass
class LLMStageResult:
    """单个LLM阶段调用的结果"""
    stage: str
    status: LLMStageStatus
    llm_output: Optional[str] = None          # LLM原始输出
    parsed_output: Optional[Dict[str, Any]] = None  # 解析后的结构化输出
    fused_output: Optional[Any] = None        # 融合到原结果的部分
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    degraded: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "llm_output": self.llm_output,
            "parsed_output": self.parsed_output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "degraded": self.degraded,
        }


# ============================================================================
# JSON解析器
# ============================================================================

_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", re.DOTALL)
_JSON_FIRST_OBJECT = re.compile(r"(\{[\s\S]*\})", re.DOTALL)
_JSON_FIRST_ARRAY = re.compile(r"(\[[\s\S]*\])", re.DOTALL)


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """
    从LLM输出中提取JSON对象

    优先级：
        1. Markdown代码块 ```json {...} ``` / ```[...]```
        2. 第一个完整的 {...} 块
        3. 整体作为JSON解析
        4. 数组自动包装为 {"items": [...]}
    """
    if not text:
        return None

    def _wrap_if_list(data):
        """如果是数组则包装为 items dict"""
        if isinstance(data, list):
            return {"items": data}
        return data

    # 1. Markdown 代码块
    matches = _JSON_PATTERN.findall(text)
    if matches:
        for match in matches:
            try:
                result = json.loads(match)
                wrapped = _wrap_if_list(result)
                if wrapped is not None:
                    return wrapped
            except json.JSONDecodeError:
                continue

    # 2. 第一个 {...} 块
    obj_matches = _JSON_FIRST_OBJECT.findall(text)
    for match in obj_matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # 3. 整体解析
    try:
        result = json.loads(text)
        return _wrap_if_list(result)
    except json.JSONDecodeError:
        pass

    # 4. 第一个 [...] 块
    arr_matches = _JSON_FIRST_ARRAY.findall(text)
    for match in arr_matches:
        try:
            data = json.loads(match)
            if isinstance(data, list):
                return {"items": data}
        except json.JSONDecodeError:
            continue

    return None


# ============================================================================
# LLM Loop - 核心类
# ============================================================================

class LLMLoop:
    """
    16阶段循环的LLM增强管理器

    使用方法:
        llm_loop = LLMLoop(enable=True)
        # 在任何阶段中:
        result = llm_loop.call_stage("goal_generator", inputs, role=LLMRole.ENHANCER)
        if result.parsed_output:
            # 融合到原结果
            pass
    """

    def __init__(self, enable: bool = True,
                 enable_stages: Optional[List[str]] = None,
                 disable_stages: Optional[List[str]] = None,
                 default_model: Optional[str] = None,
                 default_temperature: float = 0.3,
                 default_max_tokens: int = 512,
                 default_timeout: float = 30.0):
        """
        Args:
            enable: 全局开关
            enable_stages: 仅启用这些阶段（None表示全部）
            disable_stages: 禁用这些阶段
            default_model: 默认模型
            default_temperature: 默认温度
            default_max_tokens: 默认最大tokens
            default_timeout: 默认超时
        """
        self.enable = enable
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_timeout = default_timeout

        # 阶段启用集合
        self._enabled_stages: Optional[set] = None
        if enable_stages:
            self._enabled_stages = set(enable_stages)
        self._disabled_stages: set = set(disable_stages or [])

        # 提示词模板注册表
        self._templates: Dict[str, str] = {}  # stage_name -> template_name
        self._lock = threading.RLock()

        # 调用历史
        self.history: List[LLMStageResult] = []
        self._history_lock = threading.RLock()

        # 统计
        self.stats = {
            "total_calls": 0,
            "success_calls": 0,
            "cached_calls": 0,
            "degraded_calls": 0,
            "error_calls": 0,
            "skipped_calls": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
        }
        self._stats_lock = threading.RLock()

        # 注册内置的16个阶段模板
        self._register_default_templates()

        logger.info(f"[LLMLoop] 初始化完成 enable={enable} stages={enable_stages}")

    # ========================================================================
    # 阶段管理
    # ========================================================================

    def enable_stage(self, stage: str):
        """启用某个阶段"""
        with self._lock:
            self._disabled_stages.discard(stage)
            if self._enabled_stages is not None:
                self._enabled_stages.add(stage)

    def disable_stage(self, stage: str):
        """禁用某个阶段"""
        with self._lock:
            self._disabled_stages.add(stage)
            if self._enabled_stages is not None:
                self._enabled_stages.discard(stage)

    def is_stage_enabled(self, stage: str) -> bool:
        """检查阶段是否启用"""
        if not self.enable:
            return False
        with self._lock:
            if self._enabled_stages is not None:
                if stage not in self._enabled_stages:
                    return False
            if stage in self._disabled_stages:
                return False
        return True

    # ========================================================================
    # 模板管理
    # ========================================================================

    def register_template(self, stage: str, template: PromptTemplate):
        """为某个阶段注册模板"""
        with self._lock:
            register_prompt(template, overwrite=True)
            self._templates[stage] = template.name
            logger.info(f"[LLMLoop] 阶段 {stage} 注册模板 {template.name}")

    def get_template(self, stage: str) -> Optional[PromptTemplate]:
        """获取阶段对应的模板"""
        with self._lock:
            name = self._templates.get(stage)
            if name is None:
                return None
            return get_prompt(name)

    # ========================================================================
    # 核心调用
    # ========================================================================

    def call_stage(self, stage: str, inputs: Dict[str, Any],
                   role: LLMRole = LLMRole.ENHANCER,
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   timeout: Optional[float] = None,
                   use_cache: bool = True) -> LLMStageResult:
        """
        调用LLM执行某个阶段

        Args:
            stage: 阶段名
            inputs: 输入变量
            role: LLM角色
            model: 覆盖默认模型
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大tokens
            timeout: 覆盖默认超时
            use_cache: 是否使用缓存

        Returns:
            LLMStageResult
        """
        result = LLMStageResult(stage=stage, status=LLMStageStatus.SKIPPED)

        # 1. 检查启用状态
        if not self.is_stage_enabled(stage):
            result.status = LLMStageStatus.DISABLED
            return result

        # 2. 获取模板
        template = self.get_template(stage)
        if template is None:
            # 没有模板时，使用通用模板
            template = self._build_fallback_template(stage, role, inputs)

        # 3. 渲染模板
        try:
            rendered = render_prompt(template.name, inputs=inputs)
        except Exception as e:
            logger.warning(f"[LLMLoop] 阶段 {stage} 渲染失败: {e}")
            result.status = LLMStageStatus.ERROR
            result.error = f"render_error: {e}"
            self._record_call(result, stage)
            return result

        if rendered is None:
            result.status = LLMStageStatus.ERROR
            result.error = "render_failed"
            self._record_call(result, stage)
            return result

        # 4. 调用LLM
        result = self._invoke_llm(
            stage=stage,
            messages=rendered.to_messages(),
            model=model or template.model or self.default_model,
            temperature=temperature if temperature is not None else
                        template.temperature if template.temperature is not None else
                        self.default_temperature,
            max_tokens=max_tokens or template.max_tokens or self.default_max_tokens,
            timeout=timeout or self.default_timeout,
            use_cache=use_cache,
        )
        self._record_call(result, stage)
        return result

    def call_with_messages(self, stage: str, messages: List[Dict[str, str]],
                           **kwargs) -> LLMStageResult:
        """直接使用消息列表调用LLM"""
        result = LLMStageResult(stage=stage, status=LLMStageStatus.SKIPPED)
        if not self.is_stage_enabled(stage):
            result.status = LLMStageStatus.DISABLED
            return result
        result = self._invoke_llm(
            stage=stage,
            messages=messages,
            model=kwargs.get("model", self.default_model),
            temperature=kwargs.get("temperature", self.default_temperature),
            max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
            timeout=kwargs.get("timeout", self.default_timeout),
            use_cache=kwargs.get("use_cache", True),
        )
        self._record_call(result, stage)
        return result

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _invoke_llm(self, stage: str, messages: List[Dict[str, str]],
                    model: Optional[str], temperature: float, max_tokens: int,
                    timeout: float, use_cache: bool) -> LLMStageResult:
        """调用LLM核心逻辑"""
        result = LLMStageResult(stage=stage, status=LLMStageStatus.SKIPPED)
        t0 = time.time()

        # 转换为 Message 对象
        msg_objs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                msg_objs.append(Message.system(content))
            elif role == "assistant":
                msg_objs.append(Message.assistant(content))
            else:
                msg_objs.append(Message.user(content))

        # 获取LLM客户端
        try:
            client = get_llm_client()
        except Exception as e:
            result.status = LLMStageStatus.ERROR
            result.error = f"client_init_error: {e}"
            result.degraded = True
            result.latency_ms = (time.time() - t0) * 1000
            return result

        # 调用（client.chat 接收 messages 列表 + kwargs）
        # use_cache 是 client 实例属性，需要临时设置
        # timeout 也是 client 实例属性
        try:
            original_use_cache = client.use_cache
            original_timeout = getattr(client, "timeout", None)
            client.use_cache = use_cache
            if timeout:
                client.timeout = timeout

            kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if model:
                kwargs["model"] = model

            try:
                response = client.chat(msg_objs, **kwargs)
            finally:
                client.use_cache = original_use_cache
                if original_timeout is not None:
                    client.timeout = original_timeout

            result.llm_output = response.content
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens
            result.total_tokens = response.usage.total_tokens
            result.latency_ms = (time.time() - t0) * 1000
            result.status = LLMStageStatus.SUCCESS

            # 解析JSON
            parsed = parse_json_output(response.content)
            if parsed is not None:
                result.parsed_output = parsed

        except LLMError as e:
            result.status = LLMStageStatus.DEGRADED
            result.error = f"llm_error: {e}"
            result.degraded = True
            result.latency_ms = (time.time() - t0) * 1000
            logger.warning(f"[LLMLoop] 阶段 {stage} LLM失败,降级: {e}")
        except Exception as e:
            result.status = LLMStageStatus.ERROR
            result.error = f"unexpected: {e}"
            result.degraded = True
            result.latency_ms = (time.time() - t0) * 1000
            logger.error(f"[LLMLoop] 阶段 {stage} 异常: {e}")

        return result

    def _record_call(self, result: LLMStageResult, stage: str):
        """记录调用"""
        with self._history_lock:
            self.history.append(result)
            # 限制历史长度
            if len(self.history) > 1000:
                self.history = self.history[-500:]

        with self._stats_lock:
            self.stats["total_calls"] += 1
            if result.status == LLMStageStatus.SUCCESS:
                self.stats["success_calls"] += 1
            elif result.status == LLMStageStatus.DEGRADED:
                self.stats["degraded_calls"] += 1
            elif result.status == LLMStageStatus.ERROR:
                self.stats["error_calls"] += 1
            elif result.status in (LLMStageStatus.SKIPPED, LLMStageStatus.DISABLED):
                self.stats["skipped_calls"] += 1
            self.stats["total_tokens"] += result.total_tokens
            self.stats["total_latency_ms"] += result.latency_ms

    def _build_fallback_template(self, stage: str, role: LLMRole,
                                 inputs: Dict[str, Any]) -> PromptTemplate:
        """构建通用回退模板"""
        role_text = {
            LLMRole.ENHANCER: "增强原结果",
            LLMRole.REPLACER: "替换原结果",
            LLMRole.EXPLICATOR: "解释原结果",
        }.get(role, "处理")

        content_lines = [
            f"你是{stage}阶段的处理助手。",
            f"任务: 基于输入数据{role_text}。",
            "",
            "输入数据:",
        ]
        for k, v in inputs.items():
            content_lines.append(f"- {k}: {v}")

        content_lines.extend([
            "",
            "请以JSON格式输出:",
            "{",
            '  "output": "结果描述",',
            '  "confidence": 0.0-1.0,',
            '  "reasoning": "推理过程"',
            "}",
        ])

        template_name = f"__fallback_{stage}"
        template = PromptTemplate(
            name=template_name,
            description=f"{stage}阶段通用回退模板",
            sections=[
                PromptSection.role(f"你是{stage}阶段的处理助手"),
                PromptSection.task("\n".join(content_lines)),
            ],
            output_format=PromptFormat.MESSAGES,
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
            author="LLMLoop",
        )
        register_prompt(template, overwrite=True)
        with self._lock:
            self._templates[stage] = template_name
        return template

    def _register_default_templates(self):
        """注册16个阶段的默认模板"""
        defaults = {
            "perception": {
                "role": "你是环境感知专家，能从原始信号中提取语义信息。",
                "task": "基于感知输入提取关键特征。",
                "output": '{"features": [...], "context": "...", "confidence": 0.0}',
            },
            "drive_engine": {
                "role": "你是动机分析专家，能评估驱动状态。",
                "task": "基于当前感知和历史，评估三类驱动力。",
                "output": '{"safety": 0.0-1.0, "gain": 0.0-1.0, "efficiency": 0.0-1.0, "reasoning": "..."}',
            },
            "goal_generator": {
                "role": "你是目标规划专家，能根据驱动力生成合理目标。",
                "task": "基于驱动力和当前状态，生成1-3个目标。",
                "output": '{"goals": [{"type": "...", "priority": 0.0-1.0, "description": "..."}]}',
            },
            "predictor": {
                "role": "你是预测专家，能基于状态预测未来走向。",
                "task": "基于当前感知预测下一个状态。",
                "output": '{"predicted_state": "...", "confidence": 0.0-1.0, "horizon_steps": 1-10}',
            },
            "decision_maker": {
                "role": "你是决策专家，能基于多因素选择最优动作。",
                "task": "基于目标、预测、驱动选择动作。",
                "output": '{"action_type": "...", "parameters": {...}, "expected_reward": 0.0, "reasoning": "..."}',
            },
            "safety_ethics": {
                "role": "你是安全审查专家，能评估动作的潜在风险。",
                "task": "评估动作的安全性。",
                "output": '{"approved": true/false, "risk_level": "low/medium/high", "concerns": [...]}',
            },
            "executor": {
                "role": "你是动作执行专家，能将抽象动作转为具体参数。",
                "task": "将动作意图细化为可执行参数。",
                "output": '{"parameters": {...}, "estimated_success": 0.0-1.0}',
            },
            "error_evaluator": {
                "role": "你是误差分析专家，能诊断错误的根本原因。",
                "task": "分析预测误差的来源。",
                "output": '{"magnitude": 0.0-1.0, "attribution": {"model": 0.0-1.0, "data": 0.0-1.0, "action": 0.0-1.0}, "root_cause": "..."}',
            },
            "task_update": {
                "role": "你是任务管理专家，能评估任务进展。",
                "task": "基于当前奖励和误差，更新任务状态。",
                "output": '{"progress": 0.0-1.0, "status": "active/blocked/completed/failed", "next_step": "..."}',
            },
            "skill_update": {
                "role": "你是技能评估专家，能评估技能的可靠性。",
                "task": "基于使用历史评估技能价值。",
                "output": '{"q_value": 0.0-1.0, "reliability": 0.0-1.0, "should_keep": true/false}',
            },
            "coordination": {
                "role": "你是多Agent协调专家，能优化任务分配。",
                "task": "评估当前协作状态。",
                "output": '{"distribution": "fair/biased", "bottleneck": "agent_id or null", "recommendation": "..."}',
            },
            "evolution": {
                "role": "你是进化策略专家，能根据系统状态调整学习策略。",
                "task": "评估当前阶段是否需要调整进化策略。",
                "output": '{"difficulty_change": 0.0, "focus_shift": "...", "adjustments": [...]}',
            },
            "optimizer": {
                "role": "你是模型优化专家，能诊断预测模型状态。",
                "task": "分析优化器当前状态。",
                "output": '{"learning_health": "good/converging/diverging", "recommended_lr": 0.0-1.0}',
            },
            "memory": {
                "role": "你是记忆管理专家，能决定什么值得记住。",
                "task": "评估当前经验的记忆价值。",
                "output": '{"importance": 0.0-1.0, "category": "episodic/semantic/procedural", "summary": "..."}',
            },
            "metacognition": {
                "role": "你是元认知专家，能进行自我反思。",
                "task": "反思最近的表现并提出改进方向。",
                "output": '{"insights": [...], "improvements": [...], "confidence": 0.0-1.0}',
            },
            "timing_control": {
                "role": "你是时序控制专家，能优化系统节律。",
                "task": "评估当前时序配置。",
                "output": '{"cycle_time_ms": 0.0, "bottleneck": "...", "recommendation": "..."}',
            },
        }

        for stage, content in defaults.items():
            template_name = f"__stage_{stage}"
            template = PromptTemplate(
                name=template_name,
                description=f"{stage}阶段LLM模板",
                sections=[
                    PromptSection.role(content["role"]),
                    PromptSection.task(content["task"] + " 输入: {{input}}"),
                    PromptSection.output_format(content["output"]),
                ],
                variables=[
                    PromptVariable(name="input", description="阶段输入", required=True),
                ],
                output_format=PromptFormat.MESSAGES,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
                author="LLMLoop",
            )
            register_prompt(template, overwrite=True)
            with self._lock:
                self._templates[stage] = template_name

    # ========================================================================
    # 历史与统计
    # ========================================================================

    def get_history(self, stage: Optional[str] = None,
                    limit: int = 100) -> List[LLMStageResult]:
        """获取历史"""
        with self._history_lock:
            if stage is None:
                return list(self.history[-limit:])
            return [r for r in self.history if r.stage == stage][-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._stats_lock:
            stats = dict(self.stats)
        # 成功率
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["success_calls"] / stats["total_calls"]
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_latency_ms"] = 0.0
        # 与LLM客户端的全局统计合并
        try:
            client_stats = get_llm_stats()
            stats["client_total_calls"] = client_stats.get("total_calls", 0)
        except Exception:
            pass
        return stats

    def reset_stats(self):
        """重置统计"""
        with self._stats_lock:
            for k in self.stats:
                self.stats[k] = 0 if isinstance(self.stats[k], (int, float)) else (
                    {} if isinstance(self.stats[k], dict) else None
                )
            self.stats["total_calls"] = 0
            self.stats["success_calls"] = 0
            self.stats["cached_calls"] = 0
            self.stats["degraded_calls"] = 0
            self.stats["error_calls"] = 0
            self.stats["skipped_calls"] = 0
            self.stats["total_tokens"] = 0
            self.stats["total_latency_ms"] = 0.0
        with self._history_lock:
            self.history.clear()

    def get_report(self) -> Dict[str, Any]:
        """获取完整报告"""
        return {
            "stats": self.get_stats(),
            "history_count": len(self.history),
            "enabled": self.enable,
            "enabled_stages": list(self._enabled_stages) if self._enabled_stages else "all",
            "disabled_stages": list(self._disabled_stages),
            "registered_templates": dict(self._templates),
        }


# ============================================================================
# 全局实例（单例）
# ============================================================================

_global_llm_loop: Optional[LLMLoop] = None
_global_lock = threading.Lock()


def get_llm_loop(enable: Optional[bool] = None) -> LLMLoop:
    """
    获取全局LLM Loop实例

    Args:
        enable: 是否启用（仅在创建时生效）
    """
    global _global_llm_loop
    with _global_lock:
        if _global_llm_loop is None:
            _global_llm_loop = LLMLoop(enable=enable if enable is not None else True)
        return _global_llm_loop


def set_llm_loop(llm_loop: LLMLoop):
    """设置全局LLM Loop实例"""
    global _global_llm_loop
    with _global_lock:
        _global_llm_loop = llm_loop


def reset_llm_loop():
    """重置全局LLM Loop"""
    global _global_llm_loop
    with _global_lock:
        _global_llm_loop = None
