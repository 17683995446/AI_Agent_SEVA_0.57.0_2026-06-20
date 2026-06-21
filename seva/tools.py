"""
工具调用与Function Calling（阶段62）

对应第一性原理：
    1. LLM = 认知引擎（决定调用什么工具、传什么参数）
    2. SEVA = 控制骨架（执行工具、验证参数、处理结果）
    3. 工具是Agent与外部世界交互的接口
    4. 安全性：参数验证 + 执行沙箱 + 超时保护

模块组成：
    - ToolParameter: 工具参数定义 (name/type/description/required/default/enum)
    - ToolDefinition: 工具定义 (name/description/parameters/func)
    - ToolCallRequest: LLM发起的工具调用请求
    - ToolCallResult: 工具执行结果
    - ToolRegistry: 工具注册表 (注册/注销/查找/列举)
    - ToolExecutor: 工具执行器 (参数验证/执行/超时/错误处理)
    - ToolCallLoop: LLM驱动的工具调用循环 (ReAct模式)
    - 内置工具: calculator/datetime/text_utils/http_get
"""
from __future__ import annotations

import os
import re
import json
import time
import math
import logging
import threading
import traceback
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Message, LLMError, get_llm_client
from .llm_loop import parse_json_output


logger = logging.getLogger("seva.tools")


# ============================================================================
# 参数类型
# ============================================================================

class ParameterType(Enum):
    """工具参数类型"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


# ============================================================================
# 工具参数定义
# ============================================================================

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    param_type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.param_type.value,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.enum is not None:
            d["enum"] = self.enum
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.maximum is not None:
            d["maximum"] = self.maximum
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolParameter":
        return cls(
            name=data["name"],
            param_type=ParameterType(data.get("type", "string")),
            description=data.get("description", ""),
            required=data.get("required", True),
            default=data.get("default"),
            enum=data.get("enum"),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
        )

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """验证参数值"""
        if value is None:
            if self.required:
                return False, f"参数 '{self.name}' 是必填的"
            return True, None

        # bool检查优先(bool是int子类,isinstance(True, int)==True)
        if isinstance(value, bool):
            if self.param_type != ParameterType.BOOLEAN:
                return False, f"参数 '{self.name}' 应为{self.param_type.value},得到布尔值"
            # bool类型验证通过,跳过后续类型检查
        else:
            # 类型检查
            type_map = {
                ParameterType.STRING: str,
                ParameterType.NUMBER: (int, float),
                ParameterType.INTEGER: int,
                ParameterType.ARRAY: (list, tuple),
                ParameterType.OBJECT: dict,
            }
            expected_type = type_map.get(self.param_type)
            if expected_type and not isinstance(value, expected_type):
                return False, f"参数 '{self.name}' 应为{self.param_type.value},得到{type(value).__name__}"

        # enum检查
        if self.enum is not None and value not in self.enum:
            return False, f"参数 '{self.name}' 必须是 {self.enum} 之一,得到 {value}"

        # 范围检查
        if self.param_type in (ParameterType.NUMBER, ParameterType.INTEGER):
            if self.minimum is not None and value < self.minimum:
                return False, f"参数 '{self.name}' 不能小于 {self.minimum}"
            if self.maximum is not None and value > self.maximum:
                return False, f"参数 '{self.name}' 不能大于 {self.maximum}"

        return True, None


# ============================================================================
# 工具定义
# ============================================================================

@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    func: Optional[Callable] = None
    category: str = "general"
    timeout: float = 30.0
    dangerous: bool = False  # 标记危险工具(如文件写入/HTTP)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "category": self.category,
            "timeout": self.timeout,
            "dangerous": self.dangerous,
            "enabled": self.enabled,
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为OpenAI Function Calling schema"""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {
                "type": param.param_type.value,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ============================================================================
# 工具调用请求与结果
# ============================================================================

@dataclass
class ToolCallRequest:
    """LLM发起的工具调用请求"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = ""
    reasoning: Optional[str] = None  # LLM的推理过程

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "reasoning": self.reasoning,
        }


class ToolCallStatus(Enum):
    """工具调用状态"""
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"


@dataclass
class ToolCallResult:
    """工具执行结果"""
    call_id: str
    tool_name: str
    status: ToolCallStatus
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        return self.status == ToolCallStatus.SUCCESS


# ============================================================================
# 工具注册表
# ============================================================================

class ToolRegistry:
    """
    工具注册表

    职责:
        1. 注册/注销工具
        2. 按名称/类别查找
        3. 生成OpenAI schema列表
        4. 线程安全
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.RLock()
        self._register_builtin_tools()

    def register(self, tool: ToolDefinition) -> bool:
        """注册工具"""
        with self._lock:
            if tool.name in self._tools:
                logger.warning(f"[Tools] 工具 {tool.name} 已存在,覆盖")
            self._tools[tool.name] = tool
            logger.info(f"[Tools] 注册工具: {tool.name}")
            return True

    def unregister(self, name: str) -> bool:
        """注销工具"""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具"""
        with self._lock:
            return self._tools.get(name)

    def list(self, category: Optional[str] = None) -> List[str]:
        """列出工具名"""
        with self._lock:
            if category:
                return [name for name, t in self._tools.items()
                       if t.category == category and t.enabled]
            return [name for name, t in self._tools.items() if t.enabled]

    def list_all(self) -> List[ToolDefinition]:
        """列出所有工具定义"""
        with self._lock:
            return list(self._tools.values())

    def count(self) -> int:
        with self._lock:
            return len(self._tools)

    def to_openai_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """生成OpenAI Function Calling schema列表"""
        with self._lock:
            schemas = []
            for tool in self._tools.values():
                if not tool.enabled:
                    continue
                if category and tool.category != category:
                    continue
                schemas.append(tool.to_openai_schema())
            return schemas

    def clear(self):
        """清空所有工具"""
        with self._lock:
            self._tools.clear()

    # ========================================================================
    # 内置工具
    # ========================================================================

    def _register_builtin_tools(self):
        """注册内置工具"""
        # 1. 计算器
        self.register(ToolDefinition(
            name="calculator",
            description="数学计算器,支持基本运算和数学函数",
            parameters=[
                ToolParameter(name="expression", param_type=ParameterType.STRING,
                             description="数学表达式,如 '2+3*4' 或 'sqrt(16)'",
                             required=True),
            ],
            func=_tool_calculator,
            category="math",
            timeout=5.0,
        ))

        # 2. 时间日期
        self.register(ToolDefinition(
            name="datetime",
            description="获取当前日期和时间",
            parameters=[
                ToolParameter(name="format", param_type=ParameterType.STRING,
                             description="时间格式 (strftime),默认 '%Y-%m-%d %H:%M:%S'",
                             required=False, default="%Y-%m-%d %H:%M:%S"),
            ],
            func=_tool_datetime,
            category="utility",
            timeout=2.0,
        ))

        # 3. 文本处理
        self.register(ToolDefinition(
            name="text_length",
            description="计算文本长度(字符数/单词数)",
            parameters=[
                ToolParameter(name="text", param_type=ParameterType.STRING,
                             description="要计算的文本",
                             required=True),
                ToolParameter(name="mode", param_type=ParameterType.STRING,
                             description="计算模式",
                             required=False, default="chars",
                             enum=["chars", "words", "lines"]),
            ],
            func=_tool_text_length,
            category="text",
            timeout=2.0,
        ))

        # 4. 文本统计
        self.register(ToolDefinition(
            name="text_stats",
            description="文本统计信息(字符/单词/句子/段落)",
            parameters=[
                ToolParameter(name="text", param_type=ParameterType.STRING,
                             description="要统计的文本",
                             required=True),
            ],
            func=_tool_text_stats,
            category="text",
            timeout=2.0,
        ))

        # 5. HTTP GET
        self.register(ToolDefinition(
            name="http_get",
            description="发起HTTP GET请求(仅用于只读API)",
            parameters=[
                ToolParameter(name="url", param_type=ParameterType.STRING,
                             description="请求URL",
                             required=True),
                ToolParameter(name="timeout", param_type=ParameterType.NUMBER,
                             description="超时秒数",
                             required=False, default=10.0,
                             minimum=1.0, maximum=60.0),
            ],
            func=_tool_http_get,
            category="network",
            timeout=30.0,
            dangerous=True,
        ))

        # 6. JSON解析
        self.register(ToolDefinition(
            name="json_parse",
            description="解析JSON字符串",
            parameters=[
                ToolParameter(name="text", param_type=ParameterType.STRING,
                             description="JSON字符串",
                             required=True),
            ],
            func=_tool_json_parse,
            category="utility",
            timeout=2.0,
        ))


# ============================================================================
# 内置工具实现
# ============================================================================

def _tool_calculator(expression: str) -> Dict[str, Any]:
    """计算器工具"""
    # 安全检查:只允许数字、运算符、函数名、括号
    allowed = re.match(r'^[\d+\-*/().,\s\w]+$', expression)
    if not allowed:
        raise ValueError("表达式包含非法字符")

    # 允许的函数
    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "pi": math.pi, "e": math.e,
        "__builtins__": {},
    }
    try:
        result = eval(expression, safe_dict, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        raise ValueError(f"计算失败: {e}")


def _tool_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> Dict[str, Any]:
    """日期时间工具"""
    now = time.time()
    import datetime
    dt = datetime.datetime.fromtimestamp(now)
    try:
        formatted = dt.strftime(format)
    except Exception as e:
        raise ValueError(f"格式化失败: {e}")
    return {
        "timestamp": now,
        "formatted": formatted,
        "iso": dt.isoformat(),
    }


def _tool_text_length(text: str, mode: str = "chars") -> Dict[str, Any]:
    """文本长度工具"""
    if mode == "chars":
        length = len(text)
    elif mode == "words":
        length = len(text.split())
    elif mode == "lines":
        length = len(text.splitlines())
    else:
        raise ValueError(f"未知模式: {mode}")
    return {"text": text[:50] + "..." if len(text) > 50 else text,
            "mode": mode, "length": length}


def _tool_text_stats(text: str) -> Dict[str, Any]:
    """文本统计工具"""
    import re as _re
    chars = len(text)
    words = len(text.split())
    sentences = len(_re.findall(r'[.!?。！？]+', text))
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    return {
        "chars": chars,
        "words": words,
        "sentences": sentences,
        "paragraphs": paragraphs,
    }


def _tool_http_get(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """HTTP GET工具"""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL必须以http://或https://开头")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SEVA/0.62.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            return {
                "status": resp.status,
                "url": url,
                "content_length": len(content),
                "content_preview": content[:500],
            }
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP错误: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise ValueError(f"URL错误: {e.reason}")
    except Exception as e:
        raise ValueError(f"请求失败: {e}")


def _tool_json_parse(text: str) -> Dict[str, Any]:
    """JSON解析工具"""
    try:
        data = json.loads(text)
        return {"parsed": data, "type": type(data).__name__}
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")


# ============================================================================
# 工具执行器
# ============================================================================

class ToolExecutor:
    """
    工具执行器

    职责:
        1. 参数验证
        2. 执行工具(带超时)
        3. 错误处理
        4. 结果封装
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._lock = threading.RLock()
        self.stats = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "timeout_calls": 0,
            "total_latency_ms": 0.0,
        }
        self._stats_lock = threading.RLock()

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行工具调用"""
        call_id = request.call_id or f"call_{int(time.time()*1000)}"
        t0 = time.time()
        result = ToolCallResult(
            call_id=call_id,
            tool_name=request.tool_name,
            status=ToolCallStatus.SUCCESS,
        )

        # 查找工具
        tool = self.registry.get(request.tool_name)
        if tool is None:
            result.status = ToolCallStatus.NOT_FOUND
            result.error = f"工具 '{request.tool_name}' 不存在"
            result.latency_ms = (time.time() - t0) * 1000
            self._record_call(result)
            return result

        if not tool.enabled:
            result.status = ToolCallStatus.DISABLED
            result.error = f"工具 '{request.tool_name}' 已禁用"
            result.latency_ms = (time.time() - t0) * 1000
            self._record_call(result)
            return result

        # 参数验证
        for param in tool.parameters:
            value = request.arguments.get(param.name, param.default)
            valid, err = param.validate(value)
            if not valid:
                result.status = ToolCallStatus.VALIDATION_ERROR
                result.error = err
                result.latency_ms = (time.time() - t0) * 1000
                self._record_call(result)
                return result

        # 执行
        try:
            if tool.func is None:
                result.status = ToolCallStatus.EXECUTION_ERROR
                result.error = "工具没有实现函数"
            else:
                # 准备参数
                kwargs = {}
                for param in tool.parameters:
                    if param.name in request.arguments:
                        kwargs[param.name] = request.arguments[param.name]
                    elif param.default is not None:
                        kwargs[param.name] = param.default

                # 带超时执行
                exec_result = self._execute_with_timeout(
                    tool.func, kwargs, tool.timeout
                )
                result.result = exec_result
        except TimeoutError as e:
            result.status = ToolCallStatus.TIMEOUT
            result.error = f"工具执行超时: {e}"
        except Exception as e:
            result.status = ToolCallStatus.EXECUTION_ERROR
            result.error = f"{type(e).__name__}: {e}"

        result.latency_ms = (time.time() - t0) * 1000
        self._record_call(result)
        return result

    def _execute_with_timeout(self, func: Callable, kwargs: Dict,
                               timeout: float) -> Any:
        """带超时执行函数"""
        if timeout <= 0:
            return func(**kwargs)

        result_container: List[Any] = [None]
        error_container: List[Exception] = []
        done = threading.Event()

        def worker():
            try:
                result_container[0] = func(**kwargs)
            except Exception as e:
                error_container.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"执行超过 {timeout}s")
        if error_container:
            raise error_container[0]
        return result_container[0]

    def _record_call(self, result: ToolCallResult):
        """记录调用"""
        with self._stats_lock:
            self.stats["total_calls"] += 1
            self.stats["total_latency_ms"] += result.latency_ms
            if result.success:
                self.stats["success_calls"] += 1
            elif result.status == ToolCallStatus.TIMEOUT:
                self.stats["timeout_calls"] += 1
            else:
                self.stats["error_calls"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self.stats)
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["success_calls"] / stats["total_calls"]
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_latency_ms"] = 0.0
        return stats


# ============================================================================
# LLM驱动的工具调用循环
# ============================================================================

class ToolCallLoop:
    """
    LLM驱动的工具调用循环 (ReAct模式)

    流程:
        1. 用户消息 + 工具schema → LLM
        2. LLM返回工具调用请求 (JSON)
        3. 执行工具
        4. 工具结果 → LLM
        5. LLM返回最终回答 或 继续调用工具
        6. 重复直到完成或达到最大轮次
    """

    def __init__(self, registry: ToolRegistry,
                 executor: Optional[ToolExecutor] = None,
                 max_iterations: int = 5,
                 llm_client=None):
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.max_iterations = max_iterations
        self._client = llm_client
        self._lock = threading.RLock()

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def run(self, user_message: str,
            system_prompt: Optional[str] = None,
            enable_tools: Optional[List[str]] = None,
            use_cache: bool = True) -> Dict[str, Any]:
        """
        运行工具调用循环

        Args:
            user_message: 用户消息
            system_prompt: 系统提示
            enable_tools: 启用的工具列表(None=全部)
            use_cache: 是否使用LLM缓存

        Returns:
            {
                "answer": 最终回答,
                "tool_calls": [ToolCallResult...],
                "iterations": 轮次数,
                "total_tokens": 总token数,
            }
        """
        # 构造工具schema
        schemas = self.registry.to_openai_schemas()
        if enable_tools:
            schemas = [s for s in schemas
                      if s["function"]["name"] in enable_tools]

        if not schemas:
            return {
                "answer": "没有可用的工具",
                "tool_calls": [],
                "iterations": 0,
                "total_tokens": 0,
            }

        # 系统提示
        sys_content = system_prompt or (
            "你是一个能使用工具的助手。当需要使用工具时,请返回JSON格式的工具调用:\n"
            '{"tool": "工具名", "arguments": {...}, "reasoning": "为什么调用"}\n'
            "如果你已经有足够信息回答,直接返回文本回答(不包含JSON)。"
        )
        tools_desc = "\n".join(
            f"- {s['function']['name']}: {s['function']['description']}"
            for s in schemas
        )
        sys_content += f"\n\n可用工具:\n{tools_desc}"

        messages = [
            Message.system(sys_content),
            Message.user(user_message),
        ]

        tool_calls: List[ToolCallResult] = []
        total_tokens = 0
        answer = ""

        for iteration in range(self.max_iterations):
            # 调用LLM
            try:
                client = self.client
                original_cache = client.use_cache
                client.use_cache = use_cache
                try:
                    response = client.chat(messages, temperature=0.1, max_tokens=512)
                finally:
                    client.use_cache = original_cache
                total_tokens += response.usage.total_tokens
            except LLMError as e:
                logger.warning(f"[ToolLoop] LLM调用失败: {e}")
                answer = f"LLM调用失败: {e}"
                break
            except Exception as e:
                logger.error(f"[ToolLoop] 异常: {e}")
                answer = f"异常: {e}"
                break

            llm_output = response.content.strip()

            # 尝试解析工具调用
            parsed = parse_json_output(llm_output)
            if parsed and "tool" in parsed:
                # 执行工具调用
                tool_name = parsed["tool"]
                arguments = parsed.get("arguments", {})
                reasoning = parsed.get("reasoning")

                request = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=arguments,
                    reasoning=reasoning,
                )
                result = self.executor.execute(request)
                tool_calls.append(result)

                # 将工具结果加入对话
                messages.append(Message.assistant(llm_output))
                if result.success:
                    tool_msg = f"工具 {tool_name} 执行成功:\n{json.dumps(result.result, ensure_ascii=False)}"
                else:
                    tool_msg = f"工具 {tool_name} 执行失败: {result.error}"
                messages.append(Message.user(tool_msg))
                continue
            else:
                # 没有工具调用,LLM给出最终回答
                answer = llm_output
                break
        else:
            answer = "达到最大轮次限制,最后输出: " + (llm_output if 'llm_output' in dir() else "")

        return {
            "answer": answer,
            "tool_calls": [r.to_dict() for r in tool_calls],
            "iterations": len(tool_calls),
            "total_tokens": total_tokens,
        }


# ============================================================================
# 全局单例
# ============================================================================

_global_registry: Optional[ToolRegistry] = None
_global_executor: Optional[ToolExecutor] = None
_global_lock = threading.RLock()


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    global _global_registry
    with _global_lock:
        if _global_registry is None:
            _global_registry = ToolRegistry()
        return _global_registry


def get_tool_executor() -> ToolExecutor:
    """获取全局工具执行器"""
    global _global_executor
    with _global_lock:
        if _global_executor is None:
            _global_executor = ToolExecutor(get_tool_registry())
        return _global_executor


def reset_tools():
    """重置全局工具"""
    global _global_registry, _global_executor
    with _global_lock:
        _global_registry = None
        _global_executor = None
