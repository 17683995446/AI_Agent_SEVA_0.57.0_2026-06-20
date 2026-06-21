"""
Agent基类与生命周期管理（阶段64）

对应第一性原理：
    1. Agent = 认知引擎（LLM）+ 控制骨架（SEVA）+ 工具能力（Tools）+ 记忆（Memory）
    2. 生命周期是Agent的核心：创建→配置→启动→运行→暂停→恢复→停止→销毁
    3. 状态机驱动：每个状态有明确的允许动作和转换条件
    4. 配置与实现分离：同一Agent可通过不同配置适配不同场景
    5. 可观测性：所有状态转换和运行事件可被监控

模块组成：
    - AgentState: Agent状态枚举
    - AgentConfig: Agent配置 (名称/描述/模型/工具/记忆/超时等)
    - AgentEvent/AgentEventType: Agent事件 (状态转换/工具调用/错误等)
    - AgentStats: Agent运行统计
    - AgentContext: Agent运行时上下文 (会话/记忆/工具/历史)
    - BaseAgent: Agent基类 (生命周期/状态机/事件/钩子)
    - AgentRunner: Agent运行器 (启动/停止/监控)
    - 全局单例: get_agent_runner
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import Message, LLMError, get_llm_client
from ..tools import (
    ToolCallLoop, ToolCallRequest, ToolCallResult, ToolCallStatus,
    get_tool_registry, get_tool_executor,
)
from ..memory_llm import get_memory_manager, MemoryType, MemoryEntry
from ..planner import Goal, Plan, PlanResult, AutonomousAgent


logger = logging.getLogger("seva.agents.base")


# ============================================================================
# Agent状态枚举
# ============================================================================

class AgentState(Enum):
    """Agent生命周期状态"""
    CREATED = "created"            # 已创建(未配置)
    CONFIGURED = "configured"      # 已配置(未启动)
    RUNNING = "running"            # 运行中
    PAUSED = "paused"              # 已暂停
    STOPPED = "stopped"            # 已停止(可重启)
    TERMINATED = "terminated"      # 已终止(不可重启)
    ERROR = "error"                # 错误状态


# 状态转换图
# CREATED → CONFIGURED → RUNNING ↔ PAUSED
#                ↓             ↓
#             STOPPED ← ────────┘
#                ↓
#            TERMINATED
# 任何状态 → ERROR (错误)
# ERROR → CONFIGURED (恢复)
_STATE_TRANSITIONS = {
    AgentState.CREATED: {AgentState.CONFIGURED, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.CONFIGURED: {AgentState.RUNNING, AgentState.STOPPED, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.RUNNING: {AgentState.PAUSED, AgentState.STOPPED, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.PAUSED: {AgentState.RUNNING, AgentState.STOPPED, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.STOPPED: {AgentState.CONFIGURED, AgentState.RUNNING, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.ERROR: {AgentState.CONFIGURED, AgentState.TERMINATED},
    AgentState.TERMINATED: set(),  # 终态
}


# ============================================================================
# Agent事件
# ============================================================================

class AgentEventType(Enum):
    """Agent事件类型"""
    STATE_CHANGE = "state_change"        # 状态转换
    CONFIG_UPDATE = "config_update"      # 配置更新
    MESSAGE = "message"                  # 消息
    TOOL_CALL = "tool_call"              # 工具调用
    ERROR = "error"                      # 错误
    LIFECYCLE = "lifecycle"              # 生命周期事件
    METRIC = "metric"                    # 指标


@dataclass
class AgentEvent:
    """Agent事件"""
    agent_id: str
    event_type: AgentEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ============================================================================
# Agent配置
# ============================================================================

@dataclass
class AgentConfig:
    """Agent配置"""
    name: str = "base_agent"
    description: str = ""
    version: str = "1.0.0"
    # LLM配置
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    # 工具配置
    enable_tools: bool = True
    enabled_tools: Optional[List[str]] = None  # None=全部
    tool_max_iterations: int = 5
    # 记忆配置
    enable_memory: bool = True
    memory_capacity: int = 100
    # 规划配置
    enable_planning: bool = False
    max_replans: int = 2
    max_iterations: int = 10
    # 运行配置
    timeout: float = 60.0
    max_retries: int = 2
    auto_save: bool = True
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "enable_tools": self.enable_tools,
            "enabled_tools": self.enabled_tools,
            "tool_max_iterations": self.tool_max_iterations,
            "enable_memory": self.enable_memory,
            "memory_capacity": self.memory_capacity,
            "enable_planning": self.enable_planning,
            "max_replans": self.max_replans,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "auto_save": self.auto_save,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(
            name=data.get("name", "base_agent"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            model=data.get("model"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1024),
            system_prompt=data.get("system_prompt"),
            enable_tools=data.get("enable_tools", True),
            enabled_tools=data.get("enabled_tools"),
            tool_max_iterations=data.get("tool_max_iterations", 5),
            enable_memory=data.get("enable_memory", True),
            memory_capacity=data.get("memory_capacity", 100),
            enable_planning=data.get("enable_planning", False),
            max_replans=data.get("max_replans", 2),
            max_iterations=data.get("max_iterations", 10),
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 2),
            auto_save=data.get("auto_save", True),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Agent运行统计
# ============================================================================

@dataclass
class AgentStats:
    """Agent运行统计"""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    last_active_at: Optional[float] = None
    stopped_at: Optional[float] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    state_transitions: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        uptime = 0.0
        if self.started_at:
            end = self.stopped_at or time.time()
            uptime = end - self.started_at
        return {
            "created_at": self.created_at,
            "started_at": self.started_at,
            "last_active_at": self.last_active_at,
            "stopped_at": self.stopped_at,
            "uptime": uptime,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": self.total_latency_ms,
            "state_transitions": self.state_transitions,
            "errors": self.errors,
            "success_rate": (self.successful_runs / self.total_runs
                            if self.total_runs > 0 else 0.0),
            "avg_latency_ms": (self.total_latency_ms / self.total_runs
                              if self.total_runs > 0 else 0.0),
        }


# ============================================================================
# Agent运行时上下文
# ============================================================================

@dataclass
class AgentContext:
    """Agent运行时上下文"""
    agent_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({"role": role, "content": content})

    def add_user_message(self, content: str):
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.add_message("assistant", content)

    def add_system_message(self, content: str):
        self.add_message("system", content)

    def clear_messages(self):
        self.messages.clear()

    def get_messages(self, max_count: Optional[int] = None) -> List[Dict[str, str]]:
        """获取消息列表"""
        if max_count and len(self.messages) > max_count:
            return self.messages[-max_count:]
        return list(self.messages)

    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "messages": self.messages,
            "variables": self.variables,
            "session_id": self.session_id,
            "created_at": self.created_at,
        }


# ============================================================================
# Agent运行结果
# ============================================================================

@dataclass
class AgentRunResult:
    """Agent运行结果"""
    agent_id: str
    session_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Agent基类
# ============================================================================

class BaseAgent:
    """
    Agent基类

    提供完整的生命周期管理:
        - 状态机驱动 (CREATED→CONFIGURED→RUNNING↔PAUSED→STOPPED→TERMINATED)
        - 事件系统 (状态转换/工具调用/错误等)
        - 钩子机制 (before_run/after_run/on_error等)
        - 统计与监控
        - 上下文管理 (消息/变量/会话)

    子类需要实现:
        - _run_impl(input, context) -> AgentRunResult  核心运行逻辑
    """

    def __init__(self, agent_id: Optional[str] = None,
                 config: Optional[AgentConfig] = None):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config or AgentConfig()
        self.state = AgentState.CREATED
        self.stats = AgentStats()
        self.context = AgentContext(agent_id=self.agent_id)
        self._event_listeners: List[Callable[[AgentEvent], None]] = []
        self._hooks: Dict[str, List[Callable]] = {
            "before_configure": [],
            "after_configure": [],
            "before_start": [],
            "after_start": [],
            "before_run": [],
            "after_run": [],
            "before_pause": [],
            "after_pause": [],
            "before_resume": [],
            "after_resume": [],
            "before_stop": [],
            "after_stop": [],
            "on_error": [],
        }
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 默认不暂停

        # 组件引用(延迟初始化)
        self._llm_client = None
        self._tool_loop = None
        self._memory_manager = None
        self._autonomous_agent = None

    # ========================================================================
    # 状态管理
    # ========================================================================

    def _transition_to(self, new_state: AgentState) -> bool:
        """状态转换"""
        with self._lock:
            allowed = _STATE_TRANSITIONS.get(self.state, set())
            if new_state not in allowed:
                logger.warning(
                    f"[Agent {self.agent_id}] 非法状态转换: "
                    f"{self.state.value} → {new_state.value}"
                )
                return False

            old_state = self.state
            self.state = new_state
            self.stats.state_transitions += 1

            # 更新时间戳
            if new_state == AgentState.RUNNING and self.stats.started_at is None:
                self.stats.started_at = time.time()
            elif new_state == AgentState.STOPPED:
                self.stats.stopped_at = time.time()
            elif new_state in (AgentState.RUNNING, AgentState.PAUSED):
                self.stats.last_active_at = time.time()

            # 发出事件
            self._emit_event(AgentEventType.STATE_CHANGE, {
                "from": old_state.value,
                "to": new_state.value,
            })
            logger.info(f"[Agent {self.agent_id}] 状态: {old_state.value} → {new_state.value}")
            return True

    def can_transition_to(self, new_state: AgentState) -> bool:
        """检查是否可以转换到新状态"""
        with self._lock:
            return new_state in _STATE_TRANSITIONS.get(self.state, set())

    # ========================================================================
    # 生命周期方法
    # ========================================================================

    def configure(self, config: Optional[AgentConfig] = None) -> bool:
        """配置Agent"""
        if config is not None:
            self.config = config

        self._run_hooks("before_configure")
        success = self._transition_to(AgentState.CONFIGURED)
        if success:
            self._run_hooks("after_configure")
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "configured"})
        return success

    def start(self) -> bool:
        """启动Agent"""
        if self.state == AgentState.CREATED:
            self.configure()

        self._run_hooks("before_start")
        success = self._transition_to(AgentState.RUNNING)
        if success:
            self._stop_event.clear()
            self._pause_event.set()
            self._run_hooks("after_start")
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "started"})
        return success

    def pause(self) -> bool:
        """暂停Agent"""
        self._run_hooks("before_pause")
        success = self._transition_to(AgentState.PAUSED)
        if success:
            self._pause_event.clear()
            self._run_hooks("after_pause")
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "paused"})
        return success

    def resume(self) -> bool:
        """恢复Agent"""
        self._run_hooks("before_resume")
        success = self._transition_to(AgentState.RUNNING)
        if success:
            self._pause_event.set()
            self._run_hooks("after_resume")
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "resumed"})
        return success

    def stop(self) -> bool:
        """停止Agent"""
        self._run_hooks("before_stop")
        self._stop_event.set()
        self._pause_event.set()  # 解除暂停以让等待的线程退出
        success = self._transition_to(AgentState.STOPPED)
        if success:
            self._run_hooks("after_stop")
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "stopped"})
        return success

    def terminate(self) -> bool:
        """终止Agent(不可恢复)"""
        self._stop_event.set()
        self._pause_event.set()
        success = self._transition_to(AgentState.TERMINATED)
        if success:
            self._emit_event(AgentEventType.LIFECYCLE, {"action": "terminated"})
        return success

    def reset(self) -> bool:
        """重置Agent到CONFIGURED状态"""
        if self.state == AgentState.TERMINATED:
            return False
        # 先停止
        if self.state == AgentState.RUNNING:
            self.stop()
        # 转到CONFIGURED
        if self.state in (AgentState.STOPPED, AgentState.ERROR):
            return self._transition_to(AgentState.CONFIGURED)
        return False

    # ========================================================================
    # 运行方法
    # ========================================================================

    def run(self, input: Union[str, Dict[str, Any]],
            **kwargs) -> AgentRunResult:
        """
        运行Agent

        Args:
            input: 输入(字符串或字典)
            **kwargs: 额外参数

        Returns:
            运行结果
        """
        # 状态检查
        if self.state == AgentState.CREATED:
            self.configure()
        if self.state in (AgentState.CONFIGURED, AgentState.STOPPED, AgentState.PAUSED):
            self.start()
        if self.state != AgentState.RUNNING:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error=f"Agent状态不允许运行: {self.state.value}",
            )

        # 检查是否已停止
        if self._stop_event.is_set():
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error="Agent已被停止",
            )

        t0 = time.time()
        self._run_hooks("before_run", input=input)

        # 更新统计
        with self._lock:
            self.stats.total_runs += 1
            self.stats.last_active_at = time.time()

        result = AgentRunResult(
            agent_id=self.agent_id,
            session_id=self.context.session_id,
            success=False,
        )

        try:
            # 调用子类实现
            result = self._run_impl(input, **kwargs)
            result.agent_id = self.agent_id
            result.session_id = self.context.session_id

            # 更新统计
            with self._lock:
                if result.success:
                    self.stats.successful_runs += 1
                else:
                    self.stats.failed_runs += 1
                self.stats.total_tokens += result.tokens_used
                self.stats.total_tool_calls += len(result.tool_calls)
                self.stats.total_latency_ms += result.latency_ms

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {e}"
            logger.error(f"[Agent {self.agent_id}] 运行异常: {e}")
            with self._lock:
                self.stats.failed_runs += 1
                self.stats.errors += 1
            self._emit_event(AgentEventType.ERROR, {
                "error": str(e),
                "type": type(e).__name__,
            })
            self._run_hooks("on_error", error=e)
            # 转到错误状态
            self._transition_to(AgentState.ERROR)

        result.latency_ms = (time.time() - t0) * 1000
        self._run_hooks("after_run", result=result)
        return result

    def _run_impl(self, input: Union[str, Dict[str, Any]],
                  **kwargs) -> AgentRunResult:
        """
        子类实现的核心运行逻辑

        必须返回AgentRunResult
        """
        raise NotImplementedError("子类必须实现_run_impl方法")

    # ========================================================================
    # 组件访问
    # ========================================================================

    @property
    def llm_client(self):
        """LLM客户端"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def tool_loop(self) -> ToolCallLoop:
        """工具调用循环"""
        if self._tool_loop is None:
            self._tool_loop = ToolCallLoop(
                get_tool_registry(),
                get_tool_executor(),
                max_iterations=self.config.tool_max_iterations,
            )
        return self._tool_loop

    @property
    def memory_manager(self):
        """记忆管理器"""
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager()
        return self._memory_manager

    @property
    def autonomous_agent(self) -> AutonomousAgent:
        """自主Agent(用于规划)"""
        if self._autonomous_agent is None:
            from .planner import get_autonomous_agent
            self._autonomous_agent = get_autonomous_agent()
        return self._autonomous_agent

    # ========================================================================
    # 事件系统
    # ========================================================================

    def add_event_listener(self, listener: Callable[[AgentEvent], None]):
        """添加事件监听器"""
        with self._lock:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[AgentEvent], None]):
        """移除事件监听器"""
        with self._lock:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

    def _emit_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """发出事件"""
        event = AgentEvent(
            agent_id=self.agent_id,
            event_type=event_type,
            data=data,
        )
        listeners = list(self._event_listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"[Agent {self.agent_id}] 事件监听器异常: {e}")

    # ========================================================================
    # 钩子机制
    # ========================================================================

    def add_hook(self, hook_name: str, callback: Callable):
        """添加钩子"""
        with self._lock:
            if hook_name in self._hooks:
                self._hooks[hook_name].append(callback)
            else:
                logger.warning(f"未知钩子: {hook_name}")

    def remove_hook(self, hook_name: str, callback: Callable):
        """移除钩子"""
        with self._lock:
            if hook_name in self._hooks and callback in self._hooks[hook_name]:
                self._hooks[hook_name].remove(callback)

    def _run_hooks(self, hook_name: str, **kwargs):
        """运行钩子"""
        with self._lock:
            hooks = list(self._hooks.get(hook_name, []))
        for hook in hooks:
            try:
                hook(self, **kwargs)
            except Exception as e:
                logger.warning(f"[Agent {self.agent_id}] 钩子 {hook_name} 异常: {e}")

    # ========================================================================
    # 上下文管理
    # ========================================================================

    def add_message(self, role: str, content: str):
        """添加消息到上下文"""
        self.context.add_message(role, content)

    def clear_context(self):
        """清空上下文"""
        self.context.clear_messages()
        self.context.variables.clear()

    def set_variable(self, key: str, value: Any):
        """设置变量"""
        self.context.set_variable(key, value)

    def get_variable(self, key: str, default: Any = None) -> Any:
        """获取变量"""
        return self.context.get_variable(key, default)

    # ========================================================================
    # 记忆操作
    # ========================================================================

    def remember(self, content: str, memory_type: MemoryType = MemoryType.EPISODIC,
                 importance: float = 0.5):
        """存储到记忆"""
        if self.config.enable_memory:
            try:
                self.memory_manager.add(content, memory_type,
                                       importance=importance,
                                       auto_summarize=False)
            except Exception as e:
                logger.warning(f"[Agent {self.agent_id}] 记忆存储失败: {e}")

    def recall(self, query: str, limit: int = 5) -> List[Any]:
        """从记忆检索"""
        if not self.config.enable_memory:
            return []
        try:
            return self.memory_manager.retrieve(query, limit=limit)
        except Exception as e:
            logger.warning(f"[Agent {self.agent_id}] 记忆检索失败: {e}")
            return []

    # ========================================================================
    # 状态查询
    # ========================================================================

    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING

    def is_paused(self) -> bool:
        return self.state == AgentState.PAUSED

    def is_stopped(self) -> bool:
        return self.state in (AgentState.STOPPED, AgentState.TERMINATED)

    def is_terminated(self) -> bool:
        return self.state == AgentState.TERMINATED

    def get_state(self) -> AgentState:
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self.stats.to_dict()

    def get_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "state": self.state.value,
            "session_id": self.context.session_id,
            "stats": self.get_stats(),
            "config": self.config.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化Agent状态(不含运行时组件)"""
        return {
            "agent_id": self.agent_id,
            "config": self.config.to_dict(),
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "context": self.context.to_dict(),
        }


# ============================================================================
# Agent运行器
# ============================================================================

class AgentRunner:
    """
    Agent运行器

    职责:
        1. 管理多个Agent实例
        2. 启动/停止/监控Agent
        3. 收集Agent事件
        4. 提供Agent查询接口
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._events: List[AgentEvent] = []
        self._max_events: int = 1000
        self._lock = threading.RLock()

    def register(self, agent: BaseAgent) -> bool:
        """注册Agent"""
        with self._lock:
            if agent.agent_id in self._agents:
                logger.warning(f"[Runner] Agent {agent.agent_id} 已存在,覆盖")
            self._agents[agent.agent_id] = agent
            # 添加事件监听器
            agent.add_event_listener(self._on_agent_event)
            logger.info(f"[Runner] 注册Agent: {agent.agent_id}")
            return True

    def unregister(self, agent_id: str) -> bool:
        """注销Agent"""
        with self._lock:
            agent = self._agents.pop(agent_id, None)
            if agent:
                agent.remove_event_listener(self._on_agent_event)
                logger.info(f"[Runner] 注销Agent: {agent_id}")
                return True
            return False

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """获取Agent"""
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self, state: Optional[AgentState] = None) -> List[str]:
        """列出Agent ID"""
        with self._lock:
            if state:
                return [aid for aid, a in self._agents.items()
                       if a.state == state]
            return list(self._agents.keys())

    def list_all(self) -> List[BaseAgent]:
        """列出所有Agent"""
        with self._lock:
            return list(self._agents.values())

    def count(self) -> int:
        with self._lock:
            return len(self._agents)

    def start_all(self) -> Dict[str, bool]:
        """启动所有Agent"""
        results = {}
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            results[agent.agent_id] = agent.start()
        return results

    def stop_all(self) -> Dict[str, bool]:
        """停止所有Agent"""
        results = {}
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            results[agent.agent_id] = agent.stop()
        return results

    def terminate_all(self) -> Dict[str, bool]:
        """终止所有Agent"""
        results = {}
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            results[agent.agent_id] = agent.terminate()
        return results

    def get_events(self, agent_id: Optional[str] = None,
                   event_type: Optional[AgentEventType] = None,
                   limit: int = 100) -> List[AgentEvent]:
        """获取事件"""
        with self._lock:
            events = list(self._events)
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_events(self):
        """清空事件"""
        with self._lock:
            self._events.clear()

    def get_report(self) -> Dict[str, Any]:
        """获取运行器报告"""
        with self._lock:
            agents = list(self._agents.values())
        state_counts = {}
        for a in agents:
            state_counts[a.state.value] = state_counts.get(a.state.value, 0) + 1
        return {
            "total_agents": len(agents),
            "states": state_counts,
            "total_events": len(self._events),
            "agents": [a.get_info() for a in agents],
        }

    def _on_agent_event(self, event: AgentEvent):
        """Agent事件回调"""
        with self._lock:
            self._events.append(event)
            # 限制事件数量
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]


# ============================================================================
# 全局单例
# ============================================================================

_global_runner: Optional[AgentRunner] = None
_global_lock = threading.RLock()


def get_agent_runner() -> AgentRunner:
    """获取全局Agent运行器"""
    global _global_runner
    with _global_lock:
        if _global_runner is None:
            _global_runner = AgentRunner()
        return _global_runner


def reset_agent_runner():
    """重置全局Agent运行器"""
    global _global_runner
    with _global_lock:
        if _global_runner:
            _global_runner.terminate_all()
        _global_runner = None
