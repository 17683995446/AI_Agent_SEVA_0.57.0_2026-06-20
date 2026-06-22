"""
工作流Agent（阶段68）

对应第一性原理：
    1. 工作流 = 多步骤任务的有序编排
    2. 每个步骤是原子性的,可独立执行和重试
    3. 步骤间有依赖关系,形成DAG(有向无环图)
    4. 状态机驱动: 每个步骤有明确的状态转换
    5. 容错性: 失败步骤可重试或跳过,不影响整体流程

模块组成：
    - StepStatus/WorkflowStatus: 步骤与工作流状态
    - WorkflowStep: 工作流步骤
    - Workflow: 工作流定义
    - WorkflowExecutor: 工作流执行器
    - WorkflowBuilder: 工作流构建器 (流式API)
    - WorkflowAgent: 工作流Agent
    - 全局单例: get_workflow_agent
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
from ..llm_loop import parse_json_output
from .base import (
    BaseAgent, AgentConfig, AgentState, AgentRunResult,
    AgentEvent, AgentEventType,
)


logger = logging.getLogger("seva.agents.workflow")


# ============================================================================
# 枚举
# ============================================================================

class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"            # 待执行
    READY = "ready"                # 就绪(依赖已完成)
    RUNNING = "running"            # 执行中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"              # 失败
    SKIPPED = "skipped"            # 已跳过
    BLOCKED = "blocked"            # 被阻塞(依赖失败)


class WorkflowStatus(Enum):
    """工作流状态"""
    DRAFT = "draft"                # 草稿
    READY = "ready"                # 就绪
    RUNNING = "running"            # 执行中
    PAUSED = "paused"              # 已暂停
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"              # 失败
    CANCELLED = "cancelled"        # 已取消


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # 依赖步骤ID
    action: Optional[str] = None  # 动作类型 (llm/tool/agent/custom)
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_key: Optional[str] = None  # 输入变量名(从上下文获取)
    output_key: Optional[str] = None  # 输出变量名(存到上下文)
    result: Optional[Any] = None
    error: Optional[str] = None
    max_retries: int = 2
    retry_count: int = 0
    timeout: float = 60.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "action": self.action,
            "parameters": self.parameters,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "result": self.result,
            "error": self.error,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(
            id=data.get("id", f"step_{uuid.uuid4().hex[:8]}"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=StepStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            action=data.get("action"),
            parameters=data.get("parameters", {}),
            input_key=data.get("input_key"),
            output_key=data.get("output_key"),
            result=data.get("result"),
            error=data.get("error"),
            max_retries=data.get("max_retries", 2),
            retry_count=data.get("retry_count", 0),
            timeout=data.get("timeout", 60.0),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)

    @property
    def can_retry(self) -> bool:
        return self.status == StepStatus.FAILED and self.retry_count < self.max_retries

    @property
    def latency_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class Workflow:
    """工作流"""
    id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    context: Dict[str, Any] = field(default_factory=dict)  # 工作流上下文(变量)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "context": self.context,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)

    @property
    def is_terminal(self) -> bool:
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def get_ready_steps(self) -> List[WorkflowStep]:
        """获取就绪步骤(依赖已完成)"""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        failed_ids = {s.id for s in self.steps if s.status == StepStatus.FAILED}
        ready = []
        for s in self.steps:
            if s.status != StepStatus.PENDING:
                continue
            # 检查依赖
            if all(dep in completed_ids for dep in s.dependencies):
                ready.append(s)
            elif any(dep in failed_ids for dep in s.dependencies):
                s.status = StepStatus.BLOCKED
        return ready

    def all_steps_terminal(self) -> bool:
        return all(s.is_terminal for s in self.steps)

    def get_failed_steps(self) -> List[WorkflowStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def get_blocked_steps(self) -> List[WorkflowStep]:
        return [s for s in self.steps if s.status == StepStatus.BLOCKED]


# ============================================================================
# 工作流执行器
# ============================================================================

class WorkflowExecutor:
    """
    工作流执行器

    职责:
        1. 按依赖顺序执行步骤
        2. 管理步骤状态转换
        3. 处理失败(重试/跳过)
        4. 维护工作流上下文
    """

    def __init__(self):
        self._action_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self.stats = {
            "workflows_executed": 0,
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
        }
        # 注册默认动作处理器
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认动作处理器"""
        self._action_handlers["noop"] = self._handle_noop
        self._action_handlers["set_var"] = self._handle_set_var
        self._action_handlers["echo"] = self._handle_echo

    def register_handler(self, action: str, handler: Callable):
        """注册动作处理器"""
        with self._lock:
            self._action_handlers[action] = handler

    def execute(self, workflow: Workflow,
                max_iterations: int = 100) -> Dict[str, Any]:
        """
        执行工作流

        Returns:
            执行结果字典
        """
        with self._lock:
            self.stats["workflows_executed"] += 1

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = time.time()

        t0 = time.time()
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # 获取就绪步骤
            ready_steps = workflow.get_ready_steps()

            if not ready_steps:
                # 没有就绪步骤,检查是否完成
                if workflow.all_steps_terminal():
                    failed = workflow.get_failed_steps()
                    blocked = workflow.get_blocked_steps()
                    if failed or blocked:
                        workflow.status = WorkflowStatus.FAILED
                    else:
                        workflow.status = WorkflowStatus.COMPLETED
                    break
                else:
                    # 有阻塞步骤,工作流失败
                    workflow.status = WorkflowStatus.FAILED
                    break

            # 执行就绪步骤
            for step in ready_steps:
                self._execute_step(step, workflow)

            workflow.metadata["iterations"] = iterations

        workflow.completed_at = time.time()
        workflow.metadata["total_latency_ms"] = (time.time() - t0) * 1000
        workflow.metadata["iterations"] = iterations

        return {
            "status": workflow.status.value,
            "progress": workflow.progress,
            "steps_total": len(workflow.steps),
            "steps_completed": sum(1 for s in workflow.steps if s.status == StepStatus.COMPLETED),
            "steps_failed": len(workflow.get_failed_steps()),
            "context": workflow.context,
            "latency_ms": (time.time() - t0) * 1000,
        }

    def _execute_step(self, step: WorkflowStep, workflow: Workflow):
        """执行单个步骤"""
        with self._lock:
            self.stats["steps_executed"] += 1

        step.status = StepStatus.RUNNING
        step.started_at = time.time()

        try:
            # 获取输入
            input_value = None
            if step.input_key:
                input_value = workflow.context.get(step.input_key)

            # 执行动作
            handler = self._action_handlers.get(step.action)
            if handler is None:
                raise ValueError(f"未知动作: {step.action}")

            result = handler(step, workflow, input_value)

            # 存储输出
            if step.output_key:
                workflow.context[step.output_key] = result

            step.result = result
            step.status = StepStatus.COMPLETED
            with self._lock:
                self.stats["steps_succeeded"] += 1

        except Exception as e:
            step.error = f"{type(e).__name__}: {e}"
            step.status = StepStatus.FAILED
            with self._lock:
                self.stats["steps_failed"] += 1

            # 重试逻辑
            if step.can_retry:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                logger.info(f"[Workflow] 步骤 {step.name} 重试 {step.retry_count}/{step.max_retries}")

        step.completed_at = time.time()

    # 默认动作处理器
    def _handle_noop(self, step: WorkflowStep, workflow: Workflow,
                     input_value: Any) -> Any:
        """空操作"""
        return None

    def _handle_set_var(self, step: WorkflowStep, workflow: Workflow,
                        input_value: Any) -> Any:
        """设置变量"""
        var_name = step.parameters.get("var_name")
        var_value = step.parameters.get("var_value")
        if var_name:
            workflow.context[var_name] = var_value
        return var_value

    def _handle_echo(self, step: WorkflowStep, workflow: Workflow,
                     input_value: Any) -> Any:
        """回显"""
        return input_value or step.parameters.get("message", "")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 工作流构建器
# ============================================================================

class WorkflowBuilder:
    """
    工作流构建器 (流式API)

    用法:
        wf = (WorkflowBuilder("my_workflow")
              .step("step1", action="noop")
              .step("step2", action="echo", dependencies=["step1"])
              .build())
    """

    def __init__(self, name: str = "", description: str = ""):
        self._workflow = Workflow(name=name, description=description)
        self._step_map: Dict[str, str] = {}  # 步骤名 -> ID

    def step(self, name: str, action: str = "noop",
             description: str = "",
             dependencies: Optional[List[str]] = None,
             parameters: Optional[Dict[str, Any]] = None,
             input_key: Optional[str] = None,
             output_key: Optional[str] = None,
             max_retries: int = 2,
             timeout: float = 60.0) -> "WorkflowBuilder":
        """添加步骤"""
        step = WorkflowStep(
            name=name,
            description=description,
            action=action,
            parameters=parameters or {},
            input_key=input_key,
            output_key=output_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        # 转换依赖名为ID
        if dependencies:
            step.dependencies = [
                self._step_map.get(dep, dep) for dep in dependencies
            ]
        self._workflow.steps.append(step)
        self._step_map[name] = step.id
        return self

    def context(self, key: str, value: Any) -> "WorkflowBuilder":
        """设置上下文变量"""
        self._workflow.context[key] = value
        return self

    def build(self) -> Workflow:
        """构建工作流"""
        self._workflow.status = WorkflowStatus.READY
        return self._workflow


# ============================================================================
# 工作流Agent配置
# ============================================================================

@dataclass
class WorkflowAgentConfig(AgentConfig):
    """工作流Agent配置"""
    name: str = "workflow_agent"
    description: str = "工作流Agent"
    max_iterations: int = 100        # 最大迭代次数
    enable_retry: bool = True        # 启用重试
    system_prompt: Optional[str] = "你是工作流编排专家,帮助用户构建和执行多步骤任务。"


# ============================================================================
# 工作流Agent
# ============================================================================

class WorkflowAgent(BaseAgent):
    """
    工作流Agent

    特性:
        1. 构建工作流 (WorkflowBuilder)
        2. 执行工作流 (WorkflowExecutor)
        3. 自定义动作处理器
        4. 工作流状态监控
    """

    def __init__(self, agent_id: Optional[str] = None,
                 config: Optional[WorkflowAgentConfig] = None):
        super().__init__(agent_id=agent_id,
                        config=config or WorkflowAgentConfig())
        self.executor = WorkflowExecutor()
        self._workflows: Dict[str, Workflow] = {}

    @property
    def wf_config(self) -> WorkflowAgentConfig:
        return self.config

    def create_workflow(self, name: str = "",
                        description: str = "") -> WorkflowBuilder:
        """创建工作流构建器"""
        return WorkflowBuilder(name=name, description=description)

    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行工作流"""
        self._workflows[workflow.id] = workflow
        return self.executor.execute(
            workflow,
            max_iterations=self.wf_config.max_iterations,
        )

    def register_action(self, action: str, handler: Callable):
        """注册自定义动作处理器"""
        self.executor.register_handler(action, handler)

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """获取工作流"""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Workflow]:
        """列出所有工作流"""
        return list(self._workflows.values())

    def _run_impl(self, input: Union[str, Dict[str, Any]],
                  **kwargs) -> AgentRunResult:
        """工作流Agent核心实现"""
        if isinstance(input, Workflow):
            # 直接执行工作流
            workflow = input
        elif isinstance(input, dict) and "workflow" in input:
            workflow = input["workflow"]
        else:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error="输入必须是Workflow对象或包含workflow键的字典",
            )

        t0 = time.time()
        result = self.execute_workflow(workflow)

        return AgentRunResult(
            agent_id=self.agent_id,
            session_id=self.context.session_id,
            success=result["status"] == "completed",
            output=f"工作流 {workflow.name} 状态: {result['status']}\n"
                  f"进度: {result['progress']*100:.1f}%\n"
                  f"步骤: {result['steps_completed']}/{result['steps_total']}完成",
            latency_ms=(time.time() - t0) * 1000,
            metadata={
                "workflow_id": workflow.id,
                "result": result,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取详细统计"""
        base_stats = super().get_stats()
        base_stats.update({
            "executor": self.executor.get_stats(),
            "workflows_count": len(self._workflows),
        })
        return base_stats


# ============================================================================
# 全局单例
# ============================================================================

_global_workflow_agent: Optional[WorkflowAgent] = None
_global_lock = threading.RLock()


def get_workflow_agent() -> WorkflowAgent:
    """获取全局工作流Agent"""
    global _global_workflow_agent
    with _global_lock:
        if _global_workflow_agent is None:
            _global_workflow_agent = WorkflowAgent()
        return _global_workflow_agent


def reset_workflow_agent():
    """重置全局工作流Agent"""
    global _global_workflow_agent
    with _global_lock:
        _global_workflow_agent = None
