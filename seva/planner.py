"""
智能决策与自主规划（阶段63）

对应第一性原理：
    1. LLM = 认知引擎（目标分解、任务规划、反思纠错）
    2. SEVA = 控制骨架（执行计划、监控进度、处理失败）
    3. 规划是分层的：目标 → 子任务 → 执行 → 反思 → 调整
    4. 反思是进化的核心：失败 → 分析 → 调整 → 重试
    5. 整合前5个阶段：llm + prompts + llm_loop + memory + tools

模块组成：
    - Goal: 目标定义 (描述/成功标准/优先级/截止时间)
    - Task/TaskStatus: 任务定义与状态
    - Plan/PlanStatus/PlanStrategy: 计划定义与策略
    - TaskResult/PlanResult: 执行结果
    - ReflectionResult: 反思结果 (问题识别/改进建议)
    - RiskLevel: 风险等级
    - Planner: 规划器 (目标分解/计划创建/难度评估)
    - PlanExecutor: 计划执行器 (任务执行/失败处理/重规划)
    - Reflector: 反思器 (结果评估/问题识别/经验学习)
    - DecisionMaker: 决策器 (下一步决策/风险评估)
    - AutonomousAgent: 自主Agent (整合所有组件)
    - 全局单例: get_planner/get_plan_executor/get_reflector/get_autonomous_agent
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

from .llm import Message, LLMError, get_llm_client
from .llm_loop import parse_json_output
from .tools import (
    ToolCallLoop, ToolCallRequest, ToolCallResult, ToolCallStatus,
    get_tool_registry, get_tool_executor,
)
from .memory_llm import get_memory_manager, MemoryType


logger = logging.getLogger("seva.planner")


# ============================================================================
# 枚举定义
# ============================================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"              # 待执行
    IN_PROGRESS = "in_progress"      # 执行中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败
    SKIPPED = "skipped"              # 已跳过
    BLOCKED = "blocked"              # 被阻塞(依赖未完成)


class PlanStatus(Enum):
    """计划状态"""
    DRAFT = "draft"                  # 草稿
    ACTIVE = "active"                # 执行中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败
    ABANDONED = "abandoned"          # 已放弃
    REPLANNED = "replanned"          # 已重规划


class PlanStrategy(Enum):
    """计划策略"""
    SEQUENTIAL = "sequential"        # 顺序执行
    PARALLEL = "parallel"            # 并行执行(无依赖的任务)
    ADAPTIVE = "adaptive"            # 自适应(根据结果动态调整)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"                      # 低风险
    MEDIUM = "medium"                # 中等风险
    HIGH = "high"                    # 高风险
    CRITICAL = "critical"            # 严重风险


class ActionType(Enum):
    """决策动作类型"""
    EXECUTE_TASK = "execute_task"          # 执行任务
    RETRY_TASK = "retry_task"              # 重试任务
    REPLAN = "replan"                      # 重新规划
    SKIP_TASK = "skip_task"                # 跳过任务
    ABORT_PLAN = "abort_plan"              # 终止计划
    COMPLETE_PLAN = "complete_plan"        # 完成计划
    REQUEST_HELP = "request_help"          # 请求帮助


# ============================================================================
# 数据类：目标
# ============================================================================

@dataclass
class Goal:
    """目标定义"""
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    description: str = ""
    success_criteria: str = ""         # 成功标准(自然语言描述)
    priority: int = 5                  # 优先级 1(最高)-10(最低)
    deadline: Optional[float] = None   # 截止时间戳
    parent_goal_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "priority": self.priority,
            "deadline": self.deadline,
            "parent_goal_id": self.parent_goal_id,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=data.get("id", f"goal_{uuid.uuid4().hex[:8]}"),
            description=data.get("description", ""),
            success_criteria=data.get("success_criteria", ""),
            priority=data.get("priority", 5),
            deadline=data.get("deadline"),
            parent_goal_id=data.get("parent_goal_id"),
            context=data.get("context", {}),
            created_at=data.get("created_at", time.time()),
        )


# ============================================================================
# 数据类：任务
# ============================================================================

@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    description: str = ""
    goal_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID列表
    tools_needed: List[str] = field(default_factory=list)  # 需要的工具名
    expected_output: str = ""           # 期望输出描述
    actual_output: Optional[str] = None # 实际输出
    error: Optional[str] = None         # 错误信息
    max_retries: int = 2                # 最大重试次数
    retry_count: int = 0                # 当前重试次数
    difficulty: float = 0.5             # 难度评估 [0, 1]
    estimated_time: float = 0.0         # 预估时间(秒)
    actual_time: float = 0.0            # 实际时间(秒)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "tools_needed": self.tools_needed,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error": self.error,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "actual_time": self.actual_time,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data.get("id", f"task_{uuid.uuid4().hex[:8]}"),
            description=data.get("description", ""),
            goal_id=data.get("goal_id", ""),
            status=TaskStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            tools_needed=data.get("tools_needed", []),
            expected_output=data.get("expected_output", ""),
            actual_output=data.get("actual_output"),
            error=data.get("error"),
            max_retries=data.get("max_retries", 2),
            retry_count=data.get("retry_count", 0),
            difficulty=data.get("difficulty", 0.5),
            estimated_time=data.get("estimated_time", 0.0),
            actual_time=data.get("actual_time", 0.0),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.status in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED
        )

    @property
    def can_retry(self) -> bool:
        """是否可以重试"""
        return (self.status == TaskStatus.FAILED
                and self.retry_count < self.max_retries)


# ============================================================================
# 数据类：计划
# ============================================================================

@dataclass
class Plan:
    """计划定义"""
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    goal: Goal = field(default_factory=Goal)
    tasks: List[Task] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    strategy: PlanStrategy = PlanStrategy.SEQUENTIAL
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    version: int = 1                   # 版本号(重规划时递增)
    parent_plan_id: Optional[str] = None  # 父计划(重规划时关联)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
            "status": self.status.value,
            "strategy": self.strategy.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "version": self.version,
            "parent_plan_id": self.parent_plan_id,
            "metadata": self.metadata,
        }

    @property
    def progress(self) -> float:
        """完成进度 [0, 1]"""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    @property
    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.status in (
            PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.ABANDONED
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def get_ready_tasks(self) -> List[Task]:
        """获取可执行的任务(依赖已完成且状态为pending)"""
        completed_ids = {t.id for t in self.tasks
                        if t.status == TaskStatus.COMPLETED}
        ready = []
        for t in self.tasks:
            if t.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in t.dependencies):
                ready.append(t)
            elif t.status == TaskStatus.PENDING:
                # 检查是否有依赖失败
                failed_deps = [dep for dep in t.dependencies
                              if self.get_task(dep)
                              and self.get_task(dep).status == TaskStatus.FAILED]
                if failed_deps:
                    t.status = TaskStatus.BLOCKED
        return ready

    def get_blocked_tasks(self) -> List[Task]:
        """获取被阻塞的任务"""
        return [t for t in self.tasks if t.status == TaskStatus.BLOCKED]

    def get_failed_tasks(self) -> List[Task]:
        """获取失败的任务"""
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def all_tasks_terminal(self) -> bool:
        """所有任务是否都处于终态"""
        return all(t.is_terminal for t in self.tasks)


# ============================================================================
# 数据类：执行结果
# ============================================================================

@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED


@dataclass
class PlanResult:
    """计划执行结果"""
    plan_id: str
    status: PlanStatus
    task_results: List[TaskResult] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    iterations: int = 0
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    final_output: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "task_results": [r.to_dict() for r in self.task_results],
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "iterations": self.iterations,
            "reflections": self.reflections,
            "final_output": self.final_output,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        return self.status == PlanStatus.COMPLETED


# ============================================================================
# 数据类：反思结果
# ============================================================================

@dataclass
class Issue:
    """问题识别"""
    severity: RiskLevel
    description: str
    task_id: Optional[str] = None
    suggestion: str = ""


@dataclass
class ReflectionResult:
    """反思结果"""
    success: bool
    summary: str = ""
    issues: List[Issue] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    should_replan: bool = False
    replan_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "issues": [
                {"severity": i.severity.value, "description": i.description,
                 "task_id": i.task_id, "suggestion": i.suggestion}
                for i in self.issues
            ],
            "improvements": self.improvements,
            "lessons_learned": self.lessons_learned,
            "should_replan": self.should_replan,
            "replan_reason": self.replan_reason,
            "timestamp": self.timestamp,
        }


# ============================================================================
# 数据类：决策动作
# ============================================================================

@dataclass
class Action:
    """决策动作"""
    action_type: ActionType
    target_task_id: Optional[str] = None
    reason: str = ""
    risk: RiskLevel = RiskLevel.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_task_id": self.target_task_id,
            "reason": self.reason,
            "risk": self.risk.value,
            "metadata": self.metadata,
        }


# ============================================================================
# 规划器
# ============================================================================

class Planner:
    """
    规划器

    职责:
        1. 目标分解: 将复杂目标分解为可执行的子任务
        2. 计划创建: 创建包含任务依赖关系的计划
        3. 难度评估: 评估任务难度
        4. 依赖识别: 识别任务间的依赖关系
    """

    DECOMPOSE_PROMPT = """你是一个任务规划专家。请将以下目标分解为具体的子任务。

目标: {goal_description}
成功标准: {success_criteria}
可用工具: {available_tools}

请返回JSON格式:
{{
    "tasks": [
        {{
            "description": "任务描述",
            "expected_output": "期望输出",
            "tools_needed": ["工具名"],
            "difficulty": 0.5,
            "estimated_time": 10.0,
            "dependencies": []
        }}
    ]
}}

要求:
1. 每个任务应该是原子性的,可独立执行
2. 任务数量在2-8个之间
3. dependencies填写依赖的任务序号(从0开始)
4. difficulty范围0.0-1.0
5. 只返回JSON,不要其他内容"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {
            "plans_created": 0,
            "tasks_decomposed": 0,
            "decomposition_failures": 0,
        }

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def decompose_goal(self, goal: Goal,
                       available_tools: Optional[List[str]] = None) -> List[Task]:
        """
        分解目标为子任务

        Args:
            goal: 目标
            available_tools: 可用工具列表

        Returns:
            任务列表
        """
        if available_tools is None:
            registry = get_tool_registry()
            available_tools = registry.list()

        tools_str = ", ".join(available_tools) if available_tools else "(无)"
        prompt = self.DECOMPOSE_PROMPT.format(
            goal_description=goal.description,
            success_criteria=goal.success_criteria or "(未指定)",
            available_tools=tools_str,
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是任务规划专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.3,
                    max_tokens=1024,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed or "tasks" not in parsed:
                logger.warning(f"[Planner] LLM分解失败,创建单任务回退")
                self.stats["decomposition_failures"] += 1
                return self._fallback_decompose(goal)

            tasks = []
            task_id_map = {}  # 序号 -> task_id
            for i, task_data in enumerate(parsed["tasks"]):
                task = Task(
                    description=task_data.get("description", f"子任务{i+1}"),
                    goal_id=goal.id,
                    expected_output=task_data.get("expected_output", ""),
                    tools_needed=task_data.get("tools_needed", []),
                    difficulty=min(1.0, max(0.0,
                        float(task_data.get("difficulty", 0.5)))),
                    estimated_time=float(task_data.get("estimated_time", 0.0)),
                    max_retries=2,
                )
                # 转换依赖序号为任务ID
                deps = task_data.get("dependencies", [])
                task_id_map[i] = task.id
                task._pending_deps = deps  # 临时存储,后面转换
                tasks.append(task)

            # 转换依赖
            for task in tasks:
                pending = getattr(task, "_pending_deps", [])
                task.dependencies = [task_id_map.get(d, "") for d in pending
                                    if d in task_id_map]
                if hasattr(task, "_pending_deps"):
                    delattr(task, "_pending_deps")

            self.stats["tasks_decomposed"] += len(tasks)
            logger.info(f"[Planner] 目标分解: {len(tasks)}个任务")
            return tasks

        except LLMError as e:
            logger.warning(f"[Planner] LLM分解失败: {e}")
            self.stats["decomposition_failures"] += 1
            return self._fallback_decompose(goal)
        except Exception as e:
            logger.error(f"[Planner] 分解异常: {e}")
            self.stats["decomposition_failures"] += 1
            return self._fallback_decompose(goal)

    def _fallback_decompose(self, goal: Goal) -> List[Task]:
        """回退分解: 创建单个任务"""
        return [Task(
            description=goal.description,
            goal_id=goal.id,
            expected_output=goal.success_criteria or "完成目标",
            difficulty=0.7,
            estimated_time=30.0,
        )]

    def create_plan(self, goal: Goal,
                    strategy: PlanStrategy = PlanStrategy.SEQUENTIAL,
                    available_tools: Optional[List[str]] = None) -> Plan:
        """创建计划"""
        with self._lock:
            tasks = self.decompose_goal(goal, available_tools)
            plan = Plan(
                goal=goal,
                tasks=tasks,
                status=PlanStatus.ACTIVE,
                strategy=strategy,
            )
            self.stats["plans_created"] += 1
            logger.info(f"[Planner] 创建计划 {plan.id}: {len(tasks)}个任务")
            return plan

    def estimate_difficulty(self, task: Task) -> float:
        """评估任务难度"""
        # 基于工具数量和描述长度简单评估
        tool_factor = min(0.3, len(task.tools_needed) * 0.1)
        desc_factor = min(0.3, len(task.description) / 200)
        return min(1.0, 0.3 + tool_factor + desc_factor)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 计划执行器
# ============================================================================

class PlanExecutor:
    """
    计划执行器

    职责:
        1. 执行计划中的任务
        2. 处理任务失败(重试/跳过/重规划)
        3. 监控执行进度
        4. 记录执行结果
    """

    def __init__(self, planner: Optional[Planner] = None,
                 tool_loop: Optional[ToolCallLoop] = None):
        self.planner = planner or Planner()
        self._tool_loop = tool_loop
        self._lock = threading.RLock()
        self.stats = {
            "plans_executed": 0,
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "replans": 0,
        }

    @property
    def tool_loop(self) -> ToolCallLoop:
        if self._tool_loop is None:
            self._tool_loop = ToolCallLoop(
                get_tool_registry(), get_tool_executor(),
                max_iterations=5,
            )
        return self._tool_loop

    def execute_plan(self, plan: Plan,
                     max_iterations: int = 10) -> PlanResult:
        """
        执行计划

        Args:
            plan: 计划
            max_iterations: 最大迭代次数

        Returns:
            计划执行结果
        """
        with self._lock:
            self.stats["plans_executed"] += 1

        result = PlanResult(
            plan_id=plan.id,
            status=PlanStatus.ACTIVE,
        )

        t0 = time.time()
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            result.iterations = iteration

            # 获取可执行任务
            ready_tasks = plan.get_ready_tasks()

            if not ready_tasks:
                # 没有可执行任务,检查是否全部完成
                if plan.all_tasks_terminal():
                    # 检查是否有失败任务
                    failed = plan.get_failed_tasks()
                    blocked = plan.get_blocked_tasks()
                    if failed or blocked:
                        plan.status = PlanStatus.FAILED
                        result.status = PlanStatus.FAILED
                        result.final_output = f"计划失败: {len(failed)}个任务失败, {len(blocked)}个被阻塞"
                    else:
                        plan.status = PlanStatus.COMPLETED
                        result.status = PlanStatus.COMPLETED
                        # 汇总输出
                        outputs = [t.actual_output for t in plan.tasks
                                  if t.actual_output]
                        result.final_output = "\n".join(outputs) if outputs else "计划完成"
                    break
                else:
                    # 有任务未完成但无可执行任务(可能被阻塞)
                    blocked = plan.get_blocked_tasks()
                    if blocked:
                        # 尝试重规划
                        logger.info(f"[Executor] {len(blocked)}个任务被阻塞,尝试重规划")
                        replanned = self.replan(plan, blocked[0])
                        if replanned:
                            continue
                        else:
                            plan.status = PlanStatus.FAILED
                            result.status = PlanStatus.FAILED
                            result.final_output = f"计划失败: {len(blocked)}个任务被阻塞且无法重规划"
                            break
                    else:
                        break

            # 执行就绪任务
            for task in ready_tasks:
                task_result = self.execute_task(task, plan)
                result.task_results.append(task_result)
                result.total_tokens += task_result.tokens_used
                result.total_latency_ms += task_result.latency_ms

                # 处理失败
                if not task_result.success:
                    if task.can_retry:
                        logger.info(f"[Executor] 任务 {task.id} 失败,准备重试({task.retry_count}/{task.max_retries})")
                        task.status = TaskStatus.PENDING
                        task.retry_count += 1
                    else:
                        logger.warning(f"[Executor] 任务 {task.id} 失败且无法重试")
                        # 检查是否需要重规划
                        if self._should_replan(plan, task):
                            replanned = self.replan(plan, task)
                            if replanned:
                                break  # 重新开始迭代
                else:
                    logger.info(f"[Executor] 任务 {task.id} 完成")

            plan.updated_at = time.time()

        result.total_latency_ms = (time.time() - t0) * 1000

        if not plan.is_terminal:
            # 达到最大迭代仍未完成
            plan.status = PlanStatus.FAILED
            result.status = PlanStatus.FAILED
            result.final_output = f"达到最大迭代次数({max_iterations})"

        return result

    def execute_task(self, task: Task, plan: Plan) -> TaskResult:
        """执行单个任务"""
        with self._lock:
            self.stats["tasks_executed"] += 1

        t0 = time.time()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()

        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.IN_PROGRESS,
        )

        try:
            # 使用ToolCallLoop执行任务
            task_prompt = self._build_task_prompt(task, plan)
            loop_result = self.tool_loop.run(
                task_prompt,
                enable_tools=task.tools_needed or None,
                use_cache=False,
            )

            result.output = loop_result.get("answer", "")
            result.tool_calls = loop_result.get("tool_calls", [])
            result.tokens_used = loop_result.get("total_tokens", 0)

            # 判断是否成功
            if loop_result.get("iterations", 0) > 0 or result.output:
                result.status = TaskStatus.COMPLETED
                task.status = TaskStatus.COMPLETED
                task.actual_output = result.output
                with self._lock:
                    self.stats["tasks_succeeded"] += 1
            else:
                result.status = TaskStatus.FAILED
                result.error = "无输出"
                task.status = TaskStatus.FAILED
                task.error = result.error
                with self._lock:
                    self.stats["tasks_failed"] += 1

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = f"{type(e).__name__}: {e}"
            task.status = TaskStatus.FAILED
            task.error = result.error
            with self._lock:
                self.stats["tasks_failed"] += 1
            logger.error(f"[Executor] 任务 {task.id} 异常: {e}")

        task.completed_at = time.time()
        task.actual_time = task.completed_at - (task.started_at or task.completed_at)
        result.latency_ms = (time.time() - t0) * 1000

        return result

    def _build_task_prompt(self, task: Task, plan: Plan) -> str:
        """构造任务执行提示"""
        parts = [f"任务: {task.description}"]
        if task.expected_output:
            parts.append(f"期望输出: {task.expected_output}")
        if plan.goal.success_criteria:
            parts.append(f"总体目标: {plan.goal.description}")
            parts.append(f"成功标准: {plan.goal.success_criteria}")
        # 已完成任务的输出作为上下文
        completed = [t for t in plan.tasks
                    if t.status == TaskStatus.COMPLETED and t.actual_output]
        if completed:
            parts.append("已完成任务:")
            for t in completed[-3:]:  # 最近3个
                parts.append(f"  - {t.description}: {t.actual_output[:100]}")
        return "\n".join(parts)

    def _should_replan(self, plan: Plan, failed_task: Task) -> bool:
        """判断是否需要重规划"""
        # 如果失败任务有依赖者,需要重规划
        for t in plan.tasks:
            if failed_task.id in t.dependencies and not t.is_terminal:
                return True
        # 如果超过一半任务失败,需要重规划
        failed_count = len(plan.get_failed_tasks())
        if failed_count > len(plan.tasks) / 2:
            return True
        return False

    def replan(self, plan: Plan, failed_task: Task) -> bool:
        """
        重规划

        Returns:
            是否重规划成功
        """
        with self._lock:
            self.stats["replans"] += 1

        try:
            # 标记失败任务为跳过
            failed_task.status = TaskStatus.SKIPPED

            # 收集需要重新执行的任务描述(未完成的+刚失败的)
            pending_descs = []
            for t in plan.tasks:
                if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                    pending_descs.append(t.description)
                elif t.id == failed_task.id:
                    # 失败任务也需要重新执行
                    pending_descs.append(t.description)

            if not pending_descs:
                return False

            pending_desc = "; ".join(pending_descs)
            new_goal = Goal(
                description=f"重规划: {pending_desc}",
                success_criteria=plan.goal.success_criteria,
                priority=plan.goal.priority,
                parent_goal_id=plan.goal.id,
            )
            new_tasks = self.planner.decompose_goal(new_goal)
            if not new_tasks:
                return False

            # 添加新任务到计划
            for t in new_tasks:
                t.goal_id = plan.goal.id
                plan.tasks.append(t)

            plan.version += 1
            plan.updated_at = time.time()
            logger.info(f"[Executor] 重规划 v{plan.version}: 新增{len(new_tasks)}个任务")
            return True

        except Exception as e:
            logger.error(f"[Executor] 重规划失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 反思器
# ============================================================================

class Reflector:
    """
    反思器

    职责:
        1. 评估执行结果
        2. 识别问题与改进点
        3. 学习经验教训
        4. 判断是否需要重规划
    """

    REFLECT_PROMPT = """请反思以下任务执行结果,评估是否成功并识别问题。

目标: {goal}
成功标准: {criteria}

任务执行结果:
{results}

请返回JSON:
{{
    "success": true/false,
    "summary": "总体评估",
    "issues": [
        {{"severity": "low/medium/high/critical", "description": "问题描述", "suggestion": "改进建议"}}
    ],
    "improvements": ["改进点1", "改进点2"],
    "lessons_learned": ["经验教训1"],
    "should_replan": false,
    "replan_reason": "重规划原因(如需)"
}}

只返回JSON,不要其他内容。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {
            "reflections_made": 0,
            "issues_identified": 0,
            "replans_suggested": 0,
        }

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def reflect(self, plan: Plan, result: PlanResult) -> ReflectionResult:
        """反思执行结果"""
        with self._lock:
            self.stats["reflections_made"] += 1

        # 简单规则: 如果计划成功,直接返回成功
        if result.success and not plan.get_failed_tasks():
            reflection = ReflectionResult(
                success=True,
                summary="计划成功完成,所有任务均通过",
                lessons_learned=["计划执行顺利"],
            )
            result.reflections.append(reflection.to_dict())
            return reflection

        # 使用LLM进行深度反思
        results_str = self._format_results(plan, result)
        prompt = self.REFLECT_PROMPT.format(
            goal=plan.goal.description,
            criteria=plan.goal.success_criteria or "(未指定)",
            results=results_str,
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是反思分析专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.3,
                    max_tokens=768,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed:
                return self._fallback_reflect(plan, result)

            # 解析问题
            issues = []
            for issue_data in parsed.get("issues", []):
                severity_str = issue_data.get("severity", "low")
                try:
                    severity = RiskLevel(severity_str)
                except ValueError:
                    severity = RiskLevel.LOW
                issues.append(Issue(
                    severity=severity,
                    description=issue_data.get("description", ""),
                    suggestion=issue_data.get("suggestion", ""),
                ))

            with self._lock:
                self.stats["issues_identified"] += len(issues)
                if parsed.get("should_replan", False):
                    self.stats["replans_suggested"] += 1

            reflection = ReflectionResult(
                success=bool(parsed.get("success", False)),
                summary=parsed.get("summary", ""),
                issues=issues,
                improvements=parsed.get("improvements", []),
                lessons_learned=parsed.get("lessons_learned", []),
                should_replan=bool(parsed.get("should_replan", False)),
                replan_reason=parsed.get("replan_reason"),
            )

            # 存储到记忆
            self._store_to_memory(plan, reflection)
            result.reflections.append(reflection.to_dict())
            return reflection

        except LLMError as e:
            logger.warning(f"[Reflector] LLM反思失败: {e}")
            return self._fallback_reflect(plan, result)
        except Exception as e:
            logger.error(f"[Reflector] 反思异常: {e}")
            return self._fallback_reflect(plan, result)

    def _format_results(self, plan: Plan, result: PlanResult) -> str:
        """格式化结果用于LLM"""
        lines = []
        for task in plan.tasks:
            status = task.status.value
            output = (task.actual_output or "(无输出)")[:200]
            error = task.error or ""
            lines.append(f"- [{status}] {task.description}")
            if output:
                lines.append(f"  输出: {output}")
            if error:
                lines.append(f"  错误: {error}")
        return "\n".join(lines)

    def _fallback_reflect(self, plan: Plan, result: PlanResult) -> ReflectionResult:
        """回退反思: 基于规则的简单评估"""
        failed = plan.get_failed_tasks()
        blocked = plan.get_blocked_tasks()
        success = not failed and not blocked

        issues = []
        for t in failed:
            issues.append(Issue(
                severity=RiskLevel.HIGH,
                description=f"任务失败: {t.description}",
                task_id=t.id,
                suggestion="检查任务参数或增加重试次数",
            ))
        for t in blocked:
            issues.append(Issue(
                severity=RiskLevel.MEDIUM,
                description=f"任务被阻塞: {t.description}",
                task_id=t.id,
                suggestion="检查依赖任务状态",
            ))

        reflection = ReflectionResult(
            success=success,
            summary=f"{'成功' if success else '失败'}: {len(failed)}失败, {len(blocked)}阻塞",
            issues=issues,
            should_replan=bool(failed),
            replan_reason="存在失败任务" if failed else None,
        )
        result.reflections.append(reflection.to_dict())
        return reflection

    def _store_to_memory(self, plan: Plan, reflection: ReflectionResult):
        """存储反思到记忆系统"""
        try:
            manager = get_memory_manager()
            content = f"计划: {plan.goal.description}\n结果: {reflection.summary}"
            if reflection.lessons_learned:
                content += "\n教训: " + "; ".join(reflection.lessons_learned)
            manager.add(
                content,
                MemoryType.EPISODIC,
                importance=0.7 if reflection.success else 0.9,
                auto_summarize=False,
            )
        except Exception as e:
            logger.debug(f"[Reflector] 存储记忆失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 决策器
# ============================================================================

class DecisionMaker:
    """
    决策器

    职责:
        1. 决策下一步动作
        2. 评估风险
        3. 在多个选项中选择最优
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.stats = {
            "decisions_made": 0,
            "retries_decided": 0,
            "replans_decided": 0,
            "aborts_decided": 0,
        }

    def decide_next_action(self, plan: Plan,
                           last_result: Optional[TaskResult] = None) -> Action:
        """决策下一步动作"""
        with self._lock:
            self.stats["decisions_made"] += 1

        # 如果有失败结果
        if last_result and not last_result.success:
            task = plan.get_task(last_result.task_id)
            # 先检查失败比例: 多个任务失败(>=2且超过半数)时重规划
            failed_count = len(plan.get_failed_tasks())
            if failed_count >= 2 and failed_count > len(plan.tasks) * 0.5:
                with self._lock:
                    self.stats["replans_decided"] += 1
                return Action(
                    action_type=ActionType.REPLAN,
                    reason=f"失败任务过多({failed_count}/{len(plan.tasks)})",
                    risk=RiskLevel.HIGH,
                )
            # 检查是否可以重试
            if task and task.can_retry:
                with self._lock:
                    self.stats["retries_decided"] += 1
                return Action(
                    action_type=ActionType.RETRY_TASK,
                    target_task_id=task.id,
                    reason=f"任务失败,可重试({task.retry_count}/{task.max_retries})",
                    risk=RiskLevel.MEDIUM,
                )
            # 无法重试,跳过失败任务
            return Action(
                action_type=ActionType.SKIP_TASK,
                target_task_id=last_result.task_id,
                reason="任务无法重试,跳过",
                risk=RiskLevel.MEDIUM,
            )

        # 检查可执行任务
        ready = plan.get_ready_tasks()
        if ready:
            # 选择难度最低的
            task = min(ready, key=lambda t: t.difficulty)
            return Action(
                action_type=ActionType.EXECUTE_TASK,
                target_task_id=task.id,
                reason=f"执行任务(难度{task.difficulty:.1f})",
                risk=self._assess_risk(task),
            )

        # 没有就绪任务
        if plan.all_tasks_terminal():
            failed = plan.get_failed_tasks()
            if failed:
                return Action(
                    action_type=ActionType.ABORT_PLAN,
                    reason=f"计划失败: {len(failed)}个任务失败",
                    risk=RiskLevel.HIGH,
                )
            return Action(
                action_type=ActionType.COMPLETE_PLAN,
                reason="所有任务完成",
                risk=RiskLevel.LOW,
            )

        # 有阻塞任务
        blocked = plan.get_blocked_tasks()
        if blocked:
            with self._lock:
                self.stats["replans_decided"] += 1
            return Action(
                action_type=ActionType.REPLAN,
                reason=f"{len(blocked)}个任务被阻塞",
                risk=RiskLevel.HIGH,
            )

        return Action(
            action_type=ActionType.REQUEST_HELP,
            reason="无法决策下一步",
            risk=RiskLevel.MEDIUM,
        )

    def _assess_risk(self, task: Task) -> RiskLevel:
        """评估任务风险"""
        if task.difficulty >= 0.8:
            return RiskLevel.HIGH
        elif task.difficulty >= 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 自主Agent
# ============================================================================

class AutonomousAgent:
    """
    自主Agent

    整合所有组件:
        Planner + PlanExecutor + Reflector + DecisionMaker + Tools + Memory

    工作流程:
        1. 接收目标
        2. Planner分解目标,创建计划
        3. PlanExecutor执行计划
        4. Reflector反思结果
        5. 如需重规划,回到步骤2
        6. 返回最终结果
    """

    def __init__(self,
                 planner: Optional[Planner] = None,
                 executor: Optional[PlanExecutor] = None,
                 reflector: Optional[Reflector] = None,
                 decision_maker: Optional[DecisionMaker] = None):
        self.planner = planner or Planner()
        self.executor = executor or PlanExecutor(self.planner)
        self.reflector = reflector or Reflector()
        self.decision_maker = decision_maker or DecisionMaker()
        self._lock = threading.RLock()
        self._history: List[Dict[str, Any]] = []

    def run(self, goal: Union[str, Goal],
            max_replans: int = 2,
            max_iterations: int = 10) -> PlanResult:
        """
        运行自主Agent

        Args:
            goal: 目标(字符串或Goal对象)
            max_replans: 最大重规划次数
            max_iterations: 每次计划的最大迭代次数

        Returns:
            最终计划结果
        """
        if isinstance(goal, str):
            goal = Goal(description=goal)

        logger.info(f"[Agent] 启动,目标: {goal.description}")

        replan_count = 0
        current_plan: Optional[Plan] = None
        final_result: Optional[PlanResult] = None

        while replan_count <= max_replans:
            # 创建计划
            if current_plan is None:
                current_plan = self.planner.create_plan(goal)
            elif replan_count > 0:
                # 重规划: 创建新版本
                new_plan = self.planner.create_plan(
                    goal,
                    strategy=current_plan.strategy,
                )
                new_plan.parent_plan_id = current_plan.id
                new_plan.version = current_plan.version + 1
                current_plan = new_plan
                logger.info(f"[Agent] 重规划 v{current_plan.version}")

            # 执行计划
            result = self.executor.execute_plan(current_plan, max_iterations)

            # 反思
            reflection = self.reflector.reflect(current_plan, result)

            # 记录历史
            self._history.append({
                "plan_id": current_plan.id,
                "version": current_plan.version,
                "status": result.status.value,
                "reflection": reflection.to_dict(),
                "timestamp": time.time(),
            })

            final_result = result

            # 判断是否需要重规划
            if reflection.should_replan and replan_count < max_replans:
                replan_count += 1
                logger.info(f"[Agent] 反思建议重规划 ({replan_count}/{max_replans})")
                continue

            break

        # 存储最终结果到记忆
        if final_result:
            self._store_final_result(goal, current_plan, final_result)

        logger.info(f"[Agent] 完成,状态: {final_result.status.value if final_result else 'unknown'}")
        return final_result or PlanResult(
            plan_id="",
            status=PlanStatus.FAILED,
            final_output="Agent执行失败",
        )

    def _store_final_result(self, goal: Goal, plan: Plan, result: PlanResult):
        """存储最终结果到记忆"""
        try:
            manager = get_memory_manager()
            content = (
                f"目标: {goal.description}\n"
                f"结果: {result.status.value}\n"
                f"输出: {result.final_output or '(无)'}"
            )
            manager.add(
                content,
                MemoryType.EPISODIC,
                importance=0.8 if result.success else 0.6,
                auto_summarize=False,
            )
        except Exception as e:
            logger.debug(f"[Agent] 存储结果失败: {e}")

    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        with self._lock:
            return list(self._history)

    def get_report(self) -> Dict[str, Any]:
        """获取报告"""
        with self._lock:
            return {
                "planner_stats": self.planner.get_stats(),
                "executor_stats": self.executor.get_stats(),
                "reflector_stats": self.reflector.get_stats(),
                "decision_stats": self.decision_maker.get_stats(),
                "history_count": len(self._history),
            }


# ============================================================================
# 全局单例
# ============================================================================

_global_planner: Optional[Planner] = None
_global_executor: Optional[PlanExecutor] = None
_global_reflector: Optional[Reflector] = None
_global_decision_maker: Optional[DecisionMaker] = None
_global_agent: Optional[AutonomousAgent] = None
_global_lock = threading.RLock()


def get_planner() -> Planner:
    """获取全局规划器"""
    global _global_planner
    with _global_lock:
        if _global_planner is None:
            _global_planner = Planner()
        return _global_planner


def get_plan_executor() -> PlanExecutor:
    """获取全局计划执行器"""
    global _global_executor
    with _global_lock:
        if _global_executor is None:
            _global_executor = PlanExecutor(get_planner())
        return _global_executor


def get_reflector() -> Reflector:
    """获取全局反思器"""
    global _global_reflector
    with _global_lock:
        if _global_reflector is None:
            _global_reflector = Reflector()
        return _global_reflector


def get_decision_maker() -> DecisionMaker:
    """获取全局决策器"""
    global _global_decision_maker
    with _global_lock:
        if _global_decision_maker is None:
            _global_decision_maker = DecisionMaker()
        return _global_decision_maker


def get_autonomous_agent() -> AutonomousAgent:
    """获取全局自主Agent"""
    global _global_agent
    with _global_lock:
        if _global_agent is None:
            _global_agent = AutonomousAgent(
                planner=get_planner(),
                executor=get_plan_executor(),
                reflector=get_reflector(),
                decision_maker=get_decision_maker(),
            )
        return _global_agent


def reset_planner():
    """重置全局规划器"""
    global _global_planner, _global_executor, _global_agent
    with _global_lock:
        _global_planner = None
        _global_executor = None
        _global_agent = None


def reset_reflector():
    """重置全局反思器"""
    global _global_reflector, _global_agent
    with _global_lock:
        _global_reflector = None
        _global_agent = None


def reset_decision_maker():
    """重置全局决策器"""
    global _global_decision_maker, _global_agent
    with _global_lock:
        _global_decision_maker = None
        _global_agent = None


def reset_autonomous_agent():
    """重置全局自主Agent"""
    global _global_agent
    with _global_lock:
        _global_agent = None
