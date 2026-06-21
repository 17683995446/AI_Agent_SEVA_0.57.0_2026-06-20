"""
多Agent协作与编排（阶段55）
对应第一性原理：编排 = 将复杂任务分解为有依赖关系的子任务图，按拓扑顺序协调执行

提供能力:
    - TaskNode: 任务节点（工作流中的单个任务，含依赖关系）
    - TaskStatus: 任务状态（PENDING/RUNNING/SUCCESS/FAILED/SKIPPED/CANCELLED）
    - ExecutionMode: 执行模式（SEQUENTIAL/PARALLEL/PIPELINE）
    - Workflow: 工作流（DAG任务图，依赖管理，拓扑排序）
    - Orchestrator: 编排器（单例，管理工作流执行、调度、监控）
    - WorkflowResult: 工作流执行结果
    - 便捷函数: create_workflow / run_workflow / get_orchestrator 等

使用示例:
    from seva import Workflow, TaskNode, ExecutionMode

    wf = Workflow("数据处理")
    wf.add_task(TaskNode("fetch", fetch_data))
    wf.add_task(TaskNode("process", process_data), depends_on=["fetch"])
    wf.add_task(TaskNode("save", save_data), depends_on=["process"])
    result = wf.execute()
"""
from __future__ import annotations

import os
import sys
import time
import json
import threading
import uuid
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = os.path.expanduser("~/.seva")


# ============================================================================
# TaskStatus - 任务状态枚举
# ============================================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    SUCCESS = "success"      # 成功
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 跳过（依赖失败）
    CANCELLED = "cancelled"  # 取消


# ============================================================================
# ExecutionMode - 执行模式枚举
# ============================================================================

class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"      # 并行执行（无依赖的任务并行）
    PIPELINE = "pipeline"      # 管道执行（流式处理）


# ============================================================================
# WorkflowStatus - 工作流状态枚举
# ============================================================================

class WorkflowStatus(Enum):
    """工作流状态"""
    CREATED = "created"      # 已创建
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


# ============================================================================
# TaskNode - 任务节点
# ============================================================================

@dataclass
class TaskNode:
    """
    任务节点：工作流中的单个任务

    Args:
        name: 任务名称（唯一标识）
        func: 任务函数
        args: 位置参数
        kwargs: 关键字参数
        retries: 重试次数
        timeout: 超时时间（秒，0表示不限制）
        condition: 执行条件函数（返回True才执行）
    """
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    retries: int = 0
    timeout: float = 0.0
    condition: Optional[Callable[[], bool]] = None

    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    attempts: int = 0

    @property
    def is_completed(self) -> bool:
        """是否已完成（成功或失败）"""
        return self.status in (TaskStatus.SUCCESS, TaskStatus.FAILED,
                                TaskStatus.SKIPPED, TaskStatus.CANCELLED)

    @property
    def is_successful(self) -> bool:
        """是否成功"""
        return self.status == TaskStatus.SUCCESS

    @property
    def duration(self) -> float:
        """执行时长（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def execute(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行任务

        Args:
            context: 工作流上下文（前面任务的结果）

        Returns:
            任务执行结果

        Raises:
            Exception: 任务执行失败
        """
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.attempts += 1

        try:
            # 检查条件
            if self.condition is not None and not self.condition():
                self.status = TaskStatus.SKIPPED
                self.end_time = time.time()
                return None

            # 执行函数
            # 如果函数接受context参数，传入context
            try:
                sig = functools.singledispatch(self.func)
            except Exception:
                pass

            # 尝试传入context
            try:
                result = self.func(*self.args, context=context, **self.kwargs)
            except TypeError:
                # 函数不接受context参数
                result = self.func(*self.args, **self.kwargs)

            self.result = result
            self.status = TaskStatus.SUCCESS
            self.end_time = time.time()
            return result

        except Exception as e:
            self.error = str(e)
            self.end_time = time.time()

            # 重试
            if self.attempts <= self.retries:
                return self.execute(context)

            self.status = TaskStatus.FAILED
            raise

    def reset(self) -> None:
        """重置任务状态"""
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "duration": self.duration,
            "attempts": self.attempts,
            "retries": self.retries,
        }

    def __repr__(self) -> str:
        return f"TaskNode(name={self.name!r}, status={self.status.value})"


# ============================================================================
# WorkflowResult - 工作流执行结果
# ============================================================================

@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    tasks: Dict[str, TaskNode] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """总执行时长"""
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def success_count(self) -> int:
        """成功任务数"""
        return sum(1 for t in self.tasks.values() if t.is_successful)

    @property
    def failure_count(self) -> int:
        """失败任务数"""
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        """跳过任务数"""
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

    def get_result(self, task_name: str) -> Any:
        """获取某任务的结果"""
        task = self.tasks.get(task_name)
        return task.result if task else None

    def get_results(self) -> Dict[str, Any]:
        """获取所有成功任务的结果"""
        return {
            name: task.result
            for name, task in self.tasks.items()
            if task.is_successful
        }

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "duration": self.duration,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
            "error": self.error,
        }

    def summary(self) -> str:
        """摘要字符串"""
        return (f"Workflow '{self.workflow_name}' {self.status.value}: "
                f"{self.success_count}成功, {self.failure_count}失败, "
                f"{self.skipped_count}跳过, 耗时{self.duration:.2f}s")


# ============================================================================
# Workflow - 工作流
# ============================================================================

class Workflow:
    """
    工作流：DAG任务图，管理任务依赖关系和执行顺序

    使用示例:
        wf = Workflow("数据处理")
        wf.add_task(TaskNode("fetch", fetch_data))
        wf.add_task(TaskNode("process", process_data), depends_on=["fetch"])
        wf.add_task(TaskNode("save", save_data), depends_on=["process"])
        result = wf.execute()
    """

    def __init__(self, name: str = "", mode: ExecutionMode = ExecutionMode.PARALLEL,
                 max_workers: int = 4, fail_fast: bool = False):
        """
        Args:
            name: 工作流名称
            mode: 执行模式
            max_workers: 最大并行工作线程数
            fail_fast: 是否快速失败（某任务失败立即停止）
        """
        self.workflow_id: str = str(uuid.uuid4())[:8]
        self.name: str = name or self.workflow_id
        self.mode: ExecutionMode = mode
        self.max_workers: int = max(1, max_workers)
        self.fail_fast: bool = fail_fast

        self._tasks: Dict[str, TaskNode] = {}
        self._dependencies: Dict[str, Set[str]] = {}  # task -> 依赖的任务集合
        self._dependents: Dict[str, Set[str]] = {}    # task -> 依赖它的任务集合
        self._lock = threading.RLock()
        self._status: WorkflowStatus = WorkflowStatus.CREATED
        self._cancel_event = threading.Event()

    @property
    def status(self) -> WorkflowStatus:
        """工作流状态"""
        return self._status

    @property
    def task_count(self) -> int:
        """任务数"""
        return len(self._tasks)

    def add_task(self, task: TaskNode, depends_on: Optional[List[str]] = None) -> "Workflow":
        """
        添加任务

        Args:
            task: 任务节点
            depends_on: 依赖的任务名称列表

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 任务名已存在或依赖不存在
        """
        with self._lock:
            if task.name in self._tasks:
                raise ValueError(f"任务已存在: {task.name}")

            self._tasks[task.name] = task
            self._dependencies[task.name] = set()
            self._dependents[task.name] = set()

            if depends_on:
                for dep in depends_on:
                    if dep not in self._tasks:
                        raise ValueError(f"依赖任务不存在: {dep}")
                    self._dependencies[task.name].add(dep)
                    self._dependents[dep].add(task.name)

        return self

    def remove_task(self, name: str) -> bool:
        """移除任务"""
        with self._lock:
            if name not in self._tasks:
                return False
            # 移除依赖关系
            for dep in self._dependencies.get(name, set()):
                self._dependents[dep].discard(name)
            for dependent in self._dependents.get(name, set()):
                self._dependencies[dependent].discard(name)
            del self._tasks[name]
            del self._dependencies[name]
            del self._dependents[name]
            return True

    def get_task(self, name: str) -> Optional[TaskNode]:
        """获取任务"""
        with self._lock:
            return self._tasks.get(name)

    def list_tasks(self) -> List[str]:
        """列出所有任务名"""
        with self._lock:
            return list(self._tasks.keys())

    def get_dependencies(self, name: str) -> Set[str]:
        """获取任务的依赖"""
        with self._lock:
            return set(self._dependencies.get(name, set()))

    def get_dependents(self, name: str) -> Set[str]:
        """获取依赖该任务的任务"""
        with self._lock:
            return set(self._dependents.get(name, set()))

    def topological_sort(self) -> List[str]:
        """
        拓扑排序

        Returns:
            按依赖顺序排列的任务名列表

        Raises:
            ValueError: 存在循环依赖
        """
        with self._lock:
            # Kahn's algorithm
            in_degree: Dict[str, int] = {
                name: len(self._dependencies[name]) for name in self._tasks
            }
            queue: List[str] = [name for name, deg in in_degree.items() if deg == 0]
            result: List[str] = []

            while queue:
                # 按名称排序保证确定性
                queue.sort()
                name = queue.pop(0)
                result.append(name)

                for dependent in sorted(self._dependents.get(name, set())):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if len(result) != len(self._tasks):
                raise ValueError("工作流存在循环依赖")

            return result

    def validate(self) -> bool:
        """
        验证工作流（检查循环依赖等）

        Returns:
            是否有效
        """
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def execute(self) -> WorkflowResult:
        """
        执行工作流

        Returns:
            工作流执行结果
        """
        self._cancel_event.clear()
        self._status = WorkflowStatus.RUNNING
        start_time = time.time()

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.name,
            status=WorkflowStatus.RUNNING,
            start_time=start_time,
        )

        try:
            if self.mode == ExecutionMode.SEQUENTIAL:
                self._execute_sequential(result)
            elif self.mode == ExecutionMode.PIPELINE:
                self._execute_pipeline(result)
            else:
                self._execute_parallel(result)

            # 判定最终状态
            if result.failure_count > 0:
                result.status = WorkflowStatus.FAILED
            else:
                result.status = WorkflowStatus.COMPLETED

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = str(e)

        finally:
            result.end_time = time.time()
            with self._lock:
                self._status = result.status

        return result

    def _execute_sequential(self, result: WorkflowResult) -> None:
        """顺序执行"""
        order = self.topological_sort()
        context: Dict[str, Any] = {}

        for name in order:
            if self._cancel_event.is_set():
                break

            task = self._tasks[name]
            result.tasks[name] = task

            # 检查依赖是否成功
            deps = self._dependencies[name]
            failed_deps = [d for d in deps if result.tasks.get(d) and
                          result.tasks[d].status != TaskStatus.SUCCESS]
            if failed_deps:
                task.status = TaskStatus.SKIPPED
                task.error = f"依赖任务失败: {failed_deps}"
                continue

            try:
                task_result = task.execute(context=context)
                context[name] = task_result
            except Exception as e:
                if self.fail_fast:
                    raise

    def _execute_parallel(self, result: WorkflowResult) -> None:
        """并行执行（按依赖层级并行）"""
        context: Dict[str, Any] = {}
        context_lock = threading.Lock()
        completed: Set[str] = set()
        failed: Set[str] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[Future, str] = {}

            while len(completed) < len(self._tasks):
                if self._cancel_event.is_set():
                    break

                # 找出所有依赖已完成的待执行任务
                ready: List[str] = []
                with self._lock:
                    for name, task in self._tasks.items():
                        if name in completed or name in failed:
                            continue
                        if name in [f for f in futures.values()]:
                            continue
                        if task.status != TaskStatus.PENDING:
                            continue
                        deps = self._dependencies[name]

                        # 检查是否有失败的依赖
                        failed_deps = deps & failed
                        if failed_deps:
                            task.status = TaskStatus.SKIPPED
                            task.error = f"依赖任务失败: {list(failed_deps)}"
                            result.tasks[name] = task
                            completed.add(name)
                            continue

                        # 检查依赖是否全部完成
                        if deps.issubset(completed):
                            ready.append(name)

                # 提交就绪任务
                for name in ready:
                    task = self._tasks[name]
                    result.tasks[name] = task

                    def run_task(t=task, n=name):
                        try:
                            with context_lock:
                                ctx = dict(context)
                            r = t.execute(context=ctx)
                            with context_lock:
                                context[n] = r
                            return n, True
                        except Exception:
                            return n, False

                    future = executor.submit(run_task)
                    futures[future] = name

                # 等待至少一个future完成
                if futures:
                    done_futures = []
                    # 等待所有已提交的future中至少一个完成
                    for f in as_completed(list(futures.keys())):
                        done_futures.append(f)
                        break

                    for f in done_futures:
                        name, success = f.result()
                        completed.add(name)
                        if not success:
                            failed.add(name)
                        del futures[f]
                elif not ready:
                    # 没有就绪任务也没有进行中的任务
                    # 标记剩余任务为跳过
                    with self._lock:
                        for name, task in self._tasks.items():
                            if name not in completed and name not in failed:
                                task.status = TaskStatus.SKIPPED
                                task.error = "无法调度"
                                result.tasks[name] = task
                                completed.add(name)
                    break

        if self.fail_fast and failed:
            first_failed = next(iter(failed))
            task = result.tasks.get(first_failed)
            raise RuntimeError(f"任务失败: {first_failed}: {task.error if task else '未知错误'}")

    def _wait_for_first(self, futures: Dict[Future, str]) -> Tuple[Set[Future], Set[Future]]:
        """等待第一个future完成"""
        done: Set[Future] = set()
        not_done: Set[Future] = set()
        for f in as_completed(futures.keys(), timeout=None):
            done.add(f)
            break
        not_done = set(futures.keys()) - done
        return done, not_done

    def _execute_pipeline(self, result: WorkflowResult) -> None:
        """管道执行（流式处理，前一个的输出作为后一个的输入）"""
        order = self.topological_sort()
        context: Dict[str, Any] = {}

        for name in order:
            if self._cancel_event.is_set():
                break

            task = self._tasks[name]
            result.tasks[name] = task

            # 检查依赖
            deps = self._dependencies[name]
            failed_deps = [d for d in deps if result.tasks.get(d) and
                          result.tasks[d].status != TaskStatus.SUCCESS]
            if failed_deps:
                task.status = TaskStatus.SKIPPED
                task.error = f"依赖任务失败: {failed_deps}"
                continue

            try:
                # 管道模式：将依赖结果作为输入
                if deps:
                    # 取最后一个依赖的结果作为输入
                    dep_results = [context[d] for d in sorted(deps) if d in context]
                    if dep_results:
                        task.args = (dep_results[-1],) + task.args

                task_result = task.execute(context=context)
                context[name] = task_result
            except Exception as e:
                if self.fail_fast:
                    raise

    def cancel(self) -> None:
        """取消工作流"""
        self._cancel_event.set()
        with self._lock:
            self._status = WorkflowStatus.CANCELLED

    def reset(self) -> None:
        """重置工作流"""
        with self._lock:
            for task in self._tasks.values():
                task.reset()
            self._status = WorkflowStatus.CREATED
            self._cancel_event.clear()

    def get_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        with self._lock:
            return {
                "workflow_id": self.workflow_id,
                "name": self.name,
                "status": self._status.value,
                "mode": self.mode.value,
                "task_count": len(self._tasks),
                "tasks": [name for name in self._tasks],
                "max_workers": self.max_workers,
                "fail_fast": self.fail_fast,
            }

    def __repr__(self) -> str:
        return (f"Workflow(name={self.name!r}, tasks={len(self._tasks)}, "
                f"status={self._status.value})")


# ============================================================================
# Orchestrator - 编排器（单例）
# ============================================================================

class Orchestrator:
    """
    编排器：管理多个工作流的注册、执行、监控

    单例模式，全局唯一实例
    """

    _instance: Optional["Orchestrator"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "Orchestrator":
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

        self._workflows: Dict[str, Workflow] = {}
        self._results: Dict[str, WorkflowResult] = {}
        self._data_lock = threading.RLock()

    def register(self, workflow: Workflow) -> bool:
        """注册工作流"""
        with self._data_lock:
            if workflow.name in self._workflows:
                return False
            self._workflows[workflow.name] = workflow
            return True

    def unregister(self, name: str) -> bool:
        """注销工作流"""
        with self._data_lock:
            if name not in self._workflows:
                return False
            del self._workflows[name]
            self._results.pop(name, None)
            return True

    def get_workflow(self, name: str) -> Optional[Workflow]:
        """获取工作流"""
        with self._data_lock:
            return self._workflows.get(name)

    def list_workflows(self) -> List[str]:
        """列出所有工作流名"""
        with self._data_lock:
            return list(self._workflows.keys())

    def run(self, name: str) -> Optional[WorkflowResult]:
        """运行指定工作流"""
        with self._data_lock:
            workflow = self._workflows.get(name)
        if workflow is None:
            return None

        result = workflow.execute()
        with self._data_lock:
            self._results[name] = result
        return result

    def run_all(self) -> Dict[str, WorkflowResult]:
        """运行所有工作流"""
        results = {}
        with self._data_lock:
            names = list(self._workflows.keys())
        for name in names:
            results[name] = self.run(name)
        return results

    def get_result(self, name: str) -> Optional[WorkflowResult]:
        """获取工作流执行结果"""
        with self._data_lock:
            return self._results.get(name)

    def get_report(self) -> Dict[str, Any]:
        """获取编排器报告"""
        with self._data_lock:
            return {
                "total_workflows": len(self._workflows),
                "workflow_names": list(self._workflows.keys()),
                "results": {
                    name: result.to_dict()
                    for name, result in self._results.items()
                },
            }

    def format_report(self) -> str:
        """格式化报告"""
        report = self.get_report()
        lines = []
        lines.append("=" * 60)
        lines.append("SEVA 工作流编排报告")
        lines.append("=" * 60)
        lines.append(f"工作流总数: {report['total_workflows']}")
        lines.append("")

        lines.append("--- 已注册工作流 ---")
        if report["workflow_names"]:
            for name in report["workflow_names"]:
                wf = self._workflows[name]
                info = wf.get_info()
                lines.append(f"  [{info['status'].upper():9s}] {name} "
                            f"({info['task_count']}个任务, {info['mode']})")
        else:
            lines.append("  (无注册工作流)")
        lines.append("")

        lines.append("--- 执行结果 ---")
        if report["results"]:
            for name, result in report["results"].items():
                lines.append(f"  {name}: {result['status']} "
                            f"({result['success_count']}成功, "
                            f"{result['failure_count']}失败, "
                            f"耗时{result['duration']:.2f}s)")
        else:
            lines.append("  (无执行结果)")
        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """重置所有状态"""
        with self._data_lock:
            self._workflows.clear()
            self._results.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.reset()
            cls._instance = None


# ============================================================================
# 便捷函数
# ============================================================================

def get_orchestrator() -> Orchestrator:
    """获取编排器单例"""
    return Orchestrator()


def create_workflow(name: str, mode: ExecutionMode = ExecutionMode.PARALLEL,
                    **kwargs) -> Workflow:
    """创建工作流"""
    return Workflow(name=name, mode=mode, **kwargs)


def run_workflow(workflow: Workflow) -> WorkflowResult:
    """执行工作流"""
    return workflow.execute()


def register_workflow(workflow: Workflow) -> bool:
    """注册工作流到编排器"""
    return Orchestrator().register(workflow)


def run_registered_workflow(name: str) -> Optional[WorkflowResult]:
    """运行已注册的工作流"""
    return Orchestrator().run(name)


def list_workflows() -> List[str]:
    """列出所有已注册工作流"""
    return Orchestrator().list_workflows()


def get_workflow_result(name: str) -> Optional[WorkflowResult]:
    """获取工作流结果"""
    return Orchestrator().get_result(name)


def get_orchestration_report() -> str:
    """获取编排报告"""
    return Orchestrator().format_report()


def get_orchestration_status() -> Dict[str, Any]:
    """获取编排状态"""
    return Orchestrator().get_report()


def reset_orchestration() -> None:
    """重置所有编排状态"""
    Orchestrator().reset()
