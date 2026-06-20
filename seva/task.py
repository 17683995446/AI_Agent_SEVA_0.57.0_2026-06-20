"""
长任务管理模块（TaskManager）
阶段25: 长任务自驱系统
- 任务状态机: TODO → DOING → DONE/FAILED
- 目标栈: 层次化子目标管理
- HTN分解接入主循环
- 里程碑与进度跟踪
"""
import time
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from .types import Goal, GoalType, TaskState, gen_id
from .config import Config

logger = logging.getLogger("seva.task")


@dataclass
class Milestone:
    """里程碑"""
    milestone_id: str
    description: str
    target_value: float
    current_value: float = 0.0
    achieved: bool = False

    def update(self, value: float) -> bool:
        self.current_value = value
        if not self.achieved and value >= self.target_value:
            self.achieved = True
            return True
        return False


@dataclass
class Task:
    """
    长任务：包装Goal，增加状态机、进度、里程碑、子任务
    """
    task_id: str
    goal: Goal
    state: TaskState = TaskState.TODO
    progress: float = 0.0           # [0, 1]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    accumulated_reward: float = 0.0
    accumulated_error: float = 0.0
    step_count: int = 0
    max_steps: int = 1000           # 任务超时步数
    success_threshold: float = 0.8  # 完成阈值
    failure_threshold: float = -5.0  # 失败阈值（累积reward过低）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self):
        """开始执行任务"""
        if self.state == TaskState.TODO:
            self.state = TaskState.DOING
            self.started_at = time.time()
            self.updated_at = time.time()
            logger.info(f"[Task] 开始 {self.goal.type.value} id={self.task_id}")

    def update_progress(self, reward: float, error: float = 0.0):
        """更新任务进度"""
        if self.state != TaskState.DOING:
            return
        self.step_count += 1
        self.accumulated_reward += reward
        self.accumulated_error += abs(error)
        self.updated_at = time.time()

        # 基于累积reward计算进度
        if self.accumulated_reward > 0:
            self.progress = min(1.0, self.accumulated_reward /
                                max(self.success_threshold * 10, 1.0))

        # 里程碑检查
        for ms in self.milestones:
            if not ms.achieved:
                ms.update(self.accumulated_reward)

    def check_completion(self) -> Optional[TaskState]:
        """
        检查任务是否完成或失败
        返回: 新状态（若状态变化），否则None
        """
        if self.state != TaskState.DOING:
            return None

        # 失败条件
        if self.accumulated_reward < self.failure_threshold:
            self.state = TaskState.FAILED
            self.completed_at = time.time()
            logger.warning(f"[Task] 失败 {self.goal.type.value} "
                          f"reward={self.accumulated_reward:.2f}")
            return TaskState.FAILED

        # 超时
        if self.step_count >= self.max_steps:
            if self.progress >= self.success_threshold:
                self.state = TaskState.DONE
                self.completed_at = time.time()
                logger.info(f"[Task] 完成(超时但达标) {self.goal.type.value} "
                           f"progress={self.progress:.2f}")
                return TaskState.DONE
            else:
                self.state = TaskState.FAILED
                self.completed_at = time.time()
                logger.warning(f"[Task] 超时失败 {self.goal.type.value} "
                              f"progress={self.progress:.2f}")
                return TaskState.FAILED

        # 成功条件
        if self.progress >= self.success_threshold:
            self.state = TaskState.DONE
            self.completed_at = time.time()
            logger.info(f"[Task] 完成 {self.goal.type.value} "
                       f"progress={self.progress:.2f} steps={self.step_count}")
            return TaskState.DONE

        return None

    def cancel(self):
        """取消任务"""
        self.state = TaskState.CANCELLED
        self.completed_at = time.time()
        logger.info(f"[Task] 取消 {self.goal.type.value} id={self.task_id}")

    @property
    def is_active(self) -> bool:
        return self.state == TaskState.DOING

    @property
    def is_terminal(self) -> bool:
        return self.state in (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_type": self.goal.type.value,
            "state": self.state.value,
            "progress": self.progress,
            "step_count": self.step_count,
            "accumulated_reward": self.accumulated_reward,
            "milestones_achieved": sum(1 for m in self.milestones if m.achieved),
            "milestones_total": len(self.milestones),
            "has_subtasks": len(self.subtask_ids) > 0,
        }


class TaskManager:
    """
    长任务管理器
    - 维护任务栈（层次化子目标）
    - 管理任务生命周期（TODO→DOING→DONE/FAILED）
    - HTN分解接入主循环
    - 里程碑与进度跟踪
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_active_tasks = 10
        self.default_max_steps = 500
        self.default_success_threshold = 0.8

        # 任务存储
        self.tasks: Dict[str, Task] = {}  # task_id -> Task
        self.task_stack: List[str] = []   # 活跃任务ID栈（末尾=当前焦点）
        self.completed_tasks: List[str] = []  # 已完成任务ID
        self.failed_tasks: List[str] = []     # 失败任务ID

        # 统计
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0

        logger.info("[TaskManager] 初始化")

    def submit_goal(self, goal: Goal,
                    max_steps: Optional[int] = None,
                    success_threshold: Optional[float] = None,
                    milestones: Optional[List[Milestone]] = None,
                    decompose_fn: Optional[Callable[[Goal], List[Goal]]] = None
                    ) -> str:
        """
        提交目标，创建任务
        - 若提供decompose_fn，对复合目标进行HTN分解
        - 返回task_id
        """
        task = Task(
            task_id=gen_id("task"),
            goal=goal,
            max_steps=max_steps or self.default_max_steps,
            success_threshold=success_threshold or self.default_success_threshold,
            milestones=milestones or [],
        )
        self.tasks[task.task_id] = task
        self.total_tasks_created += 1

        # HTN分解：如果有分解函数且目标可分解
        if decompose_fn is not None:
            sub_goals = decompose_fn(goal)
            if len(sub_goals) > 1 or (len(sub_goals) == 1 and
                                       sub_goals[0].goal_id != goal.goal_id):
                for sg in sub_goals:
                    subtask = Task(
                        task_id=gen_id("task"),
                        goal=sg,
                        parent_task_id=task.task_id,
                        max_steps=max_steps or self.default_max_steps,
                        success_threshold=success_threshold or self.default_success_threshold,
                    )
                    self.tasks[subtask.task_id] = subtask
                    task.subtask_ids.append(subtask.task_id)
                    self.total_tasks_created += 1
                logger.info(f"[TaskManager] HTN分解 {goal.type.value} -> "
                           f"{len(task.subtask_ids)}个子任务")

        # 如果有子任务，先入栈子任务；否则入栈任务本身
        if task.subtask_ids:
            for sid in task.subtask_ids:
                self._push_task(sid)
            self._push_task(task.task_id)  # 父任务最后（栈底）
        else:
            self._push_task(task.task_id)

        return task.task_id

    def _push_task(self, task_id: str):
        """将任务压入栈"""
        if task_id not in self.task_stack:
            self.task_stack.append(task_id)
            task = self.tasks[task_id]
            if task.state == TaskState.TODO:
                task.start()

    def get_current_task(self) -> Optional[Task]:
        """获取当前焦点任务（栈顶活跃任务）"""
        while self.task_stack:
            task_id = self.task_stack[-1]
            task = self.tasks.get(task_id)
            if task is None or task.is_terminal:
                self.task_stack.pop()
                if task and task.state == TaskState.DONE:
                    self.completed_tasks.append(task_id)
                    self.total_tasks_completed += 1
                elif task and task.state == TaskState.FAILED:
                    self.failed_tasks.append(task_id)
                    self.total_tasks_failed += 1
                continue
            return task
        return None

    def update_current(self, reward: float, error: float = 0.0):
        """更新当前任务进度"""
        task = self.get_current_task()
        if task is not None:
            task.update_progress(reward, error)

            # 检查完成状态
            new_state = task.check_completion()
            if new_state is not None:
                # 子任务完成 → 更新父任务进度
                if task.parent_task_id:
                    parent = self.tasks.get(task.parent_task_id)
                    if parent:
                        # 子任务完成贡献父任务进度
                        subtasks = [self.tasks[sid] for sid in parent.subtask_ids
                                   if sid in self.tasks]
                        done_count = sum(1 for s in subtasks if s.state == TaskState.DONE)
                        parent.progress = done_count / max(len(subtasks), 1)
                        parent.accumulated_reward += reward
                        if parent.progress >= parent.success_threshold:
                            parent.state = TaskState.DONE
                            parent.completed_at = time.time()
                            logger.info(f"[TaskManager] 父任务完成 "
                                       f"{parent.goal.type.value}")

    def get_active_goals(self) -> List[Goal]:
        """获取当前活跃目标列表（用于决策模块）"""
        task = self.get_current_task()
        if task is not None:
            return [task.goal]
        return []

    def get_progress_report(self) -> Dict[str, Any]:
        """获取进度报告"""
        active = [self.tasks[tid] for tid in self.task_stack
                  if tid in self.tasks and not self.tasks[tid].is_terminal]
        return {
            "active_tasks": len(active),
            "current_task": active[-1].to_dict() if active else None,
            "total_created": self.total_tasks_created,
            "total_completed": self.total_tasks_completed,
            "total_failed": self.total_tasks_failed,
            "stack_depth": len(self.task_stack),
        }

    def cancel_all(self):
        """取消所有活跃任务"""
        for task_id in self.task_stack:
            task = self.tasks.get(task_id)
            if task and not task.is_terminal:
                task.cancel()
        self.task_stack.clear()

    def reset(self):
        """重置任务管理器"""
        self.tasks.clear()
        self.task_stack.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        logger.info("[TaskManager] 重置完成")

    def health_check(self) -> bool:
        return True
