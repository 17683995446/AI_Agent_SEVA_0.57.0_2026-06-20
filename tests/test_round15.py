"""
第十五轮测试：阶段25长任务自驱系统验证
验证TaskManager、Task状态机、HTN分解接入主循环、里程碑、目标栈
核心目标：证明Agent能自主驱动完成长任务（多周期目标持续追踪）
"""
import sys
import os
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

from seva.config import Config
from seva.core import SelfEvolvingAgentCore
from seva.environment import SimulationEnvironment
from seva.task import Task, TaskManager, Milestone
from seva.types import Goal, GoalType, TaskState, gen_id

PASS = 0
FAIL = 0
RESULTS = []


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    RESULTS.append(f"\n=== {title} ===")


# ============================================================
# 测试1：TaskState枚举完整性
# ============================================================
def test1_task_state_enum():
    section("测试1：TaskState枚举完整性")
    states = [TaskState.TODO, TaskState.DOING, TaskState.DONE,
              TaskState.FAILED, TaskState.CANCELLED]
    values = [s.value for s in states]
    report("5种状态存在", len(states) == 5,
           f"values={values}")
    report("TODO初始值", TaskState.TODO.value == "todo")
    report("DOING值", TaskState.DOING.value == "doing")
    report("DONE值", TaskState.DONE.value == "done")
    report("FAILED值", TaskState.FAILED.value == "failed")
    report("CANCELLED值", TaskState.CANCELLED.value == "cancelled")


# ============================================================
# 测试2：Task数据类基础
# ============================================================
def test2_task_dataclass():
    section("测试2：Task数据类基础")
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={"target": "energy"}, priority=0.8)
    task = Task(task_id="t1", goal=goal)

    report("初始状态TODO", task.state == TaskState.TODO)
    report("初始进度0", task.progress == 0.0)
    report("初始step_count=0", task.step_count == 0)
    report("初始reward=0", task.accumulated_reward == 0.0)
    report("无父任务", task.parent_task_id is None)
    report("无子任务", len(task.subtask_ids) == 0)
    report("无里程碑", len(task.milestones) == 0)
    report("is_active=False(TODO)", not task.is_active)
    report("is_terminal=False(TODO)", not task.is_terminal)


# ============================================================
# 测试3：Task状态机 - start/update/check_completion
# ============================================================
def test3_task_state_machine():
    section("测试3：Task状态机")
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9)
    task = Task(task_id="t1", goal=goal,
                max_steps=100, success_threshold=0.8)

    # start
    task.start()
    report("start后DOING", task.state == TaskState.DOING)
    report("start后is_active", task.is_active)
    report("started_at已设置", task.started_at is not None)

    # update_progress累积reward
    for _ in range(10):
        task.update_progress(reward=1.0, error=0.1)
    report("10步后step_count=10", task.step_count == 10)
    report("10步后reward=10", abs(task.accumulated_reward - 10.0) < 1e-6)
    report("10步后error=1.0", abs(task.accumulated_error - 1.0) < 1e-6)
    report("进度>0", task.progress > 0)

    # 持续累积直到完成
    for _ in range(20):
        task.update_progress(reward=1.0)
        state = task.check_completion()
        if state is not None:
            break
    report("累积reward达标后DONE", task.state == TaskState.DONE)
    report("is_terminal(DONE)", task.is_terminal)
    report("completed_at已设置", task.completed_at is not None)


# ============================================================
# 测试4：Task失败状态机
# ============================================================
def test4_task_failure():
    section("测试4：Task失败状态机")
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.5)
    task = Task(task_id="t1", goal=goal,
                max_steps=5, success_threshold=0.8,
                failure_threshold=-5.0)
    task.start()

    # 持续负reward触发失败
    for _ in range(20):
        task.update_progress(reward=-1.0)
        state = task.check_completion()
        if state == TaskState.FAILED:
            break
    report("负reward触发FAILED", task.state == TaskState.FAILED)
    report("FAILED后is_terminal", task.is_terminal)


# ============================================================
# 测试5：Task超时失败
# ============================================================
def test5_task_timeout():
    section("测试5：Task超时失败")
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.5)
    task = Task(task_id="t1", goal=goal,
                max_steps=3, success_threshold=0.99)
    task.start()

    # 小reward，不达标，但超时
    for _ in range(5):
        task.update_progress(reward=0.01)
        state = task.check_completion()
        if state is not None:
            break
    report("超时未达标FAILED", task.state == TaskState.FAILED)


# ============================================================
# 测试6：Task取消
# ============================================================
def test6_task_cancel():
    section("测试6：Task取消")
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.5)
    task = Task(task_id="t1", goal=goal)
    task.start()
    task.cancel()
    report("cancel后CANCELLED", task.state == TaskState.CANCELLED)
    report("CANCELLED后is_terminal", task.is_terminal)
    report("completed_at已设置", task.completed_at is not None)


# ============================================================
# 测试7：Milestone里程碑
# ============================================================
def test7_milestone():
    section("测试7：Milestone里程碑")
    ms = Milestone(milestone_id="m1", description="累积10reward",
                   target_value=10.0)
    report("初始未达成", not ms.achieved)
    report("初始current=0", ms.current_value == 0.0)

    # 未达标更新
    achieved = ms.update(5.0)
    report("5<10未达成", not achieved and not ms.achieved)
    report("current=5", ms.current_value == 5.0)

    # 达标更新
    achieved = ms.update(10.0)
    report("10>=10达成", achieved and ms.achieved)

    # 已达成后不再触发
    achieved = ms.update(20.0)
    report("已达成后update返回False", not achieved)


# ============================================================
# 测试8：Task里程碑集成
# ============================================================
def test8_task_milestones():
    section("测试8：Task里程碑集成")
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9)
    ms1 = Milestone("m1", "阶段1", 5.0)
    ms2 = Milestone("m2", "阶段2", 10.0)
    task = Task(task_id="t1", goal=goal,
                milestones=[ms1, ms2],
                max_steps=100, success_threshold=0.8)
    task.start()

    # 累积5reward → ms1达成
    for _ in range(5):
        task.update_progress(reward=1.0)
    report("ms1达成", ms1.achieved)
    report("ms2未达成", not ms2.achieved)

    # 继续累积到10 → ms2达成
    for _ in range(5):
        task.update_progress(reward=1.0)
    report("ms2达成", ms2.achieved)

    # to_dict包含里程碑信息
    d = task.to_dict()
    report("to_dict含milestones_achieved", d["milestones_achieved"] == 2)
    report("to_dict含milestones_total", d["milestones_total"] == 2)


# ============================================================
# 测试9：TaskManager基础 - submit/get_current
# ============================================================
def test9_task_manager_basic():
    section("测试9：TaskManager基础")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.8)

    task_id = mgr.submit_goal(goal)
    report("submit返回task_id", task_id is not None)
    report("tasks非空", len(mgr.tasks) == 1)
    report("total_created=1", mgr.total_tasks_created == 1)

    current = mgr.get_current_task()
    report("get_current非空", current is not None)
    report("current状态DOING", current.state == TaskState.DOING)
    report("task_stack非空", len(mgr.task_stack) == 1)


# ============================================================
# 测试10：TaskManager - 无分解时单任务
# ============================================================
def test10_task_manager_no_decompose():
    section("测试10：TaskManager无分解")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.8)

    task_id = mgr.submit_goal(goal, decompose_fn=None)
    report("无分解时单任务", len(mgr.tasks) == 1)
    report("无子任务", len(mgr.tasks[task_id].subtask_ids) == 0)


# ============================================================
# 测试11：TaskManager - HTN分解
# ============================================================
def test11_task_manager_htn_decompose():
    section("测试11：TaskManager HTN分解")
    mgr = TaskManager(Config())

    # 使用真实GoalGenerator.decompose
    from seva.goal import GoalGenerator
    gen = GoalGenerator(Config())
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9,
                expected_free_energy=0.5)

    task_id = mgr.submit_goal(goal, decompose_fn=gen.decompose)
    parent = mgr.tasks[task_id]

    report("父任务存在", parent is not None)
    report("有子任务", len(parent.subtask_ids) > 0)
    report("total_created>1", mgr.total_tasks_created > 1)
    report("task_stack包含子任务",
           any(sid in mgr.task_stack for sid in parent.subtask_ids))

    # 子任务有parent_task_id
    for sid in parent.subtask_ids:
        sub = mgr.tasks[sid]
        report(f"子任务{sid[:8]}有parent", sub.parent_task_id == task_id)


# ============================================================
# 测试12：TaskManager - update_current进度更新
# ============================================================
def test12_task_manager_update():
    section("测试12：TaskManager进度更新")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9)
    task_id = mgr.submit_goal(goal, max_steps=1000,
                              success_threshold=0.8)

    # 模拟多周期更新
    for _ in range(5):
        mgr.update_current(reward=1.0, error=0.1)

    current = mgr.get_current_task()
    report("5次更新后step_count=5", current.step_count == 5)
    report("5次更新后reward=5", abs(current.accumulated_reward - 5.0) < 1e-6)
    report("5次更新后error=0.5", abs(current.accumulated_error - 0.5) < 1e-6)


# ============================================================
# 测试13：TaskManager - 任务完成后出栈
# ============================================================
def test13_task_manager_completion():
    section("测试13：TaskManager任务完成出栈")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9)
    task_id = mgr.submit_goal(goal, max_steps=1000,
                              success_threshold=0.8)

    # 持续正reward直到完成
    for _ in range(100):
        mgr.update_current(reward=1.0)
        current = mgr.get_current_task()
        if current is None:
            break

    report("完成后current为None", mgr.get_current_task() is None)
    report("task_stack已清空", len(mgr.task_stack) == 0)
    report("total_completed=1", mgr.total_tasks_completed == 1)
    report("completed_tasks非空", len(mgr.completed_tasks) == 1)


# ============================================================
# 测试14：TaskManager - 父子任务层级完成
# ============================================================
def test14_task_manager_parent_child():
    section("测试14：TaskManager父子任务层级")
    mgr = TaskManager(Config())
    from seva.goal import GoalGenerator
    gen = GoalGenerator(Config())
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9,
                expected_free_energy=0.5)

    parent_id = mgr.submit_goal(goal, max_steps=100,
                                success_threshold=0.8,
                                decompose_fn=gen.decompose)
    parent = mgr.tasks[parent_id]
    n_subtasks = len(parent.subtask_ids)
    report(f"父任务有{n_subtasks}个子任务", n_subtasks > 0)

    # 完成所有子任务
    for _ in range(500):
        current = mgr.get_current_task()
        if current is None:
            break
        mgr.update_current(reward=1.0)

    report("所有任务完成后current=None",
           mgr.get_current_task() is None)
    report("total_completed>=1", mgr.total_tasks_completed >= 1)


# ============================================================
# 测试15：TaskManager - get_active_goals
# ============================================================
def test15_task_manager_active_goals():
    section("测试15：TaskManager活跃目标")
    mgr = TaskManager(Config())
    report("无任务时活跃目标为空", len(mgr.get_active_goals()) == 0)

    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.8)
    mgr.submit_goal(goal)
    goals = mgr.get_active_goals()
    report("有任务时活跃目标非空", len(goals) == 1)
    report("活跃目标类型匹配", goals[0].type == GoalType.MAINTAIN)


# ============================================================
# 测试16：TaskManager - get_progress_report
# ============================================================
def test16_task_manager_progress_report():
    section("测试16：TaskManager进度报告")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.8)
    mgr.submit_goal(goal)

    report_dict = mgr.get_progress_report()
    report("含active_tasks", "active_tasks" in report_dict)
    report("含current_task", "current_task" in report_dict)
    report("含total_created", "total_created" in report_dict)
    report("含total_completed", "total_completed" in report_dict)
    report("含total_failed", "total_failed" in report_dict)
    report("含stack_depth", "stack_depth" in report_dict)
    report("active_tasks=1", report_dict["active_tasks"] == 1)
    report("current_task非空", report_dict["current_task"] is not None)


# ============================================================
# 测试17：TaskManager - cancel_all
# ============================================================
def test17_task_manager_cancel_all():
    section("测试17：TaskManager取消所有")
    mgr = TaskManager(Config())
    for i in range(3):
        goal = Goal(goal_id=f"g{i}", type=GoalType.MAINTAIN,
                    parameters={}, priority=0.8)
        mgr.submit_goal(goal)

    mgr.cancel_all()
    report("cancel后task_stack空", len(mgr.task_stack) == 0)
    # 所有任务应处于CANCELLED
    all_cancelled = all(t.state == TaskState.CANCELLED
                        for t in mgr.tasks.values())
    report("所有任务CANCELLED", all_cancelled)


# ============================================================
# 测试18：TaskManager - reset
# ============================================================
def test18_task_manager_reset():
    section("测试18：TaskManager重置")
    mgr = TaskManager(Config())
    goal = Goal(goal_id="g1", type=GoalType.MAINTAIN,
                parameters={}, priority=0.8)
    mgr.submit_goal(goal)
    mgr.update_current(reward=1.0)

    mgr.reset()
    report("reset后tasks空", len(mgr.tasks) == 0)
    report("reset后task_stack空", len(mgr.task_stack) == 0)
    report("reset后completed_tasks空", len(mgr.completed_tasks) == 0)
    report("reset后failed_tasks空", len(mgr.failed_tasks) == 0)
    report("reset后total_created=0", mgr.total_tasks_created == 0)
    report("reset后total_completed=0", mgr.total_tasks_completed == 0)


# ============================================================
# 测试19：TaskManager - health_check
# ============================================================
def test19_task_manager_health_check():
    section("测试19：TaskManager健康检查")
    mgr = TaskManager(Config())
    report("health_check通过", mgr.health_check())


# ============================================================
# 测试20：Core集成 - task_manager模块注册
# ============================================================
def test20_core_task_manager_registered():
    section("测试20：Core注册task_manager")
    core = SelfEvolvingAgentCore(Config())
    report("task_manager在modules中", "task_manager" in core.modules)
    report("task_manager是TaskManager实例",
           isinstance(core.modules["task_manager"], TaskManager))


# ============================================================
# 测试21：Core集成 - 自动创建长任务
# ============================================================
def test21_core_auto_create_task():
    section("测试21：Core自动创建长任务")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 第一个周期应自动创建任务
    core.run_cycle()
    task_mgr = core.modules["task_manager"]
    report("第一周期后创建任务", task_mgr.total_tasks_created >= 1)
    report("task_stack非空", len(task_mgr.task_stack) >= 1)

    current = task_mgr.get_current_task()
    report("current任务存在", current is not None)
    report("current状态DOING", current.state == TaskState.DOING)


# ============================================================
# 测试22：Core集成 - 任务进度持续更新
# ============================================================
def test22_core_task_progress_update():
    section("测试22：Core任务进度持续更新")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    core.run_cycle()
    task_mgr = core.modules["task_manager"]
    first_task = task_mgr.get_current_task()
    first_steps = first_task.step_count if first_task else 0

    # 再运行4个周期
    for _ in range(4):
        core.run_cycle()

    current = task_mgr.get_current_task()
    report("任务持续存在", current is not None)
    report("step_count递增", current.step_count > first_steps)
    report("step_count=5", current.step_count == 5)


# ============================================================
# 测试23：Core集成 - cycle_log含task_update
# ============================================================
def test23_core_cycle_log_task_update():
    section("测试23：Core周期日志含task_update")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    core.run_cycle()
    log = core.last_cycle_log
    stages = [entry["stage"] for entry in log]
    report("cycle_log含task_update", "task_update" in stages)

    task_entry = next(e for e in log if e["stage"] == "task_update")
    report("task_update含task_active", "task_active" in task_entry["result"])
    report("task_update含current_task", "current_task" in task_entry["result"])
    report("task_update含total_completed", "total_completed" in task_entry["result"])


# ============================================================
# 测试24：Core集成 - metrics含task_stats
# ============================================================
def test24_core_metrics_task_stats():
    section("测试24：Core指标含task_stats")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    for _ in range(3):
        core.run_cycle()

    metrics = core.get_metrics()
    report("metrics含task_stats", "task_stats" in metrics)
    ts = metrics["task_stats"]
    report("task_stats含total_created", "total_created" in ts)
    report("task_stats含total_completed", "total_completed" in ts)
    report("task_stats含current_task", "current_task" in ts)
    report("total_created>=1", ts["total_created"] >= 1)


# ============================================================
# 测试25：Core集成 - 长任务自驱多周期
# ============================================================
def test25_core_long_task_self_driving():
    section("测试25：Core长任务自驱多周期")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config(), env_type="mixed", seed=42)
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 运行50个周期，验证自驱持续
    initial_cycle = core.cycle_count
    for _ in range(50):
        core.run_cycle()

    task_mgr = core.modules["task_manager"]
    report("50周期后cycle_count增加50",
           core.cycle_count - initial_cycle == 50)
    report("50周期后total_created>=1", task_mgr.total_tasks_created >= 1)
    report("50周期后total_completed+total_failed+active>0",
           task_mgr.total_tasks_completed + task_mgr.total_tasks_failed +
           len(task_mgr.task_stack) > 0)

    # 验证任务有进度积累
    metrics = core.get_metrics()
    ts = metrics["task_stats"]
    report("有任务历史记录",
           ts["total_created"] > 0 or ts["total_completed"] > 0)


# ============================================================
# 测试26：Core集成 - 任务完成后自动接续
# ============================================================
def test26_core_task_continuation():
    section("测试26：Core任务完成后自动接续")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config(), env_type="mixed", seed=42)
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 运行足够周期让任务完成或失败
    for _ in range(200):
        core.run_cycle()
        task_mgr = core.modules["task_manager"]
        # 如果当前无任务，下一周期应自动创建新任务
        if task_mgr.get_current_task() is None:
            # 运行下一周期，应自动创建新任务
            core.run_cycle()
            new_task = task_mgr.get_current_task()
            report("无任务后自动创建新任务", new_task is not None)
            report("新任务状态DOING",
                   new_task.state == TaskState.DOING if new_task else False)
            return

    # 如果200周期内没有完成，也验证任务持续活跃
    task_mgr = core.modules["task_manager"]
    report("200周期内任务持续活跃",
           task_mgr.get_current_task() is not None or
           task_mgr.total_tasks_completed > 0)


# ============================================================
# 测试27：Core集成 - reset清空任务
# ============================================================
def test27_core_reset_tasks():
    section("测试27：Core重置清空任务")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    for _ in range(3):
        core.run_cycle()

    core.reset()
    task_mgr = core.modules["task_manager"]
    report("reset后tasks空", len(task_mgr.tasks) == 0)
    report("reset后total_created=0", task_mgr.total_tasks_created == 0)


# ============================================================
# 测试28：Core集成 - 无环境时也工作
# ============================================================
def test28_core_no_env_tasks():
    section("测试28：Core无环境时长任务工作")
    core = SelfEvolvingAgentCore(Config())
    # 不附加环境
    core.inject_input(np.zeros(16))

    core.run_cycle()
    task_mgr = core.modules["task_manager"]
    report("无环境也创建任务", task_mgr.total_tasks_created >= 1)
    report("无环境任务DOING",
           task_mgr.get_current_task() is not None)


# ============================================================
# 测试29：性能测试 - 长任务开销小
# ============================================================
def test29_performance():
    section("测试29：长任务系统性能开销")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 预热
    for _ in range(10):
        core.run_cycle()

    # 测量100步
    t0 = time.time()
    for _ in range(100):
        core.run_cycle()
    elapsed = (time.time() - t0) / 100 * 1000

    report("长任务系统<30ms/周期", elapsed < 30, f"{elapsed:.2f}ms/周期")


# ============================================================
# 测试30：to_dict序列化
# ============================================================
def test30_task_to_dict():
    section("测试30：Task序列化")
    goal = Goal(goal_id="g1", type=GoalType.EXPLORE,
                parameters={}, priority=0.9)
    ms = Milestone("m1", "阶段1", 5.0)
    task = Task(task_id="t1", goal=goal, milestones=[ms],
                max_steps=100, success_threshold=0.8)
    task.start()
    task.update_progress(reward=3.0, error=0.5)

    d = task.to_dict()
    report("含task_id", "task_id" in d)
    report("含goal_type", d["goal_type"] == "explore")
    report("含state", d["state"] == "doing")
    report("含progress", "progress" in d)
    report("含step_count", d["step_count"] == 1)
    report("含accumulated_reward", abs(d["accumulated_reward"] - 3.0) < 1e-6)
    report("含milestones_achieved", d["milestones_achieved"] == 0)
    report("含milestones_total", d["milestones_total"] == 1)
    report("含has_subtasks", d["has_subtasks"] is False)


def main():
    tests = [
        test1_task_state_enum,
        test2_task_dataclass,
        test3_task_state_machine,
        test4_task_failure,
        test5_task_timeout,
        test6_task_cancel,
        test7_milestone,
        test8_task_milestones,
        test9_task_manager_basic,
        test10_task_manager_no_decompose,
        test11_task_manager_htn_decompose,
        test12_task_manager_update,
        test13_task_manager_completion,
        test14_task_manager_parent_child,
        test15_task_manager_active_goals,
        test16_task_manager_progress_report,
        test17_task_manager_cancel_all,
        test18_task_manager_reset,
        test19_task_manager_health_check,
        test20_core_task_manager_registered,
        test21_core_auto_create_task,
        test22_core_task_progress_update,
        test23_core_cycle_log_task_update,
        test24_core_metrics_task_stats,
        test25_core_long_task_self_driving,
        test26_core_task_continuation,
        test27_core_reset_tasks,
        test28_core_no_env_tasks,
        test29_performance,
        test30_task_to_dict,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [ERROR] {test.__name__} 异常: {e}")
            traceback.print_exc()
            global FAIL
            FAIL += 1
            RESULTS.append(f"  [ERROR] {test.__name__}: {e}")

    print("\n" + "="*60)
    print(f"测试总结: {PASS}通过 / {FAIL}失败")
    print("="*60)

    print("\n".join(RESULTS))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
