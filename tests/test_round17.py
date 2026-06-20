"""
第十七轮测试：阶段27多Agent协作系统验证
验证MultiAgentCoordinator、AgentInfo、AgentMessage、任务分配、协调机制
核心目标：证明多个Agent能协作完成任务
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
from seva.multiagent import (MultiAgentCoordinator, AgentInfo, AgentStatus,
                              AgentMessage, AgentMessageType)
from seva.task import Task
from seva.types import Goal, GoalType, gen_id

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


def make_goal(gtype=GoalType.EXPLORE, priority=0.8):
    return Goal(goal_id=gen_id("goal"), type=gtype,
                parameters={}, priority=priority)


def make_task(gtype=GoalType.EXPLORE):
    return Task(task_id=gen_id("task"),
                goal=make_goal(gtype))


# ============================================================
# 测试1：AgentStatus枚举
# ============================================================
def test1_agent_status_enum():
    section("测试1：AgentStatus枚举")
    statuses = [AgentStatus.IDLE, AgentStatus.BUSY,
                AgentStatus.DONE, AgentStatus.FAILED, AgentStatus.OFFLINE]
    report("5种状态", len(statuses) == 5)
    report("IDLE值", AgentStatus.IDLE.value == "idle")
    report("BUSY值", AgentStatus.BUSY.value == "busy")
    report("OFFLINE值", AgentStatus.OFFLINE.value == "offline")


# ============================================================
# 测试2：AgentMessageType枚举
# ============================================================
def test2_message_type_enum():
    section("测试2：AgentMessageType枚举")
    types = [AgentMessageType.REQUEST, AgentMessageType.RESPONSE,
             AgentMessageType.BROADCAST, AgentMessageType.COORDINATE]
    report("4种消息类型", len(types) == 4)
    report("REQUEST值", AgentMessageType.REQUEST.value == "request")
    report("BROADCAST值", AgentMessageType.BROADCAST.value == "broadcast")


# ============================================================
# 测试3：AgentInfo基础
# ============================================================
def test3_agent_info_basic():
    section("测试3：AgentInfo基础")
    agent = AgentInfo(agent_id="a1", name="Explorer",
                      capabilities={GoalType.EXPLORE})
    report("初始状态IDLE", agent.status == AgentStatus.IDLE)
    report("初始is_available", agent.is_available)
    report("初始load=0", agent.load == 0.0)
    report("初始success_rate=0", agent.success_rate == 0.0)
    report("初始total_tasks=0", agent.total_tasks_assigned == 0)
    report("有EXPLORE能力", GoalType.EXPLORE in agent.capabilities)
    report("无COLLECT能力", GoalType.COLLECT not in agent.capabilities)


# ============================================================
# 测试4：AgentInfo状态变化
# ============================================================
def test4_agent_info_status():
    section("测试4：AgentInfo状态变化")
    agent = AgentInfo(agent_id="a1", name="Worker",
                      capabilities={GoalType.EXPLORE})
    agent.status = AgentStatus.BUSY
    report("BUSY时不可用", not agent.is_available)
    report("BUSY时load=1", agent.load == 1.0)

    agent.status = AgentStatus.IDLE
    report("IDLE时可用", agent.is_available)

    agent.status = AgentStatus.OFFLINE
    report("OFFLINE时不可用", not agent.is_available)


# ============================================================
# 测试5：AgentInfo success_rate
# ============================================================
def test5_agent_info_success_rate():
    section("测试5：AgentInfo成功率")
    agent = AgentInfo(agent_id="a1", name="Worker",
                      capabilities={GoalType.EXPLORE})
    agent.total_tasks_completed = 7
    agent.total_tasks_failed = 3
    report("success_rate=0.7", abs(agent.success_rate - 0.7) < 1e-6)

    agent.total_tasks_completed = 0
    agent.total_tasks_failed = 0
    report("无任务时success_rate=0", agent.success_rate == 0.0)


# ============================================================
# 测试6：AgentInfo to_dict
# ============================================================
def test6_agent_info_to_dict():
    section("测试6：AgentInfo序列化")
    agent = AgentInfo(agent_id="a1", name="Explorer",
                      capabilities={GoalType.EXPLORE, GoalType.COLLECT})
    agent.total_tasks_completed = 5
    agent.total_tasks_failed = 1
    d = agent.to_dict()
    report("含agent_id", d["agent_id"] == "a1")
    report("含name", d["name"] == "Explorer")
    report("含capabilities", len(d["capabilities"]) == 2)
    report("含status", d["status"] == "idle")
    report("含total_assigned", d["total_assigned"] == 0)
    report("含total_completed", d["total_completed"] == 5)
    report("含success_rate", d["success_rate"] > 0.8)
    report("含is_available", d["is_available"] is True)


# ============================================================
# 测试7：AgentMessage基础
# ============================================================
def test7_agent_message():
    section("测试7：AgentMessage基础")
    msg = AgentMessage(
        message_id="m1",
        msg_type=AgentMessageType.REQUEST,
        sender_id="coordinator",
        receiver_id="a1",
        content={"task_id": "t1"}
    )
    report("msg_type=REQUEST", msg.msg_type == AgentMessageType.REQUEST)
    report("sender=coordinator", msg.sender_id == "coordinator")
    report("receiver=a1", msg.receiver_id == "a1")
    report("content有task_id", msg.content["task_id"] == "t1")
    report("timestamp已设置", msg.timestamp > 0)

    d = msg.to_dict()
    report("to_dict含message_id", d["message_id"] == "m1")
    report("to_dict含msg_type", d["msg_type"] == "request")


# ============================================================
# 测试8：Coordinator注册Agent
# ============================================================
def test8_coordinator_register():
    section("测试8：Coordinator注册Agent")
    coord = MultiAgentCoordinator(Config())
    agent = AgentInfo(agent_id="a1", name="Explorer",
                      capabilities={GoalType.EXPLORE})
    ok = coord.register_agent(agent)
    report("注册成功", ok)
    report("agents非空", len(coord.agents) == 1)
    report("message_inbox已创建", "a1" in coord.message_inbox)

    # 重复注册
    ok2 = coord.register_agent(agent)
    report("重复注册失败", not ok2)
    report("仍只有1个Agent", len(coord.agents) == 1)


# ============================================================
# 测试9：Coordinator注销Agent
# ============================================================
def test9_coordinator_unregister():
    section("测试9：Coordinator注销Agent")
    coord = MultiAgentCoordinator(Config())
    agent = AgentInfo(agent_id="a1", name="Explorer",
                      capabilities={GoalType.EXPLORE})
    coord.register_agent(agent)
    coord.unregister_agent("a1")
    report("注销后agents空", len(coord.agents) == 0)
    report("注销后message_inbox空", "a1" not in coord.message_inbox)


# ============================================================
# 测试10：Coordinator get_agent
# ============================================================
def test10_coordinator_get_agent():
    section("测试10：Coordinator获取Agent")
    coord = MultiAgentCoordinator(Config())
    agent = AgentInfo(agent_id="a1", name="Explorer",
                      capabilities={GoalType.EXPLORE})
    coord.register_agent(agent)
    retrieved = coord.get_agent("a1")
    report("获取到Agent", retrieved is not None)
    report("获取正确Agent", retrieved.agent_id == "a1")
    report("获取不存在Agent返回None", coord.get_agent("nonexistent") is None)


# ============================================================
# 测试11：Coordinator get_available_agents
# ============================================================
def test11_coordinator_available_agents():
    section("测试11：Coordinator获取可用Agent")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Idle", capabilities={GoalType.EXPLORE})
    a2 = AgentInfo(agent_id="a2", name="Busy", capabilities={GoalType.EXPLORE})
    a2.status = AgentStatus.BUSY
    coord.register_agent(a1)
    coord.register_agent(a2)
    available = coord.get_available_agents()
    report("1个可用Agent", len(available) == 1)
    report("可用Agent是a1", available[0].agent_id == "a1")


# ============================================================
# 测试12：Coordinator任务分配 - 能力匹配
# ============================================================
def test12_coordinator_assign_by_capability():
    section("测试12：Coordinator按能力分配任务")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Explorer",
                   capabilities={GoalType.EXPLORE})
    a2 = AgentInfo(agent_id="a2", name="Collector",
                   capabilities={GoalType.COLLECT})
    coord.register_agent(a1)
    coord.register_agent(a2)

    # EXPLORE任务应分配给a1
    task = make_task(GoalType.EXPLORE)
    assigned = coord.assign_task(task)
    report("EXPLORE分配给a1", assigned == "a1")
    report("a1状态BUSY", coord.agents["a1"].status == AgentStatus.BUSY)
    report("a1有current_task", coord.agents["a1"].current_task_id is not None)
    report("a1 total_assigned=1", coord.agents["a1"].total_tasks_assigned == 1)


# ============================================================
# 测试13：Coordinator任务分配 - 无可用Agent
# ============================================================
def test13_coordinator_no_available():
    section("测试13：Coordinator无可用Agent")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Busy",
                   capabilities={GoalType.EXPLORE})
    a1.status = AgentStatus.BUSY
    coord.register_agent(a1)

    task = make_task(GoalType.EXPLORE)
    assigned = coord.assign_task(task)
    report("无可用Agent返回None", assigned is None)


# ============================================================
# 测试14：Coordinator任务分配 - 无能力匹配
# ============================================================
def test14_coordinator_no_capability_match():
    section("测试14：Coordinator无能力匹配")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Collector",
                   capabilities={GoalType.COLLECT})
    coord.register_agent(a1)

    # EXPLORE任务，Agent只有COLLECT能力，但仍可用
    task = make_task(GoalType.EXPLORE)
    assigned = coord.assign_task(task)
    report("无能力匹配也分配（回退）", assigned is not None)


# ============================================================
# 测试15：Coordinator任务分配 - 负载均衡
# ============================================================
def test15_coordinator_load_balance():
    section("测试15：Coordinator负载均衡")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Agent1",
                   capabilities={GoalType.EXPLORE})
    a2 = AgentInfo(agent_id="a2", name="Agent2",
                   capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)
    coord.register_agent(a2)

    # 第一个任务分配给a1或a2
    task1 = make_task(GoalType.EXPLORE)
    assigned1 = coord.assign_task(task1)
    report("第一个任务已分配", assigned1 is not None)

    # 第二个任务应分配给另一个（a1忙了）
    task2 = make_task(GoalType.EXPLORE)
    assigned2 = coord.assign_task(task2)
    report("第二个任务已分配", assigned2 is not None)
    report("两个任务分配给不同Agent", assigned1 != assigned2)


# ============================================================
# 测试16：Coordinator任务分配 - 历史表现
# ============================================================
def test16_coordinator_history_performance():
    section("测试16：Coordinator按历史表现分配")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Good",
                   capabilities={GoalType.EXPLORE})
    a1.total_tasks_completed = 8
    a1.total_tasks_failed = 2
    a2 = AgentInfo(agent_id="a2", name="Bad",
                   capabilities={GoalType.EXPLORE})
    a2.total_tasks_completed = 2
    a2.total_tasks_failed = 8
    coord.register_agent(a1)
    coord.register_agent(a2)

    task = make_task(GoalType.EXPLORE)
    assigned = coord.assign_task(task)
    report("分配给成功率高的Agent", assigned == "a1")


# ============================================================
# 测试17：Coordinator报告结果
# ============================================================
def test17_coordinator_report_result():
    section("测试17：Coordinator报告结果")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Worker",
                   capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    task = make_task(GoalType.EXPLORE)
    coord.assign_task(task)
    report("分配后a1状态BUSY", coord.agents["a1"].status == AgentStatus.BUSY)

    coord.report_result("a1", task.task_id, True, {"found": 3})
    report("报告后a1状态IDLE", coord.agents["a1"].status == AgentStatus.IDLE)
    report("a1 total_completed=1", coord.agents["a1"].total_tasks_completed == 1)
    report("a1 current_task=None", coord.agents["a1"].current_task_id is None)
    report("coord total_completed=1", coord.total_tasks_completed == 1)
    report("task_results有结果", task.task_id in coord.task_results)


# ============================================================
# 测试18：Coordinator报告失败
# ============================================================
def test18_coordinator_report_failure():
    section("测试18：Coordinator报告失败")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="Worker",
                   capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    task = make_task(GoalType.EXPLORE)
    coord.assign_task(task)
    coord.report_result("a1", task.task_id, False, {"error": "timeout"})
    report("a1 total_failed=1", coord.agents["a1"].total_tasks_failed == 1)
    report("coord total_failed=1", coord.total_tasks_failed == 1)
    report("a1状态IDLE", coord.agents["a1"].status == AgentStatus.IDLE)


# ============================================================
# 测试19：Coordinator消息发送 - 点对点
# ============================================================
def test19_coordinator_send_message():
    section("测试19：Coordinator点对点消息")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    msg = AgentMessage(
        message_id="m1",
        msg_type=AgentMessageType.REQUEST,
        sender_id="coordinator",
        receiver_id="a1",
        content={"task_id": "t1"}
    )
    coord.send_message(msg)
    report("消息已发送", coord.total_messages_sent == 1)
    report("a1 inbox有1条消息", len(coord.message_inbox["a1"]) == 1)

    msgs = coord.get_messages("a1")
    report("get_messages返回1条", len(msgs) == 1)
    report("get_messages后inbox清空", len(coord.message_inbox["a1"]) == 0)


# ============================================================
# 测试20：Coordinator消息发送 - 广播
# ============================================================
def test20_coordinator_broadcast():
    section("测试20：Coordinator广播消息")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    a2 = AgentInfo(agent_id="a2", name="A2", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)
    coord.register_agent(a2)

    coord.broadcast("coordinator", {"type": "sync"})
    report("广播后a1有消息", len(coord.message_inbox["a1"]) == 1)
    report("广播后a2有消息", len(coord.message_inbox["a2"]) == 1)


# ============================================================
# 测试21：Coordinator get_status
# ============================================================
def test21_coordinator_status():
    section("测试21：Coordinator状态")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    a2 = AgentInfo(agent_id="a2", name="A2", capabilities={GoalType.EXPLORE})
    a2.status = AgentStatus.BUSY
    coord.register_agent(a1)
    coord.register_agent(a2)

    status = coord.get_status()
    report("含total_agents", status["total_agents"] == 2)
    report("含available_agents", status["available_agents"] == 1)
    report("含busy_agents", status["busy_agents"] == 1)
    report("含total_tasks_distributed", status["total_tasks_distributed"] == 0)
    report("含agents列表", len(status["agents"]) == 2)


# ============================================================
# 测试22：Coordinator coordinate
# ============================================================
def test22_coordinator_coordinate():
    section("测试22：Coordinator协调轮次")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    result = coord.coordinate()
    report("coordinate返回状态", "total_agents" in result)
    report("coordinate返回agents", "agents" in result)


# ============================================================
# 测试23：Coordinator get_task_result
# ============================================================
def test23_coordinator_task_result():
    section("测试23：Coordinator获取任务结果")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    task = make_task(GoalType.EXPLORE)
    coord.assign_task(task)
    coord.report_result("a1", task.task_id, True, {"data": "result"})

    result = coord.get_task_result(task.task_id)
    report("获取到结果", result is not None)
    report("结果含agent_id", result["agent_id"] == "a1")
    report("结果含success", result["success"] is True)
    report("结果含result", result["result"]["data"] == "result")


# ============================================================
# 测试24：Coordinator reset
# ============================================================
def test24_coordinator_reset():
    section("测试24：Coordinator重置")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)
    task = make_task(GoalType.EXPLORE)
    coord.assign_task(task)

    coord.reset()
    report("reset后agents空", len(coord.agents) == 0)
    report("reset后messages空", len(coord.messages) == 0)
    report("reset后task_results空", len(coord.task_results) == 0)
    report("reset后total_distributed=0", coord.total_tasks_distributed == 0)


# ============================================================
# 测试25：Coordinator health_check
# ============================================================
def test25_coordinator_health_check():
    section("测试25：Coordinator健康检查")
    coord = MultiAgentCoordinator(Config())
    report("health_check通过", coord.health_check())


# ============================================================
# 测试26：多Agent协作完整流程
# ============================================================
def test26_multi_agent_workflow():
    section("测试26：多Agent协作完整流程")
    coord = MultiAgentCoordinator(Config())

    # 注册3个不同能力的Agent
    explorer = AgentInfo(agent_id="explorer", name="Explorer",
                         capabilities={GoalType.EXPLORE})
    collector = AgentInfo(agent_id="collector", name="Collector",
                          capabilities={GoalType.COLLECT})
    maintainer = AgentInfo(agent_id="maintainer", name="Maintainer",
                           capabilities={GoalType.MAINTAIN})
    coord.register_agent(explorer)
    coord.register_agent(collector)
    coord.register_agent(maintainer)
    report("注册3个Agent", coord.get_status()["total_agents"] == 3)

    # 分配EXPLORE任务
    explore_task = make_task(GoalType.EXPLORE)
    explore_agent = coord.assign_task(explore_task)
    report("EXPLORE分配给explorer", explore_agent == "explorer")

    # 分配COLLECT任务
    collect_task = make_task(GoalType.COLLECT)
    collect_agent = coord.assign_task(collect_task)
    report("COLLECT分配给collector", collect_agent == "collector")

    # 分配MAINTAIN任务
    maintain_task = make_task(GoalType.MAINTAIN)
    maintain_agent = coord.assign_task(maintain_task)
    report("MAINTAIN分配给maintainer", maintain_agent == "maintainer")

    # 报告结果
    coord.report_result("explorer", explore_task.task_id, True, {"areas": 5})
    coord.report_result("collector", collect_task.task_id, True, {"resources": 10})
    coord.report_result("maintainer", maintain_task.task_id, False, {"error": "low_energy"})

    status = coord.get_status()
    report("2个任务成功", status["total_tasks_completed"] == 2)
    report("1个任务失败", status["total_tasks_failed"] == 1)
    report("3个任务已分配", status["total_tasks_distributed"] == 3)


# ============================================================
# 测试27：Core集成 - coordinator模块注册
# ============================================================
def test27_core_coordinator_registered():
    section("测试27：Core注册coordinator")
    core = SelfEvolvingAgentCore(Config())
    report("coordinator在modules中", "coordinator" in core.modules)
    report("coordinator是MultiAgentCoordinator实例",
           isinstance(core.modules["coordinator"], MultiAgentCoordinator))


# ============================================================
# 测试28：Core集成 - cycle_log含coordination
# ============================================================
def test28_core_cycle_log_coordination():
    section("测试28：Core周期日志含coordination")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))
    core.run_cycle()
    log = core.last_cycle_log
    stages = [e["stage"] for e in log]
    report("cycle_log含coordination", "coordination" in stages)

    coord_entry = next(e for e in log if e["stage"] == "coordination")
    report("coordination含total_agents", "total_agents" in coord_entry["result"])
    report("coordination含available_agents", "available_agents" in coord_entry["result"])
    report("coordination含busy_agents", "busy_agents" in coord_entry["result"])


# ============================================================
# 测试29：Core集成 - metrics含coord_stats
# ============================================================
def test29_core_metrics_coord():
    section("测试29：Core指标含coord_stats")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))
    core.run_cycle()
    metrics = core.get_metrics()
    report("metrics含coord_stats", "coord_stats" in metrics)
    cs = metrics["coord_stats"]
    report("coord_stats含total_agents", "total_agents" in cs)
    report("coord_stats含available_agents", "available_agents" in cs)


# ============================================================
# 测试30：Core集成 - 通过coordinator分配任务
# ============================================================
def test30_core_distribute_task():
    section("测试30：Core通过coordinator分配任务")
    core = SelfEvolvingAgentCore(Config())
    coord = core.modules["coordinator"]

    # 注册一个Agent
    agent = AgentInfo(agent_id="worker1", name="Worker",
                      capabilities={GoalType.EXPLORE})
    coord.register_agent(agent)

    # 创建任务并分配
    task = make_task(GoalType.EXPLORE)
    assigned_to = coord.assign_task(task)
    report("任务已分配", assigned_to == "worker1")

    # 报告结果
    coord.report_result("worker1", task.task_id, True, {"explored": 5})
    status = coord.get_status()
    report("任务完成", status["total_tasks_completed"] == 1)


# ============================================================
# 测试31：Core集成 - reset清空coordinator
# ============================================================
def test31_core_reset_coordinator():
    section("测试31：Core重置清空coordinator")
    core = SelfEvolvingAgentCore(Config())
    coord = core.modules["coordinator"]
    agent = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(agent)

    core.reset()
    report("reset后coordinator agents空",
           len(core.modules["coordinator"].agents) == 0)


# ============================================================
# 测试32：性能测试
# ============================================================
def test32_performance():
    section("测试32：多Agent系统性能开销")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))

    # 预热
    for _ in range(10):
        core.run_cycle()

    # 测量100步
    t0 = time.time()
    for _ in range(100):
        core.run_cycle()
    elapsed = (time.time() - t0) / 100 * 1000

    report("多Agent系统<30ms/周期", elapsed < 30, f"{elapsed:.2f}ms/周期")


# ============================================================
# 测试33：消息队列上限
# ============================================================
def test33_message_queue_limit():
    section("测试33：消息队列上限")
    coord = MultiAgentCoordinator(Config())
    a1 = AgentInfo(agent_id="a1", name="A1", capabilities={GoalType.EXPLORE})
    coord.register_agent(a1)

    # 发送超过max_messages条消息
    for i in range(coord.max_messages + 100):
        coord.send_message(AgentMessage(
            message_id=f"m{i}",
            msg_type=AgentMessageType.BROADCAST,
            sender_id="coordinator",
            receiver_id=None,
            content={"idx": i}
        ))

    report("消息队列不超过上限", len(coord.messages) <= coord.max_messages)


# ============================================================
# 测试34：多Agent并行任务
# ============================================================
def test34_parallel_tasks():
    section("测试34：多Agent并行任务")
    coord = MultiAgentCoordinator(Config())

    # 注册5个Agent
    for i in range(5):
        agent = AgentInfo(agent_id=f"a{i}", name=f"Agent{i}",
                          capabilities={GoalType.EXPLORE})
        coord.register_agent(agent)

    # 分配5个任务
    task_ids = []
    for _ in range(5):
        task = make_task(GoalType.EXPLORE)
        assigned = coord.assign_task(task)
        if assigned:
            task_ids.append((task.task_id, assigned))

    report("5个任务全部分配", len(task_ids) == 5)
    report("5个Agent都忙碌",
           coord.get_status()["busy_agents"] == 5)
    report("0个可用Agent",
           coord.get_status()["available_agents"] == 0)

    # 所有Agent报告完成
    for tid, aid in task_ids:
        coord.report_result(aid, tid, True, {"done": True})

    report("5个任务完成",
           coord.get_status()["total_tasks_completed"] == 5)
    report("5个Agent都空闲",
           coord.get_status()["available_agents"] == 5)


def main():
    tests = [
        test1_agent_status_enum,
        test2_message_type_enum,
        test3_agent_info_basic,
        test4_agent_info_status,
        test5_agent_info_success_rate,
        test6_agent_info_to_dict,
        test7_agent_message,
        test8_coordinator_register,
        test9_coordinator_unregister,
        test10_coordinator_get_agent,
        test11_coordinator_available_agents,
        test12_coordinator_assign_by_capability,
        test13_coordinator_no_available,
        test14_coordinator_no_capability_match,
        test15_coordinator_load_balance,
        test16_coordinator_history_performance,
        test17_coordinator_report_result,
        test18_coordinator_report_failure,
        test19_coordinator_send_message,
        test20_coordinator_broadcast,
        test21_coordinator_status,
        test22_coordinator_coordinate,
        test23_coordinator_task_result,
        test24_coordinator_reset,
        test25_coordinator_health_check,
        test26_multi_agent_workflow,
        test27_core_coordinator_registered,
        test28_core_cycle_log_coordination,
        test29_core_metrics_coord,
        test30_core_distribute_task,
        test31_core_reset_coordinator,
        test32_performance,
        test33_message_queue_limit,
        test34_parallel_tasks,
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
