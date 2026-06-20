"""
第十六轮测试：阶段26技能学习与增强验证
验证SkillLibrary、SkillAcquisition、Skill数据类、核心集成
核心目标：证明Agent能从经验中学习技能并用于决策
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
from seva.skill import Skill, SkillLibrary, SkillAcquisition
from seva.types import (Goal, GoalType, ActionType, Action, Drives,
                        PerceptionState, ExperienceRecord, gen_id)

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


def make_drives(safety=0.5, gain=0.5, efficiency=0.5):
    return Drives(safety=safety, gain=gain, efficiency=efficiency)


def make_perception():
    return PerceptionState(internal=np.zeros(16),
                           external=np.zeros(32),
                           precision=np.ones(16),
                           timestamp=time.time())


def make_record(goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                reward=1.0, params=None):
    goal = Goal(goal_id=gen_id("goal"), type=goal_type,
                parameters={}, priority=0.8)
    action = Action(action_id=gen_id("act"), type=action_type,
                    parameters=params or {}, expected_reward=reward,
                    source="test")
    return ExperienceRecord(
        record_id=gen_id("exp"), timestamp=time.time(),
        state=make_perception(), drives=make_drives(),
        goal=goal, prediction=None, action=action,
        error=None, attribution=None, reward=reward
    )


# ============================================================
# 测试1：Skill数据类基础
# ============================================================
def test1_skill_dataclass():
    section("测试1：Skill数据类基础")
    skill = Skill(
        skill_id="s1", name="explore_move",
        goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
        action_parameters={"direction": "up"}
    )
    report("初始q_value=0", skill.q_value == 0.0)
    report("初始success_count=0", skill.success_count == 0)
    report("初始use_count=0", skill.use_count == 0)
    report("初始success_rate=0", skill.success_rate == 0.0)
    report("初始value=0", skill.value == 0.0)
    report("初始不可靠", not skill.is_reliable)


# ============================================================
# 测试2：Skill TD学习更新
# ============================================================
def test2_skill_update():
    section("测试2：Skill TD学习更新")
    skill = Skill(
        skill_id="s1", name="test_skill",
        goal_type=GoalType.COLLECT, action_type=ActionType.COLLECT,
    )
    # 正reward更新
    skill.update(reward=1.0, success=True, learning_rate=0.1)
    report("1次更新后q_value=0.1", abs(skill.q_value - 0.1) < 1e-6)
    report("1次更新后success_count=1", skill.success_count == 1)
    report("1次更新后use_count=1", skill.use_count == 1)
    report("1次更新后total_reward=1.0", abs(skill.total_reward - 1.0) < 1e-6)

    # 继续更新
    skill.update(reward=1.0, success=True, learning_rate=0.1)
    report("2次更新后q_value=0.19", abs(skill.q_value - 0.19) < 1e-6)
    report("2次更新后success_count=2", skill.success_count == 2)

    # 负reward更新
    skill.update(reward=-1.0, success=False, learning_rate=0.1)
    report("3次更新后fail_count=1", skill.fail_count == 1)
    report("3次更新后q_value=0.071", abs(skill.q_value - 0.071) < 1e-4)


# ============================================================
# 测试3：Skill属性
# ============================================================
def test3_skill_properties():
    section("测试3：Skill属性")
    skill = Skill(
        skill_id="s1", name="test",
        goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
        q_value=0.8
    )
    skill.success_count = 7
    skill.fail_count = 3
    report("success_rate=0.7", abs(skill.success_rate - 0.7) < 1e-6)
    report("value=0.56", abs(skill.value - 0.56) < 1e-6)
    report("is_reliable(use>=3)", skill.is_reliable)

    # 不足3次
    skill2 = Skill(skill_id="s2", name="test2",
                   goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE)
    skill2.success_count = 1
    skill2.fail_count = 1
    report("use<3不可靠", not skill2.is_reliable)


# ============================================================
# 测试4：Skill to_action
# ============================================================
def test4_skill_to_action():
    section("测试4：Skill转动作")
    skill = Skill(
        skill_id="s1", name="collect_skill",
        goal_type=GoalType.COLLECT, action_type=ActionType.COLLECT,
        action_parameters={"target": "resource"},
        q_value=0.9
    )
    action = skill.to_action()
    report("action类型COLLECT", action.type == ActionType.COLLECT)
    report("action参数正确", action.parameters == {"target": "resource"})
    report("action source=skill", action.source == "skill")
    report("action expected_reward=q_value", abs(action.expected_reward - 0.9) < 1e-6)
    report("action有id", action.action_id is not None)


# ============================================================
# 测试5：Skill to_dict
# ============================================================
def test5_skill_to_dict():
    section("测试5：Skill序列化")
    skill = Skill(
        skill_id="s1", name="test_skill",
        goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
        q_value=0.5
    )
    skill.success_count = 4
    skill.fail_count = 1
    skill.use_count = 5
    d = skill.to_dict()
    report("含skill_id", d["skill_id"] == "s1")
    report("含name", d["name"] == "test_skill")
    report("含goal_type", d["goal_type"] == "explore")
    report("含action_type", d["action_type"] == "move")
    report("含q_value", d["q_value"] == 0.5)
    report("含success_rate", d["success_rate"] == 0.8)
    report("含value", abs(d["value"] - 0.4) < 1e-6)
    report("含use_count", d["use_count"] == 5)
    report("含is_reliable", d["is_reliable"] is True)


# ============================================================
# 测试6：SkillLibrary基础 - add
# ============================================================
def test6_library_add():
    section("测试6：SkillLibrary添加技能")
    lib = SkillLibrary(Config())
    skill = Skill(
        skill_id="s1", name="test",
        goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
    )
    sid = lib.add(skill)
    report("add返回skill_id", sid == "s1")
    report("skills非空", len(lib.skills) == 1)
    report("技能存在", "s1" in lib.skills)


# ============================================================
# 测试7：SkillLibrary - 去重
# ============================================================
def test7_library_dedup():
    section("测试7：SkillLibrary去重")
    lib = SkillLibrary(Config())
    skill1 = Skill(skill_id="s1", name="test1",
                   goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                   action_parameters={"direction": "up"})
    skill2 = Skill(skill_id="s2", name="test2",
                   goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                   action_parameters={"direction": "up"})
    lib.add(skill1)
    sid2 = lib.add(skill2)
    report("重复技能合并", sid2 == "s1")
    report("仍只有1个技能", len(lib.skills) == 1)


# ============================================================
# 测试8：SkillLibrary - retrieve
# ============================================================
def test8_library_retrieve():
    section("测试8：SkillLibrary检索")
    lib = SkillLibrary(Config())
    # 添加一个可靠技能
    skill = Skill(skill_id="s1", name="explore_move",
                  goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                  q_value=0.8)
    skill.success_count = 5
    skill.fail_count = 2
    lib.add(skill)

    goal = make_goal(GoalType.EXPLORE)
    retrieved = lib.retrieve(goal)
    report("检索到匹配技能", retrieved is not None)
    report("检索到正确技能", retrieved.skill_id == "s1")

    # 不匹配的目标
    goal2 = make_goal(GoalType.COLLECT)
    retrieved2 = lib.retrieve(goal2)
    report("不匹配目标返回None", retrieved2 is None)


# ============================================================
# 测试9：SkillLibrary - retrieve按价值排序
# ============================================================
def test9_library_retrieve_by_value():
    section("测试9：SkillLibrary按价值检索")
    lib = SkillLibrary(Config())
    # 低价值技能
    skill1 = Skill(skill_id="s1", name="low",
                   goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                   q_value=0.2)
    skill1.success_count = 3
    skill1.fail_count = 1
    lib.add(skill1)
    # 高价值技能
    skill2 = Skill(skill_id="s2", name="high",
                   goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                   action_parameters={"direction": "up"},
                   q_value=0.9)
    skill2.success_count = 4
    skill2.fail_count = 1
    lib.add(skill2)

    goal = make_goal(GoalType.EXPLORE)
    retrieved = lib.retrieve(goal)
    report("检索到高价值技能", retrieved is not None)
    report("检索到s2", retrieved.skill_id == "s2")


# ============================================================
# 测试10：SkillLibrary - update
# ============================================================
def test10_library_update():
    section("测试10：SkillLibrary更新Q值")
    lib = SkillLibrary(Config())
    skill = Skill(skill_id="s1", name="test",
                  goal_type=GoalType.COLLECT, action_type=ActionType.COLLECT,
                  q_value=0.5)
    lib.add(skill)

    lib.update("s1", reward=1.0, success=True)
    updated = lib.skills["s1"]
    report("更新后use_count=1", updated.use_count == 1)
    report("更新后success_count=1", updated.success_count == 1)
    report("Q值增加", updated.q_value > 0.5)
    report("total_reward=1.0", abs(updated.total_reward - 1.0) < 1e-6)


# ============================================================
# 测试11：SkillLibrary - prune
# ============================================================
def test11_library_prune():
    section("测试11：SkillLibrary修剪")
    lib = SkillLibrary(Config())
    # 低价值技能（使用>5次，价值极低）
    bad_skill = Skill(skill_id="s1", name="bad",
                      goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                      q_value=-1.5)
    bad_skill.success_count = 1
    bad_skill.fail_count = 9
    bad_skill.use_count = 10
    lib.add(bad_skill)
    # 好技能
    good_skill = Skill(skill_id="s2", name="good",
                       goal_type=GoalType.COLLECT, action_type=ActionType.COLLECT,
                       q_value=0.8)
    good_skill.success_count = 5
    good_skill.fail_count = 1
    lib.add(good_skill)

    lib.prune()
    report("低价值技能被修剪", "s1" not in lib.skills)
    report("好技能保留", "s2" in lib.skills)


# ============================================================
# 测试12：SkillLibrary - get_stats
# ============================================================
def test12_library_stats():
    section("测试12：SkillLibrary统计")
    lib = SkillLibrary(Config())
    skill = Skill(skill_id="s1", name="test",
                  goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
                  q_value=0.5)
    skill.success_count = 4
    skill.fail_count = 1
    skill.use_count = 5
    lib.add(skill)

    stats = lib.get_stats()
    report("含total_skills", stats["total_skills"] == 1)
    report("含reliable_skills", stats["reliable_skills"] == 1)
    report("含total_uses", stats["total_uses"] == 5)
    report("含total_successes", stats["total_successes"] == 4)
    report("含avg_q_value", abs(stats["avg_q_value"] - 0.5) < 1e-6)
    report("含best_skill", stats["best_skill"] is not None)


# ============================================================
# 测试13：SkillLibrary - get_all_skills
# ============================================================
def test13_library_get_all():
    section("测试13：SkillLibrary获取所有技能")
    lib = SkillLibrary(Config())
    s1 = Skill(skill_id="s1", name="low",
               goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE, q_value=0.1)
    s1.success_count = 3
    s2 = Skill(skill_id="s2", name="high",
               goal_type=GoalType.COLLECT, action_type=ActionType.COLLECT, q_value=0.9)
    s2.success_count = 3
    lib.add(s1)
    lib.add(s2)

    all_skills = lib.get_all_skills()
    report("返回2个技能", len(all_skills) == 2)
    report("按价值排序", all_skills[0].q_value >= all_skills[1].q_value)


# ============================================================
# 测试14：SkillLibrary - reset
# ============================================================
def test14_library_reset():
    section("测试14：SkillLibrary重置")
    lib = SkillLibrary(Config())
    skill = Skill(skill_id="s1", name="test",
                  goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE)
    lib.add(skill)
    lib.reset()
    report("reset后skills空", len(lib.skills) == 0)


# ============================================================
# 测试15：SkillLibrary - health_check
# ============================================================
def test15_library_health_check():
    section("测试15：SkillLibrary健康检查")
    lib = SkillLibrary(Config())
    report("health_check通过", lib.health_check())


# ============================================================
# 测试16：SkillAcquisition - observe
# ============================================================
def test16_acquisition_observe():
    section("测试16：SkillAcquisition观察经验")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    record = make_record(GoalType.EXPLORE, ActionType.MOVE, reward=1.0)
    acq.observe(record)
    report("buffer有1条", len(acq.pattern_buffer) == 1)
    report("追踪1个模式", len(acq._pattern_counts) == 1)

    # 观察更多相同模式
    for _ in range(4):
        acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE, reward=1.0))
    report("5条相同模式", acq._pattern_counts[(GoalType.EXPLORE, ActionType.MOVE)] == 5)


# ============================================================
# 测试17：SkillAcquisition - detect_and_create
# ============================================================
def test17_acquisition_detect():
    section("测试17：SkillAcquisition检测并创建技能")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    # 观察5次相同模式（超过阈值3）
    for _ in range(5):
        acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE, reward=1.0))

    new_id = acq.detect_and_create()
    report("创建新技能", new_id is not None)
    report("技能库有1个技能", len(lib.skills) == 1)
    skill = lib.get_all_skills()[0]
    report("技能goal_type=EXPLORE", skill.goal_type == GoalType.EXPLORE)
    report("技能action_type=MOVE", skill.action_type == ActionType.MOVE)
    report("技能q_value=1.0", abs(skill.q_value - 1.0) < 1e-6)
    report("技能use_count=5", skill.use_count == 5)


# ============================================================
# 测试18：SkillAcquisition - 不重复创建
# ============================================================
def test18_acquisition_no_duplicate():
    section("测试18：SkillAcquisition不重复创建")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    for _ in range(5):
        acq.observe(make_record(GoalType.COLLECT, ActionType.COLLECT, reward=1.0))
    acq.detect_and_create()
    report("第一次创建1个技能", len(lib.skills) == 1)

    # 再观察并检测，不应创建重复技能
    for _ in range(5):
        acq.observe(make_record(GoalType.COLLECT, ActionType.COLLECT, reward=1.0))
    acq.detect_and_create()
    report("不重复创建", len(lib.skills) == 1)


# ============================================================
# 测试19：SkillAcquisition - 多模式检测
# ============================================================
def test19_acquisition_multi_pattern():
    section("测试19：SkillAcquisition多模式检测")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    # 两种模式
    for _ in range(4):
        acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE, reward=0.5))
    for _ in range(4):
        acq.observe(make_record(GoalType.COLLECT, ActionType.COLLECT, reward=1.0))

    acq.detect_and_create()
    report("创建2个技能", len(lib.skills) == 2)
    skills = lib.get_all_skills()
    goal_types = {s.goal_type for s in skills}
    report("含EXPLORE技能", GoalType.EXPLORE in goal_types)
    report("含COLLECT技能", GoalType.COLLECT in goal_types)


# ============================================================
# 测试20：SkillAcquisition - 阈值不足不创建
# ============================================================
def test20_acquisition_below_threshold():
    section("测试20：SkillAcquisition阈值不足不创建")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    # 只观察2次（阈值=3）
    for _ in range(2):
        acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE, reward=1.0))
    acq.detect_and_create()
    report("阈值不足不创建", len(lib.skills) == 0)


# ============================================================
# 测试21：SkillAcquisition - 负reward环境也能学习
# ============================================================
def test21_acquisition_negative_reward():
    section("测试21：SkillAcquisition负reward环境学习")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    # 所有reward都是-0.01（步进成本）
    for _ in range(5):
        acq.observe(make_record(GoalType.MAINTAIN, ActionType.WAIT, reward=-0.01))

    acq.detect_and_create()
    report("负reward也创建技能", len(lib.skills) == 1)
    skill = lib.get_all_skills()[0]
    report("q_value=-0.01", abs(skill.q_value - (-0.01)) < 1e-6)


# ============================================================
# 测试22：SkillAcquisition - get_stats
# ============================================================
def test23_acquisition_stats():
    section("测试22：SkillAcquisition统计")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    for _ in range(5):
        acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE, reward=1.0))

    stats = acq.get_stats()
    report("含buffer_size", stats["buffer_size"] == 5)
    report("含tracked_patterns", stats["tracked_patterns"] == 1)
    report("含patterns_above_threshold", stats["patterns_above_threshold"] == 1)


# ============================================================
# 测试23：SkillAcquisition - reset
# ============================================================
def test24_acquisition_reset():
    section("测试23：SkillAcquisition重置")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)
    acq.observe(make_record())
    acq.reset()
    report("reset后buffer空", len(acq.pattern_buffer) == 0)
    report("reset后patterns空", len(acq._pattern_counts) == 0)


# ============================================================
# 测试24：Core集成 - 模块注册
# ============================================================
def test25_core_modules_registered():
    section("测试24：Core注册技能模块")
    core = SelfEvolvingAgentCore(Config())
    report("skill_library在modules中", "skill_library" in core.modules)
    report("skill_acquisition在modules中", "skill_acquisition" in core.modules)
    report("skill_library是SkillLibrary实例",
           isinstance(core.modules["skill_library"], SkillLibrary))
    report("skill_acquisition是SkillAcquisition实例",
           isinstance(core.modules["skill_acquisition"], SkillAcquisition))


# ============================================================
# 测试25：Core集成 - cycle_log含skill_update
# ============================================================
def test26_core_cycle_log_skill():
    section("测试25：Core周期日志含skill_update")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    core.run_cycle()
    log = core.last_cycle_log
    stages = [e["stage"] for e in log]
    report("cycle_log含skill_update", "skill_update" in stages)

    skill_entry = next(e for e in log if e["stage"] == "skill_update")
    report("skill_update含skill_used", "skill_used" in skill_entry["result"])
    report("skill_update含total_skills", "total_skills" in skill_entry["result"])
    report("skill_update含reliable_skills", "reliable_skills" in skill_entry["result"])


# ============================================================
# 测试26：Core集成 - metrics含skill_stats
# ============================================================
def test27_core_metrics_skill():
    section("测试26：Core指标含skill_stats")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    core.run_cycle()
    metrics = core.get_metrics()
    report("metrics含skill_stats", "skill_stats" in metrics)
    ss = metrics["skill_stats"]
    report("skill_stats含total_skills", "total_skills" in ss)
    report("skill_stats含reliable_skills", "reliable_skills" in ss)
    report("skill_stats含total_uses", "total_uses" in ss)


# ============================================================
# 测试27：Core集成 - 手动注入技能后被使用
# ============================================================
def test28_core_skill_used():
    section("测试27：Core使用注入技能")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 手动注入一个可靠技能
    lib = core.modules["skill_library"]
    skill = Skill(
        skill_id="s1", name="maintain_wait",
        goal_type=GoalType.MAINTAIN, action_type=ActionType.WAIT,
        q_value=0.5
    )
    skill.success_count = 5
    skill.fail_count = 1
    skill.use_count = 6
    lib.add(skill)

    core.run_cycle()
    log = core.last_cycle_log
    decision_entry = next(e for e in log if e["stage"] == "decision_maker")
    report("使用了技能", decision_entry["result"]["skill_used"] is not None)
    report("action source=skill", decision_entry["result"]["source"] == "skill")

    # 验证技能Q值被更新
    skill_after = lib.skills["s1"]
    report("技能use_count增加", skill_after.use_count > 6)


# ============================================================
# 测试28：Core集成 - 技能Q值通过TD学习更新
# ============================================================
def test29_core_skill_q_update():
    section("测试28：Core技能Q值TD更新")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    lib = core.modules["skill_library"]
    skill = Skill(
        skill_id="s1", name="maintain_wait",
        goal_type=GoalType.MAINTAIN, action_type=ActionType.WAIT,
        q_value=0.0
    )
    skill.success_count = 5
    skill.fail_count = 1
    skill.use_count = 6
    lib.add(skill)

    initial_q = lib.skills["s1"].q_value
    # 运行5个周期，技能应被反复使用和更新
    for _ in range(5):
        core.run_cycle()
    updated_q = lib.skills["s1"].q_value
    report("Q值被更新", updated_q != initial_q)
    report("use_count增加", lib.skills["s1"].use_count > 6)


# ============================================================
# 测试29：Core集成 - 技能获取自动触发
# ============================================================
def test30_core_auto_acquisition():
    section("测试29：Core自动技能获取")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 运行足够周期让技能获取触发（acquisition_interval=50）
    for _ in range(100):
        core.run_cycle()

    # 验证skill_acquisition有观察经验
    acq = core.modules["skill_acquisition"]
    report("acquisition有观察经验", len(acq.pattern_buffer) > 0)
    report("acquisition追踪了模式", len(acq._pattern_counts) > 0)


# ============================================================
# 测试30：Core集成 - reset清空技能
# ============================================================
def test31_core_reset_skills():
    section("测试30：Core重置清空技能")
    core = SelfEvolvingAgentCore(Config())
    lib = core.modules["skill_library"]
    skill = Skill(skill_id="s1", name="test",
                  goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE)
    lib.add(skill)

    core.reset()
    report("reset后skill_library空", len(core.modules["skill_library"].skills) == 0)
    report("reset后skill_acquisition空",
           len(core.modules["skill_acquisition"].pattern_buffer) == 0)


# ============================================================
# 测试31：Core集成 - 无环境时技能系统工作
# ============================================================
def test32_core_no_env_skills():
    section("测试31：Core无环境时技能系统工作")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))

    core.run_cycle()
    log = core.last_cycle_log
    stages = [e["stage"] for e in log]
    report("无环境也有skill_update", "skill_update" in stages)


# ============================================================
# 测试32：Core集成 - decision_maker回退
# ============================================================
def test33_core_fallback_to_decision():
    section("测试32：Core无技能时回退到决策")
    core = SelfEvolvingAgentCore(Config())
    env = SimulationEnvironment(Config())
    core.attach_environment(env)
    obs = env.reset()
    core.inject_input(obs)

    # 不注入技能，应回退到decision_maker
    core.run_cycle()
    log = core.last_cycle_log
    decision_entry = next(e for e in log if e["stage"] == "decision_maker")
    report("无技能时source=decision",
           decision_entry["result"]["source"] in ("decision", "safe_fallback"))
    report("无技能时skill_used=None",
           decision_entry["result"]["skill_used"] is None)


# ============================================================
# 测试33：性能测试
# ============================================================
def test34_performance():
    section("测试33：技能系统性能开销")
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

    report("技能系统<30ms/周期", elapsed < 30, f"{elapsed:.2f}ms/周期")


# ============================================================
# 测试34：技能库容量限制
# ============================================================
def test35_library_capacity():
    section("测试34：SkillLibrary容量限制")
    lib = SkillLibrary(Config())
    # 添加超过max_skills个技能
    for i in range(105):
        skill = Skill(
            skill_id=f"s{i}", name=f"skill_{i}",
            goal_type=GoalType.EXPLORE, action_type=ActionType.MOVE,
            action_parameters={"idx": i},  # 唯一参数避免去重
            q_value=0.5
        )
        skill.success_count = 3
        lib.add(skill)
    report("不超过max_skills", len(lib.skills) <= lib.max_skills)


# ============================================================
# 测试35：技能获取 - 不同参数同模式
# ============================================================
def test36_acquisition_different_params():
    section("测试35：SkillAcquisition不同参数同模式")
    lib = SkillLibrary(Config())
    acq = SkillAcquisition(Config(), lib)

    # 相同(goal_type, action_type)但不同参数
    acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE,
                           reward=0.5, params={"direction": "up"}))
    acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE,
                           reward=0.5, params={"direction": "down"}))
    acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE,
                           reward=0.5, params={"direction": "left"}))
    acq.observe(make_record(GoalType.EXPLORE, ActionType.MOVE,
                           reward=0.5, params={"direction": "right"}))

    # 应识别为同一模式（4次 >= 阈值3）
    report("不同参数识别为同模式",
           acq._pattern_counts[(GoalType.EXPLORE, ActionType.MOVE)] == 4)

    acq.detect_and_create()
    report("创建1个技能", len(lib.skills) == 1)
    skill = lib.get_all_skills()[0]
    report("技能有参数", len(skill.action_parameters) > 0)


def main():
    tests = [
        test1_skill_dataclass,
        test2_skill_update,
        test3_skill_properties,
        test4_skill_to_action,
        test5_skill_to_dict,
        test6_library_add,
        test7_library_dedup,
        test8_library_retrieve,
        test9_library_retrieve_by_value,
        test10_library_update,
        test11_library_prune,
        test12_library_stats,
        test13_library_get_all,
        test14_library_reset,
        test15_library_health_check,
        test16_acquisition_observe,
        test17_acquisition_detect,
        test18_acquisition_no_duplicate,
        test19_acquisition_multi_pattern,
        test20_acquisition_below_threshold,
        test21_acquisition_negative_reward,
        test23_acquisition_stats,
        test24_acquisition_reset,
        test25_core_modules_registered,
        test26_core_cycle_log_skill,
        test27_core_metrics_skill,
        test28_core_skill_used,
        test29_core_skill_q_update,
        test30_core_auto_acquisition,
        test31_core_reset_skills,
        test32_core_no_env_skills,
        test33_core_fallback_to_decision,
        test34_performance,
        test35_library_capacity,
        test36_acquisition_different_params,
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
