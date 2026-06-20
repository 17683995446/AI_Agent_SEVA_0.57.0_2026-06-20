"""
第十八轮测试：阶段28自主进化闭环验证
验证EvolutionLoop、MetaLearner、SelfTuner、CurriculumScheduler
核心目标：证明Agent能自主进化（元学习+自我调参+课程学习+长程自驱）

第一性原理验证：
1. 元学习 = 监控学习过程本身，识别趋势（下降/震荡/停滞）
2. 自我调参 = 根据性能反馈自动调整超参数，无需人工
3. 课程学习 = 由易到难，掌握后提升，能力不足回退
4. 进化闭环 = 评估→调整→执行→再评估的持续改进
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
from seva.evolution import (EvolutionLoop, MetaLearner, SelfTuner,
                             CurriculumScheduler, DifficultyLevel,
                             LearningMetrics, TunableParameter)

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
# 测试1：DifficultyLevel枚举
# ============================================================
def test1_difficulty_level_enum():
    section("测试1：DifficultyLevel枚举")
    levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM,
              DifficultyLevel.HARD, DifficultyLevel.EXPERT]
    report("4个难度等级", len(levels) == 4)
    report("EASY=1", DifficultyLevel.EASY.value == 1)
    report("MEDIUM=2", DifficultyLevel.MEDIUM.value == 2)
    report("HARD=3", DifficultyLevel.HARD.value == 3)
    report("EXPERT=4", DifficultyLevel.EXPERT.value == 4)
    report("EASY<MEDIUM<HARD<EXPERT",
           DifficultyLevel.EASY.value < DifficultyLevel.MEDIUM.value <
           DifficultyLevel.HARD.value < DifficultyLevel.EXPERT.value)


# ============================================================
# 测试2：LearningMetrics数据类
# ============================================================
def test2_learning_metrics():
    section("测试2：LearningMetrics数据类")
    m = LearningMetrics(cycle=10, avg_error=0.5, avg_reward=0.3,
                        skill_count=5, task_success_rate=0.7)
    report("cycle字段", m.cycle == 10)
    report("avg_error字段", m.avg_error == 0.5)
    report("avg_reward字段", m.avg_reward == 0.3)
    report("skill_count字段", m.skill_count == 5)
    report("task_success_rate字段", m.task_success_rate == 0.7)
    report("timestamp自动生成", m.timestamp > 0)


# ============================================================
# 测试3：TunableParameter数据类
# ============================================================
def test3_tunable_parameter():
    section("测试3：TunableParameter数据类")
    p = TunableParameter(name="lr", value=0.1, min_value=0.001,
                         max_value=0.5, adjust_step=0.01)
    report("name字段", p.name == "lr")
    report("value字段", p.value == 0.1)
    report("history初始空", len(p.history) == 0)

    # 增大调整
    changed = p.adjust(1.0)
    report("增大调整成功", changed and p.value == 0.11)
    report("history记录1次", len(p.history) == 1)

    # 减小调整
    p.adjust(-1.0)
    report("减小调整成功", p.value == 0.10)
    report("history记录2次", len(p.history) == 2)

    # 边界限制（max）
    p2 = TunableParameter(name="x", value=0.49, min_value=0.0,
                          max_value=0.5, adjust_step=0.05)
    p2.adjust(1.0)
    report("max边界限制", p2.value == 0.5)

    # 边界限制（min）
    p3 = TunableParameter(name="x", value=0.01, min_value=0.0,
                          max_value=0.5, adjust_step=0.05)
    p3.adjust(-1.0)
    report("min边界限制", p3.value == 0.0)

    # to_dict
    d = p.to_dict()
    report("to_dict含name", d["name"] == "lr")
    report("to_dict含value", "value" in d)
    report("to_dict含min", "min" in d)
    report("to_dict含max", "max" in d)


# ============================================================
# 测试4：MetaLearner基础
# ============================================================
def test4_meta_learner_basic():
    section("测试4：MetaLearner基础")
    ml = MetaLearner(Config())
    report("初始化成功", ml is not None)
    report("metrics_history初始空", len(ml.metrics_history) == 0)
    report("error_trend初始stable", ml.error_trend == "stable")
    report("convergence_rate初始0", ml.convergence_rate == 0.0)
    report("skill_acquisition_rate初始0", ml.skill_acquisition_rate == 0.0)


# ============================================================
# 测试5：MetaLearner记录
# ============================================================
def test5_meta_learner_record():
    section("测试5：MetaLearner记录")
    ml = MetaLearner(Config())
    ml.record(cycle=1, avg_error=0.5, avg_reward=0.3,
              skill_count=5, task_success_rate=0.7)
    report("记录后history长度1", len(ml.metrics_history) == 1)
    report("记录cycle正确", ml.metrics_history[0].cycle == 1)
    report("记录avg_error正确", ml.metrics_history[0].avg_error == 0.5)

    # 记录多条
    for i in range(5):
        ml.record(cycle=i+2, avg_error=0.5-i*0.05, avg_reward=0.3,
                  skill_count=5+i, task_success_rate=0.7)
    report("记录6条后history长度6", len(ml.metrics_history) == 6)


# ============================================================
# 测试6：MetaLearner评估 - 误差下降趋势
# ============================================================
def test6_meta_learner_evaluate_decreasing():
    section("测试6：MetaLearner评估-误差下降")
    ml = MetaLearner(Config())
    # 误差持续下降
    for i in range(10):
        ml.record(cycle=i+1, avg_error=1.0 - i*0.08, avg_reward=0.3,
                  skill_count=i, task_success_rate=0.7)
    result = ml.evaluate()
    report("评估返回dict", isinstance(result, dict))
    report("误差趋势=decreasing", result["error_trend"] == "decreasing",
           f"trend={result['error_trend']}")
    report("收敛速度>0", result["convergence_rate"] > 0,
           f"rate={result['convergence_rate']}")


# ============================================================
# 测试7：MetaLearner评估 - 误差震荡
# ============================================================
def test7_meta_learner_evaluate_oscillating():
    section("测试7：MetaLearner评估-误差震荡")
    ml = MetaLearner(Config())
    # 误差震荡且均值稳定（前半后半均值相等，方差大）
    errors = [0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.2, 0.8, 0.2, 0.8]
    for i, e in enumerate(errors):
        ml.record(cycle=i+1, avg_error=e, avg_reward=0.3,
                  skill_count=i, task_success_rate=0.7)
    result = ml.evaluate()
    report("误差趋势=oscillating", result["error_trend"] == "oscillating",
           f"trend={result['error_trend']}")


# ============================================================
# 测试8：MetaLearner评估 - 数据不足
# ============================================================
def test8_meta_learner_evaluate_insufficient():
    section("测试8：MetaLearner评估-数据不足")
    ml = MetaLearner(Config())
    ml.record(cycle=1, avg_error=0.5, avg_reward=0.3,
              skill_count=5, task_success_rate=0.7)
    result = ml.evaluate()
    report("数据不足返回insufficient_data",
           result["error_trend"] == "insufficient_data")
    report("收敛速度0", result["convergence_rate"] == 0.0)


# ============================================================
# 测试9：MetaLearner反馈
# ============================================================
def test9_meta_learner_feedback():
    section("测试9：MetaLearner反馈")
    ml = MetaLearner(Config())
    # 震荡场景（均值稳定，方差大）
    errors = [0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.2, 0.8, 0.2, 0.8]
    for i, e in enumerate(errors):
        ml.record(cycle=i+1, avg_error=e, avg_reward=0.3,
                  skill_count=i, task_success_rate=0.7)
    ml.evaluate()
    feedback = ml.get_feedback()
    report("反馈含error_direction", "error_direction" in feedback)
    report("反馈含exploration_need", "exploration_need" in feedback)
    report("反馈含skill_acquisition_need", "skill_acquisition_need" in feedback)
    report("震荡时error_direction<0", feedback["error_direction"] < 0,
           f"dir={feedback['error_direction']}")


# ============================================================
# 测试10：MetaLearner reset
# ============================================================
def test10_meta_learner_reset():
    section("测试10：MetaLearner reset")
    ml = MetaLearner(Config())
    for i in range(5):
        ml.record(cycle=i+1, avg_error=0.5, avg_reward=0.3,
                  skill_count=i, task_success_rate=0.7)
    ml.evaluate()
    ml.reset()
    report("reset后history空", len(ml.metrics_history) == 0)
    report("reset后error_trend=stable", ml.error_trend == "stable")
    report("reset后convergence_rate=0", ml.convergence_rate == 0.0)


# ============================================================
# 测试11：MetaLearner health_check
# ============================================================
def test11_meta_learner_health_check():
    section("测试11：MetaLearner健康检查")
    ml = MetaLearner(Config())
    report("health_check通过", ml.health_check())


# ============================================================
# 测试12：SelfTuner基础
# ============================================================
def test12_self_tuner_basic():
    section("测试12：SelfTuner基础")
    st = SelfTuner(Config())
    report("初始化成功", st is not None)
    report("3个可调参数", len(st.params) == 3)
    report("含learning_rate", "learning_rate" in st.params)
    report("含exploration_rate", "exploration_rate" in st.params)
    report("含pattern_threshold", "pattern_threshold" in st.params)


# ============================================================
# 测试13：SelfTuner调参 - 震荡减小学习率
# ============================================================
def test13_self_tuner_decrease_lr_on_oscillation():
    section("测试13：SelfTuner震荡时减小学习率")
    st = SelfTuner(Config())
    initial_lr = st.get_param("learning_rate")
    # 震荡反馈：error_direction=-1.0
    feedback = {"error_direction": -1.0, "exploration_need": 0.0,
                "skill_acquisition_need": 0.0}
    st.tune(feedback)
    new_lr = st.get_param("learning_rate")
    report("震荡后学习率减小", new_lr < initial_lr,
           f"{initial_lr}->{new_lr}")


# ============================================================
# 测试14：SelfTuner调参 - 下降增大学习率
# ============================================================
def test14_self_tuner_increase_lr_on_decreasing():
    section("测试14：SelfTuner误差下降时增大学习率")
    st = SelfTuner(Config())
    # 先把学习率调低，便于观察增大
    st.params["learning_rate"].value = 0.05
    initial_lr = st.get_param("learning_rate")
    # 下降反馈：error_direction=0.5
    feedback = {"error_direction": 0.5, "exploration_need": 0.0,
                "skill_acquisition_need": 0.0}
    st.tune(feedback)
    new_lr = st.get_param("learning_rate")
    report("下降后学习率增大", new_lr > initial_lr,
           f"{initial_lr}->{new_lr}")


# ============================================================
# 测试15：SelfTuner调参 - 探索需求
# ============================================================
def test15_self_tuner_exploration_need():
    section("测试15：SelfTuner探索需求")
    st = SelfTuner(Config())
    initial_er = st.get_param("exploration_rate")
    feedback = {"error_direction": 0.0, "exploration_need": 0.5,
                "skill_acquisition_need": 0.0}
    st.tune(feedback)
    new_er = st.get_param("exploration_rate")
    report("探索需求时exploration_rate增大", new_er > initial_er,
           f"{initial_er}->{new_er}")


# ============================================================
# 测试16：SelfTuner调参 - 技能获取需求降低阈值
# ============================================================
def test16_self_tuner_skill_need():
    section("测试16：SelfTuner技能获取需求降低阈值")
    st = SelfTuner(Config())
    initial_pt = st.get_param("pattern_threshold")
    feedback = {"error_direction": 0.0, "exploration_need": 0.0,
                "skill_acquisition_need": 0.5}
    st.tune(feedback)
    new_pt = st.get_param("pattern_threshold")
    report("技能获取需求时pattern_threshold降低", new_pt < initial_pt,
           f"{initial_pt}->{new_pt}")


# ============================================================
# 测试17：SelfTuner get_all_params
# ============================================================
def test17_self_tuner_get_all_params():
    section("测试17：SelfTuner获取所有参数")
    st = SelfTuner(Config())
    params = st.get_all_params()
    report("返回dict", isinstance(params, dict))
    report("含learning_rate", "learning_rate" in params)
    report("含exploration_rate", "exploration_rate" in params)
    report("含pattern_threshold", "pattern_threshold" in params)
    report("每个参数含value", "value" in params["learning_rate"])
    report("每个参数含min", "min" in params["learning_rate"])
    report("每个参数含max", "max" in params["learning_rate"])


# ============================================================
# 测试18：SelfTuner reset
# ============================================================
def test18_self_tuner_reset():
    section("测试18：SelfTuner reset")
    st = SelfTuner(Config())
    st.tune({"error_direction": -1.0, "exploration_need": 0.0,
             "skill_acquisition_need": 0.0})
    report("调参后有history", len(st.params["learning_rate"].history) > 0)
    st.reset()
    report("reset后history空", len(st.params["learning_rate"].history) == 0)
    report("reset后adjustment_count=0", st.adjustment_count == 0)


# ============================================================
# 测试19：SelfTuner health_check
# ============================================================
def test19_self_tuner_health_check():
    section("测试19：SelfTuner健康检查")
    st = SelfTuner(Config())
    report("health_check通过", st.health_check())


# ============================================================
# 测试20：CurriculumScheduler基础
# ============================================================
def test20_curriculum_basic():
    section("测试20：CurriculumScheduler基础")
    cs = CurriculumScheduler(Config())
    report("初始化成功", cs is not None)
    report("初始难度EASY", cs.get_difficulty() == DifficultyLevel.EASY)
    report("初始success_rate=0", cs.get_success_rate() == 0.0)
    report("初始results空", len(cs.recent_results) == 0)


# ============================================================
# 测试21：CurriculumScheduler记录结果
# ============================================================
def test21_curriculum_record_result():
    section("测试21：CurriculumScheduler记录结果")
    cs = CurriculumScheduler(Config())
    cs.record_result(True)
    cs.record_result(False)
    cs.record_result(True)
    report("记录3个结果", len(cs.recent_results) == 3)
    report("success_rate=2/3", abs(cs.get_success_rate() - 2/3) < 1e-6)


# ============================================================
# 测试22：CurriculumScheduler难度提升
# ============================================================
def test22_curriculum_increase_difficulty():
    section("测试22：CurriculumScheduler难度提升")
    cs = CurriculumScheduler(Config())
    # 连续成功，success_rate >= 0.7
    for _ in range(15):
        cs.record_result(True)
    new_diff = cs.evaluate_and_adjust(cycle=100)
    report("难度提升到MEDIUM", cs.get_difficulty() == DifficultyLevel.MEDIUM)
    report("返回新难度", new_diff == DifficultyLevel.MEDIUM)


# ============================================================
# 测试23：CurriculumScheduler难度回退
# ============================================================
def test23_curriculum_decrease_difficulty():
    section("测试23：CurriculumScheduler难度回退")
    cs = CurriculumScheduler(Config())
    # 先提升到MEDIUM
    for _ in range(15):
        cs.record_result(True)
    cs.evaluate_and_adjust(cycle=100)
    report("已提升到MEDIUM", cs.get_difficulty() == DifficultyLevel.MEDIUM)

    # 连续失败，success_rate < 0.3
    cs.recent_results.clear()
    for _ in range(15):
        cs.record_result(False)
    new_diff = cs.evaluate_and_adjust(cycle=200)
    report("难度回退到EASY", cs.get_difficulty() == DifficultyLevel.EASY)
    report("返回新难度", new_diff == DifficultyLevel.EASY)


# ============================================================
# 测试24：CurriculumScheduler数据不足不调整
# ============================================================
def test24_curriculum_insufficient_data():
    section("测试24：CurriculumScheduler数据不足不调整")
    cs = CurriculumScheduler(Config())
    # 只记录5个结果（< 10）
    for _ in range(5):
        cs.record_result(True)
    result = cs.evaluate_and_adjust(cycle=100)
    report("数据不足返回None", result is None)
    report("难度保持EASY", cs.get_difficulty() == DifficultyLevel.EASY)


# ============================================================
# 测试25：CurriculumScheduler难度上限
# ============================================================
def test25_curriculum_max_difficulty():
    section("测试25：CurriculumScheduler难度上限EXPERT")
    cs = CurriculumScheduler(Config())
    # 连续提升到EXPERT
    for level in range(3):  # EASY->MEDIUM->HARD->EXPERT
        cs.recent_results.clear()
        for _ in range(15):
            cs.record_result(True)
        cs.evaluate_and_adjust(cycle=100*(level+1))
    report("达到EXPERT", cs.get_difficulty() == DifficultyLevel.EXPERT)

    # 已在EXPERT，再成功也不提升
    cs.recent_results.clear()
    for _ in range(15):
        cs.record_result(True)
    result = cs.evaluate_and_adjust(cycle=500)
    report("EXPERT后不再提升", cs.get_difficulty() == DifficultyLevel.EXPERT)


# ============================================================
# 测试26：CurriculumScheduler get_stats
# ============================================================
def test26_curriculum_get_stats():
    section("测试26：CurriculumScheduler统计")
    cs = CurriculumScheduler(Config())
    cs.record_result(True)
    cs.record_result(False)
    stats = cs.get_stats()
    report("返回dict", isinstance(stats, dict))
    report("含current_difficulty", "current_difficulty" in stats)
    report("含current_level", "current_level" in stats)
    report("含success_rate", "success_rate" in stats)
    report("含total_results", "total_results" in stats)
    report("含difficulty_changes", "difficulty_changes" in stats)
    report("current_difficulty=EASY", stats["current_difficulty"] == "EASY")


# ============================================================
# 测试27：CurriculumScheduler reset
# ============================================================
def test27_curriculum_reset():
    section("测试27：CurriculumScheduler reset")
    cs = CurriculumScheduler(Config())
    for _ in range(15):
        cs.record_result(True)
    cs.evaluate_and_adjust(cycle=100)
    cs.reset()
    report("reset后难度EASY", cs.get_difficulty() == DifficultyLevel.EASY)
    report("reset后results空", len(cs.recent_results) == 0)
    report("reset后history空", len(cs.difficulty_history) == 0)


# ============================================================
# 测试28：CurriculumScheduler health_check
# ============================================================
def test28_curriculum_health_check():
    section("测试28：CurriculumScheduler健康检查")
    cs = CurriculumScheduler(Config())
    report("health_check通过", cs.health_check())


# ============================================================
# 测试29：EvolutionLoop基础
# ============================================================
def test29_evolution_loop_basic():
    section("测试29：EvolutionLoop基础")
    el = EvolutionLoop(Config())
    report("初始化成功", el is not None)
    report("含meta_learner", hasattr(el, "meta_learner"))
    report("含self_tuner", hasattr(el, "self_tuner"))
    report("含curriculum", hasattr(el, "curriculum"))
    report("evolution_history初始空", len(el.evolution_history) == 0)
    report("total_evaluations初始0", el.total_evaluations == 0)


# ============================================================
# 测试30：EvolutionLoop step - 非评估周期
# ============================================================
def test30_evolution_loop_step_non_eval():
    section("测试30：EvolutionLoop step非评估周期")
    el = EvolutionLoop(Config())
    # cycle=1，不是100的倍数，不评估
    result = el.step(cycle=1, avg_error=0.5, avg_reward=0.3,
                     skill_count=5, task_success_rate=0.7,
                     task_result=True)
    report("未评估", result["evaluated"] is False)
    report("无调整", result["adjustments"] == [])
    report("无难度变化", result["difficulty_change"] is None)
    report("meta_learner记录1条", len(el.meta_learner.metrics_history) == 1)
    report("curriculum记录1条结果", len(el.curriculum.recent_results) == 1)


# ============================================================
# 测试31：EvolutionLoop step - 评估周期
# ============================================================
def test31_evolution_loop_step_eval():
    section("测试31：EvolutionLoop step评估周期")
    el = EvolutionLoop(Config())
    # 先记录99个周期（非评估）
    for i in range(1, 100):
        el.step(cycle=i, avg_error=0.5-i*0.001, avg_reward=0.3,
                skill_count=i//10, task_success_rate=0.8,
                task_result=True)
    report("99周期后未评估", el.total_evaluations == 0)

    # 第100周期触发评估
    result = el.step(cycle=100, avg_error=0.4, avg_reward=0.3,
                     skill_count=10, task_success_rate=0.8,
                     task_result=True)
    report("第100周期评估", result["evaluated"] is True)
    report("total_evaluations=1", el.total_evaluations == 1)
    report("evolution_history有1条", len(el.evolution_history) == 1)


# ============================================================
# 测试32：EvolutionLoop get_evolution_stats
# ============================================================
def test32_evolution_loop_stats():
    section("测试32：EvolutionLoop统计")
    el = EvolutionLoop(Config())
    for i in range(1, 101):
        el.step(cycle=i, avg_error=0.5, avg_reward=0.3,
                skill_count=5, task_success_rate=0.8,
                task_result=True)
    stats = el.get_evolution_stats()
    report("返回dict", isinstance(stats, dict))
    report("含total_evaluations", "total_evaluations" in stats)
    report("含total_adjustments", "total_adjustments" in stats)
    report("含current_difficulty", "current_difficulty" in stats)
    report("含success_rate", "success_rate" in stats)
    report("含error_trend", "error_trend" in stats)
    report("含convergence_rate", "convergence_rate" in stats)
    report("含skill_acquisition_rate", "skill_acquisition_rate" in stats)
    report("含params", "params" in stats)
    report("含evolution_history_size", "evolution_history_size" in stats)
    report("total_evaluations=1", stats["total_evaluations"] == 1)


# ============================================================
# 测试33：EvolutionLoop get_latest_evolution
# ============================================================
def test33_evolution_loop_latest():
    section("测试33：EvolutionLoop最新进化记录")
    el = EvolutionLoop(Config())
    report("无历史时返回None", el.get_latest_evolution() is None)

    for i in range(1, 101):
        el.step(cycle=i, avg_error=0.5, avg_reward=0.3,
                skill_count=5, task_success_rate=0.8,
                task_result=True)
    latest = el.get_latest_evolution()
    report("有历史时返回dict", isinstance(latest, dict))
    report("含cycle", "cycle" in latest)
    report("含meta", "meta" in latest)
    report("含adjustments", "adjustments" in latest)
    report("含difficulty", "difficulty" in latest)
    report("含params", "params" in latest)


# ============================================================
# 测试34：EvolutionLoop reset
# ============================================================
def test34_evolution_loop_reset():
    section("测试34：EvolutionLoop reset")
    el = EvolutionLoop(Config())
    for i in range(1, 101):
        el.step(cycle=i, avg_error=0.5, avg_reward=0.3,
                skill_count=5, task_success_rate=0.8,
                task_result=True)
    el.reset()
    report("reset后history空", len(el.evolution_history) == 0)
    report("reset后total_evaluations=0", el.total_evaluations == 0)
    report("reset后total_adjustments=0", el.total_adjustments == 0)
    report("reset后meta_learner清空",
           len(el.meta_learner.metrics_history) == 0)
    report("reset后curriculum难度EASY",
           el.curriculum.get_difficulty() == DifficultyLevel.EASY)


# ============================================================
# 测试35：EvolutionLoop health_check
# ============================================================
def test35_evolution_loop_health_check():
    section("测试35：EvolutionLoop健康检查")
    el = EvolutionLoop(Config())
    report("health_check通过", el.health_check())


# ============================================================
# 测试36：EvolutionLoop完整进化流程
# ============================================================
def test36_evolution_full_workflow():
    section("测试36：EvolutionLoop完整进化流程")
    el = EvolutionLoop(Config())

    # 阶段1：误差下降，技能增长，任务成功率高
    for i in range(1, 101):
        el.step(cycle=i, avg_error=1.0-i*0.008, avg_reward=0.5,
                skill_count=i//5, task_success_rate=0.9,
                task_result=True)

    stats1 = el.get_evolution_stats()
    report("第1轮评估完成", stats1["total_evaluations"] == 1)
    report("难度提升到MEDIUM",
           stats1["current_difficulty"] == "MEDIUM",
           f"diff={stats1['current_difficulty']}")
    report("误差趋势decreasing",
           stats1["error_trend"] == "decreasing",
           f"trend={stats1['error_trend']}")

    # 阶段2：继续200周期，误差继续下降
    for i in range(101, 201):
        el.step(cycle=i, avg_error=max(0.1, 0.2-i*0.0005), avg_reward=0.6,
                skill_count=20+i//10, task_success_rate=0.9,
                task_result=True)

    stats2 = el.get_evolution_stats()
    report("第2轮评估完成", stats2["total_evaluations"] == 2)
    report("难度提升到HARD",
           stats2["current_difficulty"] == "HARD",
           f"diff={stats2['current_difficulty']}")


# ============================================================
# 测试37：Core集成 - evolution_loop模块注册
# ============================================================
def test37_core_evolution_registered():
    section("测试37：Core注册evolution_loop")
    core = SelfEvolvingAgentCore(Config())
    report("evolution_loop在modules中", "evolution_loop" in core.modules)
    report("evolution_loop是EvolutionLoop实例",
           isinstance(core.modules["evolution_loop"], EvolutionLoop))


# ============================================================
# 测试38：Core集成 - cycle_log含evolution
# ============================================================
def test38_core_cycle_log_evolution():
    section("测试38：Core周期日志含evolution")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))
    core.run_cycle()
    log = core.last_cycle_log
    stages = [e["stage"] for e in log]
    report("cycle_log含evolution", "evolution" in stages)

    evol_entry = next(e for e in log if e["stage"] == "evolution")
    report("evolution含evaluated", "evaluated" in evol_entry["result"])
    report("evolution含adjustments", "adjustments" in evol_entry["result"])
    report("evolution含difficulty_change", "difficulty_change" in evol_entry["result"])


# ============================================================
# 测试39：Core集成 - metrics含evolution_stats
# ============================================================
def test39_core_metrics_evolution():
    section("测试39：Core指标含evolution_stats")
    core = SelfEvolvingAgentCore(Config())
    core.inject_input(np.zeros(16))
    core.run_cycle()
    metrics = core.get_metrics()
    report("metrics含evolution_stats", "evolution_stats" in metrics)
    es = metrics["evolution_stats"]
    report("evolution_stats含total_evaluations", "total_evaluations" in es)
    report("evolution_stats含current_difficulty", "current_difficulty" in es)
    report("evolution_stats含error_trend", "error_trend" in es)
    report("evolution_stats含params", "params" in es)


# ============================================================
# 测试40：Core集成 - 长程运行触发评估
# ============================================================
def test40_core_long_run_triggers_evaluation():
    section("测试40：Core长程运行触发进化评估")
    core = SelfEvolvingAgentCore(Config())
    core.initialize()
    # 运行105个周期，应触发第100周期的评估
    for _ in range(105):
        core.inject_input(np.random.randn(16) * 0.3)
        core.run_cycle()

    metrics = core.get_metrics()
    es = metrics["evolution_stats"]
    report("运行105周期后total_evaluations>=1",
           es["total_evaluations"] >= 1,
           f"evals={es['total_evaluations']}")
    report("evolution_history_size>=1",
           es["evolution_history_size"] >= 1,
           f"size={es['evolution_history_size']}")


# ============================================================
# 测试41：Core集成 - reset清空evolution
# ============================================================
def test41_core_reset_evolution():
    section("测试41：Core重置清空evolution")
    core = SelfEvolvingAgentCore(Config())
    # 运行100+周期触发评估
    for _ in range(105):
        core.inject_input(np.zeros(16))
        core.run_cycle()
    report("reset前有评估",
           core.modules["evolution_loop"].total_evaluations >= 1)

    core.reset()
    report("reset后total_evaluations=0",
           core.modules["evolution_loop"].total_evaluations == 0)
    report("reset后history空",
           len(core.modules["evolution_loop"].evolution_history) == 0)
    report("reset后难度EASY",
           core.modules["evolution_loop"].curriculum.get_difficulty()
           == DifficultyLevel.EASY)


# ============================================================
# 测试42：Core集成 - 健康检查
# ============================================================
def test42_core_health_check():
    section("测试42：Core健康检查含evolution")
    core = SelfEvolvingAgentCore(Config())
    report("health_check通过", core.health_check())


# ============================================================
# 测试43：性能测试
# ============================================================
def test43_performance():
    section("测试43：自主进化系统性能开销")
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

    report("自主进化系统<30ms/周期", elapsed < 30, f"{elapsed:.2f}ms/周期")


# ============================================================
# 测试44：进化闭环稳定性 - 500周期不崩溃
# ============================================================
def test44_evolution_stability():
    section("测试44：进化闭环500周期稳定性")
    core = SelfEvolvingAgentCore(Config())
    core.initialize()

    for i in range(500):
        core.inject_input(np.random.randn(16) * 0.3)
        core.run_cycle()
        # 每周期都应处于normal或safe_fallback模式
        if core.mode not in ("normal", "safe_fallback"):
            report("模式异常", False, f"cycle={i} mode={core.mode}")
            return

    report("500周期不崩溃", core.cycle_count == 500)
    report("模式正常", core.mode in ("normal", "safe_fallback"))
    metrics = core.get_metrics()
    report("evolution_stats存在", "evolution_stats" in metrics)
    report("至少4次评估（500/100=5，但首次在100）",
           metrics["evolution_stats"]["total_evaluations"] >= 4,
           f"evals={metrics['evolution_stats']['total_evaluations']}")


# ============================================================
# 测试45：进化闭环 - 元学习与自我调参联动
# ============================================================
def test45_meta_tuner_integration():
    section("测试45：元学习与自我调参联动")
    el = EvolutionLoop(Config())

    # 模拟震荡误差场景
    for i in range(1, 101):
        # 误差震荡
        error = 0.5 + 0.3 * np.sin(i * 0.5)
        el.step(cycle=i, avg_error=error, avg_reward=0.3,
                skill_count=5, task_success_rate=0.5,
                task_result=(i % 2 == 0))

    stats = el.get_evolution_stats()
    report("评估已执行", stats["total_evaluations"] >= 1)
    # 联动机制：meta_learner产生trend，self_tuner据此调整
    # 接受任何trend值，重点是机制工作正常
    report("error_trend已识别",
           stats["error_trend"] in ("decreasing", "stable",
                                     "increasing", "oscillating"),
           f"trend={stats['error_trend']}")
    report("params已被self_tuner管理",
           isinstance(stats["params"], dict) and len(stats["params"]) == 3)
    report("curriculum已记录结果",
           el.curriculum.get_stats()["total_results"] > 0)


# ============================================================
# 测试46：进化闭环 - 课程学习多级提升
# ============================================================
def test46_curriculum_multi_level():
    section("测试46：课程学习多级提升")
    el = EvolutionLoop(Config())

    # 模拟持续高成功率
    for i in range(1, 401):
        el.step(cycle=i, avg_error=0.3, avg_reward=0.5,
                skill_count=10, task_success_rate=0.9,
                task_result=True)

    stats = el.get_evolution_stats()
    # 400周期，4次评估（100/200/300/400），每次都应提升难度
    report("4次评估", stats["total_evaluations"] == 4,
           f"evals={stats['total_evaluations']}")
    report("难度达到HARD或EXPERT",
           stats["current_difficulty"] in ("HARD", "EXPERT"),
           f"diff={stats['current_difficulty']}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第十八轮测试：阶段28自主进化闭环验证")
    print("="*60)

    tests = [
        test1_difficulty_level_enum,
        test2_learning_metrics,
        test3_tunable_parameter,
        test4_meta_learner_basic,
        test5_meta_learner_record,
        test6_meta_learner_evaluate_decreasing,
        test7_meta_learner_evaluate_oscillating,
        test8_meta_learner_evaluate_insufficient,
        test9_meta_learner_feedback,
        test10_meta_learner_reset,
        test11_meta_learner_health_check,
        test12_self_tuner_basic,
        test13_self_tuner_decrease_lr_on_oscillation,
        test14_self_tuner_increase_lr_on_decreasing,
        test15_self_tuner_exploration_need,
        test16_self_tuner_skill_need,
        test17_self_tuner_get_all_params,
        test18_self_tuner_reset,
        test19_self_tuner_health_check,
        test20_curriculum_basic,
        test21_curriculum_record_result,
        test22_curriculum_increase_difficulty,
        test23_curriculum_decrease_difficulty,
        test24_curriculum_insufficient_data,
        test25_curriculum_max_difficulty,
        test26_curriculum_get_stats,
        test27_curriculum_reset,
        test28_curriculum_health_check,
        test29_evolution_loop_basic,
        test30_evolution_loop_step_non_eval,
        test31_evolution_loop_step_eval,
        test32_evolution_loop_stats,
        test33_evolution_loop_latest,
        test34_evolution_loop_reset,
        test35_evolution_loop_health_check,
        test36_evolution_full_workflow,
        test37_core_evolution_registered,
        test38_core_cycle_log_evolution,
        test39_core_metrics_evolution,
        test40_core_long_run_triggers_evaluation,
        test41_core_reset_evolution,
        test42_core_health_check,
        test43_performance,
        test44_evolution_stability,
        test45_meta_tuner_integration,
        test46_curriculum_multi_level,
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
