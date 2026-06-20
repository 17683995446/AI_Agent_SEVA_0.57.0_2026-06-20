"""
阶段34测试：预设场景与开箱即用
==============================

验证预设场景模块的完整功能：
1. Scenario数据类
2. PresetScenarios预设场景集合
3. ScenarioRunner场景运行器
4. 便捷函数（run_scenario/list_scenarios/compare_scenarios/quick_demo）
5. CLI场景命令
6. 开箱即用性验证

对应第一性原理：易用 = 开箱即用，无需配置即可运行常见场景
"""
import os
import sys
import json
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import (Scenario, PresetScenarios, ScenarioRunner,
                  run_scenario, list_scenarios, list_scenarios_detailed,
                  compare_scenarios, quick_demo, run_all_scenarios,
                  get_scenario_report)
from seva.scenarios import PresetScenarios as PS


# ============================================================
# 测试1-5：Scenario数据类
# ============================================================
def test1_scenario_creation():
    """测试1: Scenario可创建"""
    s = Scenario(name="test", description="测试场景")
    assert s.name == "test"
    assert s.description == "测试场景"
    assert s.preset == "easy"  # 默认值
    assert s.env_type == "mixed"
    assert s.steps == 100
    assert s.seed == 42


def test2_scenario_custom_values():
    """测试2: Scenario支持自定义值"""
    s = Scenario(
        name="custom",
        description="自定义场景",
        preset="standard",
        env_type="drift",
        steps=200,
        seed=123,
        overrides={"core": {"max_cycles": 500}},
        tags=["test", "custom"],
        expected_min_reward=-50.0,
    )
    assert s.preset == "standard"
    assert s.env_type == "drift"
    assert s.steps == 200
    assert s.seed == 123
    assert s.overrides == {"core": {"max_cycles": 500}}
    assert s.tags == ["test", "custom"]
    assert s.expected_min_reward == -50.0


def test3_scenario_to_dict():
    """测试3: Scenario转字典"""
    s = Scenario(name="test", description="测试", tags=["a", "b"])
    d = s.to_dict()
    assert d["name"] == "test"
    assert d["description"] == "测试"
    assert d["tags"] == ["a", "b"]
    assert "preset" in d
    assert "env_type" in d
    assert "steps" in d


def test4_scenario_defaults():
    """测试4: Scenario默认值正确"""
    s = Scenario(name="t", description="d")
    assert s.overrides == {}
    assert s.tags == []
    assert s.expected_min_reward == -1000.0


def test5_scenario_field_types():
    """测试5: Scenario字段类型正确"""
    s = Scenario(name="t", description="d")
    assert isinstance(s.overrides, dict)
    assert isinstance(s.tags, list)


# ============================================================
# 测试6-12：PresetScenarios预设场景集合
# ============================================================
def test6_preset_scenarios_list():
    """测试6: 预设场景列表完整"""
    scenarios = PS.list_scenarios()
    assert len(scenarios) == 8
    assert "quick_validation" in scenarios
    assert "standard_run" in scenarios
    assert "performance_benchmark" in scenarios
    assert "drift_adaptation" in scenarios
    assert "adversarial_robustness" in scenarios
    assert "research_long_run" in scenarios
    assert "production_stability" in scenarios
    assert "evolution_demo" in scenarios


def test7_preset_scenarios_get():
    """测试7: 可获取每个预设场景"""
    for name in PS.list_scenarios():
        s = PS.get(name)
        assert isinstance(s, Scenario)
        assert s.name == name
        assert len(s.description) > 0
        assert s.preset in ["easy", "standard", "research", "production"]
        assert s.env_type in ["mixed", "drift", "adversarial", "blank"]
        assert s.steps > 0


def test8_preset_scenarios_get_invalid():
    """测试8: 获取无效场景抛出异常"""
    with pytest.raises(ValueError):
        PS.get("invalid_scenario")


def test9_preset_scenarios_list_detailed():
    """测试9: 详细场景列表"""
    detailed = PS.list_with_descriptions()
    assert len(detailed) == 8
    for item in detailed:
        assert "name" in item
        assert "description" in item
        assert "preset" in item
        assert "env_type" in item
        assert "steps" in item
        assert "tags" in item


def test10_preset_scenarios_list_by_tag():
    """测试10: 按标签筛选场景"""
    quick = PS.list_by_tag("quick")
    assert "quick_validation" in quick

    test_scenarios = PS.list_by_tag("test")
    assert "drift_adaptation" in test_scenarios
    assert "adversarial_robustness" in test_scenarios


def test11_quick_validation_scenario():
    """测试11: quick_validation场景配置正确"""
    s = PS.get("quick_validation")
    assert s.preset == "easy"
    assert s.env_type == "mixed"
    assert s.steps == 10
    assert "quick" in s.tags
    assert "validation" in s.tags


def test12_evolution_demo_scenario():
    """测试12: evolution_demo场景有覆盖配置"""
    s = PS.get("evolution_demo")
    assert "evolution" in s.overrides
    assert s.overrides["evolution"]["evaluation_interval"] == 50


# ============================================================
# 测试13-20：ScenarioRunner场景运行器
# ============================================================
def test13_runner_creation():
    """测试13: ScenarioRunner可创建"""
    runner = ScenarioRunner()
    assert runner.results == {}
    assert runner.verbose is False


def test14_runner_run_single():
    """测试14: 运行单个场景"""
    runner = ScenarioRunner()
    scenario = PS.get("quick_validation")
    result = runner.run(scenario)

    assert result["scenario_name"] == "quick_validation"
    assert result["steps_completed"] > 0
    assert "total_reward" in result
    assert "elapsed_time" in result
    assert "passed" in result
    assert isinstance(result["passed"], bool)


def test15_runner_results_stored():
    """测试15: 运行结果被存储"""
    runner = ScenarioRunner()
    scenario = PS.get("quick_validation")
    runner.run(scenario)
    assert "quick_validation" in runner.results


def test16_runner_run_multiple():
    """测试16: 运行多个场景"""
    runner = ScenarioRunner()
    results = runner.run_multiple(["quick_validation"])
    assert len(results) >= 1
    assert "quick_validation" in results


def test17_runner_compare():
    """测试17: 对比多个场景"""
    runner = ScenarioRunner()
    report = runner.compare(["quick_validation"])
    assert "scenarios" in report
    assert "results" in report
    assert "summary" in report
    assert "quick_validation" in report["results"]


def test18_runner_summary():
    """测试18: 摘要生成正确"""
    runner = ScenarioRunner()
    runner.compare(["quick_validation"])
    summary = runner._generate_summary(runner.results)
    assert summary["total_scenarios"] == 1
    assert summary["passed"] >= 0
    assert "best_reward" in summary
    assert "worst_reward" in summary
    assert "avg_reward" in summary


def test19_runner_report():
    """测试19: 生成文本报告"""
    runner = ScenarioRunner()
    runner.run(PS.get("quick_validation"))
    report = runner.get_report()
    assert isinstance(report, str)
    assert "场景运行报告" in report
    assert "quick_validation" in report
    assert "摘要" in report


def test20_runner_report_empty():
    """测试20: 无结果时报告"""
    runner = ScenarioRunner()
    report = runner.get_report()
    assert "无运行结果" in report


# ============================================================
# 测试21-28：便捷函数
# ============================================================
def test21_run_scenario():
    """测试21: run_scenario便捷函数"""
    result = run_scenario("quick_validation")
    assert result["scenario_name"] == "quick_validation"
    assert result["steps_completed"] > 0
    assert "total_reward" in result


def test22_run_scenario_invalid():
    """测试22: run_scenario无效场景"""
    with pytest.raises(ValueError):
        run_scenario("invalid")


def test23_list_scenarios():
    """测试23: list_scenarios函数"""
    scenarios = list_scenarios()
    assert isinstance(scenarios, list)
    assert len(scenarios) == 8
    assert "quick_validation" in scenarios


def test24_list_scenarios_detailed():
    """测试24: list_scenarios_detailed函数"""
    detailed = list_scenarios_detailed()
    assert len(detailed) == 8
    assert all("name" in d for d in detailed)


def test25_compare_scenarios():
    """测试25: compare_scenarios函数"""
    report = compare_scenarios(["quick_validation"])
    assert "scenarios" in report
    assert "results" in report
    assert "summary" in report


def test26_compare_scenarios_multiple():
    """测试26: 对比多个场景"""
    report = compare_scenarios(["quick_validation", "standard_run"])
    assert len(report["scenarios"]) == 2
    assert report["summary"]["total_scenarios"] == 2


def test27_quick_demo():
    """测试27: quick_demo函数"""
    stats = quick_demo(steps=10, verbose=False)
    assert stats["steps_completed"] > 0
    assert "total_real_reward" in stats


def test28_get_scenario_report():
    """测试28: get_scenario_report函数"""
    results = run_all_scenarios.__wrapped__() if hasattr(
        run_all_scenarios, "__wrapped__") else None
    # 直接使用run_scenario生成结果
    runner = ScenarioRunner()
    runner.run(PS.get("quick_validation"))
    report = get_scenario_report(runner.results)
    assert isinstance(report, str)
    assert "quick_validation" in report


# ============================================================
# 测试29-35：场景运行结果完整性
# ============================================================
def test29_result_contains_all_fields():
    """测试29: 运行结果包含所有字段"""
    result = run_scenario("quick_validation")
    required_fields = [
        "scenario_name", "description", "preset", "env_type",
        "steps", "seed", "tags", "steps_completed", "total_reward",
        "avg_cycle_time", "avg_prediction_error", "env_done", "mode",
        "elapsed_time", "tasks_completed", "skills_total",
        "evolution_evaluations", "current_difficulty", "passed",
    ]
    for field in required_fields:
        assert field in result, f"结果缺少字段: {field}"


def test30_result_passed_logic():
    """测试30: passed字段逻辑正确"""
    result = run_scenario("quick_validation")
    scenario = PS.get("quick_validation")
    expected = result["total_reward"] >= scenario.expected_min_reward
    assert result["passed"] == expected


def test31_result_elapsed_time_positive():
    """测试31: 运行时间为正"""
    result = run_scenario("quick_validation")
    assert result["elapsed_time"] > 0


def test32_result_steps_completed():
    """测试32: 完成步数合理"""
    result = run_scenario("quick_validation")
    scenario = PS.get("quick_validation")
    assert result["steps_completed"] <= scenario.steps
    assert result["steps_completed"] > 0


def test33_result_preset_matches():
    """测试33: 结果中预设与场景一致"""
    result = run_scenario("quick_validation")
    scenario = PS.get("quick_validation")
    assert result["preset"] == scenario.preset
    assert result["env_type"] == scenario.env_type


def test34_result_tags_copied():
    """测试34: 结果中标签被复制"""
    result = run_scenario("quick_validation")
    scenario = PS.get("quick_validation")
    assert result["tags"] == scenario.tags


def test35_result_mode_valid():
    """测试35: 运行模式有效"""
    result = run_scenario("quick_validation")
    assert result["mode"] in ["normal", "safe", "emergency"]


# ============================================================
# 测试36-42：CLI场景命令
# ============================================================
def run_cli(*args):
    """运行CLI命令"""
    cmd = [sys.executable, "-m", "seva"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


def test36_cli_scenarios_list():
    """测试36: seva scenarios命令"""
    code, out, err = run_cli("scenarios")
    assert code == 0
    assert "预设场景" in out
    assert "quick_validation" in out
    assert "standard_run" in out
    assert "8" in out  # 8个场景


def test37_cli_scenario_run():
    """测试37: seva scenario run命令"""
    code, out, err = run_cli("scenario", "run", "quick_validation")
    assert code == 0
    assert "运行场景" in out
    assert "quick_validation" in out
    assert "运行结果" in out
    assert "总奖励" in out


def test38_cli_scenario_run_invalid():
    """测试38: seva scenario run无效场景"""
    code, out, err = run_cli("scenario", "run", "invalid_scenario")
    assert code == 1
    assert "错误" in out or "未知" in out


def test39_cli_scenario_compare():
    """测试39: seva scenario compare命令"""
    code, out, err = run_cli("scenario", "compare",
                             "quick_validation", "standard_run")
    assert code == 0
    assert "对比场景" in out
    assert "quick_validation" in out
    assert "standard_run" in out
    assert "摘要" in out


def test40_cli_scenario_run_with_output():
    """测试40: seva scenario run --output"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "result.json")
        code, out, err = run_cli("scenario", "run", "quick_validation",
                                 "--output", output_path)
        assert code == 0
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert data["scenario_name"] == "quick_validation"


def test41_cli_scenario_compare_with_output():
    """测试41: seva scenario compare --output"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "compare.json")
        code, out, err = run_cli("scenario", "compare",
                                 "quick_validation", "standard_run",
                                 "--output", output_path)
        assert code == 0
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "results" in data


def test42_cli_scenario_list():
    """测试42: seva scenario list命令"""
    code, out, err = run_cli("scenario", "list")
    assert code == 0
    assert "预设场景" in out


# ============================================================
# 测试43-48：开箱即用性验证
# ============================================================
def test43_one_line_run_scenario():
    """测试43: 一行代码运行场景（开箱即用）"""
    # 用户只需一行代码
    result = run_scenario("quick_validation")
    assert result is not None
    assert "total_reward" in result


def test44_one_line_list_scenarios():
    """测试44: 一行代码列出场景"""
    scenarios = list_scenarios()
    assert len(scenarios) == 8


def test45_one_line_quick_demo():
    """测试45: 一行代码快速演示"""
    stats = quick_demo(steps=5, verbose=False)
    assert stats is not None
    assert "steps_completed" in stats


def test46_no_config_required():
    """测试46: 无需配置即可运行"""
    # 所有预设场景都无需额外配置
    for name in list_scenarios():
        s = PS.get(name)
        # 场景自身已封装所有配置
        assert s.preset in ["easy", "standard", "research", "production"]
        assert s.env_type in ["mixed", "drift", "adversarial", "blank"]
        assert s.steps > 0


def test47_scenario_covers_all_presets():
    """测试47: 场景覆盖所有预设模板"""
    presets_used = set()
    for name in list_scenarios():
        s = PS.get(name)
        presets_used.add(s.preset)
    assert "easy" in presets_used
    assert "standard" in presets_used
    assert "research" in presets_used
    assert "production" in presets_used


def test48_scenario_covers_all_env_types():
    """测试48: 场景覆盖多种环境类型"""
    env_types_used = set()
    for name in list_scenarios():
        s = PS.get(name)
        env_types_used.add(s.env_type)
    assert "mixed" in env_types_used
    assert "drift" in env_types_used
    assert "adversarial" in env_types_used


# ============================================================
# 测试49-55：场景分类完整性
# ============================================================
def test49_validation_scenario():
    """测试49: 验证类场景"""
    s = PS.get("quick_validation")
    assert "validation" in s.tags or "smoke" in s.tags
    assert s.steps <= 20  # 验证场景应快速


def test50_benchmark_scenario():
    """测试50: 基准类场景"""
    s = PS.get("performance_benchmark")
    assert "benchmark" in s.tags or "performance" in s.tags
    assert s.steps >= 100  # 基准场景应较长


def test51_test_scenarios():
    """测试51: 测试类场景"""
    for name in ["drift_adaptation", "adversarial_robustness"]:
        s = PS.get(name)
        assert "test" in s.tags


def test52_research_scenario():
    """测试52: 研究类场景"""
    s = PS.get("research_long_run")
    assert s.preset == "research"
    assert "research" in s.tags


def test53_production_scenario():
    """测试53: 生产类场景"""
    s = PS.get("production_stability")
    assert s.preset == "production"
    assert "production" in s.tags


def test54_evolution_demo_scenario():
    """测试54: 进化演示场景"""
    s = PS.get("evolution_demo")
    assert "evolution" in s.tags
    assert "demo" in s.tags
    assert s.steps >= 200  # 进化需要较长运行


def test55_standard_run_scenario():
    """测试55: 标准运行场景"""
    s = PS.get("standard_run")
    assert s.preset == "standard"
    assert "standard" in s.tags


# ============================================================
# 测试56-60：对比功能深度验证
# ============================================================
def test56_compare_summary_fields():
    """测试56: 对比摘要包含所有字段"""
    report = compare_scenarios(["quick_validation", "standard_run"])
    summary = report["summary"]
    required = [
        "total_scenarios", "passed", "failed",
        "best_reward", "worst_reward", "avg_reward",
        "total_time", "avg_time",
        "fastest_scenario", "best_scenario",
    ]
    for field in required:
        assert field in summary, f"摘要缺少字段: {field}"


def test57_compare_best_scenario():
    """测试57: 最佳场景识别正确"""
    report = compare_scenarios(["quick_validation", "standard_run"])
    summary = report["summary"]
    best = summary["best_scenario"]
    best_reward = report["results"][best]["total_reward"]
    assert best_reward == summary["best_reward"]


def test58_compare_fastest_scenario():
    """测试58: 最快场景识别正确"""
    report = compare_scenarios(["quick_validation", "standard_run"])
    summary = report["summary"]
    fastest = summary["fastest_scenario"]
    fastest_time = report["results"][fastest]["elapsed_time"]
    # 最快场景的时间应是最小的
    all_times = [r["elapsed_time"] for r in report["results"].values()]
    assert fastest_time == min(all_times)


def test59_compare_passed_failed_count():
    """测试59: 通过/失败计数正确"""
    report = compare_scenarios(["quick_validation", "standard_run"])
    summary = report["summary"]
    assert summary["passed"] + summary["failed"] == summary["total_scenarios"]


def test60_compare_three_scenarios():
    """测试60: 对比三个场景"""
    report = compare_scenarios([
        "quick_validation", "standard_run", "performance_benchmark"
    ])
    assert report["summary"]["total_scenarios"] == 3


# ============================================================
# 测试61-65：模块导出与集成
# ============================================================
def test61_module_import():
    """测试61: 模块可从seva导入"""
    from seva import scenarios
    assert hasattr(scenarios, "Scenario")
    assert hasattr(scenarios, "PresetScenarios")
    assert hasattr(scenarios, "ScenarioRunner")


def test62_all_exports_available():
    """测试62: 所有导出可用"""
    from seva import (Scenario, PresetScenarios, ScenarioRunner,
                      run_scenario, list_scenarios,
                      list_scenarios_detailed, compare_scenarios,
                      quick_demo, run_all_scenarios,
                      get_scenario_report)
    assert Scenario is not None
    assert PresetScenarios is not None
    assert ScenarioRunner is not None
    assert callable(run_scenario)
    assert callable(list_scenarios)
    assert callable(compare_scenarios)
    assert callable(quick_demo)


def test63_scenario_in_all():
    """测试63: Scenario在__all__中"""
    import seva
    assert "Scenario" in seva.__all__
    assert "PresetScenarios" in seva.__all__
    assert "ScenarioRunner" in seva.__all__
    assert "run_scenario" in seva.__all__
    assert "list_scenarios" in seva.__all__
    assert "compare_scenarios" in seva.__all__
    assert "quick_demo" in seva.__all__


def test64_version_updated():
    """测试64: 版本号已更新"""
    import seva
    assert seva.__version__ >= "0.34.0"


def test65_scenario_runner_run_all():
    """测试65: ScenarioRunner.run_all方法存在"""
    runner = ScenarioRunner()
    assert hasattr(runner, "run_all")
    assert hasattr(runner, "run_multiple")
    assert hasattr(runner, "run")
    assert hasattr(runner, "compare")
    assert hasattr(runner, "get_report")


# ============================================================
# 性能测试
# ============================================================
def test66_quick_validation_fast():
    """测试66: quick_validation应在5秒内完成"""
    import time
    start = time.time()
    run_scenario("quick_validation")
    elapsed = time.time() - start
    assert elapsed < 5, f"quick_validation运行时间过长: {elapsed:.2f}s"


def test67_quick_demo_fast():
    """测试67: quick_demo应在10秒内完成"""
    import time
    start = time.time()
    quick_demo(steps=10, verbose=False)
    elapsed = time.time() - start
    assert elapsed < 10


def test68_compare_fast():
    """测试68: 对比两个场景应在30秒内完成"""
    import time
    start = time.time()
    compare_scenarios(["quick_validation", "standard_run"])
    elapsed = time.time() - start
    assert elapsed < 30


# ============================================================
# 完整性测试
# ============================================================
def test69_all_scenarios_runnable():
    """测试69: 所有场景可运行（除长运行场景外）"""
    # 跳过耗时较长的场景
    skip = {"research_long_run", "evolution_demo",
            "performance_benchmark", "production_stability"}
    for name in list_scenarios():
        if name in skip:
            continue
        result = run_scenario(name)
        assert result["steps_completed"] > 0, f"{name}未运行任何步"


def test70_scenario_descriptions_chinese():
    """测试70: 场景描述使用中文"""
    for name in list_scenarios():
        s = PS.get(name)
        # 描述中应包含中文
        chinese_count = sum(1 for c in s.description
                            if '\u4e00' <= c <= '\u9fff')
        assert chinese_count > 0, f"{name}描述缺少中文"


def test71_scenario_tags_not_empty():
    """测试71: 所有场景都有标签"""
    for name in list_scenarios():
        s = PS.get(name)
        assert len(s.tags) > 0, f"{name}缺少标签"


def test72_scenario_names_unique():
    """测试72: 场景名称唯一"""
    names = list_scenarios()
    assert len(names) == len(set(names))


def test73_complete_scenario_system():
    """测试73: 场景系统完整性（综合验证）"""
    # 1. 列出场景
    scenarios = list_scenarios()
    assert len(scenarios) == 8

    # 2. 运行场景
    result = run_scenario("quick_validation")
    assert result["passed"] in [True, False]

    # 3. 对比场景
    report = compare_scenarios(["quick_validation"])
    assert "summary" in report

    # 4. 快速演示
    stats = quick_demo(steps=5, verbose=False)
    assert stats["steps_completed"] > 0

    # 5. CLI可用
    code, out, err = run_cli("scenarios")
    assert code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
