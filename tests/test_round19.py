"""
第十九轮测试：阶段29统一API与快速启动验证
验证quickstart、SEVAFactory、PresetConfigs、run、get_status
核心目标：证明用户能一行代码启动Agent，无需样板代码

第一性原理验证：
1. 易用 = 一行启动：quickstart(steps=100)即可
2. 易用 = 预设模板：不用懂内部配置就能用
3. 易用 = 工厂模式：统一创建入口
4. 易用 = 状态可见：get_status返回易读摘要
"""
import sys
import os
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

from seva.api import (quickstart, run, get_status, create_config,
                      SEVAFactory, PresetConfigs)
from seva.config import Config
from seva.core import SelfEvolvingAgentCore
from seva.environment import SimulationEnvironment

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
# 测试1：PresetConfigs基础
# ============================================================
def test1_preset_configs_basic():
    section("测试1：PresetConfigs基础")
    presets = PresetConfigs.list_presets()
    report("4个预设", len(presets) == 4)
    report("含easy", "easy" in presets)
    report("含standard", "standard" in presets)
    report("含research", "research" in presets)
    report("含production", "production" in presets)


# ============================================================
# 测试2：PresetConfigs获取
# ============================================================
def test2_preset_configs_get():
    section("测试2：PresetConfigs获取")
    easy = PresetConfigs.get("easy")
    report("easy返回dict", isinstance(easy, dict))
    report("easy含core", "core" in easy)
    report("easy含perception", "perception" in easy)
    report("easy含memory", "memory" in easy)
    report("easy含environment", "environment" in easy)

    standard = PresetConfigs.get("standard")
    report("standard返回dict", isinstance(standard, dict))
    # easy比standard规模小
    report("easy.input_dim < standard.input_dim",
           easy["perception"]["input_dim"] < standard["perception"]["input_dim"])
    report("easy.grid_size < standard.grid_size",
           easy["environment"]["grid_size"] < standard["environment"]["grid_size"])


# ============================================================
# 测试3：PresetConfigs未知预设
# ============================================================
def test3_preset_configs_unknown():
    section("测试3：PresetConfigs未知预设")
    try:
        PresetConfigs.get("unknown")
        report("未知预设抛ValueError", False)
    except ValueError as e:
        report("未知预设抛ValueError", True)
        report("错误信息含可用预设", "easy" in str(e))


# ============================================================
# 测试4：PresetConfigs各模板差异
# ============================================================
def test4_preset_configs_diff():
    section("测试4：各预设模板差异")
    easy = PresetConfigs.get("easy")
    standard = PresetConfigs.get("standard")
    research = PresetConfigs.get("research")
    production = PresetConfigs.get("production")

    # research规模最大
    report("research.input_dim最大",
           research["perception"]["input_dim"] >= standard["perception"]["input_dim"])
    report("research.grid_size最大",
           research["environment"]["grid_size"] >= standard["environment"]["grid_size"])

    # production安全更严格
    report("production熔断阈值更小",
           production["safety"]["circuit_breaker_threshold"] <=
           standard.get("safety", {}).get("circuit_breaker_threshold", 10))


# ============================================================
# 测试5：SEVAFactory创建Agent
# ============================================================
def test5_factory_create():
    section("测试5：SEVAFactory创建Agent")
    agent = SEVAFactory.create("easy")
    report("返回SelfEvolvingAgentCore",
           isinstance(agent, SelfEvolvingAgentCore))
    report("已initialize", agent.is_running)
    report("已附加环境", agent.environment is not None)
    report("有pending_input", hasattr(agent, "_pending_input")
           and agent._pending_input is not None)


# ============================================================
# 测试6：SEVAFactory不同环境类型
# ============================================================
def test6_factory_env_types():
    section("测试6：SEVAFactory不同环境类型")
    for env_type in ["blank", "mixed", "drift", "adversarial"]:
        agent = SEVAFactory.create("easy", env_type=env_type)
        report(f"创建{env_type}环境",
               agent.environment is not None and
               agent.environment.env_type == env_type)


# ============================================================
# 测试7：SEVAFactory配置覆盖
# ============================================================
def test7_factory_overrides():
    section("测试7：SEVAFactory配置覆盖")
    agent = SEVAFactory.create("easy",
                               overrides={"core": {"max_cycles": 500}})
    report("覆盖生效", agent.config.get("core", "max_cycles") == 500)
    # 其他配置保持easy预设
    report("其他配置保持预设",
           agent.config.get("perception", "input_dim") ==
           PresetConfigs.get("easy")["perception"]["input_dim"])


# ============================================================
# 测试8：SEVAFactory不同种子
# ============================================================
def test8_factory_seed():
    section("测试8：SEVAFactory不同种子")
    agent1 = SEVAFactory.create("easy", seed=42)
    agent2 = SEVAFactory.create("easy", seed=42)
    agent3 = SEVAFactory.create("easy", seed=123)

    # 同种子应产生相同初始观测
    report("同种子初始观测相同",
           np.allclose(agent1._pending_input, agent2._pending_input))
    # 不同种子应产生不同初始观测
    report("不同种子初始观测不同",
           not np.allclose(agent1._pending_input, agent3._pending_input))


# ============================================================
# 测试9：SEVAFactory create_without_env
# ============================================================
def test9_factory_without_env():
    section("测试9：SEVAFactory无环境创建")
    agent = SEVAFactory.create_without_env("easy")
    report("返回SelfEvolvingAgentCore",
           isinstance(agent, SelfEvolvingAgentCore))
    report("已initialize", agent.is_running)
    report("未附加环境", agent.environment is None)


# ============================================================
# 测试10：quickstart一行启动
# ============================================================
def test10_quickstart():
    section("测试10：quickstart一行启动")
    agent, stats = quickstart(steps=20, preset="easy")
    report("返回agent", isinstance(agent, SelfEvolvingAgentCore))
    report("返回stats", isinstance(stats, dict))
    report("stats含steps_completed", "steps_completed" in stats)
    report("stats含total_real_reward", "total_real_reward" in stats)
    report("stats含mode", "mode" in stats)
    report("运行20步", stats["steps_completed"] == 20)
    report("mode为normal", stats["mode"] == "normal")


# ============================================================
# 测试11：quickstart不同预设
# ============================================================
def test11_quickstart_presets():
    section("测试11：quickstart不同预设")
    for preset in ["easy", "standard"]:
        agent, stats = quickstart(steps=10, preset=preset)
        report(f"{preset}预设运行成功",
               stats["steps_completed"] == 10 and stats["mode"] == "normal")


# ============================================================
# 测试12：quickstart verbose模式
# ============================================================
def test12_quickstart_verbose():
    section("测试12：quickstart verbose模式")
    # verbose不应影响功能
    agent, stats = quickstart(steps=10, preset="easy", verbose=True)
    report("verbose模式运行成功", stats["steps_completed"] == 10)


# ============================================================
# 测试13：run函数
# ============================================================
def test13_run_function():
    section("测试13：run函数")
    agent = SEVAFactory.create("easy")
    stats = run(agent, steps=15)
    report("返回stats", isinstance(stats, dict))
    report("运行15步", stats["steps_completed"] == 15)
    report("stats含avg_cycle_time", "avg_cycle_time" in stats)
    report("stats含task_stats", "task_stats" in stats)
    report("stats含skill_stats", "skill_stats" in stats)
    report("stats含evolution_stats", "evolution_stats" in stats)


# ============================================================
# 测试14：run函数无环境报错
# ============================================================
def test14_run_no_env():
    section("测试14：run函数无环境报错")
    agent = SEVAFactory.create_without_env("easy")
    try:
        run(agent, steps=10)
        report("无环境抛RuntimeError", False)
    except RuntimeError as e:
        report("无环境抛RuntimeError", True)
        report("错误信息含attach_environment", "attach_environment" in str(e))


# ============================================================
# 测试15：run函数verbose
# ============================================================
def test15_run_verbose():
    section("测试15：run函数verbose")
    agent = SEVAFactory.create("easy")
    stats = run(agent, steps=10, verbose=True)
    report("verbose运行成功", stats["steps_completed"] == 10)


# ============================================================
# 测试16：create_config函数
# ============================================================
def test16_create_config():
    section("测试16：create_config函数")
    config = create_config("easy")
    report("返回Config", isinstance(config, Config))
    report("应用easy预设",
           config.get("perception", "input_dim") ==
           PresetConfigs.get("easy")["perception"]["input_dim"])

    # 带覆盖
    config2 = create_config("easy", overrides={"core": {"max_cycles": 200}})
    report("覆盖生效", config2.get("core", "max_cycles") == 200)


# ============================================================
# 测试17：get_status函数
# ============================================================
def test17_get_status():
    section("测试17：get_status函数")
    agent = SEVAFactory.create("easy")
    run(agent, steps=10)
    status = get_status(agent)
    report("返回dict", isinstance(status, dict))
    report("含cycle_count", "cycle_count" in status)
    report("含mode", "mode" in status)
    report("含healthy", "healthy" in status)
    report("含total_reward", "total_reward" in status)
    report("含avg_cycle_time_ms", "avg_cycle_time_ms" in status)
    report("含env_attached", "env_attached" in status)
    report("含env_done", "env_done" in status)
    report("含tasks_completed", "tasks_completed" in status)
    report("含skills_total", "skills_total" in status)
    report("含evolution_evaluations", "evolution_evaluations" in status)
    report("含current_difficulty", "current_difficulty" in status)
    report("cycle_count=10", status["cycle_count"] == 10)
    report("healthy=True", status["healthy"] is True)
    report("env_attached=True", status["env_attached"] is True)


# ============================================================
# 测试18：get_status无环境
# ============================================================
def test18_get_status_no_env():
    section("测试18：get_status无环境")
    agent = SEVAFactory.create_without_env("easy")
    agent.inject_input(np.zeros(16))
    agent.run_cycle()
    status = get_status(agent)
    report("env_attached=False", status["env_attached"] is False)


# ============================================================
# 测试19：包级导入
# ============================================================
def test19_package_import():
    section("测试19：包级导入")
    import seva
    report("seva.quickstart可导入", hasattr(seva, "quickstart"))
    report("seva.run可导入", hasattr(seva, "run"))
    report("seva.get_status可导入", hasattr(seva, "get_status"))
    report("seva.create_config可导入", hasattr(seva, "create_config"))
    report("seva.SEVAFactory可导入", hasattr(seva, "SEVAFactory"))
    report("seva.PresetConfigs可导入", hasattr(seva, "PresetConfigs"))
    report("seva.Config可导入", hasattr(seva, "Config"))
    report("seva.SelfEvolvingAgentCore可导入",
           hasattr(seva, "SelfEvolvingAgentCore"))
    report("seva.__version__存在", hasattr(seva, "__version__"))


# ============================================================
# 测试20：一行启动端到端
# ============================================================
def test20_one_line_e2e():
    section("测试20：一行启动端到端")
    # 最简使用：一行启动
    from seva import quickstart
    agent, stats = quickstart(steps=50, preset="easy")
    report("一行启动运行50步", stats["steps_completed"] == 50)
    report("无崩溃", stats["mode"] == "normal")
    report("有reward统计", isinstance(stats["total_real_reward"], (int, float)))

    # 查看状态
    status = get_status(agent)
    report("状态查询成功", status["cycle_count"] == 50)
    report("难度初始EASY", status["current_difficulty"] == "EASY")


# ============================================================
# 测试21：长时运行稳定性
# ============================================================
def test21_long_run_stability():
    section("测试21：长时运行稳定性")
    agent, stats = quickstart(steps=200, preset="easy")
    report("200步运行成功", stats["steps_completed"] == 200)
    report("mode正常", stats["mode"] == "normal")
    report("avg_cycle_time<30ms",
           stats["avg_cycle_time"] * 1000 < 30,
           f"{stats['avg_cycle_time']*1000:.2f}ms")


# ============================================================
# 测试22：不同环境类型运行
# ============================================================
def test22_different_envs():
    section("测试22：不同环境类型运行")
    for env_type in ["blank", "mixed"]:
        agent, stats = quickstart(steps=30, preset="easy", env_type=env_type)
        report(f"{env_type}环境运行成功",
               stats["steps_completed"] == 30 and stats["mode"] == "normal")


# ============================================================
# 测试23：预设配置正确性
# ============================================================
def test23_preset_correctness():
    section("测试23：预设配置正确性")
    # easy预设应使用小规模
    agent_easy = SEVAFactory.create("easy")
    report("easy input_dim=16",
           agent_easy.config.get("perception", "input_dim") == 16)

    # standard预设应使用中规模
    agent_std = SEVAFactory.create("standard")
    report("standard input_dim=32",
           agent_std.config.get("perception", "input_dim") == 32)

    # research预设应使用大规模
    agent_res = SEVAFactory.create("research")
    report("research input_dim=64",
           agent_res.config.get("perception", "input_dim") == 64)


# ============================================================
# 测试24：API一致性
# ============================================================
def test24_api_consistency():
    section("测试24：API一致性")
    # quickstart和Factory+run应产生一致行为
    agent1, stats1 = quickstart(steps=20, preset="easy", seed=42)

    agent2 = SEVAFactory.create("easy", seed=42)
    stats2 = run(agent2, steps=20)

    report("两种方式步数一致",
           stats1["steps_completed"] == stats2["steps_completed"])
    report("两种方式mode一致", stats1["mode"] == stats2["mode"])


# ============================================================
# 测试25：性能测试
# ============================================================
def test25_performance():
    section("测试25：API性能开销")
    t0 = time.time()
    agent, stats = quickstart(steps=100, preset="easy")
    elapsed = time.time() - t0
    report("100步quickstart<5秒", elapsed < 5, f"{elapsed:.2f}s")
    report("avg_cycle_time<30ms",
           stats["avg_cycle_time"] * 1000 < 30,
           f"{stats['avg_cycle_time']*1000:.2f}ms")


# ============================================================
# 测试26：reset后重新运行
# ============================================================
def test26_reset_rerun():
    section("测试26：reset后重新运行")
    agent = SEVAFactory.create("easy")
    run(agent, steps=10)
    report("首次运行10步", agent.cycle_count == 10)

    agent.reset()
    report("reset后cycle_count=0", agent.cycle_count == 0)

    # 重新附加环境运行
    env = SimulationEnvironment(agent.config, env_type="mixed", seed=42)
    agent.attach_environment(env)
    agent.inject_input(env.reset())
    stats = run(agent, steps=10)
    report("重新运行10步", stats["steps_completed"] == 10)


# ============================================================
# 测试27：错误处理
# ============================================================
def test27_error_handling():
    section("测试27：API错误处理")
    # 无效预设
    try:
        SEVAFactory.create("invalid")
        report("无效预设抛ValueError", False)
    except ValueError:
        report("无效预设抛ValueError", True)

    # 无效环境类型（SimulationEnvironment会接受但可能行为异常）
    try:
        agent = SEVAFactory.create("easy", env_type="invalid_env")
        report("无效环境类型不崩溃", agent is not None)
    except Exception as e:
        report("无效环境类型不崩溃", False, str(e))


# ============================================================
# 测试28：stats完整性
# ============================================================
def test28_stats_completeness():
    section("测试28：stats完整性")
    agent, stats = quickstart(steps=10, preset="easy")
    required_keys = ["steps_completed", "total_real_reward", "total_cycles",
                     "env_done", "mode", "avg_cycle_time",
                     "avg_prediction_error", "task_stats", "skill_stats",
                     "evolution_stats"]
    for key in required_keys:
        report(f"stats含{key}", key in stats)


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第十九轮测试：阶段29统一API与快速启动验证")
    print("="*60)

    tests = [
        test1_preset_configs_basic,
        test2_preset_configs_get,
        test3_preset_configs_unknown,
        test4_preset_configs_diff,
        test5_factory_create,
        test6_factory_env_types,
        test7_factory_overrides,
        test8_factory_seed,
        test9_factory_without_env,
        test10_quickstart,
        test11_quickstart_presets,
        test12_quickstart_verbose,
        test13_run_function,
        test14_run_no_env,
        test15_run_verbose,
        test16_create_config,
        test17_get_status,
        test18_get_status_no_env,
        test19_package_import,
        test20_one_line_e2e,
        test21_long_run_stability,
        test22_different_envs,
        test23_preset_correctness,
        test24_api_consistency,
        test25_performance,
        test26_reset_rerun,
        test27_error_handling,
        test28_stats_completeness,
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
