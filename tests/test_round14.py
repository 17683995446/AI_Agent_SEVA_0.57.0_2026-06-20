"""
第十四轮测试：阶段24环境闭环真实化验证
验证执行器接入SimulationEnvironment，使用真实reward
核心目标：证明Agent在真实环境中获得真实反馈而非伪reward
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
from seva.executor import Executor
from seva.types import Action, ActionType

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
# 测试1：执行器无环境时向后兼容
# ============================================================
def test1_executor_backward_compat():
    section("测试1：执行器无环境向后兼容")

    config = Config()
    executor = Executor(config)

    # MOVE动作
    action = Action(action_id="a1", type=ActionType.MOVE,
                    parameters={"direction": "up"}, expected_reward=0.5,
                    source="test")
    result = executor.execute(action, environment=None)
    report("MOVE无环境", result["success"] and result["effect"] == "moved_up",
           f"effect={result.get('effect')}")
    report("MOVE伪reward", abs(result["reward"] - 0.5) < 1e-6,
           f"reward={result['reward']}")
    report("MOVE无next_obs", result["next_observation"] is None)

    # COLLECT动作
    action2 = Action(action_id="a2", type=ActionType.COLLECT,
                     parameters={}, expected_reward=0.0, source="test")
    result2 = executor.execute(action2, environment=None)
    report("COLLECT无环境", result2["success"] and result2["effect"] == "collected_resource")
    report("COLLECT伪reward=1.0", abs(result2["reward"] - 1.0) < 1e-6,
           f"reward={result2['reward']}")

    # WAIT动作
    action3 = Action(action_id="a3", type=ActionType.WAIT,
                     parameters={}, expected_reward=0.3, source="test")
    result3 = executor.execute(action3, environment=None)
    report("WAIT无环境", result3["success"] and result3["reward"] == 0.0)

    # SELF_HARM动作
    action4 = Action(action_id="a4", type=ActionType.SELF_HARM,
                     parameters={}, expected_reward=0.0, source="test")
    result4 = executor.execute(action4, environment=None)
    report("SELF_HARM被拦截", not result4["success"])


# ============================================================
# 测试2：执行器有环境时真实reward
# ============================================================
def test2_executor_with_env():
    section("测试2：执行器有环境真实reward")

    config = Config()
    executor = Executor(config)
    env = SimulationEnvironment(config, env_type="mixed", seed=42)
    env.reset()

    # MOVE动作 - 环境应推进
    initial_pos = env.agent_pos.copy()
    action = Action(action_id="a1", type=ActionType.MOVE,
                    parameters={"direction": "up"}, expected_reward=99.0,
                    source="test")
    result = executor.execute(action, environment=env)

    report("MOVE有环境success", result["success"])
    report("MOVE真实reward≠expected", abs(result["reward"] - 99.0) > 0.01,
           f"reward={result['reward']:.3f} (expected_reward=99.0)")
    report("MOVE有next_obs", result["next_observation"] is not None,
           f"obs_shape={result['next_observation'].shape}")
    report("MOVE环境推进", env.step_count == 1, f"step={env.step_count}")
    report("MOVE位置变化", not np.array_equal(initial_pos, env.agent_pos),
           f"pos={env.agent_pos.tolist()}")

    # COLLECT动作 - 需要在资源附近
    # 先移动到资源附近
    if env.resources:
        target = env.resources[0]
        env.agent_pos = target.copy()
        action2 = Action(action_id="a2", type=ActionType.COLLECT,
                         parameters={}, expected_reward=0.0, source="test")
        result2 = executor.execute(action2, environment=env)
        report("COLLECT有环境", result2["success"])
        report("COLLECT真实reward", "collected_resource" in result2.get("side_effects", []),
               f"reward={result2['reward']:.3f} side_effects={result2['side_effects']}")

    # WAIT动作
    action3 = Action(action_id="a3", type=ActionType.WAIT,
                     parameters={}, expected_reward=5.0, source="test")
    result3 = executor.execute(action3, environment=env)
    report("WAIT有环境", result3["success"])
    report("WAIT真实reward≠5.0", abs(result3["reward"] - 5.0) > 0.01,
           f"reward={result3['reward']:.3f}")


# ============================================================
# 测试3：核心循环环境闭环
# ============================================================
def test3_core_env_loop():
    section("测试3：核心循环环境闭环")

    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=42)

    # 附加环境
    core.attach_environment(env)
    report("环境已附加", core.environment is not None)

    # 注入初始观测
    obs = env.reset()
    core.inject_input(obs)

    # 运行10个周期
    rewards = []
    for i in range(10):
        result = core.run_cycle()
        exec_stage = [s for s in result["log"] if s["stage"] == "executor"][0]
        rewards.append(exec_stage["result"]["reward"])
        report(f"周期{i}环境连接", exec_stage["result"].get("env_connected", False))

    report("10周期全部环境连接", all(True for r in rewards))
    report("环境step_count=10", env.step_count == 10, f"step={env.step_count}")
    report("reward来自环境", any(abs(r) > 0.001 for r in rewards),
           f"rewards={[f'{r:.3f}' for r in rewards]}")

    # 验证下一周期输入来自环境观测
    report("pending_input已更新", core._pending_input is not None)
    report("pending_input是环境观测", core._pending_input.shape[0] == config.get("perception", "input_dim"))


# ============================================================
# 测试4：run_with_environment便捷方法
# ============================================================
def test4_run_with_environment():
    section("测试4：run_with_environment便捷方法")

    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=42)

    stats = core.run_with_environment(env, num_steps=200)

    report("运行200步", stats["steps_completed"] == 200,
           f"steps={stats['steps_completed']}")
    report("真实reward累积", stats["total_real_reward"] != 0.0,
           f"total_reward={stats['total_real_reward']:.3f}")
    report("动作分布非空", len(stats["action_distribution"]) > 0,
           f"dist={stats['action_distribution']}")
    report("环境状态返回", "step" in stats["env_state"],
           f"env_state={stats['env_state']}")

    # 分离环境后回退到伪reward
    core.detach_environment()
    report("环境已分离", core.environment is None)


# ============================================================
# 测试5：多环境类型测试
# ============================================================
def test5_multi_env_types():
    section("测试5：多环境类型测试")

    config = Config()
    env_types = ["blank", "mixed", "drift", "adversarial"]

    for env_type in env_types:
        core = SelfEvolvingAgentCore(config)
        core.initialize()
        env = SimulationEnvironment(config, env_type=env_type, seed=42)
        stats = core.run_with_environment(env, num_steps=100)

        report(f"{env_type}环境100步", stats["steps_completed"] == 100,
               f"reward={stats['total_real_reward']:.3f}")
        report(f"{env_type}不崩溃", core.mode == "normal",
               f"mode={core.mode}")


# ============================================================
# 测试6：环境reward vs 伪reward对比
# ============================================================
def test6_real_vs_pseudo_reward():
    section("测试6：真实reward vs 伪reward对比")

    config = Config()

    # 无环境运行
    core1 = SelfEvolvingAgentCore(config)
    core1.initialize()
    np.random.seed(42)
    for _ in range(100):
        core1.inject_input(np.random.randn(config.get("perception", "input_dim")))
        core1.run_cycle()
    pseudo_reward = core1.metrics["total_reward"]

    # 有环境运行
    core2 = SelfEvolvingAgentCore(config)
    core2.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=42)
    core2.run_with_environment(env, num_steps=100)
    real_reward = core2.modules["executor"].get_stats()["total_real_reward"]

    report("伪reward≠真实reward", abs(pseudo_reward - real_reward) > 0.1,
           f"pseudo={pseudo_reward:.3f} real={real_reward:.3f}")
    report("真实reward来自环境", core2.modules["executor"].execute_count == 100,
           f"exec_count={core2.modules['executor'].execute_count}")


# ============================================================
# 测试7：环境done处理
# ============================================================
def test7_env_done_handling():
    section("测试7：环境done处理")

    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 创建max_steps很小的环境
    env = SimulationEnvironment(config, env_type="mixed", seed=42)
    env.max_steps = 10  # 只运行10步

    stats = core.run_with_environment(env, num_steps=100)

    report("环境done后停止", stats["env_done"] == True,
           f"env_done={stats['env_done']}")
    report("步数不超过max_steps", stats["steps_completed"] <= 15,
           f"steps={stats['steps_completed']} max_steps=10")


# ============================================================
# 测试8：性能基准
# ============================================================
def test8_performance():
    section("测试8：环境闭环性能基准")

    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=42)

    obs = env.reset()
    core.attach_environment(env)
    core.inject_input(obs)

    # 预热
    for _ in range(10):
        core.run_cycle()

    # 测量100步
    t0 = time.time()
    for _ in range(100):
        core.run_cycle()
    elapsed = (time.time() - t0) / 100 * 1000

    report("环境闭环<30ms/周期", elapsed < 30,
           f"{elapsed:.2f}ms/周期")


def main():
    tests = [
        test1_executor_backward_compat,
        test2_executor_with_env,
        test3_core_env_loop,
        test4_run_with_environment,
        test5_multi_env_types,
        test6_real_vs_pseudo_reward,
        test7_env_done_handling,
        test8_performance,
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
