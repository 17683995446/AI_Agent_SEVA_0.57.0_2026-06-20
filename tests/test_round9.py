"""
第九轮测试：混乱场景极限压力测试
对应阶段13
测试重点：高频分布漂移+资源枯竭+事件风暴+对抗注入的组合混乱场景
1. 5000步混合混乱环境（漂移+对抗+噪声）
2. 资源枯竭场景（资源耗尽后行为）
3. 事件风暴（高频事件冲击）
4. 对抗注入持续攻击
5. 多环境快速切换
6. 极端输入风暴（NaN/Inf/超大值混合）
7. 长时间运行内存稳定性
8. 熔断-恢复-再熔断循环
"""
import sys
import os
import time
import tracemalloc
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

from seva.config import Config
from seva.core import SelfEvolvingAgentCore
from seva.bus import MessageBus, SystemMessage, MessageType
from seva.perception import PerceptionModule
from seva.drive import DriveEngine
from seva.goal import GoalGenerator
from seva.predictor import MultiLevelPredictor
from seva.error_eval import ErrorEvaluator
from seva.optimizer import DualModeOptimizer
from seva.memory import MemoryAndReview
from seva.safety import SafetyEthicsModule
from seva.decision import DecisionMaker
from seva.executor import Executor
from seva.environment import SimulationEnvironment
from seva.types import (Action, ActionType, Goal, GoalType, Drives,
                        PerceptionState, Prediction, ErrorSignal,
                        ErrorAttribution, ExperienceRecord, gen_id)

PASS = 0
FAIL = 0
RESULTS = []


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    RESULTS.append(f"\n=== {title} ===")


# ============================================================
# 测试1：5000步混合混乱环境
# ============================================================
def test1_chaos_mixed_5000_steps():
    section("测试1：5000步混合混乱环境")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 混合输入：正常+噪声+偶尔NaN/Inf/超大值
    input_dim = config.get("perception", "input_dim")
    rng = np.random.default_rng(42)

    cycle_times = []
    errors_count = 0
    mode_changes = 0
    last_mode = "normal"

    t_start = time.time()
    for step in range(5000):
        # 混乱输入生成
        r = rng.random()
        if r < 0.7:
            # 70%正常输入
            raw = rng.normal(0, 1, input_dim)
        elif r < 0.8:
            # 10%大噪声
            raw = rng.normal(0, 5, input_dim)
        elif r < 0.85:
            # 5%含NaN
            raw = rng.normal(0, 1, input_dim)
            raw[rng.integers(0, input_dim)] = np.nan
        elif r < 0.9:
            # 5%含Inf
            raw = rng.normal(0, 1, input_dim)
            raw[rng.integers(0, input_dim)] = np.inf
        elif r < 0.95:
            # 5%超大值
            raw = rng.normal(0, 1, input_dim) * 1e6
        else:
            # 5%全零
            raw = np.zeros(input_dim)

        core.inject_input(raw)
        result = core.run_cycle()
        cycle_times.append(result["time"])

        if result["mode"] != "normal":
            errors_count += 1
        if result["mode"] != last_mode:
            mode_changes += 1
            last_mode = result["mode"]

    total_time = time.time() - t_start

    report("5000步完成", core.cycle_count == 5000)
    report("无崩溃", core.cycle_count == 5000)
    report("总时间合理", total_time < 120, f"time={total_time:.1f}s")
    report("平均周期时间<20ms", np.mean(cycle_times) < 0.020,
           f"avg={np.mean(cycle_times)*1000:.2f}ms")
    report("p99周期时间<100ms", np.percentile(cycle_times, 99) < 0.1,
           f"p99={np.percentile(cycle_times, 99)*1000:.2f}ms")
    report("记忆累积", core.modules["memory"].episodic_count() > 0)
    report("metrics不超限", len(core.metrics["cycle_times"]) <= 10000)
    report("总奖励有限", np.isfinite(core.metrics["total_reward"]),
           f"reward={core.metrics['total_reward']:.2f}")


# ============================================================
# 测试2：资源枯竭场景
# ============================================================
def test2_resource_depletion():
    section("测试2：资源枯竭场景")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=42)
    env.reset()

    # 清空所有资源模拟枯竭
    env.resources = []

    total_reward = 0.0
    collect_attempts = 0
    successful_collects = 0

    for step in range(500):
        obs = np.random.randn(config.get("perception", "input_dim")) * 0.3
        core.inject_input(obs)
        core.run_cycle()
        action = core.modules["decision_maker"].get_last_action()

        if action.type == ActionType.COLLECT:
            collect_attempts += 1

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if info.get("collected"):
            successful_collects += 1

    report("500步资源枯竭完成", True)
    report("总奖励有限", np.isfinite(total_reward))
    report("Agent不崩溃", core.mode == "normal", f"mode={core.mode}")
    report("无资源收集成功", successful_collects == 0,
           f"collected={successful_collects}")


# ============================================================
# 测试3：事件风暴（高频事件冲击）
# ============================================================
def test3_event_storm():
    section("测试3：事件风暴（高频事件冲击）")
    config = Config()
    bus = MessageBus(capacity=10000, enable_recording=True)
    core = SelfEvolvingAgentCore(config, bus=bus)
    core.initialize()

    received = []
    bus.subscribe("executor", lambda m: received.append(m))

    # 事件风暴：短时间内发布大量消息
    storm_size = 5000
    t0 = time.time()
    for i in range(storm_size):
        msg = SystemMessage(
            msg_id=f"storm_{i}",
            timestamp=time.time(),
            source_module="storm_generator",
            target_module="executor",
            msg_type=MessageType.EVENT,
            priority=np.random.randint(0, 11),
            payload={"event_id": i, "data": np.random.randn(10).tolist()}
        )
        bus.publish(msg)
    publish_time = time.time() - t0

    # 排空派发
    t0 = time.time()
    delivered = bus.drain()
    drain_time = time.time() - t0

    report(f"发布{storm_size}条消息", True)
    report("发布时间<2s", publish_time < 2.0, f"time={publish_time:.2f}s")
    report("派发时间<2s", drain_time < 2.0, f"time={drain_time:.2f}s")
    report("全部派发", delivered == storm_size, f"delivered={delivered}")
    report("全部接收", len(received) == storm_size,
           f"received={len(received)}")

    # 风暴后Agent仍能正常运行
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    result = core.run_cycle()
    report("风暴后Agent正常", result is not None)


# ============================================================
# 测试4：对抗注入持续攻击
# ============================================================
def test4_adversarial_persistent():
    section("测试4：对抗注入持续攻击")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="adversarial", seed=42)
    env.reset()

    total_reward = 0.0
    cycle_times = []

    for step in range(2000):
        # 对抗环境每100步注入扰动
        obs = np.random.randn(config.get("perception", "input_dim")) * 0.3
        # 额外注入对抗噪声
        if step % 50 == 0:
            obs += np.random.randn(len(obs)) * 2.0

        core.inject_input(obs)
        result = core.run_cycle()
        cycle_times.append(result["time"])

        action = core.modules["decision_maker"].get_last_action()
        env_obs, reward, done, info = env.step(action)
        total_reward += reward

    report("2000步对抗攻击完成", core.cycle_count == 2000)
    report("Agent不崩溃", core.mode == "normal", f"mode={core.mode}")
    report("总奖励有限", np.isfinite(total_reward),
           f"reward={total_reward:.2f}")
    report("平均周期时间<20ms", np.mean(cycle_times) < 0.020,
           f"avg={np.mean(cycle_times)*1000:.2f}ms")
    # 阶段18: k-means巩固会修剪记忆（每簇保留~20%），不再等于2000
    report("记忆累积", core.modules["memory"].episodic_count() > 0,
           f"episodic={core.modules['memory'].episodic_count()}")


# ============================================================
# 测试5：多环境快速切换
# ============================================================
def test5_rapid_env_switching():
    section("测试5：多环境快速切换")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    env_types = ["blank", "mixed", "drift", "adversarial"]
    envs = {t: SimulationEnvironment(config, env_type=t, seed=42) for t in env_types}
    for env in envs.values():
        env.reset()

    total_reward = 0.0
    for step in range(2000):
        # 每50步切换环境
        env_type = env_types[(step // 50) % len(env_types)]
        env = envs[env_type]

        obs = np.random.randn(config.get("perception", "input_dim")) * 0.3
        core.inject_input(obs)
        core.run_cycle()
        action = core.modules["decision_maker"].get_last_action()
        env_obs, reward, done, info = env.step(action)
        total_reward += reward

    report("2000步多环境切换完成", core.cycle_count == 2000)
    report("Agent不崩溃", core.mode == "normal")
    report("总奖励有限", np.isfinite(total_reward))
    # 阶段18: k-means巩固会修剪记忆（每簇保留~20%），不再等于2000
    report("记忆累积", core.modules["memory"].episodic_count() > 0,
           f"episodic={core.modules['memory'].episodic_count()}")


# ============================================================
# 测试6：极端输入风暴
# ============================================================
def test6_extreme_input_storm():
    section("测试6：极端输入风暴")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    # 各种极端输入快速交替
    extreme_inputs = [
        np.full(input_dim, np.nan),
        np.full(input_dim, np.inf),
        np.full(input_dim, -np.inf),
        np.full(input_dim, 1e10),
        np.full(input_dim, -1e10),
        np.full(input_dim, 1e-10),
        np.zeros(input_dim),
        np.ones(input_dim) * 10.0,
        np.ones(input_dim) * -10.0,
        np.random.randn(input_dim) * 100,
    ]

    for cycle in range(1000):
        raw = extreme_inputs[cycle % len(extreme_inputs)]
        core.inject_input(raw.copy())
        core.run_cycle()

    report("1000步极端输入风暴完成", core.cycle_count == 1000)
    report("Agent不崩溃", core.mode == "normal", f"mode={core.mode}")

    # 验证所有内部状态有限
    last_state = core.modules["perception"].get_current_perception()
    report("感知状态有限", last_state is not None and
           np.all(np.isfinite(last_state.internal)))

    last_pred = core.modules["predictor"].get_last_prediction()
    report("预测状态有限", last_pred is not None and
           np.all(np.isfinite(last_pred.predicted_state)))

    report("总奖励有限", np.isfinite(core.metrics["total_reward"]))


# ============================================================
# 测试7：长时间运行内存稳定性
# ============================================================
def test7_long_run_memory_stability():
    section("测试7：长时间运行内存稳定性")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # 运行3000步
    for step in range(3000):
        core.inject_input(np.random.randn(input_dim) * 0.5)
        core.run_cycle()

    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')

    # 计算总内存增长
    total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
    tracemalloc.stop()

    report("3000步完成", core.cycle_count == 3000)
    report("内存增长<50MB", total_growth < 50 * 1024 * 1024,
           f"growth={total_growth/1024/1024:.1f}MB")
    report("metrics列表不超限", len(core.metrics["cycle_times"]) <= 10000)
    report("记忆不超容量", core.modules["memory"].episodic_count() <=
           config.get("memory", "episodic_capacity"))


# ============================================================
# 测试8：熔断-恢复-再熔断循环
# ============================================================
def test8_circuit_breaker_cycles():
    section("测试8：熔断-恢复-再熔断循环")
    config = Config()
    safety = SafetyEthicsModule(config)

    # 模拟3轮熔断-恢复
    for cycle in range(3):
        # 触发熔断
        for i in range(30):
            action = Action(
                action_id=gen_id("act"), type=ActionType.SELF_HARM,
                parameters={}, expected_reward=0.0, source="test"
            )
            safety.review_action(action)

        report(f"第{cycle+1}轮熔断激活", safety.circuit_breaker_active)

        # 熔断时正常动作被否决
        normal_action = Action(
            action_id=gen_id("act"), type=ActionType.WAIT,
            parameters={}, expected_reward=0.0, source="test"
        )
        review = safety.review_action(normal_action)
        report(f"第{cycle+1}轮熔断时否决正常动作", not review.is_approved)

        # 重置熔断
        safety.reset_circuit_breaker()
        report(f"第{cycle+1}轮熔断重置", not safety.circuit_breaker_active)

        # 重置后正常动作恢复
        review = safety.review_action(normal_action)
        report(f"第{cycle+1}轮恢复后批准正常动作", review.is_approved)


# ============================================================
# 测试9：巩固-回放循环压力
# ============================================================
def test9_consolidation_replay_stress():
    section("测试9：巩固-回放循环压力")
    config = Config()
    memory = MemoryAndReview(config)
    feature_dim = config.get("perception", "feature_dim")

    # 存储3000条记忆
    for i in range(3000):
        state = PerceptionState(
            internal=np.random.randn(feature_dim) * 0.1,
            external=np.zeros(32),
            precision=np.ones(feature_dim) * 0.5
        )
        record = ExperienceRecord(
            record_id=f"exp_{i}", timestamp=time.time(),
            state=state, drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
            goal=None, prediction=None, action=None, error=None,
            attribution=None, reward=float(i % 10)
        )
        memory.store_experience(record)

    report("存储3000条记忆", memory.episodic_count() == 3000)

    # 多次巩固-回放循环
    for cycle in range(5):
        t0 = time.time()
        memory.consolidate()
        consolidate_time = time.time() - t0

        report(f"第{cycle+1}轮巩固完成", memory.episodic_count() > 0)
        report(f"第{cycle+1}轮巩固时间<1s", consolidate_time < 1.0,
               f"time={consolidate_time:.2f}s")

        # 回放
        replayed = memory.replay(batch_size=32)
        report(f"第{cycle+1}轮回放", len(replayed) <= 32)

        # 检索
        results = memory.retrieve_episodic(np.random.randn(feature_dim), k=5)
        report(f"第{cycle+1}轮检索", len(results) <= 5)

    report("5轮巩固后记忆不空", memory.episodic_count() > 0)


# ============================================================
# 测试10：全链路混乱场景（环境+输入+消息风暴）
# ============================================================
def test10_full_chain_chaos():
    section("测试10：全链路混乱场景")
    config = Config()
    bus = MessageBus(capacity=5000, enable_recording=False)
    core = SelfEvolvingAgentCore(config, bus=bus)
    core.initialize()
    env = SimulationEnvironment(config, env_type="adversarial", seed=42)
    env.reset()
    input_dim = config.get("perception", "input_dim")

    # 订阅消息
    msg_count = [0]
    bus.subscribe("executor", lambda m: msg_count.__setitem__(0, msg_count[0] + 1))

    # 同时进行：环境交互 + 消息发布 + 极端输入
    total_reward = 0.0
    for step in range(2000):
        # 混乱输入
        r = np.random.random()
        if r < 0.8:
            obs = np.random.randn(input_dim) * 0.5
        elif r < 0.9:
            obs = np.random.randn(input_dim) * 5.0
        else:
            obs = np.full(input_dim, np.nan)

        core.inject_input(obs)

        # 同时发布消息
        if step % 10 == 0:
            msg = SystemMessage(
                msg_id=f"chaos_{step}", timestamp=time.time(),
                source_module="chaos", target_module="executor",
                msg_type=MessageType.EVENT, priority=np.random.randint(0, 11),
                payload={"step": step}
            )
            bus.publish(msg)

        core.run_cycle()
        action = core.modules["decision_maker"].get_last_action()
        env_obs, reward, done, info = env.step(action)
        total_reward += reward

        # 定期排空消息
        if step % 100 == 0:
            bus.drain()

    report("2000步全链路混乱完成", core.cycle_count == 2000)
    report("Agent不崩溃", core.mode == "normal", f"mode={core.mode}")
    report("总奖励有限", np.isfinite(total_reward),
           f"reward={total_reward:.2f}")
    report("记忆累积", core.modules["memory"].episodic_count() > 0)
    report("消息总线未溢出", bus.get_stats()["queue_size"] <= 5000)


# ============================================================
# 测试11：预测模型长期学习稳定性
# ============================================================
def test11_predictor_long_term_stability():
    section("测试11：预测模型长期学习稳定性")
    config = Config()
    predictor = MultiLevelPredictor(config)
    feature_dim = config.get("perception", "feature_dim")

    # 模拟长期预测学习
    confidences = []
    norms = []

    for step in range(3000):
        # 缓慢漂移的输入
        t = step / 3000.0
        base = np.sin(np.linspace(0, 10 * np.pi, feature_dim)) * (1 + t)
        noise = np.random.randn(feature_dim) * 0.1
        internal = base + noise

        state = PerceptionState(
            internal=internal,
            external=np.zeros(32),
            precision=np.ones(feature_dim) * 0.5
        )
        pred = predictor.predict(state)
        confidences.append(pred.confidence)
        norms.append(np.linalg.norm(pred.predicted_state))

    report("3000步预测完成", len(confidences) == 3000)
    report("置信度全有限", all(np.isfinite(c) for c in confidences))
    report("预测范数全有限", all(np.isfinite(n) for n in norms))
    report("置信度在[0,1]", all(0 <= c <= 1 for c in confidences))
    report("预测范数有界", max(norms) < 200,
           f"max_norm={max(norms):.2f}")

    # 验证隐藏状态未爆炸
    report("GRU隐藏状态有限",
           np.all(np.isfinite(predictor.bottom.hidden)))


# ============================================================
# 测试12：驱力引擎长期稳定性
# ============================================================
def test12_drive_long_term_stability():
    section("测试12：驱力引擎长期稳定性")
    config = Config()
    drive = DriveEngine(config)
    feature_dim = config.get("perception", "feature_dim")

    safety_values = []
    gain_values = []
    eff_values = []

    for step in range(2000):
        # 模拟波动环境
        if step % 200 < 100:
            # 安全期
            external = np.random.randn(32) * 0.1
        else:
            # 危险期
            external = np.random.randn(32) * 0.1 - 2.0

        state = PerceptionState(
            internal=np.random.randn(feature_dim) * 0.1,
            external=external,
            precision=np.ones(feature_dim) * 0.5
        )
        drives = drive.compute_drives(state, [])
        safety_values.append(drives.safety)
        gain_values.append(drives.gain)
        eff_values.append(drives.efficiency)

    report("2000步驱力计算完成", len(safety_values) == 2000)
    report("安全驱力全有限", all(np.isfinite(s) for s in safety_values))
    report("收益驱力全有限", all(np.isfinite(g) for g in gain_values))
    report("能效驱力全有限", all(np.isfinite(e) for e in eff_values))
    report("驱力在[0,1]",
           all(0 <= s <= 1 for s in safety_values) and
           all(0 <= g <= 1 for g in gain_values) and
           all(0 <= e <= 1 for e in eff_values))

    # 验证安全驱力在危险期下降
    danger_safety = np.mean(safety_values[100:200])
    safe_safety = np.mean(safety_values[0:100])
    report("危险期安全驱力下降", danger_safety < safe_safety,
           f"safe={safe_safety:.3f} danger={danger_safety:.3f}")

    # 神经调质有限
    nm = drive.get_neuromodulator_levels()
    report("神经调质有限",
           all(np.isfinite(x) for x in [nm.dopamine, nm.serotonin,
               nm.norepinephrine, nm.acetylcholine]))


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第九轮测试：混乱场景极限压力测试")
    print("="*60)

    tests = [
        test1_chaos_mixed_5000_steps,
        test2_resource_depletion,
        test3_event_storm,
        test4_adversarial_persistent,
        test5_rapid_env_switching,
        test6_extreme_input_storm,
        test7_long_run_memory_stability,
        test8_circuit_breaker_cycles,
        test9_consolidation_replay_stress,
        test10_full_chain_chaos,
        test11_predictor_long_term_stability,
        test12_drive_long_term_stability,
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
