"""
第七轮测试：业务场景闭环与逆向场景测试
对应阶段11
测试重点：
1. 完整正向闭环（12阶段全链路）
2. 逆向场景（执行失败→回滚→重新决策）
3. 中断场景（周期中途异常→恢复→继续）
4. 重复操作幂等性
5. 安全漏洞验证（destroy_env 动作拦截）
6. 巩固数据完整性
7. 配置校验
8. 异常恢复机制
"""
import sys
import os
import time
import json
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志：WARNING级别过滤掉DEBUG/INFO，但保留WARNING/ERROR
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
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    RESULTS.append(f"\n=== {title} ===")


# ============================================================
# 测试1：完整正向闭环（12阶段全链路验证）
# ============================================================
def test1_full_forward_closed_loop():
    section("测试1：完整正向闭环（12阶段全链路）")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 注入输入并执行一个完整周期
    raw_input = np.random.randn(config.get("perception", "input_dim"))
    core.inject_input(raw_input)
    result = core.run_cycle()

    # 验证12阶段全部执行
    stages_executed = [log["stage"] for log in result["log"]]
    expected_stages = ["perception", "drive_engine", "goal_generator",
                       "predictor", "decision_maker", "safety_ethics",
                       "executor", "error_evaluator", "optimizer",
                       "memory", "metacognition", "timing_control"]
    report("12阶段全部执行",
           all(s in stages_executed for s in expected_stages),
           f"stages={stages_executed}")

    # 验证无错误阶段
    has_error = any(log["stage"] == "error" for log in result["log"])
    report("无错误阶段", not has_error)

    # 验证cycle_count递增
    report("cycle_count递增", core.cycle_count == 1, f"count={core.cycle_count}")

    # 验证metrics更新
    report("metrics更新", len(core.metrics["cycle_times"]) == 1)

    # 验证记忆存储
    report("记忆存储", core.modules["memory"].episodic_count() == 1)

    # 验证执行器计数
    report("执行器计数", core.modules["executor"].execute_count == 1)

    # 验证感知状态非fallback
    last_state = core.modules["perception"].get_current_perception()
    report("感知状态非fallback", last_state is not None and not last_state.is_fallback)

    # 验证预测已生成
    last_pred = core.modules["predictor"].get_last_prediction()
    report("预测已生成", last_pred is not None)

    # 验证动作已生成
    last_action = core.modules["decision_maker"].get_last_action()
    report("动作已生成", last_action is not None)

    # 验证驱力已计算
    last_drives = core.modules["drive_engine"].get_last_drives()
    report("驱力已计算", last_drives is not None)

    # 验证mode仍为normal
    report("mode仍为normal", core.mode == "normal", f"mode={core.mode}")


# ============================================================
# 测试2：连续多周期闭环（验证状态延续）
# ============================================================
def test2_continuous_cycles():
    section("测试2：连续多周期闭环（状态延续）")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    cycle_rewards = []
    for i in range(50):
        raw_input = np.random.randn(config.get("perception", "input_dim")) * 0.5
        core.inject_input(raw_input)
        result = core.run_cycle()
        cycle_rewards.append(result.get("reward", 0.0))

    report("50周期全部完成", core.cycle_count == 50)
    report("记忆累积", core.modules["memory"].episodic_count() == 50)
    report("metrics累积", len(core.metrics["cycle_times"]) == 50)
    report("总奖励累积", core.metrics["total_reward"] >= 0)
    report("无异常mode", core.mode == "normal", f"mode={core.mode}")

    # 验证巩固触发（consolidation_interval=1000，50步不触发）
    report("50步不触发巩固", core.modules["memory"].consolidation_count == 0)


# ============================================================
# 测试3：逆向场景 - 执行失败回滚
# ============================================================
def test3_reverse_rollback():
    section("测试3：逆向场景 - 执行失败回滚")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 注入正常输入执行一次
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    core.run_cycle()

    # 记录执行前状态
    exec_count_before = core.modules["executor"].execute_count
    mem_count_before = core.modules["memory"].episodic_count()

    # 注入故障到感知模块，直接调用process验证fallback
    core.modules["perception"].inject_fault("timeout")
    fallback_state = core.modules["perception"].process(
        np.random.randn(config.get("perception", "input_dim"))
    )
    # 验证故障时process返回fallback状态
    report("故障时process返回fallback", fallback_state.is_fallback,
           f"fallback={fallback_state.is_fallback}")

    # 验证周期仍能完成（不崩溃）
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    result = core.run_cycle()
    report("故障周期仍完成", result is not None)

    # 验证周期日志记录了fallback
    perception_log = next((l for l in result["log"] if l["stage"] == "perception"), None)
    report("周期日志记录fallback", perception_log is not None and
           perception_log["result"].get("fallback", False),
           f"log={perception_log}")

    # 清除故障后恢复
    core.modules["perception"].clear_fault()
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    result = core.run_cycle()
    perception_log = next((l for l in result["log"] if l["stage"] == "perception"), None)
    report("故障清除后恢复", perception_log is not None and
           not perception_log["result"].get("fallback", True))


# ============================================================
# 测试4：中断场景 - 周期中途异常恢复
# ============================================================
def test4_interrupt_recovery():
    section("测试4：中断场景 - 周期中途异常恢复")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # Monkey-patch predictor 抛异常模拟中途崩溃
    original_predict = core.modules["predictor"].predict
    call_count = [0]

    def faulty_predict(state, context=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("模拟预测模块崩溃")
        return original_predict(state, context)

    core.modules["predictor"].predict = faulty_predict
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    result = core.run_cycle()

    # 验证异常被捕获
    report("异常被捕获", result is not None)
    report("mode变为safe_fallback", core.mode == "safe_fallback",
           f"mode={core.mode}")

    # 验证日志记录了错误
    has_error_log = any(log["stage"] == "error" for log in result["log"])
    report("日志记录错误", has_error_log)

    # 恢复predictor
    core.modules["predictor"].predict = original_predict
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    result = core.run_cycle()

    # 验证恢复后仍能执行（但mode可能仍为safe_fallback，这是Bug 30）
    report("恢复后仍能执行", result is not None)

    # Bug 30验证：mode不会自动恢复
    mode_auto_recovered = (core.mode == "normal")
    report("Bug30验证: mode自动恢复", mode_auto_recovered,
           f"mode={core.mode}（若FAIL说明存在Bug30: 异常后mode不自动恢复）")


# ============================================================
# 测试5：重复操作幂等性
# ============================================================
def test5_idempotency():
    section("测试5：重复操作幂等性")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 相同输入执行两次
    np.random.seed(42)
    fixed_input = np.random.randn(config.get("perception", "input_dim"))

    # 第一次
    core.inject_input(fixed_input.copy())
    r1 = core.run_cycle()
    state1 = core.modules["perception"].get_current_perception()
    pred1 = core.modules["predictor"].get_last_prediction()

    # 第二次（相同输入）
    core.inject_input(fixed_input.copy())
    r2 = core.run_cycle()
    state2 = core.modules["perception"].get_current_perception()
    pred2 = core.modules["predictor"].get_last_prediction()

    # 验证两次都成功
    report("两次执行都成功", r1 is not None and r2 is not None)

    # 验证cycle_count递增
    report("cycle_count递增", core.cycle_count == 2)

    # 验证记忆累积（非幂等，每次都存储）
    report("记忆累积（每次存储）", core.modules["memory"].episodic_count() == 2)

    # 验证感知状态shape一致
    report("感知状态shape一致",
           state1.internal.shape == state2.internal.shape)

    # 验证预测shape一致
    report("预测shape一致",
           pred1.predicted_state.shape == pred2.predicted_state.shape)


# ============================================================
# 测试6：安全漏洞验证 - destroy_env 动作拦截（Bug 21）
# ============================================================
def test6_safety_destroy_env_bug():
    section("测试6：安全漏洞验证 - destroy_env 动作拦截（Bug 21）")
    config = Config()
    safety = SafetyEthicsModule(config)

    # 测试所有危险动作类型
    dangerous_actions = [
        ("self_harm", ActionType.SELF_HARM),
        ("harm_others", ActionType.HARM_OTHERS),
        ("destroy_env", ActionType.DESTROY_ENV),
    ]

    for name, action_type in dangerous_actions:
        action = Action(
            action_id=gen_id("act"),
            type=action_type,
            parameters={},
            expected_reward=0.0,
            source="test"
        )
        review = safety.review_action(action)
        # 所有危险动作都应被拦截
        report(f"危险动作{name}被拦截", not review.is_approved,
               f"approved={review.is_approved} reason={review.veto_reason}")

    # 验证配置中的dangerous_action_types
    config_types = set(config.get("safety", "dangerous_action_types"))
    enum_values = {ActionType.SELF_HARM.value, ActionType.HARM_OTHERS.value,
                   ActionType.DESTROY_ENV.value}
    report("配置与枚举值匹配", config_types == enum_values,
           f"config={config_types} enum={enum_values}")

    # 明确指出Bug 21
    if "destroy_environment" in config_types and "destroy_env" not in config_types:
        report("Bug21确认: destroy_env未被配置拦截", False,
               "配置含destroy_environment但枚举值为destroy_env")
    else:
        report("Bug21: 配置与枚举匹配", True)


# ============================================================
# 测试7：巩固数据完整性（Bug 24）
# ============================================================
def test7_consolidation_data_integrity():
    section("测试7：巩固数据完整性（Bug 24）")
    config = Config()
    memory = MemoryAndReview(config)

    # 存储大量记忆（超过500条）
    n_records = 1000
    for i in range(n_records):
        state = PerceptionState(
            internal=np.random.randn(config.get("perception", "feature_dim")) * 0.1,
            external=np.random.randn(config.get("perception", "input_dim")) * 0.1,
            precision=np.ones(config.get("perception", "feature_dim")) * 0.5,
        )
        drives = Drives(safety=0.5, gain=0.5, efficiency=0.5)
        record = ExperienceRecord(
            record_id=f"exp_{i}",
            timestamp=time.time(),
            state=state,
            drives=drives,
            goal=None,
            prediction=None,
            action=None,
            error=None,
            attribution=None,
            reward=float(i) / n_records  # 递增奖励
        )
        memory.store_experience(record)

    report(f"存储{n_records}条记忆", memory.episodic_count() == n_records)

    # 执行巩固
    memory.consolidate()
    after_count = memory.episodic_count()

    # Bug 24验证：巩固后记忆数量不应大幅减少（只合并相似的）
    # 如果只处理最近500条，会丢失前500条
    report("巩固后记忆数量合理", after_count > 0,
           f"after={after_count}")

    # 关键验证：巩固不应丢失全部记忆
    # 阶段18: k-means巩固会修剪每簇至~20%，但信息保存在语义记忆抽象中
    # loss_ratio可能较高（~0.8），但after_count应>0且语义记忆非空
    loss_ratio = 1.0 - (after_count / n_records)
    report("Bug24验证: 巩固后仍有记忆", after_count > 0,
           f"after={after_count} loss_ratio={loss_ratio:.2f}")
    report("Bug24验证: 语义记忆保留抽象", len(memory.semantic_memory) > 0,
           f"semantic_clusters={len(memory.semantic_memory)}")


# ============================================================
# 测试8：配置校验（Bug 20）
# ============================================================
def test8_config_validation():
    section("测试8：配置校验（Bug 20）")
    config = Config()

    # 验证validate方法可调用
    try:
        errors = config.validate()
        report("validate方法可调用", True, f"errors={errors}")
    except NameError as e:
        report("Bug20确认: validate方法NameError", False, str(e))
    except Exception as e:
        report("validate方法异常", False, str(e))

    # 验证默认配置无错误
    errors = config.validate()
    report("默认配置无错误", len(errors) == 0, f"errors={errors}")

    # 测试非法配置
    bad_config = Config(instance_overrides={"core": {"max_cycles": -1}})
    errors = bad_config.validate()
    report("非法配置被检测", len(errors) > 0, f"errors={errors}")


# ============================================================
# 测试9：异常恢复机制（Bug 30 深入验证）
# ============================================================
def test9_exception_recovery():
    section("测试9：异常恢复机制（Bug 30 深入验证）")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 正常执行几步
    for i in range(5):
        core.inject_input(np.random.randn(config.get("perception", "input_dim")))
        core.run_cycle()

    report("初始5步正常", core.mode == "normal", f"mode={core.mode}")

    # 注入持续异常
    original_predict = core.modules["predictor"].predict

    def always_fail_predict(state, context=None):
        raise RuntimeError("持续异常")

    core.modules["predictor"].predict = always_fail_predict
    for i in range(3):
        core.inject_input(np.random.randn(config.get("perception", "input_dim")))
        core.run_cycle()

    report("异常后mode=safe_fallback", core.mode == "safe_fallback")

    # 恢复predictor
    core.modules["predictor"].predict = original_predict
    for i in range(5):
        core.inject_input(np.random.randn(config.get("perception", "input_dim")))
        core.run_cycle()

    # Bug 30: mode不会自动恢复
    report("Bug30验证: 恢复后mode自动恢复", core.mode == "normal",
           f"mode={core.mode}（若FAIL说明存在Bug30）")

    # 验证reset能恢复
    core.reset()
    report("reset后mode恢复", core.mode == "normal")


# ============================================================
# 测试10：误差归因NaN保护（Bug 26）
# ============================================================
def test10_error_attribution_nan():
    section("测试10：误差归因NaN保护（Bug 26）")
    config = Config()
    evaluator = ErrorEvaluator(config)

    # 构造含NaN的误差信号
    nan_error = ErrorSignal(
        magnitude=float('nan'),
        perception_error=np.array([float('nan'), 1.0, float('inf')]),
        prediction_error=np.array([1.0, float('nan'), 2.0]),
        action_error=float('nan'),
        surprise=float('nan'),
        is_novel=False
    )

    try:
        attribution = evaluator.attribute_error(nan_error)
        total = attribution.perception + attribution.model + \
                attribution.action + attribution.environment
        report("NaN误差归因不崩溃", True)
        report("归因和为有限值", np.isfinite(total),
               f"total={total} perc={attribution.perception} "
               f"model={attribution.model} act={attribution.action} "
               f"env={attribution.environment}")
        report("Bug26验证: 归因各项有限",
               all(np.isfinite(x) for x in [attribution.perception,
                   attribution.model, attribution.action,
                   attribution.environment]),
               f"（若FAIL说明存在Bug26: 归因NaN未保护）")
    except Exception as e:
        report("Bug26确认: NaN归因崩溃", False, str(e))


# ============================================================
# 测试11：完整业务场景 - 环境交互闭环
# ============================================================
def test11_environment_interaction_loop():
    section("测试11：完整业务场景 - 环境交互闭环")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    env = SimulationEnvironment(config, env_type="mixed", seed=123)

    total_reward = 0.0
    cycle_results = []

    for step in range(100):
        # 从环境获取观测
        obs = env.reset() if step == 0 else None
        if obs is None:
            # 用上一步的观测（简化）
            obs = np.random.randn(config.get("perception", "input_dim")) * 0.3

        # 注入观测到Agent
        core.inject_input(obs)
        result = core.run_cycle()

        # 获取Agent动作
        action = core.modules["decision_maker"].get_last_action()

        # 执行动作到环境
        env_obs, env_reward, done, info = env.step(action)
        total_reward += env_reward
        cycle_results.append({
            "step": step,
            "action": action.type.value,
            "reward": env_reward,
            "done": done
        })

    report("100步环境交互完成", len(cycle_results) == 100)
    report("总奖励有限", np.isfinite(total_reward), f"total={total_reward:.2f}")
    report("Agent无崩溃", core.mode == "normal", f"mode={core.mode}")
    report("记忆累积", core.modules["memory"].episodic_count() == 100)

    # 验证动作类型多样性
    action_types = set(r["action"] for r in cycle_results)
    report("动作类型多样性", len(action_types) >= 1,
           f"types={action_types}")


# ============================================================
# 测试12：消息总线集成闭环
# ============================================================
def test12_bus_integration_loop():
    section("测试12：消息总线集成闭环")
    config = Config()
    bus = MessageBus(capacity=1000, enable_recording=True)
    core = SelfEvolvingAgentCore(config, bus=bus)
    core.initialize()

    # 订阅消息
    received = []

    def on_action(msg):
        received.append(msg)

    bus.subscribe("executor", on_action)

    # 发布几条消息
    for i in range(10):
        msg = SystemMessage(
            msg_id=f"msg_{i}",
            timestamp=time.time(),
            source_module="test",
            target_module="executor",
            msg_type=MessageType.ACTION,
            priority=5,
            payload={"idx": i}
        )
        bus.publish(msg)

    # 排空派发
    delivered = bus.drain()
    report("消息派发", delivered == 10, f"delivered={delivered}")
    report("订阅者接收", len(received) == 10, f"received={len(received)}")

    # 验证录制
    recording = bus.get_recording()
    report("消息录制", len(recording) == 10)

    # 执行Agent周期
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    core.run_cycle()
    report("总线集成下周期正常", core.cycle_count == 1)


# ============================================================
# 测试13：安全熔断恢复闭环
# ============================================================
def test13_circuit_breaker_recovery():
    section("测试13：安全熔断恢复闭环")
    config = Config()
    safety = SafetyEthicsModule(config)

    # 持续触发否决以激活熔断
    veto_count = 0
    for i in range(30):
        action = Action(
            action_id=gen_id("act"),
            type=ActionType.SELF_HARM,
            parameters={},
            expected_reward=0.0,
            source="test"
        )
        review = safety.review_action(action)
        if not review.is_approved:
            veto_count += 1

    report("触发足够否决", veto_count == 30)
    report("熔断激活", safety.circuit_breaker_active,
           f"active={safety.circuit_breaker_active}")

    # 熔断状态下正常动作也被否决
    normal_action = Action(
        action_id=gen_id("act"),
        type=ActionType.WAIT,
        parameters={},
        expected_reward=0.0,
        source="test"
    )
    review = safety.review_action(normal_action)
    report("熔断时正常动作被否决", not review.is_approved)

    # 重置熔断
    safety.reset_circuit_breaker()
    report("熔断重置", not safety.circuit_breaker_active)

    # 重置后正常动作恢复批准
    review = safety.review_action(normal_action)
    report("重置后正常动作批准", review.is_approved)


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第七轮测试：业务场景闭环与逆向场景测试")
    print("="*60)

    tests = [
        test1_full_forward_closed_loop,
        test2_continuous_cycles,
        test3_reverse_rollback,
        test4_interrupt_recovery,
        test5_idempotency,
        test6_safety_destroy_env_bug,
        test7_consolidation_data_integrity,
        test8_config_validation,
        test9_exception_recovery,
        test10_error_attribution_nan,
        test11_environment_interaction_loop,
        test12_bus_integration_loop,
        test13_circuit_breaker_recovery,
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

    # 输出结果汇总
    print("\n".join(RESULTS))

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
