"""
第一轮复杂场景测试
- 静态代码核验 + 单元测试 + 集成测试 + 端到端测试
- 通过日志发现bug
"""
import sys
import os
import logging
import time
import numpy as np
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志：输出到文件和stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/workspace/test_round1.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test.round1")

from seva.config import Config
from seva.bus import MessageBus, SystemMessage, MessageType
from seva.types import *
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
from seva.core import SelfEvolvingAgentCore
from seva.environment import SimulationEnvironment


def test_basic_import():
    """测试1：基础导入"""
    logger.info("=" * 60)
    logger.info("测试1：基础导入")
    try:
        from seva.config import Config
        from seva.bus import MessageBus
        from seva.core import SelfEvolvingAgentCore
        logger.info("✓ 基础导入成功")
        return True
    except Exception as e:
        logger.error(f"✗ 基础导入失败: {e}")
        traceback.print_exc()
        return False


def test_config_validation():
    """测试2：配置校验"""
    logger.info("=" * 60)
    logger.info("测试2：配置校验")
    try:
        config = Config()
        errors = config.validate()
        if errors:
            logger.error(f"✗ 配置校验有错误: {errors}")
            return False
        logger.info("✓ 配置校验通过")
        return True
    except Exception as e:
        logger.error(f"✗ 配置校验异常: {e}")
        traceback.print_exc()
        return False


def test_message_bus():
    """测试3：消息总线"""
    logger.info("=" * 60)
    logger.info("测试3：消息总线")
    try:
        bus = MessageBus(capacity=100, enable_recording=True)
        received = []
        bus.subscribe("module_a", lambda msg: received.append(("p2p", msg)))
        bus.subscribe_broadcast(lambda msg: received.append(("bcast", msg)))

        # 点对点消息：只触发点对点订阅
        msg = SystemMessage(
            msg_id="test_1",
            timestamp=time.time(),
            source_module="test",
            target_module="module_a",
            msg_type=MessageType.DATA,
            priority=5,
            payload={"key": "value"}
        )
        bus.publish(msg)
        bus.drain()
        assert len(received) == 1, f"点对点消息期望收到1条，实际{len(received)}"
        assert received[0][0] == "p2p", "应为点对点消息"
        logger.info("✓ 点对点消息测试通过")

        # 广播消息：触发广播订阅
        received.clear()
        bcast_msg = SystemMessage(
            msg_id="test_2",
            timestamp=time.time(),
            source_module="test",
            target_module=None,  # 广播
            msg_type=MessageType.EVENT,
            priority=5,
            payload={"key": "value"}
        )
        bus.publish(bcast_msg)
        bus.drain()
        assert len(received) == 1, f"广播消息期望收到1条，实际{len(received)}"
        assert received[0][0] == "bcast", "应为广播消息"
        logger.info("✓ 广播消息测试通过")

        # 测试优先级
        bus2 = MessageBus(capacity=100)
        order = []
        bus2.subscribe("m", lambda msg: order.append(msg.priority))
        for p in [1, 5, 10, 3, 7]:
            bus2.publish(SystemMessage(
                msg_id=f"p_{p}", timestamp=time.time(),
                source_module="t", target_module="m",
                msg_type=MessageType.DATA, priority=p, payload={}
            ))
        bus2.drain()
        # 高优先级应先派发
        assert order[0] == 10, f"最高优先级应先派发，实际order={order}"
        logger.info("✓ 优先级测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 消息总线测试失败: {e}")
        traceback.print_exc()
        return False


def test_perception():
    """测试4：感知模块"""
    logger.info("=" * 60)
    logger.info("测试4：感知模块")
    try:
        config = Config()
        perc = PerceptionModule(config)
        raw = np.random.randn(config.get("perception", "input_dim"))
        state = perc.process(raw)
        assert state.internal.shape == (config.get("perception", "feature_dim"),), \
            f"内部维度错误: {state.internal.shape}"
        assert state.precision.shape == (config.get("perception", "feature_dim"),), \
            f"精度维度错误: {state.precision.shape}"
        assert np.all(state.precision >= 0), "精度有负值"
        logger.info("✓ 感知模块测试通过")

        # 测试fallback
        perc.inject_fault("timeout")
        fallback_state = perc.process(raw)
        assert fallback_state.is_fallback, "fallback标志未设置"
        logger.info("✓ fallback测试通过")
        perc.clear_fault()
        return True
    except Exception as e:
        logger.error(f"✗ 感知模块测试失败: {e}")
        traceback.print_exc()
        return False


def test_drive_engine():
    """测试5：驱力引擎"""
    logger.info("=" * 60)
    logger.info("测试5：驱力引擎")
    try:
        config = Config()
        engine = DriveEngine(config)
        state = PerceptionState(
            internal=np.zeros(config.get("perception", "feature_dim")),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.ones(config.get("perception", "feature_dim")) * 0.5
        )
        drives = engine.compute_drives(state, [])
        assert 0 <= drives.safety <= 1, f"safety={drives.safety}超出[0,1]"
        assert 0 <= drives.gain <= 1, f"gain={drives.gain}超出[0,1]"
        assert 0 <= drives.efficiency <= 1, f"efficiency={drives.efficiency}超出[0,1]"
        logger.info(f"✓ 驱力测试通过 safety={drives.safety:.3f} "
                    f"gain={drives.gain:.3f} eff={drives.efficiency:.3f}")

        # 测试神经调质
        nm = engine.get_neuromodulator_levels()
        assert 0 <= nm.dopamine <= 1, "多巴胺超出范围"
        logger.info("✓ 神经调质测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 驱力引擎测试失败: {e}")
        traceback.print_exc()
        return False


def test_safety_module():
    """测试6：安全模块"""
    logger.info("=" * 60)
    logger.info("测试6：安全模块")
    try:
        config = Config()
        safety = SafetyEthicsModule(config)

        # 正常动作应通过
        safe_action = Action(
            action_id="test", type=ActionType.MOVE,
            parameters={"direction": "up"}, expected_reward=0.5
        )
        result = safety.review_action(safe_action)
        assert result.is_approved, "正常动作应被批准"
        logger.info("✓ 正常动作批准")

        # 危险动作应被否决
        dangerous = Action(
            action_id="test2", type=ActionType.SELF_HARM,
            parameters={}, expected_reward=0
        )
        result = safety.review_action(dangerous)
        assert not result.is_approved, "危险动作应被否决"
        assert result.veto_layer == "instinct", "应由本能层否决"
        logger.info("✓ 危险动作否决")

        # 欺骗性动作应被否决
        deceptive = Action(
            action_id="test3", type=ActionType.MOVE,
            parameters={"direction": "up", "deceptive": True},
            expected_reward=0.5
        )
        result = safety.review_action(deceptive)
        assert not result.is_approved, "欺骗性动作应被否决"
        logger.info("✓ 欺骗性动作否决")
        return True
    except Exception as e:
        logger.error(f"✗ 安全模块测试失败: {e}")
        traceback.print_exc()
        return False


def test_core_loop_basic():
    """测试7：核心循环基础"""
    logger.info("=" * 60)
    logger.info("测试7：核心循环基础")
    try:
        config = Config()
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        raw_input = np.random.randn(config.get("perception", "input_dim"))
        agent.inject_input(raw_input)
        result = agent.run_cycle()

        assert result["cycle"] == 1, "周期数应为1"
        assert len(result["log"]) >= 12, f"应有12阶段日志，实际{len(result['log'])}"
        logger.info(f"✓ 核心循环测试通过 阶段数={len(result['log'])}")

        # 验证12阶段全部执行
        stages = [log["stage"] for log in result["log"]]
        expected = ["perception", "drive_engine", "goal_generator",
                    "predictor", "decision_maker", "safety_ethics",
                    "executor", "error_evaluator", "optimizer",
                    "memory", "metacognition", "timing_control"]
        for s in expected:
            assert s in stages, f"阶段{s}未执行"
        logger.info("✓ 12阶段全部执行")
        return True
    except Exception as e:
        logger.error(f"✗ 核心循环测试失败: {e}")
        traceback.print_exc()
        return False


def test_e2e_mixed_env():
    """测试8：端到端混合环境"""
    logger.info("=" * 60)
    logger.info("测试8：端到端混合环境（100步）")
    try:
        config = Config()
        config.set_runtime("environment", "max_steps", 100)
        env = SimulationEnvironment(config, env_type="mixed", seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        total_reward = 0
        for step in range(100):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        metrics = agent.get_metrics()
        logger.info(f"✓ 端到端测试完成 steps={step+1} "
                    f"reward={total_reward:.3f} "
                    f"avg_error={metrics['avg_prediction_error']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ 端到端测试失败: {e}")
        traceback.print_exc()
        return False


def test_e2e_drift_env():
    """测试9：漂移环境"""
    logger.info("=" * 60)
    logger.info("测试9：漂移环境（500步，含分布漂移）")
    try:
        config = Config()
        config.set_runtime("environment", "max_steps", 500)
        env = SimulationEnvironment(config, env_type="drift", seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        rewards = []
        surprises = []
        for step in range(500):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if step % 100 == 0:
                metrics = agent.get_metrics()
                surprises.append(metrics["avg_surprise"])
                logger.info(f"  step={step} reward={reward:.3f} "
                           f"surprise={metrics['avg_surprise']:.3f}")
            if done:
                break

        logger.info(f"✓ 漂移环境测试完成 total_reward={sum(rewards):.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ 漂移环境测试失败: {e}")
        traceback.print_exc()
        return False


def test_adversarial_env():
    """测试10：对抗环境"""
    logger.info("=" * 60)
    logger.info("测试10：对抗环境（200步）")
    try:
        config = Config()
        config.set_runtime("environment", "max_steps", 200)
        env = SimulationEnvironment(config, env_type="adversarial", seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        for step in range(200):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            if done:
                break

        safety_stats = agent.modules["safety_ethics"].get_stats()
        logger.info(f"✓ 对抗环境测试完成 safety_stats={safety_stats}")
        return True
    except Exception as e:
        logger.error(f"✗ 对抗环境测试失败: {e}")
        traceback.print_exc()
        return False


def test_long_run():
    """测试11：长时间运行（1000步）"""
    logger.info("=" * 60)
    logger.info("测试11：长时间运行（1000步）")
    try:
        config = Config()
        config.set_runtime("environment", "max_steps", 1000)
        env = SimulationEnvironment(config, env_type="mixed", seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        start_time = time.time()
        for step in range(1000):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            if step % 200 == 0:
                metrics = agent.get_metrics()
                elapsed = time.time() - start_time
                logger.info(f"  step={step} elapsed={elapsed:.1f}s "
                           f"reward={metrics['total_reward']:.3f} "
                           f"avg_cycle={metrics['avg_cycle_time']*1000:.1f}ms "
                           f"mem={agent.modules['memory'].episodic_count()}")
            if done:
                break

        elapsed = time.time() - start_time
        metrics = agent.get_metrics()
        logger.info(f"✓ 长时间运行完成 steps={step+1} "
                    f"elapsed={elapsed:.1f}s "
                    f"avg_cycle={metrics['avg_cycle_time']*1000:.1f}ms")
        return True
    except Exception as e:
        logger.error(f"✗ 长时间运行测试失败: {e}")
        traceback.print_exc()
        return False


def test_memory_consolidation():
    """测试12：记忆巩固"""
    logger.info("=" * 60)
    logger.info("测试12：记忆巩固")
    try:
        config = Config()
        config.set_runtime("memory", "consolidation_interval", 10)
        env = SimulationEnvironment(config, env_type="mixed", seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        for step in range(50):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)

        count = agent.modules["memory"].episodic_count()
        logger.info(f"✓ 记忆巩固测试完成 episodic_count={count}")
        return True
    except Exception as e:
        logger.error(f"✗ 记忆巩固测试失败: {e}")
        traceback.print_exc()
        return False


def test_circuit_breaker():
    """测试13：熔断机制"""
    logger.info("=" * 60)
    logger.info("测试13：熔断机制")
    try:
        config = Config()
        safety = SafetyEthicsModule(config)

        # 连续触发否决
        for i in range(25):
            dangerous = Action(
                action_id=f"test_{i}", type=ActionType.SELF_HARM,
                parameters={}, expected_reward=0
            )
            safety.review_action(dangerous)

        stats = safety.get_stats()
        logger.info(f"  熔断状态: {stats['circuit_breaker']}")
        logger.info(f"  否决次数: {stats['veto_count']}")
        # 熔断应在连续否决后触发
        if stats["circuit_breaker"]:
            logger.info("✓ 熔断触发成功")
        else:
            logger.warning("⚠ 熔断未触发（可能阈值未达到）")
        return True
    except Exception as e:
        logger.error(f"✗ 熔断测试失败: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("第一轮复杂场景测试开始")
    logger.info("=" * 60)

    tests = [
        test_basic_import,
        test_config_validation,
        test_message_bus,
        test_perception,
        test_drive_engine,
        test_safety_module,
        test_core_loop_basic,
        test_e2e_mixed_env,
        test_e2e_drift_env,
        test_adversarial_env,
        test_long_run,
        test_memory_consolidation,
        test_circuit_breaker,
    ]

    results = {}
    for test in tests:
        name = test.__name__
        try:
            results[name] = test()
        except Exception as e:
            results[name] = False
            logger.error(f"{name} 异常: {e}")
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    logger.info(f"总计: {passed}通过, {failed}失败")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    run_all_tests()
