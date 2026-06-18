"""
第三轮测试：极端边界与异常容错测试
对应设计文档：异常容错、降级兜底、熔断机制、数值稳定性
测试目标：异常场景100%兜底，无崩溃报错
"""
import sys
import os
import time
import threading
import traceback
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva.config import Config
from seva.bus import MessageBus, SystemMessage, MessageType
from seva.perception import PerceptionModule
from seva.predictor import MultiLevelPredictor
from seva.error_eval import ErrorEvaluator
from seva.optimizer import DualModeOptimizer
from seva.memory import MemoryAndReview, SimpleHNSW
from seva.safety import SafetyEthicsModule
from seva.decision import DecisionMaker
from seva.executor import Executor
from seva.drive import DriveEngine
from seva.goal import GoalGenerator
from seva.core import SelfEvolvingAgentCore
from seva.types import (PerceptionState, Drives, Goal, GoalType, Prediction,
                        Action, ActionType, ErrorSignal, ErrorAttribution,
                        ExperienceRecord, gen_id)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("test.round3")

# 测试结果统计
_results = []


def record(name, passed, detail=""):
    _results.append((name, passed, detail))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} {name} {detail}")


def section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 测试1：感知模块极端输入（NaN/Inf/超大值/空数组/错误形状）
# ============================================================
def test_perception_extreme_inputs():
    section("测试1：感知模块极端输入容错")
    config = Config()
    perc = PerceptionModule(config)
    input_dim = config.get("perception", "input_dim")
    feature_dim = config.get("perception", "feature_dim")

    cases = [
        ("NaN输入", np.full(input_dim, np.nan)),
        ("+Inf输入", np.full(input_dim, np.inf)),
        ("-Inf输入", np.full(input_dim, -np.inf)),
        ("超大值1e10", np.full(input_dim, 1e10)),
        ("超大值1e20", np.full(input_dim, 1e20)),
        ("超小值-1e10", np.full(input_dim, -1e10)),
        ("空数组", np.array([])),
        ("错误形状(2D)", np.ones((5, 5))),
        ("部分NaN", np.array([np.nan if i % 2 == 0 else 1.0 for i in range(input_dim)])),
        ("混合极端值", np.array([np.nan, np.inf, -np.inf, 1e10, -1e10, 0.0, 1.0, -1.0]
                                + [0.5] * (input_dim - 8))),
    ]

    all_pass = True
    for name, raw in cases:
        try:
            state = perc.process(raw)
            # 验证输出不含NaN/Inf
            has_bad = (np.any(np.isnan(state.internal)) or
                       np.any(np.isinf(state.internal)) or
                       np.any(np.isnan(state.precision)) or
                       np.any(np.isinf(state.precision)))
            shape_ok = state.internal.shape == (feature_dim,)
            if has_bad or not shape_ok:
                record(f"感知-{name}", False, f"输出含异常值或形状错误 shape={state.internal.shape}")
                all_pass = False
            else:
                record(f"感知-{name}", True, f"norm={np.linalg.norm(state.internal):.4f}")
        except Exception as e:
            record(f"感知-{name}", False, f"异常: {e}")
            all_pass = False

    record("test_perception_extreme_inputs", all_pass)


# ============================================================
# 测试2：预测模块极端输入
# ============================================================
def test_predictor_extreme_inputs():
    section("测试2：预测模块极端输入容错")
    config = Config()
    perc = PerceptionModule(config)
    pred = MultiLevelPredictor(config)
    feature_dim = config.get("perception", "feature_dim")

    # 先正常预热
    for _ in range(5):
        s = perc.process(np.random.randn(config.get("perception", "input_dim")))
        pred.predict(s)

    cases = [
        ("NaN状态", PerceptionState(
            internal=np.full(feature_dim, np.nan),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.ones(feature_dim) * 0.5)),
        ("Inf状态", PerceptionState(
            internal=np.full(feature_dim, np.inf),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.ones(feature_dim) * 0.5)),
        ("超大值1e15", PerceptionState(
            internal=np.full(feature_dim, 1e15),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.ones(feature_dim) * 0.5)),
        ("零精度", PerceptionState(
            internal=np.ones(feature_dim),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.zeros(feature_dim))),
        ("NaN精度", PerceptionState(
            internal=np.ones(feature_dim),
            external=np.zeros(config.get("perception", "input_dim")),
            precision=np.full(feature_dim, np.nan))),
    ]

    all_pass = True
    for name, state in cases:
        try:
            p = pred.predict(state)
            has_bad = (np.any(np.isnan(p.predicted_state)) or
                       np.any(np.isinf(p.predicted_state)) or
                       np.isnan(p.confidence))
            norm = np.linalg.norm(p.predicted_state)
            if has_bad or norm > 1000:
                record(f"预测-{name}", False, f"输出异常 norm={norm}")
                all_pass = False
            else:
                record(f"预测-{name}", True, f"norm={norm:.4f} conf={p.confidence:.4f}")
        except Exception as e:
            record(f"预测-{name}", False, f"异常: {e}")
            all_pass = False

    record("test_predictor_extreme_inputs", all_pass)


# ============================================================
# 测试3：误差评估模块极端输入
# ============================================================
def test_error_evaluator_extreme_inputs():
    section("测试3：误差评估模块极端输入容错")
    config = Config()
    evaluator = ErrorEvaluator(config)
    feature_dim = config.get("perception", "feature_dim")

    cases = [
        ("NaN状态", PerceptionState(
            internal=np.full(feature_dim, np.nan),
            external=np.zeros(feature_dim),
            precision=np.ones(feature_dim) * 0.5)),
        ("Inf状态+None预测", PerceptionState(
            internal=np.full(feature_dim, np.inf),
            external=np.zeros(feature_dim),
            precision=np.ones(feature_dim) * 0.5)),
        ("超大状态+超大预测", (
            PerceptionState(internal=np.full(feature_dim, 1e10),
                          external=np.zeros(feature_dim),
                          precision=np.ones(feature_dim) * 0.5),
            Prediction(predicted_state=np.full(feature_dim, 1e10),
                      confidence=0.5,
                      variance=np.ones(feature_dim) * 1e-10,
                      horizon=1))),
        ("零方差预测", (
            PerceptionState(internal=np.ones(feature_dim),
                          external=np.zeros(feature_dim),
                          precision=np.ones(feature_dim) * 0.5),
            Prediction(predicted_state=np.zeros(feature_dim),
                      confidence=0.5,
                      variance=np.zeros(feature_dim),  # 零方差
                      horizon=1))),
    ]

    all_pass = True
    for name, case in cases:
        try:
            if isinstance(case, tuple):
                state, pred = case
                err = evaluator.evaluate(state, pred, None, actual_state=state)
            else:
                err = evaluator.evaluate(case, None, None)
            bad = (np.isnan(err.magnitude) or np.isinf(err.magnitude) or
                   np.isnan(err.surprise) or np.isinf(err.surprise) or
                   err.surprise > 1000)
            if bad:
                record(f"误差-{name}", False, f"mag={err.magnitude} surprise={err.surprise}")
                all_pass = False
            else:
                record(f"误差-{name}", True, f"mag={err.magnitude:.4f} surprise={err.surprise:.4f}")
        except Exception as e:
            record(f"误差-{name}", False, f"异常: {e}")
            all_pass = False

    record("test_error_evaluator_extreme_inputs", all_pass)


# ============================================================
# 测试4：优化器极端参数
# ============================================================
def test_optimizer_extreme_params():
    section("测试4：优化器极端参数容错")
    config = Config()
    opt = DualModeOptimizer(config)
    dim = 32

    cases = [
        ("NaN参数", np.full(dim, np.nan), np.ones(dim)),
        ("Inf参数", np.full(dim, np.inf), np.ones(dim)),
        ("超大参数1e10", np.full(dim, 1e10), np.ones(dim)),
        ("NaN观测", np.ones(dim), np.full(dim, np.nan)),
        ("Inf观测", np.ones(dim), np.full(dim, np.inf)),
        ("超大观测1e15", np.ones(dim), np.full(dim, 1e15)),
        ("维度不匹配", np.ones(dim * 2), np.ones(dim)),
        ("空数组", np.array([]), np.array([])),
    ]

    all_pass = True
    for name, params, obs in cases:
        try:
            new_params = opt.update_online(params, obs)
            bad = np.any(np.isnan(new_params)) or np.any(np.isinf(new_params))
            norm = np.linalg.norm(new_params) if new_params.size > 0 else 0
            if bad or norm > 1000:
                record(f"优化器-{name}", False, f"输出异常 norm={norm}")
                all_pass = False
            else:
                record(f"优化器-{name}", True, f"norm={norm:.4f}")
        except Exception as e:
            record(f"优化器-{name}", False, f"异常: {e}")
            all_pass = False

    # 渐进融合极端测试
    fusion_cases = [
        ("融合-NaN", np.full(dim, np.nan), np.ones(dim)),
        ("融合-Inf", np.full(dim, np.inf), np.ones(dim)),
        ("融合-超大", np.full(dim, 1e10), np.full(dim, -1e10)),
        ("融合-维度不匹配", np.ones(dim * 3), np.ones(dim)),
    ]
    for name, old, new in fusion_cases:
        try:
            fused = opt.progressive_fusion(old, new)
            bad = np.any(np.isnan(fused)) or np.any(np.isinf(fused))
            norm = np.linalg.norm(fused)
            if bad or norm > 1000:
                record(f"优化器-{name}", False, f"输出异常 norm={norm}")
                all_pass = False
            else:
                record(f"优化器-{name}", True, f"norm={norm:.4f}")
        except Exception as e:
            record(f"优化器-{name}", False, f"异常: {e}")
            all_pass = False

    record("test_optimizer_extreme_params", all_pass)


# ============================================================
# 测试5：记忆模块极端输入与HNSW边界
# ============================================================
def test_memory_extreme_inputs():
    section("测试5：记忆模块极端输入容错")
    config = Config()
    mem = MemoryAndReview(config)
    feature_dim = config.get("perception", "feature_dim")

    # 极端向量添加到HNSW
    hnsw = SimpleHNSW(feature_dim, capacity=100)
    extreme_vectors = [
        ("NaN向量", np.full(feature_dim, np.nan)),
        ("Inf向量", np.full(feature_dim, np.inf)),
        ("超大向量1e10", np.full(feature_dim, 1e10)),
        ("零向量", np.zeros(feature_dim)),
        ("维度不匹配", np.ones(feature_dim * 2)),
        ("空向量", np.array([])),
    ]

    all_pass = True
    for name, vec in extreme_vectors:
        try:
            hnsw.add(vec, {"name": name})
            # 搜索
            results = hnsw.search(np.ones(feature_dim), k=3)
            # 验证搜索结果不含NaN距离
            bad = any(np.isnan(d) or np.isinf(d) for d, _ in results)
            if bad:
                record(f"HNSW-{name}", False, "搜索结果含NaN/Inf距离")
                all_pass = False
            else:
                record(f"HNSW-{name}", True, f"返回{len(results)}条")
        except Exception as e:
            record(f"HNSW-{name}", False, f"异常: {e}")
            all_pass = False

    # 巩固极端情况：空记忆、极少记忆
    try:
        mem2 = MemoryAndReview(config)
        mem2.consolidate()  # 空记忆巩固
        record("巩固-空记忆", True, "无异常")
    except Exception as e:
        record("巩固-空记忆", False, f"异常: {e}")
        all_pass = False

    # 构造含NaN状态的记录并巩固
    try:
        mem3 = MemoryAndReview(config)
        for i in range(15):
            rec = ExperienceRecord(
                record_id=gen_id("exp"),
                timestamp=time.time(),
                state=PerceptionState(
                    internal=np.full(feature_dim, np.nan) if i % 3 == 0 else np.ones(feature_dim) * i,
                    external=np.zeros(feature_dim),
                    precision=np.ones(feature_dim) * 0.5
                ),
                drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
                goal=None, prediction=None, action=None, error=None,
                attribution=None, reward=float(i)
            )
            mem3.store_experience(rec)
        mem3.consolidate()
        record("巩固-含NaN状态", True, f"巩固后{mem3.episodic_count()}条")
    except Exception as e:
        record("巩固-含NaN状态", False, f"异常: {e}")
        all_pass = False

    record("test_memory_extreme_inputs", all_pass)


# ============================================================
# 测试6：安全模块熔断与恢复
# ============================================================
def test_safety_circuit_breaker():
    section("测试6：安全模块熔断机制")
    config = Config()
    safety = SafetyEthicsModule(config)

    # 持续提交危险动作触发熔断
    dangerous_action = Action(
        action_id=gen_id("act"),
        type=ActionType.SELF_HARM,
        parameters={},
        expected_reward=0.0,
        source="decision"
    )
    safe_action = Action(
        action_id=gen_id("act"),
        type=ActionType.WAIT,
        parameters={},
        expected_reward=0.0,
        source="decision"
    )

    all_pass = True
    # 提交25次危险动作（阈值是最近20次否决率>0.5）
    veto_count = 0
    for i in range(25):
        review = safety.review_action(dangerous_action, {})
        if not review.is_approved:
            veto_count += 1
        if safety.circuit_breaker_active:
            logger.info(f"  熔断在第{i+1}次触发")
            break

    if safety.circuit_breaker_active:
        record("熔断触发", True, f"第{i+1}次触发，否决{veto_count}次")
    else:
        record("熔断触发", False, f"25次后未触发，否决{veto_count}次")
        all_pass = False

    # 熔断状态下安全动作也应被否决
    review = safety.review_action(safe_action, {})
    if not review.is_approved:
        record("熔断期安全动作被否决", True, "符合预期")
    else:
        record("熔断期安全动作被否决", False, "熔断期不应批准任何动作")
        all_pass = False

    # 重置熔断
    safety.reset_circuit_breaker()
    review = safety.review_action(safe_action, {})
    if review.is_approved:
        record("熔断重置后恢复", True, "安全动作重新被批准")
    else:
        record("熔断重置后恢复", False, "重置后仍被否决")
        all_pass = False

    record("test_safety_circuit_breaker", all_pass)


# ============================================================
# 测试7：消息总线拥塞与TTL过期
# ============================================================
def test_bus_congestion_and_ttl():
    section("测试7：消息总线拥塞与TTL")
    # 小容量总线测试拥塞
    bus = MessageBus(capacity=100, enable_recording=True)

    received = []
    def callback(msg):
        received.append(msg)

    bus.subscribe("target", callback)

    all_pass = True

    # 测试1：TTL过期消息不入队
    ttl_msg = SystemMessage(
        msg_id=gen_id("msg"), timestamp=time.time(),
        source_module="test", target_module="target",
        msg_type=MessageType.DATA, priority=5,
        payload={}, ttl=0  # TTL=0应被丢弃
    )
    ok = bus.publish(ttl_msg)
    if not ok:
        record("TTL=0消息被丢弃", True)
    else:
        record("TTL=0消息被丢弃", False, "TTL=0不应入队")
        all_pass = False

    # 测试2：填满队列触发驱逐
    for i in range(150):  # 超过capacity=100
        msg = SystemMessage(
            msg_id=gen_id("msg"), timestamp=time.time(),
            source_module="test", target_module="target",
            msg_type=MessageType.DATA, priority=i % 10,  # 不同优先级
            payload={"idx": i}, ttl=100
        )
        bus.publish(msg)

    stats = bus.get_stats()
    if stats["dropped"] > 0:
        record("队列满触发驱逐", True, f"dropped={stats['dropped']} size={stats['queue_size']}")
    else:
        record("队列满触发驱逐", False, f"无驱逐 dropped={stats['dropped']}")
        all_pass = False

    # 测试3：drain派发
    delivered = bus.drain()
    if delivered > 0:
        record("drain派发", True, f"派发{delivered}条，回调收到{len(received)}条")
    else:
        record("drain派发", False, "未派发任何消息")
        all_pass = False

    # 测试4：回调异常不影响其他派发
    bus2 = MessageBus(capacity=50)
    received2 = []
    def bad_callback(msg):
        raise ValueError("故意抛异常")
    def good_callback(msg):
        received2.append(msg)

    bus2.subscribe("target", bad_callback)
    bus2.subscribe("target", good_callback)

    for i in range(5):
        msg = SystemMessage(
            msg_id=gen_id("msg"), timestamp=time.time(),
            source_module="test", target_module="target",
            msg_type=MessageType.DATA, priority=5,
            payload={"idx": i}, ttl=100
        )
        bus2.publish(msg)
    bus2.drain()

    if len(received2) == 5:
        record("回调异常隔离", True, "坏回调不影响好回调")
    else:
        record("回调异常隔离", False, f"好回调只收到{len(received2)}/5")
        all_pass = False

    record("test_bus_congestion_and_ttl", all_pass)


# ============================================================
# 测试8：消息总线并发安全
# ============================================================
def test_bus_concurrency():
    section("测试8：消息总线并发安全")
    bus = MessageBus(capacity=5000, enable_recording=False)
    received = []
    received_lock = threading.Lock()

    def callback(msg):
        with received_lock:
            received.append(msg)

    bus.subscribe("target", callback)

    errors = []
    def producer(start, end):
        try:
            for i in range(start, end):
                msg = SystemMessage(
                    msg_id=gen_id("msg"), timestamp=time.time(),
                    source_module=f"producer_{threading.get_ident()}",
                    target_module="target",
                    msg_type=MessageType.DATA, priority=i % 10,
                    payload={"idx": i}, ttl=100
                )
                bus.publish(msg)
        except Exception as e:
            errors.append(str(e))

    threads = []
    n_threads = 4
    per_thread = 500
    for t in range(n_threads):
        th = threading.Thread(target=producer, args=(t * per_thread, (t + 1) * per_thread))
        threads.append(th)

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    all_pass = True
    if errors:
        record("并发发布无异常", False, f"{len(errors)}个异常: {errors[0]}")
        all_pass = False
    else:
        record("并发发布无异常", True, f"{n_threads}线程x{per_thread}条")

    # drain派发
    delivered = bus.drain()
    with received_lock:
        recv_count = len(received)

    if recv_count == n_threads * per_thread:
        record("并发派发完整", True, f"全部{recv_count}条送达")
    else:
        record("并发派发完整", False, f"只收到{recv_count}/{n_threads*per_thread}")
        all_pass = False

    record("test_bus_concurrency", all_pass)


# ============================================================
# 测试9：核心循环模块故障注入
# ============================================================
def test_core_module_fault_injection():
    section("测试9：核心循环模块故障注入")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    all_pass = True

    # 故障1：感知模块timeout
    core.modules["perception"].inject_fault("timeout")
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    try:
        result = core.run_cycle()
        # 应该使用fallback状态，不崩溃
        if result["mode"] == "safe_fallback" or result["cycle"] > 0:
            record("感知timeout降级", True, f"mode={result['mode']}")
        else:
            record("感知timeout降级", False, "未正确处理")
            all_pass = False
    except Exception as e:
        record("感知timeout降级", False, f"异常: {e}")
        all_pass = False
    core.modules["perception"].clear_fault()

    # 故障2：预测模块抛异常
    original_predict = core.modules["predictor"].predict
    def bad_predict(state, context=None):
        raise RuntimeError("预测模块故意故障")
    core.modules["predictor"].predict = bad_predict
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    try:
        result = core.run_cycle()
        # 核心循环应该捕获异常并进入safe_fallback
        if result["mode"] == "safe_fallback":
            record("预测故障降级", True, "进入safe_fallback")
        else:
            record("预测故障降级", False, f"mode={result['mode']}")
            all_pass = False
    except Exception as e:
        record("预测故障降级", False, f"异常未捕获: {e}")
        all_pass = False
    core.modules["predictor"].predict = original_predict

    # 故障3：执行器抛异常
    original_execute = core.modules["executor"].execute
    def bad_execute(action, environment=None):
        raise RuntimeError("执行器故意故障")
    core.modules["executor"].execute = bad_execute
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    try:
        result = core.run_cycle()
        record("执行器故障降级", True, f"mode={result['mode']}")
    except Exception as e:
        record("执行器故障降级", False, f"异常未捕获: {e}")
        all_pass = False
    core.modules["executor"].execute = original_execute

    # 故障4：记忆模块抛异常
    original_store = core.modules["memory"].store_experience
    def bad_store(record):
        raise RuntimeError("记忆模块故意故障")
    core.modules["memory"].store_experience = bad_store
    core.inject_input(np.random.randn(config.get("perception", "input_dim")))
    try:
        result = core.run_cycle()
        record("记忆故障降级", True, f"mode={result['mode']}")
    except Exception as e:
        record("记忆故障降级", False, f"异常未捕获: {e}")
        all_pass = False
    core.modules["memory"].store_experience = original_store

    record("test_core_module_fault_injection", all_pass)


# ============================================================
# 测试10：核心循环极端输入
# ============================================================
def test_core_extreme_inputs():
    section("测试10：核心循环极端输入")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    all_pass = True
    cases = [
        ("NaN输入", np.full(input_dim, np.nan)),
        ("Inf输入", np.full(input_dim, np.inf)),
        ("超大值1e10", np.full(input_dim, 1e10)),
        ("空数组", np.array([])),
        ("错误形状", np.ones((10, 10))),
        ("全零", np.zeros(input_dim)),
    ]

    for name, raw in cases:
        try:
            core.inject_input(raw)
            result = core.run_cycle()
            # 检查metrics无NaN
            m = core.get_metrics()
            bad = (np.isnan(m["avg_prediction_error"]) if m["avg_prediction_error"] != 0 else False)
            if bad:
                record(f"核心-{name}", False, "metrics含NaN")
                all_pass = False
            else:
                record(f"核心-{name}", True, f"mode={result['mode']}")
        except Exception as e:
            record(f"核心-{name}", False, f"异常: {e}")
            all_pass = False

    record("test_core_extreme_inputs", all_pass)


# ============================================================
# 测试11：长时间极端波动场景
# ============================================================
def test_long_extreme_fluctuation():
    section("测试11：长时间极端波动场景（500步）")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    all_pass = True
    errors = []
    nan_cycles = 0
    inf_cycles = 0
    crash = False

    for step in range(500):
        # 极端波动：每步随机切换不同极端输入
        choice = step % 6
        if choice == 0:
            raw = np.random.randn(input_dim)  # 正常
        elif choice == 1:
            raw = np.full(input_dim, 1e5)  # 超大
        elif choice == 2:
            raw = np.full(input_dim, -1e5)  # 超小
        elif choice == 3:
            raw = np.random.randn(input_dim) * 1e3  # 大幅波动
        elif choice == 4:
            raw = np.zeros(input_dim)  # 全零
        else:
            # 部分NaN
            raw = np.random.randn(input_dim)
            raw[::10] = np.nan

        try:
            core.inject_input(raw)
            result = core.run_cycle()
            # 检查误差是否含NaN/Inf
            if core.metrics["prediction_errors"]:
                last_err = core.metrics["prediction_errors"][-1]
                if np.isnan(last_err):
                    nan_cycles += 1
                elif np.isinf(last_err) or last_err > 1e6:
                    inf_cycles += 1
        except Exception as e:
            errors.append(f"step{step}: {e}")
            crash = True
            break

    if crash:
        record("500步极端波动", False, f"step{step}崩溃: {errors[0]}")
        all_pass = False
    elif nan_cycles > 0 or inf_cycles > 0:
        record("500步极端波动", False, f"NaN周期={nan_cycles} Inf周期={inf_cycles}")
        all_pass = False
    else:
        m = core.get_metrics()
        record("500步极端波动", True,
               f"cycles={m['cycle_count']} avg_err={m['avg_prediction_error']:.4f} "
               f"avg_surprise={m['avg_surprise']:.4f} mode={m['mode']}")

    record("test_long_extreme_fluctuation", all_pass)


# ============================================================
# 测试12：决策与执行模块边界
# ============================================================
def test_decision_executor_boundary():
    section("测试12：决策与执行模块边界")
    config = Config()
    dec = DecisionMaker(config)
    exe = Executor(config)
    feature_dim = config.get("perception", "feature_dim")

    all_pass = True

    # 空目标列表
    try:
        state = PerceptionState(
            internal=np.zeros(feature_dim),
            external=np.zeros(feature_dim),
            precision=np.ones(feature_dim) * 0.5
        )
        drives = Drives(safety=0.5, gain=0.5, efficiency=0.5)
        action = dec.select_action(drives, [], state)
        result = exe.execute(action)
        if result["success"]:
            record("空目标列表", True, f"action={action.type.value}")
        else:
            record("空目标列表", False, "执行失败")
            all_pass = False
    except Exception as e:
        record("空目标列表", False, f"异常: {e}")
        all_pass = False

    # 所有目标类型遍历
    goal_types = [
        (GoalType.COLLECT, {"target": "res"}),
        (GoalType.AVOID, {}),
        (GoalType.EXPLORE, {"direction": "up"}),
        (GoalType.MAINTAIN, {}),
        (GoalType.LEARN, {"focus": "all"}),
        (GoalType.COMMUNICATE, {}),
    ]
    for gt, params in goal_types:
        try:
            goal = Goal(goal_id=gen_id("goal"), type=gt, parameters=params, priority=0.8)
            action = dec.select_action(drives, [goal], state)
            result = exe.execute(action)
            if result["success"]:
                record(f"目标类型-{gt.value}", True, f"-> action={action.type.value}")
            else:
                record(f"目标类型-{gt.value}", False, f"执行失败: {result.get('error')}")
                all_pass = False
        except Exception as e:
            record(f"目标类型-{gt.value}", False, f"异常: {e}")
            all_pass = False

    # 危险动作到达执行器（应被标记失败但不崩溃）
    try:
        dangerous = Action(
            action_id=gen_id("act"),
            type=ActionType.SELF_HARM,
            parameters={},
            expected_reward=0.0,
            source="decision"
        )
        result = exe.execute(dangerous)
        if not result["success"] and "dangerous" in result.get("error", ""):
            record("危险动作执行器拦截", True, "正确标记失败")
        else:
            record("危险动作执行器拦截", False, "未正确拦截")
            all_pass = False
    except Exception as e:
        record("危险动作执行器拦截", False, f"异常: {e}")
        all_pass = False

    record("test_decision_executor_boundary", all_pass)


# ============================================================
# 主函数
# ============================================================
def main():
    print("============================================================")
    print("第三轮测试：极端边界与异常容错（共12项）")
    print("============================================================")

    tests = [
        test_perception_extreme_inputs,
        test_predictor_extreme_inputs,
        test_error_evaluator_extreme_inputs,
        test_optimizer_extreme_params,
        test_memory_extreme_inputs,
        test_safety_circuit_breaker,
        test_bus_congestion_and_ttl,
        test_bus_concurrency,
        test_core_module_fault_injection,
        test_core_extreme_inputs,
        test_long_extreme_fluctuation,
        test_decision_executor_boundary,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"测试{test.__name__}本身崩溃: {e}")
            traceback.print_exc()
            record(test.__name__, False, f"测试本身崩溃: {e}")

    # 汇总
    print("============================================================")
    print("第三轮测试结果汇总:")
    passed = sum(1 for _, p, _ in _results if p)
    failed = sum(1 for _, p, _ in _results if not p)
    for name, p, detail in _results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status} {name} {detail}")
    print(f"总计: {passed}通过, {failed}失败")
    print("============================================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
