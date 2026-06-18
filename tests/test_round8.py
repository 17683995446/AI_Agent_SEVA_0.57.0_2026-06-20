"""
第八轮测试：接口全量遍历与边界组合测试
对应阶段12
测试重点：
1. 所有模块接口参数边界组合
2. 配置项极端值
3. 消息总线TTL/优先级/容量边界
4. 数据结构边界（空列表、单元素、超大）
5. 类型边界（None、错误类型、混合类型）
6. 数值边界（0、负数、极大、极小、NaN、Inf）
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
from seva.bus import MessageBus, SystemMessage, MessageType, AgentMessage
from seva.perception import PerceptionModule, PredictiveCodingLayer, PrecisionEstimator
from seva.drive import DriveEngine
from seva.goal import GoalGenerator
from seva.predictor import MultiLevelPredictor, GRULayer, TransformerLayer, GNNLayer
from seva.error_eval import ErrorEvaluator
from seva.optimizer import DualModeOptimizer
from seva.memory import MemoryAndReview, SimpleHNSW
from seva.safety import SafetyEthicsModule
from seva.decision import DecisionMaker
from seva.executor import Executor
from seva.environment import SimulationEnvironment
from seva.types import (Action, ActionType, Goal, GoalType, Drives,
                        PerceptionState, Prediction, ErrorSignal,
                        ErrorAttribution, ExperienceRecord, Neuromodulators,
                        SafetyReviewResult, gen_id)

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
# 测试1：感知模块接口边界组合
# ============================================================
def test1_perception_boundaries():
    section("测试1：感知模块接口边界组合")
    config = Config()
    perc = PerceptionModule(config)
    input_dim = config.get("perception", "input_dim")
    feature_dim = config.get("perception", "feature_dim")

    # 空输入
    try:
        state = perc.process(np.array([]))
        report("空输入处理", state.internal.shape == (feature_dim,))
    except Exception as e:
        report("空输入处理", False, str(e))

    # 单元素输入
    try:
        state = perc.process(np.array([1.0]))
        report("单元素输入", state.internal.shape == (feature_dim,))
    except Exception as e:
        report("单元素输入", False, str(e))

    # 超大输入
    try:
        state = perc.process(np.random.randn(input_dim * 10))
        report("超大输入", state.internal.shape == (feature_dim,))
    except Exception as e:
        report("超大输入", False, str(e))

    # 全零输入
    try:
        state = perc.process(np.zeros(input_dim))
        report("全零输入", True)
    except Exception as e:
        report("全零输入", False, str(e))

    # 全NaN输入
    try:
        state = perc.process(np.full(input_dim, np.nan))
        report("全NaN输入", not np.any(np.isnan(state.internal)))
    except Exception as e:
        report("全NaN输入", False, str(e))

    # 全Inf输入
    try:
        state = perc.process(np.full(input_dim, np.inf))
        report("全Inf输入", not np.any(np.isinf(state.internal)))
    except Exception as e:
        report("全Inf输入", False, str(e))

    # 极大值输入
    try:
        state = perc.process(np.full(input_dim, 1e10))
        report("极大值输入", np.all(np.isfinite(state.internal)))
    except Exception as e:
        report("极大值输入", False, str(e))

    # 极小值输入
    try:
        state = perc.process(np.full(input_dim, 1e-10))
        report("极小值输入", True)
    except Exception as e:
        report("极小值输入", False, str(e))

    # 混合类型输入（int + float）
    try:
        state = perc.process(np.array([1, 2, 3] + [0.0] * (input_dim - 3)))
        report("混合类型输入", True)
    except Exception as e:
        report("混合类型输入", False, str(e))

    # 2D输入
    try:
        state = perc.process(np.random.randn(4, 8))
        report("2D输入", state.internal.shape == (feature_dim,))
    except Exception as e:
        report("2D输入", False, str(e))

    # None输入（防御性处理，转为零向量）
    try:
        state = perc.process(None)
        report("None输入优雅处理", state.internal.shape == (feature_dim,))
    except Exception as e:
        report("None输入优雅处理", False, str(e))

    # 列表输入（非numpy）
    try:
        state = perc.process([0.0] * input_dim)
        report("列表输入", True)
    except Exception as e:
        report("列表输入", False, str(e))


# ============================================================
# 测试2：预测模块接口边界组合
# ============================================================
def test2_predictor_boundaries():
    section("测试2：预测模块接口边界组合")
    config = Config()
    predictor = MultiLevelPredictor(config)
    feature_dim = config.get("perception", "feature_dim")

    # 正常状态
    state = PerceptionState(
        internal=np.random.randn(feature_dim),
        external=np.zeros(32),
        precision=np.ones(feature_dim) * 0.5
    )
    pred = predictor.predict(state)
    report("正常预测", pred.predicted_state.shape == (feature_dim,))

    # 全零状态
    state_zero = PerceptionState(
        internal=np.zeros(feature_dim),
        external=np.zeros(32),
        precision=np.ones(feature_dim)
    )
    pred = predictor.predict(state_zero)
    report("全零状态预测", np.all(np.isfinite(pred.predicted_state)))

    # 全NaN状态
    state_nan = PerceptionState(
        internal=np.full(feature_dim, np.nan),
        external=np.zeros(32),
        precision=np.ones(feature_dim)
    )
    try:
        pred = predictor.predict(state_nan)
        report("全NaN状态预测", np.all(np.isfinite(pred.predicted_state)))
    except Exception as e:
        report("全NaN状态预测", False, str(e))

    # 全Inf状态
    state_inf = PerceptionState(
        internal=np.full(feature_dim, np.inf),
        external=np.zeros(32),
        precision=np.ones(feature_dim)
    )
    try:
        pred = predictor.predict(state_inf)
        report("全Inf状态预测", np.all(np.isfinite(pred.predicted_state)))
    except Exception as e:
        report("全Inf状态预测", False, str(e))

    # 极大值状态
    state_big = PerceptionState(
        internal=np.full(feature_dim, 1e10),
        external=np.zeros(32),
        precision=np.ones(feature_dim)
    )
    try:
        pred = predictor.predict(state_big)
        report("极大值状态预测", np.all(np.isfinite(pred.predicted_state)))
    except Exception as e:
        report("极大值状态预测", False, str(e))

    # GRU层直接测试
    gru = GRULayer(feature_dim)
    out = gru.forward(np.zeros(feature_dim))
    report("GRU零输入", out.shape == (feature_dim,))
    out = gru.forward(np.full(feature_dim, np.nan))
    report("GRU NaN输入", np.all(np.isfinite(out)))

    # Transformer层直接测试
    trans = TransformerLayer(feature_dim, horizon=5)
    preds = trans.forward([])
    report("Transformer空序列", len(preds) == 5)
    preds = trans.forward([np.zeros(feature_dim)])
    report("Transformer单元素序列", len(preds) == 5)
    preds = trans.forward([np.full(feature_dim, np.nan)])
    report("Transformer NaN序列", all(np.all(np.isfinite(p)) for p in preds))

    # GNN层直接测试
    gnn = GNNLayer(feature_dim)
    out = gnn.predict([], [])
    report("GNN空图", out.shape == (feature_dim,))
    out = gnn.predict(["a", "b"], [("a", "b")])
    report("GNN正常图", out.shape == (feature_dim,))


# ============================================================
# 测试3：误差评估接口边界组合
# ============================================================
def test3_error_eval_boundaries():
    section("测试3：误差评估接口边界组合")
    config = Config()
    evaluator = ErrorEvaluator(config)
    feature_dim = config.get("perception", "feature_dim")

    # 正常评估
    state = PerceptionState(
        internal=np.random.randn(feature_dim),
        external=np.zeros(32),
        precision=np.ones(feature_dim) * 0.5
    )
    pred = Prediction(
        predicted_state=np.random.randn(feature_dim),
        confidence=0.5,
        variance=np.ones(feature_dim) * 0.1,
        horizon=1
    )
    action = Action(action_id="a1", type=ActionType.WAIT, parameters={})
    error = evaluator.evaluate(state, pred, action, actual_state=state,
                                context={"actual_reward": 0.0})
    report("正常评估", error.magnitude >= 0)

    # prediction=None
    error = evaluator.evaluate(state, None, action, actual_state=state)
    report("prediction=None", error.magnitude >= 0)

    # action=None
    error = evaluator.evaluate(state, pred, None, actual_state=state)
    report("action=None", error.magnitude >= 0)

    # actual_state=None
    error = evaluator.evaluate(state, pred, action, actual_state=None)
    report("actual_state=None", error.magnitude >= 0)

    # context=None
    error = evaluator.evaluate(state, pred, action, actual_state=state,
                                context=None)
    report("context=None", error.magnitude >= 0)

    # 全部None（除state）
    error = evaluator.evaluate(state, None, None, actual_state=None)
    report("全部None", error.magnitude >= 0)

    # compute_surprise边界
    s = evaluator.compute_surprise(np.zeros(feature_dim), None)
    report("surprise无预测", s >= 0 and np.isfinite(s))

    s = evaluator.compute_surprise(np.full(feature_dim, np.nan), pred)
    report("surprise NaN观测", np.isfinite(s))

    s = evaluator.compute_surprise(np.zeros(feature_dim), pred)
    report("surprise正常", np.isfinite(s))

    # attribute_error边界
    error_zero = ErrorSignal(
        magnitude=0.0,
        perception_error=np.zeros(feature_dim),
        prediction_error=np.zeros(feature_dim),
        action_error=0.0,
        surprise=0.0
    )
    attr = evaluator.attribute_error(error_zero)
    total = attr.perception + attr.model + attr.action + attr.environment
    report("归因全零误差", abs(total - 1.0) < 1e-4, f"total={total}")


# ============================================================
# 测试4：优化器接口边界组合
# ============================================================
def test4_optimizer_boundaries():
    section("测试4：优化器接口边界组合")
    config = Config()
    opt = DualModeOptimizer(config)

    # 正常更新
    params = np.random.randn(32)
    obs = np.random.randn(32)
    new_params = opt.update_online(params, obs)
    report("正常在线更新", new_params.shape == params.shape)

    # 空数组
    try:
        new_params = opt.update_online(np.array([]), np.array([]))
        report("空数组更新", True)
    except Exception as e:
        report("空数组更新", False, str(e))

    # 维度不匹配
    try:
        new_params = opt.update_online(np.random.randn(32), np.random.randn(64))
        report("维度不匹配更新", new_params.shape == (32,) or new_params.shape == (64,))
    except Exception as e:
        report("维度不匹配更新", False, str(e))

    # NaN参数
    try:
        new_params = opt.update_online(np.full(32, np.nan), np.random.randn(32))
        report("NaN参数更新", np.all(np.isfinite(new_params)))
    except Exception as e:
        report("NaN参数更新", False, str(e))

    # Inf观测
    try:
        new_params = opt.update_online(np.random.randn(32), np.full(32, np.inf))
        report("Inf观测更新", np.all(np.isfinite(new_params)))
    except Exception as e:
        report("Inf观测更新", False, str(e))

    # learning_rate边界
    new_params = opt.update_online(params, obs, learning_rate=0.0)
    report("lr=0更新", np.allclose(new_params, params))

    new_params = opt.update_online(params, obs, learning_rate=1.0)
    report("lr=1更新", np.all(np.isfinite(new_params)))

    new_params = opt.update_online(params, obs, learning_rate=-1.0)
    report("lr=-1更新", np.all(np.isfinite(new_params)))

    # progressive_fusion边界
    fused = opt.progressive_fusion(np.zeros(32), np.ones(32), alpha=0.0)
    report("fusion alpha=0", np.allclose(fused, np.zeros(32)))

    fused = opt.progressive_fusion(np.zeros(32), np.ones(32), alpha=1.0)
    report("fusion alpha=1", np.allclose(fused, np.ones(32)))

    fused = opt.progressive_fusion(np.zeros(32), np.ones(32), alpha=0.5)
    report("fusion alpha=0.5", np.allclose(fused, np.full(32, 0.5)))

    fused = opt.progressive_fusion(np.zeros(32), np.ones(32), alpha=2.0)
    report("fusion alpha=2截断", np.allclose(fused, np.ones(32)))

    fused = opt.progressive_fusion(np.zeros(32), np.ones(32), alpha=-1.0)
    report("fusion alpha=-1截断", np.allclose(fused, np.zeros(32)))

    # 离线训练
    model_params = {"layer1": np.random.randn(10, 10)}
    dataset = [{"layer1": np.random.randn(10, 10)} for _ in range(5)]
    updated = opt.train_offline_epoch(model_params, dataset)
    report("离线训练", "layer1" in updated)

    # 空数据集
    try:
        updated = opt.train_offline_epoch(model_params, [])
        report("空数据集训练", "layer1" in updated)
    except Exception as e:
        report("空数据集训练", False, str(e))


# ============================================================
# 测试5：记忆模块接口边界组合
# ============================================================
def test5_memory_boundaries():
    section("测试5：记忆模块接口边界组合")
    config = Config()
    memory = MemoryAndReview(config)
    feature_dim = config.get("perception", "feature_dim")

    # 空记忆检索
    results = memory.retrieve_episodic(np.zeros(feature_dim), k=5)
    report("空记忆检索", len(results) == 0)

    # 空记忆回放
    replayed = memory.replay()
    report("空记忆回放", len(replayed) == 0)

    # 空记忆巩固
    memory.consolidate()
    report("空记忆巩固", True)

    # 单条记忆
    state = PerceptionState(
        internal=np.random.randn(feature_dim),
        external=np.zeros(32),
        precision=np.ones(feature_dim)
    )
    record = ExperienceRecord(
        record_id="r1", timestamp=time.time(), state=state,
        drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
        goal=None, prediction=None, action=None, error=None,
        attribution=None, reward=0.5
    )
    memory.store_experience(record)
    report("单条记忆存储", memory.episodic_count() == 1)

    # 检索单条
    results = memory.retrieve_episodic(state.internal, k=5)
    report("单条记忆检索", len(results) == 1)

    # 回放单条
    replayed = memory.replay(batch_size=10)
    report("单条记忆回放", len(replayed) == 1)

    # HNSW边界
    hnsw = SimpleHNSW(feature_dim, capacity=100)
    # 空搜索
    results = hnsw.search(np.zeros(feature_dim), k=5)
    report("HNSW空搜索", len(results) == 0)

    # 添加单条
    hnsw.add(np.random.randn(feature_dim), "item1")
    results = hnsw.search(np.zeros(feature_dim), k=5)
    report("HNSW单条搜索", len(results) == 1)

    # k=0
    results = hnsw.search(np.zeros(feature_dim), k=0)
    report("HNSW k=0", len(results) == 0)

    # k=负数
    try:
        results = hnsw.search(np.zeros(feature_dim), k=-1)
        report("HNSW k=-1", True)
    except Exception as e:
        report("HNSW k=-1", False, str(e))

    # 容量限制
    hnsw_small = SimpleHNSW(feature_dim, capacity=3)
    for i in range(10):
        hnsw_small.add(np.random.randn(feature_dim), f"item{i}")
    report("HNSW容量限制", hnsw_small.size() == 3)

    # 语义记忆
    memory.update_semantic("key1", "value1")
    report("语义记忆更新", memory.get_semantic("key1") == "value1")
    report("语义记忆不存在key", memory.get_semantic("nonexistent") is None)


# ============================================================
# 测试6：安全模块接口边界组合
# ============================================================
def test6_safety_boundaries():
    section("测试6：安全模块接口边界组合")
    config = Config()
    safety = SafetyEthicsModule(config)

    # 所有动作类型测试
    for action_type in ActionType:
        action = Action(
            action_id=gen_id("act"), type=action_type,
            parameters={}, expected_reward=0.0, source="test"
        )
        review = safety.review_action(action)
        is_dangerous = action_type.value in config.get("safety", "dangerous_action_types")
        if is_dangerous:
            report(f"危险动作{action_type.value}被拦截", not review.is_approved)
        else:
            report(f"正常动作{action_type.value}批准", review.is_approved)

    # 欺骗性参数
    action = Action(
        action_id=gen_id("act"), type=ActionType.MOVE,
        parameters={"deceptive": True}, expected_reward=0.5, source="test"
    )
    review = safety.review_action(action)
    report("欺骗性参数被拦截", not review.is_approved)

    # 极端收益驱力+高奖励
    drives = Drives(safety=0.5, gain=0.99, efficiency=0.5)
    action = Action(
        action_id=gen_id("act"), type=ActionType.COLLECT,
        parameters={}, expected_reward=3.0, source="test"
    )
    review = safety.review_action(action, context={"drives": drives})
    report("极端收益驱力+高奖励被拦截", not review.is_approved)

    # context=None
    action = Action(
        action_id=gen_id("act"), type=ActionType.WAIT,
        parameters={}, expected_reward=0.0, source="test"
    )
    review = safety.review_action(action, context=None)
    report("context=None", review.is_approved)

    # judge接口
    result = safety.judge({"action_type": "self_harm"})
    report("judge危险动作", result["judgment"] == "unsafe")
    result = safety.judge({"action_type": "move"})
    report("judge正常动作", result["judgment"] == "safe")
    result = safety.judge({})
    report("judge空参数", result["judgment"] == "safe")


# ============================================================
# 测试7：决策模块接口边界组合
# ============================================================
def test7_decision_boundaries():
    section("测试7：决策模块接口边界组合")
    config = Config()
    decision = DecisionMaker(config)
    feature_dim = config.get("perception", "feature_dim")

    state = PerceptionState(
        internal=np.random.randn(feature_dim),
        external=np.zeros(32),
        precision=np.ones(feature_dim) * 0.5
    )
    drives = Drives(safety=0.5, gain=0.5, efficiency=0.5)

    # 空目标列表
    action = decision.select_action(drives, [], state)
    report("空目标列表", action.type == ActionType.WAIT)

    # 单目标
    goal = Goal(goal_id="g1", type=GoalType.COLLECT, parameters={},
                priority=0.8)
    action = decision.select_action(drives, [goal], state)
    report("单目标COLLECT", action.type == ActionType.COLLECT)

    # 多目标（选最高优先级）
    goals = [
        Goal(goal_id="g1", type=GoalType.EXPLORE, parameters={},
             priority=0.3),
        Goal(goal_id="g2", type=GoalType.AVOID, parameters={},
             priority=0.9),
        Goal(goal_id="g3", type=GoalType.MAINTAIN, parameters={},
             priority=0.5),
    ]
    action = decision.select_action(drives, goals, state)
    report("多目标选最高优先级", action.type == ActionType.MOVE)  # AVOID->MOVE

    # 所有目标类型
    for goal_type, expected_action in [
        (GoalType.COLLECT, ActionType.COLLECT),
        (GoalType.AVOID, ActionType.MOVE),
        (GoalType.EXPLORE, ActionType.MOVE),
        (GoalType.MAINTAIN, ActionType.WAIT),
        (GoalType.LEARN, ActionType.OBSERVE),
        (GoalType.COMMUNICATE, ActionType.WAIT),  # 默认
    ]:
        goal = Goal(goal_id="g", type=goal_type, parameters={},
                    priority=0.5)
        action = decision.select_action(drives, [goal], state)
        report(f"目标{goal_type.value}->动作", action.type == expected_action,
               f"got={action.type.value}")

    # prediction=None
    action = decision.select_action(drives, [goal], state, prediction=None)
    report("prediction=None", action is not None)


# ============================================================
# 测试8：执行器接口边界组合
# ============================================================
def test8_executor_boundaries():
    section("测试8：执行器接口边界组合")
    config = Config()
    executor = Executor(config)

    # 所有动作类型执行
    for action_type in ActionType:
        action = Action(
            action_id=gen_id("act"), type=action_type,
            parameters={}, expected_reward=0.5, source="test"
        )
        result = executor.execute(action)
        report(f"执行{action_type.value}", result is not None)

    # MOVE带方向参数
    for direction in ["up", "down", "left", "right", "away_from_threat", "invalid"]:
        action = Action(
            action_id=gen_id("act"), type=ActionType.MOVE,
            parameters={"direction": direction}, expected_reward=0.5, source="test"
        )
        result = executor.execute(action)
        report(f"MOVE方向={direction}", result["success"])

    # 空参数
    action = Action(
        action_id=gen_id("act"), type=ActionType.MOVE,
        parameters={}, expected_reward=0.5, source="test"
    )
    result = executor.execute(action)
    report("MOVE空参数", result["success"])

    # SELF_HARM不应执行到这里
    action = Action(
        action_id=gen_id("act"), type=ActionType.SELF_HARM,
        parameters={}, expected_reward=0.0, source="test"
    )
    result = executor.execute(action)
    report("SELF_HARM到达执行器报错", not result["success"])


# ============================================================
# 测试9：消息总线接口边界组合
# ============================================================
def test9_bus_boundaries():
    section("测试9：消息总线接口边界组合")
    bus = MessageBus(capacity=10, enable_recording=True)

    # TTL=0消息被丢弃
    msg = SystemMessage(
        msg_id="m1", timestamp=time.time(), source_module="test",
        target_module="dest", msg_type=MessageType.DATA,
        priority=5, payload={}, ttl=0
    )
    ok = bus.publish(msg)
    report("TTL=0消息丢弃", not ok)

    # 正常消息
    msg = SystemMessage(
        msg_id="m2", timestamp=time.time(), source_module="test",
        target_module="dest", msg_type=MessageType.DATA,
        priority=5, payload={"x": 1}
    )
    ok = bus.publish(msg)
    report("正常消息入队", ok)

    # 容量限制+优先级驱逐
    bus2 = MessageBus(capacity=5, enable_recording=False)
    # 填满低优先级
    for i in range(5):
        msg = SystemMessage(
            msg_id=f"low_{i}", timestamp=time.time(), source_module="test",
            target_module="dest", msg_type=MessageType.DATA,
            priority=1, payload={"i": i}
        )
        bus2.publish(msg)
    report("填满5条低优先级", bus2.get_stats()["queue_size"] == 5)

    # 发布高优先级，应驱逐低优先级
    msg = SystemMessage(
        msg_id="high", timestamp=time.time(), source_module="test",
        target_module="dest", msg_type=MessageType.DATA,
        priority=10, payload={"high": True}
    )
    bus2.publish(msg)
    stats = bus2.get_stats()
    report("高优先级驱逐低优先级", stats["dropped"] == 1)

    # 优先级排序派发
    bus3 = MessageBus(capacity=100, enable_recording=False)
    received = []
    bus3.subscribe("dest", lambda m: received.append(m.priority))

    for prio in [5, 1, 10, 3, 8]:
        msg = SystemMessage(
            msg_id=f"m_{prio}", timestamp=time.time(), source_module="test",
            target_module="dest", msg_type=MessageType.DATA,
            priority=prio, payload={}
        )
        bus3.publish(msg)
    bus3.drain()
    # 高优先级先派发
    report("优先级排序派发", received == [10, 8, 5, 3, 1],
           f"order={received}")

    # 广播消息
    bus4 = MessageBus(capacity=100, enable_recording=False)
    broadcast_received = []
    bus4.subscribe_broadcast(lambda m: broadcast_received.append(m))
    msg = SystemMessage(
        msg_id="b1", timestamp=time.time(), source_module="test",
        target_module=None, msg_type=MessageType.EVENT,
        priority=5, payload={}
    )
    bus4.publish(msg)
    bus4.drain()
    report("广播消息接收", len(broadcast_received) == 1)

    # 回调异常不影响其他消息
    bus5 = MessageBus(capacity=100, enable_recording=False)
    bus5.subscribe("dest", lambda m: (_ for _ in ()).throw(ValueError("回调异常")))
    bus5.subscribe("dest", lambda m: None)  # 正常回调
    msg = SystemMessage(
        msg_id="m", timestamp=time.time(), source_module="test",
        target_module="dest", msg_type=MessageType.DATA,
        priority=5, payload={}
    )
    bus5.publish(msg)
    try:
        bus5.drain()
        report("回调异常不崩溃", True)
    except Exception as e:
        report("回调异常不崩溃", False, str(e))

    # clear
    bus.clear()
    report("clear清空", bus.get_stats()["queue_size"] == 0)


# ============================================================
# 测试10：配置项极端值
# ============================================================
def test10_config_extremes():
    section("测试10：配置项极端值")
    # 极小input_dim
    try:
        config = Config(instance_overrides={"perception": {"input_dim": 1}})
        errors = config.validate()
        report("input_dim=1", len(errors) == 0)
    except Exception as e:
        report("input_dim=1", False, str(e))

    # input_dim=0
    config = Config(instance_overrides={"perception": {"input_dim": 0}})
    errors = config.validate()
    report("input_dim=0被检测", len(errors) > 0)

    # max_cycles=0
    config = Config(instance_overrides={"core": {"max_cycles": 0}})
    errors = config.validate()
    report("max_cycles=0被检测", len(errors) > 0)

    # max_cycles=-1
    config = Config(instance_overrides={"core": {"max_cycles": -1}})
    errors = config.validate()
    report("max_cycles=-1被检测", len(errors) > 0)

    # exploration_rate=2.0
    config = Config(instance_overrides={"goal_generator": {"exploration_rate": 2.0}})
    errors = config.validate()
    report("exploration_rate=2.0被检测", len(errors) > 0)

    # exploration_rate=-1.0
    config = Config(instance_overrides={"goal_generator": {"exploration_rate": -1.0}})
    errors = config.validate()
    report("exploration_rate=-1.0被检测", len(errors) > 0)

    # fusion_alpha=2.0
    config = Config(instance_overrides={"optimizer": {"fusion_alpha": 2.0}})
    errors = config.validate()
    report("fusion_alpha=2.0被检测", len(errors) > 0)

    # episodic_capacity=0
    config = Config(instance_overrides={"memory": {"episodic_capacity": 0}})
    errors = config.validate()
    report("episodic_capacity=0被检测", len(errors) > 0)

    # 运行时设置
    config = Config()
    config.set_runtime("custom", "key", "value")
    report("运行时设置自定义section", config.get("custom", "key") == "value")

    # 环境变量覆盖
    os.environ["SEVA_CORE__MAX_CYCLES"] = "999999"
    config = Config()
    report("环境变量覆盖", config.get("core", "max_cycles") == 999999)
    del os.environ["SEVA_CORE__MAX_CYCLES"]


# ============================================================
# 测试11：环境接口边界组合
# ============================================================
def test11_environment_boundaries():
    section("测试11：环境接口边界组合")
    config = Config()

    # 四种环境类型
    for env_type in ["blank", "mixed", "drift", "adversarial"]:
        try:
            env = SimulationEnvironment(config, env_type=env_type, seed=42)
            obs = env.reset()
            report(f"环境{env_type}初始化", obs is not None)
        except Exception as e:
            report(f"环境{env_type}初始化", False, str(e))

    # 无效环境类型
    try:
        env = SimulationEnvironment(config, env_type="invalid", seed=42)
        obs = env.reset()
        report("无效环境类型不崩溃", True)
    except Exception as e:
        report("无效环境类型不崩溃", False, str(e))

    # step各种动作
    env = SimulationEnvironment(config, env_type="mixed", seed=42)
    env.reset()
    for action_type in ActionType:
        action = Action(
            action_id=gen_id("act"), type=action_type,
            parameters={"direction": "up"}, expected_reward=0.0, source="test"
        )
        try:
            obs, reward, done, info = env.step(action)
            report(f"环境step {action_type.value}", True)
        except Exception as e:
            report(f"环境step {action_type.value}", False, str(e))

    # step None动作
    try:
        obs, reward, done, info = env.step(None)
        report("环境step None", True)
    except Exception as e:
        report("环境step None", False, str(e))

    # get_state
    state = env.get_state()
    report("get_state", "step" in state and "agent_pos" in state)

    # 多次reset
    for i in range(3):
        env.reset()
    report("多次reset", env.step_count == 0)


# ============================================================
# 测试12：数据结构边界
# ============================================================
def test12_data_structure_boundaries():
    section("测试12：数据结构边界")
    feature_dim = 128

    # Drives边界
    drives = Drives(safety=0.0, gain=0.0, efficiency=0.0)
    report("Drives全0", drives.safety == 0.0)

    drives = Drives(safety=1.0, gain=1.0, efficiency=1.0)
    report("Drives全1", drives.safety == 1.0)

    drives = Drives(safety=-1.0, gain=2.0, efficiency=0.5)
    report("Drives越界截断", drives.safety == 0.0 and drives.gain == 1.0)

    drives = Drives(safety=float('nan'), gain=float('inf'), efficiency=-float('inf'))
    report("Drives NaN/Inf", drives.safety == 0.5 and drives.gain == 0.5)

    # Goal边界
    goal = Goal(goal_id="g", type=GoalType.EXPLORE, parameters={},
                priority=0.0)
    report("Goal priority=0", goal.priority == 0.0)

    goal = Goal(goal_id="g", type=GoalType.EXPLORE, parameters={},
                priority=1.0)
    report("Goal priority=1", goal.priority == 1.0)

    goal = Goal(goal_id="g", type=GoalType.EXPLORE, parameters={},
                priority=2.0)
    report("Goal priority=2截断", goal.priority == 1.0)

    goal = Goal(goal_id="g", type=GoalType.EXPLORE, parameters={},
                priority=-1.0)
    report("Goal priority=-1截断", goal.priority == 0.0)

    # ErrorAttribution归一化
    attr = ErrorAttribution(perception=1.0, model=1.0, action=1.0, environment=1.0)
    total = attr.perception + attr.model + attr.action + attr.environment
    report("ErrorAttribution归一化", abs(total - 1.0) < 1e-6)

    attr = ErrorAttribution(perception=0.0, model=0.0, action=0.0, environment=0.0)
    report("ErrorAttribution全0", True)  # 不崩溃即可

    # Neuromodulators
    nm = Neuromodulators()
    nm.reset_to_baseline()
    report("Neuromodulators重置", nm.dopamine == nm.dopamine_baseline)


# ============================================================
# 测试13：核心循环接口边界组合
# ============================================================
def test13_core_boundaries():
    section("测试13：核心循环接口边界组合")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()

    # 无注入输入执行（使用零向量）
    result = core.run_cycle()
    report("无注入输入执行", result is not None)

    # 多次无注入执行
    for i in range(5):
        core.run_cycle()
    report("多次无注入执行", core.cycle_count == 6)

    # get_metrics
    metrics = core.get_metrics()
    report("get_metrics", "cycle_count" in metrics and "mode" in metrics)

    # health_check
    report("health_check", core.health_check())

    # reset
    core.reset()
    report("reset后cycle_count=0", core.cycle_count == 0)
    report("reset后mode=normal", core.mode == "normal")

    # get_handled_messages
    msgs = core.get_handled_messages()
    report("get_handled_messages", isinstance(msgs, list))


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第八轮测试：接口全量遍历与边界组合测试")
    print("="*60)

    tests = [
        test1_perception_boundaries,
        test2_predictor_boundaries,
        test3_error_eval_boundaries,
        test4_optimizer_boundaries,
        test5_memory_boundaries,
        test6_safety_boundaries,
        test7_decision_boundaries,
        test8_executor_boundaries,
        test9_bus_boundaries,
        test10_config_extremes,
        test11_environment_boundaries,
        test12_data_structure_boundaries,
        test13_core_boundaries,
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
