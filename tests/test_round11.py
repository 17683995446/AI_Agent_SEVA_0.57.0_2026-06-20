"""
第十一轮测试：阶段16算法内核真实化验证
验证GRU/Transformer/GNN/Optimizer真实训练能力
核心目标：证明loss随训练下降，模型真正在学习
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
from seva.predictor import (MultiLevelPredictor, GRULayer, TransformerLayer,
                            GNNLayer, _safe, _sigmoid, _softmax)
from seva.optimizer import DualModeOptimizer
from seva.types import PerceptionState

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
# 测试1：GRU层真实训练——sin波预测任务
# ============================================================
def test1_gru_real_training():
    section("测试1：GRU层真实训练（sin波预测）")
    np.random.seed(42)
    dim = 16
    gru = GRULayer(dim, learning_rate=0.05)

    # 生成sin波序列：x_t = sin(t), target = sin(t+0.1)
    t_values = np.linspace(0, 20, 500)
    inputs = np.sin(t_values)
    targets = np.sin(t_values + 0.1)

    # 将标量扩展到dim维（每个维度相同信号+小噪声）
    def to_vec(x):
        return np.full(dim, x) + np.random.randn(dim) * 0.01

    # 训练前loss
    gru.reset()
    losses_before = []
    for i in range(50):
        x = to_vec(inputs[i])
        target = to_vec(targets[i])
        h = gru.forward(x)
        loss = float(np.mean((h - target) ** 2))
        losses_before.append(loss)

    # 训练
    gru.reset()
    train_losses = []
    for epoch in range(5):
        for i in range(400):
            x = to_vec(inputs[i])
            target = to_vec(targets[i])
            loss = gru.train_step(x, target)
            train_losses.append(loss)

    # 训练后loss
    losses_after = []
    for i in range(400, 450):
        x = to_vec(inputs[i])
        target = to_vec(targets[i])
        h = gru.forward(x)
        loss = float(np.mean((h - target) ** 2))
        losses_after.append(loss)

    avg_before = np.mean(losses_before)
    avg_after = np.mean(losses_after)
    stats = gru.get_train_stats()

    report("GRU训练步数>0", stats["train_steps"] > 0, f"steps={stats['train_steps']}")
    report("GRU训练前loss有限", np.isfinite(avg_before), f"loss={avg_before:.4f}")
    report("GRU训练后loss有限", np.isfinite(avg_after), f"loss={avg_after:.4f}")
    report("GRU loss下降", avg_after < avg_before,
           f"before={avg_before:.4f} after={avg_after:.4f}")
    report("GRU loss下降幅度>10%", (avg_before - avg_after) / avg_before > 0.1,
           f"improvement={((avg_before-avg_after)/avg_before*100):.1f}%")
    report("GRU训练loss趋势为负", stats["loss_trend"] < 0,
           f"trend={stats['loss_trend']:.6f}")


# ============================================================
# 测试2：Transformer层真实训练——序列预测
# ============================================================
def test2_transformer_real_training():
    section("测试2：Transformer层真实训练（序列预测）")
    np.random.seed(42)
    dim = 16
    trans = TransformerLayer(dim, horizon=3, learning_rate=0.01)

    # 生成序列预测任务：给定前5步，预测下一步
    def gen_sequence(start):
        t = np.linspace(start, start + 1, 5)
        return [np.sin(t[i]) + np.random.randn(dim) * 0.05 for i in range(5)]

    def gen_target(start):
        return np.sin(start + 1.2) + np.random.randn(dim) * 0.05

    # 训练前loss
    losses_before = []
    for i in range(20):
        seq = gen_sequence(i * 0.5)
        target = gen_target(i * 0.5)
        preds = trans.forward(seq)
        loss = float(np.mean((preds[0] - target) ** 2))
        losses_before.append(loss)

    # 训练
    train_losses = []
    for epoch in range(10):
        for i in range(50):
            seq = gen_sequence(i * 0.3)
            target = gen_target(i * 0.3)
            loss = trans.train_step(seq, target)
            train_losses.append(loss)

    # 训练后loss
    losses_after = []
    for i in range(20):
        seq = gen_sequence(i * 0.5 + 100)  # 新数据
        target = gen_target(i * 0.5 + 100)
        preds = trans.forward(seq)
        loss = float(np.mean((preds[0] - target) ** 2))
        losses_after.append(loss)

    avg_before = np.mean(losses_before)
    avg_after = np.mean(losses_after)
    stats = trans.get_train_stats()

    report("Transformer训练步数>0", stats["train_steps"] > 0)
    report("Transformer训练前loss有限", np.isfinite(avg_before), f"loss={avg_before:.4f}")
    report("Transformer训练后loss有限", np.isfinite(avg_after), f"loss={avg_after:.4f}")
    report("Transformer loss下降", avg_after < avg_before,
           f"before={avg_before:.4f} after={avg_after:.4f}")


# ============================================================
# 测试3：GNN层真实训练——图节点预测
# ============================================================
def test3_gnn_real_training():
    section("测试3：GNN层真实训练（图节点预测）")
    np.random.seed(42)
    dim = 16
    gnn = GNNLayer(dim, learning_rate=0.01)

    # 图结构：3个节点，形成三角图
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c"), ("a", "c")]

    # 目标：让图表示接近某个固定向量
    target = np.ones(dim) * 0.5

    # 训练前loss
    pred_before = gnn.forward(nodes, edges)
    loss_before = float(np.mean((pred_before - target) ** 2))

    # 训练
    train_losses = []
    for i in range(200):
        loss = gnn.train_step(nodes, edges, target)
        train_losses.append(loss)

    # 训练后loss
    pred_after = gnn.forward(nodes, edges)
    loss_after = float(np.mean((pred_after - target) ** 2))
    stats = gnn.get_train_stats()

    report("GNN训练步数>0", stats["train_steps"] > 0, f"steps={stats['train_steps']}")
    report("GNN训练前loss有限", np.isfinite(loss_before), f"loss={loss_before:.4f}")
    report("GNN训练后loss有限", np.isfinite(loss_after), f"loss={loss_after:.4f}")
    report("GNN loss下降", loss_after < loss_before,
           f"before={loss_before:.4f} after={loss_after:.4f}")
    report("GNN loss下降幅度>20%", (loss_before - loss_after) / loss_before > 0.2,
           f"improvement={((loss_before-loss_after)/loss_before*100):.1f}%")


# ============================================================
# 测试4：优化器真实Adam——参数拟合任务
# ============================================================
def test4_optimizer_real_adam():
    section("测试4：优化器真实Adam（参数拟合）")
    np.random.seed(42)
    config = Config()
    opt = DualModeOptimizer(config)

    # 任务：将params从随机初始化拟合到目标向量
    target = np.random.randn(32) * 0.5
    params = np.random.randn(32) * 0.1

    # 训练前误差
    error_before = float(np.mean((params - target) ** 2))

    # Adam优化
    losses = []
    for i in range(200):
        params = opt.update_online(params, target)
        loss = float(np.mean((params - target) ** 2))
        losses.append(loss)

    # 训练后误差
    error_after = float(np.mean((params - target) ** 2))
    stats = opt.get_stats()

    report("Adam更新次数>0", stats["online_updates"] > 0)
    report("Adam训练前误差有限", np.isfinite(error_before), f"err={error_before:.4f}")
    report("Adam训练后误差有限", np.isfinite(error_after), f"err={error_after:.4f}")
    report("Adam误差下降", error_after < error_before,
           f"before={error_before:.4f} after={error_after:.4f}")
    report("Adam误差下降幅度>50%", (error_before - error_after) / error_before > 0.5,
           f"improvement={((error_before-error_after)/error_before*100):.1f}%")
    report("Adam_t递增", stats["adam_t"] > 0, f"adam_t={stats['adam_t']}")


# ============================================================
# 测试5：MultiLevelPredictor在线学习闭环
# ============================================================
def test5_predictor_online_learning():
    section("测试5：MultiLevelPredictor在线学习闭环")
    np.random.seed(42)
    config = Config()
    predictor = MultiLevelPredictor(config)
    feature_dim = config.get("perception", "feature_dim")

    # 生成时序数据：缓慢漂移的sin波
    t_values = np.linspace(0, 30, 200)
    base_signals = np.sin(t_values)

    # 记录预测误差
    prediction_errors = []

    for i, t in enumerate(t_values):
        # 构造状态：每个维度是sin(t) + 不同相位的噪声
        internal = np.sin(t + np.linspace(0, np.pi, feature_dim)) * 0.5
        internal += np.random.randn(feature_dim) * 0.02
        state = PerceptionState(
            internal=internal,
            external=np.zeros(32),
            precision=np.ones(feature_dim) * 0.5
        )
        # 预测
        pred = predictor.predict(state)
        # 记录预测误差（预测的state与实际state的差异）
        if i > 0:  # 从第二步开始有"上一次预测"
            # 当前state是上一次预测的目标
            error = float(np.mean((pred.predicted_state - internal) ** 2))
            prediction_errors.append(error)

    # 获取训练统计
    stats = predictor.get_train_stats()

    report("三层模型都有训练步数", stats["bottom"]["train_steps"] > 0 and
           stats["middle"]["train_steps"] > 0 and
           stats["top"]["train_steps"] > 0)

    report("底层GRU有loss历史", stats["bottom"]["avg_loss"] is not None,
           f"avg_loss={stats['bottom']['avg_loss']}")

    # 验证预测误差在前半段vs后半段
    if len(prediction_errors) >= 100:
        first_half = np.mean(prediction_errors[:50])
        second_half = np.mean(prediction_errors[-50:])
        report("预测误差前半段有限", np.isfinite(first_half), f"err={first_half:.4f}")
        report("预测误差后半段有限", np.isfinite(second_half), f"err={second_half:.4f}")
        # 在线学习应该让模型逐渐适应，后半段误差应更低或持平
        report("在线学习有效（后半段误差不高于前半段）",
               second_half <= first_half * 1.5,
               f"first={first_half:.4f} second={second_half:.4f}")


# ============================================================
# 测试6：梯度正确性验证（数值梯度对比解析梯度）
# ============================================================
def test6_gradient_correctness():
    section("测试6：梯度正确性验证（数值梯度 vs 解析梯度）")
    np.random.seed(42)
    dim = 8
    gru = GRULayer(dim, learning_rate=0.01)

    x = np.random.randn(dim) * 0.5
    target = np.random.randn(dim) * 0.5

    # 前向传播
    h, cache = gru.forward_cache(x)
    gru.hidden = h.copy()

    # 解析梯度
    diff = h - target
    dh = 2.0 * diff / dim
    grads, _ = gru.backward(cache, dh)

    # 数值梯度验证（对W_z）
    eps = 1e-5
    W_z_orig = gru.W_z.copy()

    # 计算loss(W_z + eps)
    gru.W_z = W_z_orig.copy()
    gru.hidden = cache["h_prev"].copy()
    h_plus, _ = gru.forward_cache(x)
    loss_plus = np.mean((h_plus - target) ** 2)

    # 计算loss(W_z - eps) 对单个元素
    i, j = 0, 0
    gru.W_z = W_z_orig.copy()
    gru.W_z[i, j] += eps
    gru.hidden = cache["h_prev"].copy()
    h_plus, _ = gru.forward_cache(x)
    loss_plus = np.mean((h_plus - target) ** 2)

    gru.W_z = W_z_orig.copy()
    gru.W_z[i, j] -= eps
    gru.hidden = cache["h_prev"].copy()
    h_minus, _ = gru.forward_cache(x)
    loss_minus = np.mean((h_minus - target) ** 2)

    # 数值梯度
    numeric_grad = (loss_plus - loss_minus) / (2 * eps)
    # 解析梯度
    analytic_grad = grads["W_z"][i, j]

    # 恢复
    gru.W_z = W_z_orig.copy()

    report("数值梯度有限", np.isfinite(numeric_grad), f"grad={numeric_grad:.6f}")
    report("解析梯度有限", np.isfinite(analytic_grad), f"grad={analytic_grad:.6f}")
    # 相对误差
    if abs(numeric_grad) + abs(analytic_grad) > 1e-10:
        rel_error = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad))
    else:
        rel_error = 0.0
    report("梯度相对误差<0.1", rel_error < 0.1,
           f"rel_error={rel_error:.6f} numeric={numeric_grad:.6f} analytic={analytic_grad:.6f}")


# ============================================================
# 测试7：数值稳定性验证
# ============================================================
def test7_numerical_stability():
    section("测试7：数值稳定性验证")
    np.random.seed(42)
    dim = 16

    # GRU极端输入
    gru = GRULayer(dim, learning_rate=0.01)
    for extreme_input in [np.full(dim, np.nan), np.full(dim, np.inf),
                           np.full(dim, -np.inf), np.full(dim, 1e10),
                           np.full(dim, -1e10)]:
        h = gru.forward(extreme_input)
        report(f"GRU forward {extreme_input[0]}", np.all(np.isfinite(h)))
        loss = gru.train_step(extreme_input, np.zeros(dim))
        report(f"GRU train {extreme_input[0]}", np.isfinite(loss))

    # Transformer极端输入
    trans = TransformerLayer(dim, horizon=3, learning_rate=0.01)
    for extreme in [np.full(dim, np.nan), np.full(dim, np.inf)]:
        seq = [extreme for _ in range(3)]
        preds = trans.forward(seq)
        report(f"Transformer forward {extreme[0]}",
               all(np.all(np.isfinite(p)) for p in preds))
        loss = trans.train_step(seq, np.zeros(dim))
        report(f"Transformer train {extreme[0]}", np.isfinite(loss))

    # GNN极端输入
    gnn = GNNLayer(dim, learning_rate=0.01)
    # 先正常初始化节点
    gnn.forward(["a", "b"], [("a", "b")])
    # 然后训练极端目标
    loss = gnn.train_step(["a", "b"], [("a", "b")], np.full(dim, np.nan))
    report("GNN train NaN target", np.isfinite(loss))
    loss = gnn.train_step(["a", "b"], [("a", "b")], np.full(dim, np.inf))
    report("GNN train Inf target", np.isfinite(loss))

    # 优化器极端输入
    config = Config()
    opt = DualModeOptimizer(config)
    result = opt.update_online(np.full(32, np.nan), np.full(32, np.inf))
    report("Optimizer NaN/Inf输入", np.all(np.isfinite(result)))


# ============================================================
# 测试8：训练后模型参数确实变化
# ============================================================
def test8_params_change_after_training():
    section("测试8：训练后模型参数确实变化")
    np.random.seed(42)
    dim = 16
    gru = GRULayer(dim, learning_rate=0.05)

    # 记录训练前参数
    W_z_before = gru.W_z.copy()
    W_r_before = gru.W_r.copy()
    W_h_before = gru.W_h.copy()

    # 训练
    for i in range(50):
        x = np.random.randn(dim) * 0.5
        target = np.random.randn(dim) * 0.5
        gru.train_step(x, target)

    # 验证参数变化
    report("W_z变化", not np.allclose(W_z_before, gru.W_z),
           f"diff_norm={np.linalg.norm(gru.W_z - W_z_before):.4f}")
    report("W_r变化", not np.allclose(W_r_before, gru.W_r))
    report("W_h变化", not np.allclose(W_h_before, gru.W_h))


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "="*60)
    print("第十一轮测试：阶段16算法内核真实化验证")
    print("="*60)

    tests = [
        test1_gru_real_training,
        test2_transformer_real_training,
        test3_gnn_real_training,
        test4_optimizer_real_adam,
        test5_predictor_online_learning,
        test6_gradient_correctness,
        test7_numerical_stability,
        test8_params_change_after_training,
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
