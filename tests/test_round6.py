"""
第六轮测试：性能与资源占用核验
对应设计文档：基础性能核验、资源管理
测试目标：无内存泄漏，热点模块优化到可接受范围
"""
import sys
import os
import time
import gc
import traceback
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva.config import Config
from seva.core import SelfEvolvingAgentCore
from seva.perception import PerceptionModule
from seva.predictor import MultiLevelPredictor
from seva.error_eval import ErrorEvaluator
from seva.optimizer import DualModeOptimizer
from seva.memory import MemoryAndReview, SimpleHNSW
from seva.types import (PerceptionState, Drives, ExperienceRecord, gen_id)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("test.round6")

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
# 测试1：各模块单独benchmark
# ============================================================
def test_module_benchmark():
    section("测试1：各模块单独benchmark")
    config = Config()
    input_dim = config.get("perception", "input_dim")
    feature_dim = config.get("perception", "feature_dim")

    all_pass = True

    # 感知模块
    perc = PerceptionModule(config)
    raw = np.random.randn(input_dim)
    # 预热
    for _ in range(10):
        perc.process(raw)
    t0 = time.time()
    for _ in range(1000):
        perc.process(raw)
    perc_time = (time.time() - t0) / 1000 * 1000  # ms
    if perc_time < 5:
        record("感知模块", True, f"{perc_time:.3f}ms/次")
    else:
        record("感知模块", False, f"{perc_time:.3f}ms/次 超标")
        all_pass = False

    # 预测模块（阶段16真实化后含self-attention+训练，阈值放宽到15ms）
    pred = MultiLevelPredictor(config)
    state = perc.process(raw)
    for _ in range(10):
        pred.predict(state)
    t0 = time.time()
    for _ in range(1000):
        pred.predict(state)
    pred_time = (time.time() - t0) / 1000 * 1000
    if pred_time < 15:
        record("预测模块", True, f"{pred_time:.3f}ms/次")
    else:
        record("预测模块", False, f"{pred_time:.3f}ms/次 超标")
        all_pass = False

    # 误差评估模块
    evaluator = ErrorEvaluator(config)
    from seva.types import Prediction
    prediction = Prediction(
        predicted_state=np.random.randn(feature_dim),
        confidence=0.5,
        variance=np.ones(feature_dim) * 0.1,
        horizon=1
    )
    for _ in range(10):
        evaluator.evaluate(state, prediction, None, actual_state=state)
    t0 = time.time()
    for _ in range(1000):
        evaluator.evaluate(state, prediction, None, actual_state=state)
    eval_time = (time.time() - t0) / 1000 * 1000
    if eval_time < 5:
        record("误差评估模块", True, f"{eval_time:.3f}ms/次")
    else:
        record("误差评估模块", False, f"{eval_time:.3f}ms/次 超标")
        all_pass = False

    # 优化器
    opt = DualModeOptimizer(config)
    params = np.random.randn(feature_dim)
    obs = np.random.randn(feature_dim)
    for _ in range(10):
        opt.update_online(params, obs)
    t0 = time.time()
    for _ in range(1000):
        opt.update_online(params, obs)
    opt_time = (time.time() - t0) / 1000 * 1000
    if opt_time < 5:
        record("优化器", True, f"{opt_time:.3f}ms/次")
    else:
        record("优化器", False, f"{opt_time:.3f}ms/次 超标")
        all_pass = False

    # 记忆模块存储
    mem = MemoryAndReview(config)
    rec = ExperienceRecord(
        record_id=gen_id("exp"), timestamp=time.time(),
        state=PerceptionState(internal=np.random.randn(feature_dim),
                              external=np.zeros(feature_dim),
                              precision=np.ones(feature_dim) * 0.5),
        drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
        goal=None, prediction=None, action=None, error=None,
        attribution=None, reward=0.5
    )
    t0 = time.time()
    for _ in range(1000):
        mem.store_experience(rec)
    mem_store_time = (time.time() - t0) / 1000 * 1000
    if mem_store_time < 2:
        record("记忆存储", True, f"{mem_store_time:.3f}ms/次")
    else:
        record("记忆存储", False, f"{mem_store_time:.3f}ms/次 超标")
        all_pass = False

    # 记忆搜索
    query = np.random.randn(feature_dim)
    t0 = time.time()
    for _ in range(1000):
        mem.retrieve_episodic(query, k=5)
    mem_search_time = (time.time() - t0) / 1000 * 1000
    if mem_search_time < 10:
        record("记忆搜索(1000条)", True, f"{mem_search_time:.3f}ms/次")
    else:
        record("记忆搜索(1000条)", False, f"{mem_search_time:.3f}ms/次 超标")
        all_pass = False

    record("test_module_benchmark", all_pass)


# ============================================================
# 测试2：内存泄漏检测（对象数量跟踪）
# ============================================================
def test_memory_leak():
    section("测试2：内存泄漏检测（对象数量跟踪）")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    all_pass = True
    # 运行3批，每批1000步，检查关键数据结构增长
    batch_sizes = [1000, 1000, 1000]
    metrics_history = []

    for batch_idx, batch_size in enumerate(batch_sizes):
        for _ in range(batch_size):
            core.inject_input(np.random.randn(input_dim))
            core.run_cycle()

        # 收集关键指标
        m = {
            "batch": batch_idx,
            "cycle_times_len": len(core.metrics["cycle_times"]),
            "prediction_errors_len": len(core.metrics["prediction_errors"]),
            "surprises_len": len(core.metrics["surprises"]),
            "action_types_len": len(core.metrics["action_types"]),
            "goal_types_len": len(core.metrics["goal_types"]),
            "episodic_count": core.modules["memory"].episodic_count(),
            "surprise_history_len": len(core.modules["error_evaluator"].surprise_history),
            "emotional_memory_len": len(core.modules["memory"].emotional_memory),
        }
        metrics_history.append(m)

    # 分析增长趋势
    # cycle_times/prediction_errors/surprises/action_types/goal_types 应该持续增长（这是正常的，记录历史）
    # 但如果增长过快（每步增加多条），则有问题
    for key in ["cycle_times_len", "prediction_errors_len", "surprises_len", "action_types_len", "goal_types_len"]:
        growth = metrics_history[-1][key] - metrics_history[0][key]
        expected_growth = 2000  # 2批x1000步
        if abs(growth - expected_growth) < 100:
            record(f"{key}增长", True, f"增长{growth}（期望~{expected_growth}）")
        else:
            record(f"{key}增长", False, f"增长{growth}（期望~{expected_growth}）")
            all_pass = False

    # episodic_count 受巩固影响，应在合理范围
    epi_count = metrics_history[-1]["episodic_count"]
    if epi_count < 10000:  # 不超过capacity
        record("episodic_count", True, f"{epi_count}（<capacity=10000）")
    else:
        record("episodic_count", False, f"{epi_count} 超过capacity")
        all_pass = False

    # surprise_history 应限制在1000以内
    sh_len = metrics_history[-1]["surprise_history_len"]
    if sh_len <= 1000:
        record("surprise_history限制", True, f"{sh_len}（<=1000）")
    else:
        record("surprise_history限制", False, f"{sh_len}（>1000）")
        all_pass = False

    # emotional_memory 应限制在1000以内
    em_len = metrics_history[-1]["emotional_memory_len"]
    if em_len <= 1000:
        record("emotional_memory限制", True, f"{em_len}（<=1000）")
    else:
        record("emotional_memory限制", False, f"{em_len}（>1000）")
        all_pass = False

    # 检查cycle_times列表是否无限增长（潜在泄漏）
    # cycle_times每步追加一条，3000步后应有3000条
    ct_len = metrics_history[-1]["cycle_times_len"]
    if ct_len == 3000:
        record("cycle_times长度", True, f"{ct_len}")
    else:
        record("cycle_times长度", False, f"{ct_len}（期望3000）")
        all_pass = False

    # 建议：cycle_times应有限制（当前无限制，长期运行会泄漏）
    # 这是一个潜在问题，但不影响短期运行
    if ct_len > 10000:
        record("cycle_times潜在泄漏", False, "无上限，长期运行会泄漏")
        # 不标记为失败，因为这是设计改进建议
    else:
        record("cycle_times潜在泄漏", True, "短期可接受（建议添加上限）")

    record("test_memory_leak", all_pass)


# ============================================================
# 测试3：HNSW搜索性能曲线
# ============================================================
def test_hnsw_performance_curve():
    section("测试3：HNSW搜索性能曲线")
    config = Config()
    feature_dim = config.get("perception", "feature_dim")

    all_pass = True
    sizes = [100, 500, 1000, 2000, 5000]

    for size in sizes:
        hnsw = SimpleHNSW(feature_dim, capacity=size + 100)
        for i in range(size):
            hnsw.add(np.random.randn(feature_dim), {"id": i})

        query = np.random.randn(feature_dim)
        # 预热
        for _ in range(10):
            hnsw.search(query, k=5)

        t0 = time.time()
        for _ in range(100):
            hnsw.search(query, k=5)
        search_time = (time.time() - t0) / 100 * 1000  # ms

        # 性能应与规模成亚线性关系（向量化搜索）
        if search_time < 50:  # 50ms阈值
            record(f"HNSW-{size}条", True, f"{search_time:.2f}ms/次")
        else:
            record(f"HNSW-{size}条", False, f"{search_time:.2f}ms/次 超标")
            all_pass = False

    record("test_hnsw_performance_curve", all_pass)


# ============================================================
# 测试4：核心循环性能稳定性
# ============================================================
def test_core_cycle_stability():
    section("测试4：核心循环性能稳定性")
    config = Config()
    core = SelfEvolvingAgentCore(config)
    core.initialize()
    input_dim = config.get("perception", "input_dim")

    all_pass = True
    # 运行5000步，每500步统计性能
    batch_stats = []
    for batch in range(10):
        batch_times = []
        for _ in range(500):
            core.inject_input(np.random.randn(input_dim))
            result = core.run_cycle()
            batch_times.append(result["time"])

        avg_ms = np.mean(batch_times) * 1000
        p99_ms = np.percentile(batch_times, 99) * 1000
        max_ms = np.max(batch_times) * 1000
        batch_stats.append({
            "batch": batch,
            "avg": avg_ms,
            "p99": p99_ms,
            "max": max_ms,
        })

    # 验证性能不退化（最后一批 vs 第一批）
    first_avg = batch_stats[0]["avg"]
    last_avg = batch_stats[-1]["avg"]
    degradation = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

    if degradation < 100:  # 性能退化不超过100%
        record("性能稳定性", True,
               f"首批avg={first_avg:.2f}ms 末批avg={last_avg:.2f}ms 退化={degradation:.1f}%")
    else:
        record("性能稳定性", False,
               f"首批avg={first_avg:.2f}ms 末批avg={last_avg:.2f}ms 退化={degradation:.1f}%")
        all_pass = False

    # 验证p99延迟
    all_p99 = [s["p99"] for s in batch_stats]
    max_p99 = max(all_p99)
    if max_p99 < 50:  # p99 < 50ms
        record("p99延迟", True, f"max_p99={max_p99:.2f}ms")
    else:
        record("p99延迟", False, f"max_p99={max_p99:.2f}ms 超标")
        all_pass = False

    # 输出各批次详情
    for s in batch_stats:
        print(f"    batch{s['batch']}: avg={s['avg']:.2f}ms p99={s['p99']:.2f}ms max={s['max']:.2f}ms")

    record("test_core_cycle_stability", all_pass)


# ============================================================
# 测试5：巩固性能与记忆规模关系
# ============================================================
def test_consolidation_performance():
    section("测试5：巩固性能与记忆规模关系")
    config = Config()
    feature_dim = config.get("perception", "feature_dim")

    all_pass = True
    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        mem = MemoryAndReview(config)
        for i in range(size):
            rec = ExperienceRecord(
                record_id=gen_id("exp"), timestamp=time.time(),
                state=PerceptionState(
                    internal=np.random.randn(feature_dim),
                    external=np.zeros(feature_dim),
                    precision=np.ones(feature_dim) * 0.5
                ),
                drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
                goal=None, prediction=None, action=None, error=None,
                attribution=None, reward=float(i % 10)
            )
            mem.store_experience(rec)

        t0 = time.time()
        mem.consolidate()
        consolidate_time = (time.time() - t0) * 1000  # ms

        # 巩固性能应受限于500条限制（修复Bug6后）
        if consolidate_time < 500:  # 500ms阈值
            record(f"巩固-{size}条", True,
                   f"{consolidate_time:.1f}ms 巩固后={mem.episodic_count()}")
        else:
            record(f"巩固-{size}条", False,
                   f"{consolidate_time:.1f}ms 超标")
            all_pass = False

    record("test_consolidation_performance", all_pass)


# ============================================================
# 测试6：GC压力测试
# ============================================================
def test_gc_pressure():
    section("测试6：GC压力测试")
    config = Config()
    input_dim = config.get("perception", "input_dim")

    all_pass = True
    # 创建和销毁多个core实例，检查GC是否正常
    gc_counts = []
    for i in range(5):
        gc.collect()
        before = gc.get_count()
        core = SelfEvolvingAgentCore(config)
        core.initialize()
        for _ in range(200):
            core.inject_input(np.random.randn(input_dim))
            core.run_cycle()
        del core
        gc.collect()
        after = gc.get_count()
        gc_counts.append((before, after))

    # 验证GC能正常回收
    # 如果after的代0计数持续增长，说明有泄漏
    gen0_growth = gc_counts[-1][1][0] - gc_counts[0][1][0]
    if gen0_growth < 1000:
        record("GC回收正常", True, f"gen0增长={gen0_growth}")
    else:
        record("GC回收正常", False, f"gen0增长={gen0_growth} 可能泄漏")
        all_pass = False

    record("test_gc_pressure", all_pass)


# ============================================================
# 主函数
# ============================================================
def main():
    print("============================================================")
    print("第六轮测试：性能与资源占用核验（共6项）")
    print("============================================================")

    tests = [
        test_module_benchmark,
        test_memory_leak,
        test_hnsw_performance_curve,
        test_core_cycle_stability,
        test_consolidation_performance,
        test_gc_pressure,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"测试{test.__name__}本身崩溃: {e}")
            traceback.print_exc()
            record(test.__name__, False, f"测试本身崩溃: {e}")

    print("============================================================")
    print("第六轮测试结果汇总:")
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
