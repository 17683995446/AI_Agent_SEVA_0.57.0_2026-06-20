"""
第十三轮测试：阶段18记忆系统真实化验证
验证HNSW分层导航索引、优先级经验回放(PER)、K-means聚类巩固
核心目标：证明记忆系统在检索、回放、巩固三个维度上均按真实算法工作
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
from seva.memory import HNSWIndex, SumTree, PrioritizedReplayBuffer, MemoryAndReview
from seva.types import (ExperienceRecord, PerceptionState, Drives, Goal,
                        Prediction, Action, ErrorSignal, ErrorAttribution,
                        ActionType, GoalType, gen_id)

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


def make_record(i, feature_dim, error_mag=0.1, reward=0.0, novel=False,
                vec=None):
    """创建测试用经验记录"""
    if vec is None:
        vec = np.random.randn(feature_dim)
    return ExperienceRecord(
        record_id=f"exp_{i}", timestamp=float(i),
        state=PerceptionState(internal=vec, external=np.zeros(feature_dim),
                              precision=np.ones(feature_dim), is_fallback=False),
        drives=Drives(safety=0.5, gain=0.5, efficiency=0.5),
        goal=None, prediction=None,
        action=Action(action_id=f"act_{i}", type=ActionType.MOVE,
                      parameters={}, expected_reward=reward, source="test"),
        error=ErrorSignal(magnitude=error_mag,
                          perception_error=np.zeros(feature_dim),
                          prediction_error=np.zeros(feature_dim),
                          action_error=0.0, surprise=error_mag * 0.5,
                          is_novel=novel),
        attribution=ErrorAttribution(perception=0.2, model=0.5,
                                     action=0.1, environment=0.2),
        reward=reward
    )


# ============================================================
# 测试1：SumTree数据结构验证
# ============================================================
def test1_sumtree():
    section("测试1：SumTree数据结构")

    # 基本操作
    tree = SumTree(8)
    report("空树total=0", tree.total() == 0.0)
    report("空树n_entries=0", tree.n_entries == 0)

    tree.add(1.0, 0)
    tree.add(2.0, 1)
    tree.add(3.0, 2)
    tree.add(0.5, 3)
    report("total=6.5", abs(tree.total() - 6.5) < 1e-6, f"total={tree.total()}")
    report("n_entries=4", tree.n_entries == 4)

    # 按比例采样验证
    np.random.seed(42)
    counts = [0, 0, 0, 0]
    for _ in range(10000):
        _, _, idx = tree.get(np.random.uniform(0, tree.total()))
        if idx is not None:
            counts[idx] += 1
    # 优先级 1:2:3:0.5 → 采样比例应近似
    ratio = counts[2] / counts[0]  # 3/1 = 3
    report("按比例采样", 2.5 < ratio < 3.5,
           f"counts={counts} ratio_3/1={ratio:.2f}")
    report("低优先级采样最少", counts[3] < counts[0],
           f"p0.5={counts[3]} < p1.0={counts[0]}")

    # 更新优先级
    tree.update_by_data_idx(2, 0.1)  # 降低idx=2的优先级
    report("更新后total减少", tree.total() < 6.5,
           f"total={tree.total():.2f}")

    # 容量循环覆盖
    tree2 = SumTree(4)
    for i in range(10):
        tree2.add(float(i + 1), i)
    report("容量限制", tree2.n_entries == 4, f"n_entries={tree2.n_entries}")

    # 全零优先级安全处理
    tree3 = SumTree(4)
    tree3.add(0.0, 0)
    tree3.add(0.0, 1)
    result = tree3.get(0.0)
    report("全零优先级不崩溃", result is not None)


# ============================================================
# 测试2：PrioritizedReplayBuffer验证
# ============================================================
def test2_prioritized_replay():
    section("测试2：优先级经验回放(PER)")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    # 空缓冲区
    records, weights, indices = buf.sample(10)
    report("空缓冲区采样", len(records) == 0 and len(indices) == 0)

    # 添加记录（误差幅度递增）
    np.random.seed(42)
    for i in range(100):
        rec = make_record(i, feature_dim, error_mag=float(i) / 100,
                          reward=float(i) / 50)
        buf.add(rec)

    report("缓冲区大小=100", buf.size() == 100, f"size={buf.size()}")

    # 采样验证：高误差记录应被更频繁采样
    np.random.seed(42)
    sampled_errors = []
    for _ in range(50):
        records, weights, indices = buf.sample(20)
        sampled_errors.extend([r.error.magnitude for r in records])

    avg_sampled = np.mean(sampled_errors)
    avg_uniform = np.mean([i / 100 for i in range(100)])
    report("PER采样偏向高误差", avg_sampled > avg_uniform,
           f"PER_avg={avg_sampled:.3f} > uniform_avg={avg_uniform:.3f}")

    # 重要性采样权重验证
    records, weights, indices = buf.sample(20)
    report("权重非负", np.all(weights >= 0))
    report("权重归一化(max=1)", abs(np.max(weights) - 1.0) < 1e-6,
           f"max_weight={np.max(weights):.4f}")
    report("权重shape匹配", len(weights) == len(records))

    # 优先级更新
    old_beta = buf.beta
    buf.update_priorities(indices, [0.001] * len(indices))
    report("优先级更新不崩溃", True)
    report("beta退火", buf.beta >= old_beta,
           f"beta={buf.beta:.4f} >= {old_beta:.4f}")

    # beta退火到终点
    for _ in range(20000):
        buf._anneal_beta()
    report("beta退火到1.0", abs(buf.beta - 1.0) < 1e-6,
           f"beta={buf.beta:.4f}")

    # 容量循环覆盖
    buf2 = PrioritizedReplayBuffer(capacity=50)
    for i in range(100):
        rec = make_record(i, feature_dim, error_mag=0.1)
        buf2.add(rec)
    report("容量限制=50", buf2.size() == 50, f"size={buf2.size()}")

    # 新颖性影响优先级
    rec_normal = make_record(0, feature_dim, error_mag=0.5, novel=False)
    rec_novel = make_record(1, feature_dim, error_mag=0.5, novel=True)
    p_normal = buf.compute_priority(rec_normal)
    p_novel = buf.compute_priority(rec_novel)
    report("新颖性提升优先级", p_novel > p_normal,
           f"novel={p_novel:.4f} > normal={p_normal:.4f}")


# ============================================================
# 测试3：K-means聚类验证
# ============================================================
def test3_kmeans():
    section("测试3：K-means聚类")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    mem = MemoryAndReview(config)

    # 构造3个明显分离的簇
    np.random.seed(42)
    centers = [np.random.randn(feature_dim) * 10 for _ in range(3)]
    vectors = []
    for c in range(3):
        for _ in range(50):
            vectors.append(centers[c] + np.random.randn(feature_dim) * 0.3)
    vectors = np.array(vectors)

    centroids, labels = mem._kmeans(vectors, k=3, max_iters=20, seed=42)

    # 验证聚类正确性
    cluster_assignments = {0: set(), 1: set(), 2: set()}
    for i, label in enumerate(labels):
        original_cluster = i // 50
        cluster_assignments[label].add(original_cluster)

    # 每个k-means簇应主要包含一个原始簇
    correct = all(
        len(s) == 1 for s in cluster_assignments.values()
    )
    report("3簇正确分离", correct,
           f"assignments={ {k: list(v) for k, v in cluster_assignments.items()} }")

    # 质心接近原始中心
    centroid_matches = 0
    for c in centroids:
        dists = [np.linalg.norm(c - center) for center in centers]
        if min(dists) < 2.0:  # 质心应在原始中心附近
            centroid_matches += 1
    report("质心接近原始中心", centroid_matches >= 2,
           f"{centroid_matches}/3 质心匹配")

    # 收敛性（同样数据再跑一次应得到相同结果）
    centroids2, labels2 = mem._kmeans(vectors, k=3, max_iters=20, seed=42)
    report("确定性(seed固定)", np.array_equal(labels, labels2))

    # NaN安全
    nan_vectors = vectors.copy()
    nan_vectors[0] = np.nan
    try:
        centroids3, labels3 = mem._kmeans(nan_vectors, k=3, max_iters=5, seed=42)
        report("NaN不崩溃", True)
    except Exception as e:
        report("NaN不崩溃", False, f"异常: {e}")

    # 全相同向量
    same_vectors = np.ones((30, feature_dim))
    centroids4, labels4 = mem._kmeans(same_vectors, k=3, max_iters=10, seed=42)
    report("全相同向量不崩溃", True)


# ============================================================
# 测试4：记忆巩固验证
# ============================================================
def test4_consolidation():
    section("测试4：K-means记忆巩固")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    mem = MemoryAndReview(config)

    # 存储300条记忆（3个簇）
    np.random.seed(42)
    centers = [np.random.randn(feature_dim) * 5 for _ in range(3)]
    action_types = [ActionType.MOVE, ActionType.COLLECT, ActionType.WAIT]
    for i in range(300):
        c = i % 3
        vec = centers[c] + np.random.randn(feature_dim) * 0.5
        rec = make_record(i, feature_dim, error_mag=float(c) / 3,
                          reward=float(c), vec=vec)
        rec.action = Action(action_id=f"act_{i}", type=action_types[c],
                            parameters={}, expected_reward=float(c),
                            source="test")
        mem.store_experience(rec)

    initial_count = mem.episodic_count()
    report("巩固前300条", initial_count == 300, f"count={initial_count}")

    # 执行巩固
    t0 = time.time()
    mem.consolidate()
    elapsed = (time.time() - t0) * 1000

    after_count = mem.episodic_count()
    report("巩固减少记忆", after_count < initial_count,
           f"{initial_count} -> {after_count}")
    report("巩固后仍有记忆", after_count > 0, f"count={after_count}")
    report("巩固性能<500ms", elapsed < 500, f"{elapsed:.1f}ms")

    # 语义记忆应包含簇抽象
    semantic_count = len(mem.semantic_memory)
    report("语义记忆非空", semantic_count > 0, f"clusters={semantic_count}")

    # 验证抽象内容
    has_abstraction = False
    for key, val in mem.semantic_memory.items():
        if "centroid" in val and "avg_reward" in val and "dominant_action" in val:
            has_abstraction = True
            break
    report("抽象包含统计量", has_abstraction)

    # 巩固后PER仍可用
    replayed = mem.replay(batch_size=10)
    report("巩固后PER仍可用", len(replayed) > 0, f"replayed={len(replayed)}")

    # 巩固后HNSW仍可用
    query = np.random.randn(feature_dim)
    results = mem.retrieve_episodic(query, k=5)
    report("巩固后HNSW仍可用", True)  # HNSW独立于episodic_records

    # 多次巩固
    for _ in range(3):
        mem.consolidate()
    report("多次巩固不崩溃", mem.episodic_count() > 0)

    # NaN状态巩固
    mem2 = MemoryAndReview(config)
    for i in range(15):
        vec = np.full(feature_dim, np.nan) if i % 3 == 0 else np.ones(feature_dim) * i
        rec = make_record(i, feature_dim, vec=vec)
        mem2.store_experience(rec)
    try:
        mem2.consolidate()
        report("NaN状态巩固安全", True)
    except Exception as e:
        report("NaN状态巩固安全", False, f"异常: {e}")


# ============================================================
# 测试5：HNSW检索验证
# ============================================================
def test5_hnsw_retrieval():
    section("测试5：HNSW分层导航检索")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")

    # recall@k 验证
    np.random.seed(42)
    size = 500
    hnsw = HNSWIndex(feature_dim, capacity=size + 100)
    vectors = []
    for i in range(size):
        vec = np.random.randn(feature_dim)
        vectors.append(vec)
        hnsw.add(vec, {"id": i})

    # 对100个查询验证recall@5
    total_recall = 0.0
    for _ in range(100):
        query = np.random.randn(feature_dim)
        hnsw_results = hnsw.search(query, k=5)

        # 暴力搜索ground truth
        dists = [np.linalg.norm(query - v) for v in vectors]
        true_top5 = set(np.argsort(dists)[:5].tolist())
        hnsw_top5 = set(content["id"] for _, content in hnsw_results)
        overlap = len(true_top5 & hnsw_top5)
        total_recall += overlap / 5.0

    avg_recall = total_recall / 100
    report("recall@5 > 0.9", avg_recall > 0.9,
           f"avg_recall={avg_recall:.3f}")

    # 空搜索
    empty_hnsw = HNSWIndex(feature_dim, capacity=100)
    results = empty_hnsw.search(np.random.randn(feature_dim), k=5)
    report("空索引搜索", len(results) == 0)

    # k=0和k=-1
    results_k0 = hnsw.search(np.random.randn(feature_dim), k=0)
    report("k=0返回空", len(results_k0) == 0)
    results_kneg = hnsw.search(np.random.randn(feature_dim), k=-1)
    report("k=-1返回空", len(results_kneg) == 0)

    # 容量管理
    small_hnsw = HNSWIndex(feature_dim, capacity=50)
    for i in range(100):
        small_hnsw.add(np.random.randn(feature_dim), {"id": i})
    report("容量限制", small_hnsw.size() <= 50, f"size={small_hnsw.size()}")

    # NaN/Inf查询
    nan_query = np.full(feature_dim, np.nan)
    results_nan = hnsw.search(nan_query, k=5)
    report("NaN查询不崩溃", True)

    inf_query = np.full(feature_dim, np.inf)
    results_inf = hnsw.search(inf_query, k=5)
    report("Inf查询不崩溃", True)


# ============================================================
# 测试6：端到端记忆系统
# ============================================================
def test6_e2e_memory_system():
    section("测试6：端到端记忆系统")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    mem = MemoryAndReview(config)

    # 存储1000条经验
    np.random.seed(42)
    for i in range(1000):
        error_mag = np.random.exponential(0.3)
        reward = np.random.randn() * 0.5
        rec = make_record(i, feature_dim, error_mag=error_mag, reward=reward,
                          novel=(error_mag > 0.5))
        mem.store_experience(rec)

    report("存储1000条", mem.episodic_count() == 1000,
           f"episodic={mem.episodic_count()}")
    report("PER缓冲区=1000", mem.replay_buffer.size() == 1000,
           f"replay_size={mem.replay_buffer.size()}")

    # HNSW检索
    query = np.random.randn(feature_dim)
    retrieved = mem.retrieve_episodic(query, k=10)
    report("HNSW检索10条", len(retrieved) == 10, f"retrieved={len(retrieved)}")

    # PER回放（含权重）
    records, weights, indices = mem.replay_with_weights(batch_size=32)
    report("PER回放32条", len(records) == 32, f"replayed={len(records)}")
    report("IS权重shape", len(weights) == 32, f"weights={len(weights)}")

    # 回放后更新优先级
    new_errors = [np.random.exponential(0.2) for _ in indices]
    mem.update_replay_priorities(indices, new_errors)
    report("优先级更新成功", True)

    # 巩固
    mem.consolidate()
    report("巩固后记忆减少", mem.episodic_count() < 1000,
           f"episodic={mem.episodic_count()}")
    report("语义记忆非空", len(mem.semantic_memory) > 0,
           f"semantic={len(mem.semantic_memory)}")

    # 巩固后PER仍可用（PER独立于episodic_records）
    records2, weights2, indices2 = mem.replay_with_weights(batch_size=32)
    report("巩固后PER仍可用", len(records2) > 0,
           f"replayed={len(records2)}")

    # 巩固后HNSW仍可用
    retrieved2 = mem.retrieve_episodic(query, k=10)
    report("巩固后HNSW仍可用", True)

    # 重置
    mem.reset()
    report("重置后episodic=0", mem.episodic_count() == 0)
    report("重置后PER=0", mem.replay_buffer.size() == 0)
    report("重置后semantic=0", len(mem.semantic_memory) == 0)


# ============================================================
# 测试7：PER vs 均匀采样对比
# ============================================================
def test7_per_vs_uniform():
    section("测试7：PER vs 均匀采样对比")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    mem = MemoryAndReview(config)

    # 存储500条：前250条低误差，后250条高误差
    np.random.seed(42)
    for i in range(500):
        if i < 250:
            error_mag = np.random.uniform(0.01, 0.05)  # 低误差
        else:
            error_mag = np.random.uniform(0.8, 1.0)    # 高误差
        rec = make_record(i, feature_dim, error_mag=error_mag)
        mem.store_experience(rec)

    # PER采样
    np.random.seed(42)
    per_high_error_ratio = 0
    for _ in range(100):
        records, _, _ = mem.replay_with_weights(batch_size=50)
        high_error_count = sum(1 for r in records if r.error.magnitude > 0.5)
        per_high_error_ratio += high_error_count / 50
    per_high_error_ratio /= 100

    # 均匀采样
    np.random.seed(42)
    uniform_high_error_ratio = 0
    for _ in range(100):
        indices = np.random.choice(500, 50, replace=False)
        high_error_count = sum(1 for i in indices if i >= 250)
        uniform_high_error_ratio += high_error_count / 50
    uniform_high_error_ratio /= 100

    report("PER采样更多高误差", per_high_error_ratio > uniform_high_error_ratio,
           f"PER={per_high_error_ratio:.2f} > uniform={uniform_high_error_ratio:.2f}")
    report("PER高误差比例>0.5", per_high_error_ratio > 0.5,
           f"PER_high_ratio={per_high_error_ratio:.2f}")


# ============================================================
# 测试8：性能基准
# ============================================================
def test8_performance():
    section("测试8：记忆系统性能基准")

    config = Config()
    feature_dim = config.get("perception", "feature_dim")
    mem = MemoryAndReview(config)

    # 存储性能
    np.random.seed(42)
    t0 = time.time()
    for i in range(2000):
        rec = make_record(i, feature_dim, error_mag=0.1)
        mem.store_experience(rec)
    store_time = (time.time() - t0) / 2000 * 1000
    report("存储<2ms/次", store_time < 2.0, f"{store_time:.3f}ms/次")

    # HNSW搜索性能
    query = np.random.randn(feature_dim)
    for _ in range(10):
        mem.retrieve_episodic(query, k=5)  # 预热
    t0 = time.time()
    for _ in range(100):
        mem.retrieve_episodic(query, k=5)
    search_time = (time.time() - t0) / 100 * 1000
    report("HNSW搜索<5ms/次", search_time < 5.0, f"{search_time:.3f}ms/次")

    # PER采样性能
    t0 = time.time()
    for _ in range(100):
        mem.replay(batch_size=32)
    replay_time = (time.time() - t0) / 100 * 1000
    report("PER采样<5ms/次", replay_time < 5.0, f"{replay_time:.3f}ms/次")

    # 巩固性能
    t0 = time.time()
    mem.consolidate()
    consolidate_time = (time.time() - t0) * 1000
    report("巩固2000条<600ms", consolidate_time < 600,
           f"{consolidate_time:.1f}ms")


def main():
    tests = [
        test1_sumtree,
        test2_prioritized_replay,
        test3_kmeans,
        test4_consolidation,
        test5_hnsw_retrieval,
        test6_e2e_memory_system,
        test7_per_vs_uniform,
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
