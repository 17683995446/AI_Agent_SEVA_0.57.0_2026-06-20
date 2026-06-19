"""
记忆与离线复盘模块（MemoryAndReview）
对应设计文档第14章、第30章
五类记忆 + 真实HNSW检索 + 优先级经验回放 + 聚类巩固
"""
import numpy as np
import logging
import heapq
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from .types import ExperienceRecord, gen_id
from .config import Config

logger = logging.getLogger("seva.memory")


class HNSWIndex:
    """
    真实HNSW分层导航小世界索引
    基于 Malkov & Yashunin (2016) "Efficient and robust approximate nearest
    neighbor search using Hierarchical Navigable Small World graphs"

    核心特性：
    - 多层图结构：上层稀疏（快速定位），下层稠密（精确搜索）
    - 贪心搜索（上层）+ 束搜索（底层）
    - 邻居选择启发式（Algorithm 4）：促进多样性
    - 软删除 + 周期性重建：支持容量管理

    复杂度：O(log N) 搜索，对比暴力搜索 O(N)
    """

    def __init__(self, dim: int, capacity: int = 10000,
                 M: int = 16, ef_construction: int = 32,
                 seed: int = 42):
        self.dim = dim
        self.capacity = capacity
        self.M = M
        self.M_max = M          # layer > 0 的最大连接数
        self.M_max0 = 2 * M     # layer 0 的最大连接数
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(max(M, 2))
        self.rng = np.random.RandomState(seed)

        # 节点存储
        self.vectors: List[np.ndarray] = []
        self.contents: List[Any] = []
        self.levels: List[int] = []
        self.graph: List[List[List[int]]] = []  # graph[node][layer] = [neighbor_ids]
        self.deleted: List[bool] = []

        # 索引状态
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self._size: int = 0
        self._next_id: int = 0
        self._deleted_count: int = 0
        self._eviction_ptr: int = 0

        logger.info(f"[HNSWIndex] 初始化 dim={dim} capacity={capacity} "
                    f"M={M} ef_construction={ef_construction}")

    def _random_level(self) -> int:
        """随机生成层级（指数衰减分布：P(L) ∝ exp(-L/ml)）"""
        r = self.rng.random_sample()
        return int(-np.log(r + 1e-30) * self.ml)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """向量预处理：NaN/Inf处理 + 裁剪 + 维度对齐"""
        vector = np.asarray(vector, dtype=np.float64)
        vector = np.nan_to_num(vector, nan=0.0, posinf=10.0, neginf=-10.0)
        vector = np.clip(vector, -10.0, 10.0)
        if vector.shape != (self.dim,):
            vector = np.resize(vector, self.dim)
        return vector

    def _dist_sq(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """平方L2距离（避免sqrt，保持单调性）"""
        diff = vec_a - vec_b
        return float(np.dot(diff, diff))

    def _search_layer(self, query: np.ndarray, entry_points: List[int],
                      ef: int, layer: int) -> List[Tuple[float, int]]:
        """
        在指定层进行束搜索（beam search）
        返回 [(distance, node_id), ...] 按距离升序，最多ef个

        算法：维护两个优先队列
        - candidates: min-heap，待探索的节点（按距离升序）
        - results: max-heap，当前top-ef结果（按距离降序，便于淘汰最远的）
        """
        visited = set(entry_points)
        candidates = []  # min-heap (dist, node_id)
        results = []     # max-heap (-dist, node_id)

        for ep in entry_points:
            d = self._dist_sq(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        # 安全上限：防止病态图（如全相同向量）导致O(N)搜索
        max_visits = ef * 20
        visit_count = 0

        while candidates and visit_count < max_visits:
            d_c, c = heapq.heappop(candidates)
            d_f = -results[0][0]  # results中最远的距离
            # 标准终止：最近候选比最远结果更远时停止
            if d_c > d_f and len(results) >= ef:
                break
            # 优化：所有结果已在距离0（全相同向量场景），无需继续
            if d_c == 0.0 and d_f == 0.0 and len(results) >= ef:
                break

            visit_count += 1
            if layer >= len(self.graph[c]):
                continue
            neighbors = self.graph[c][layer]

            # 过滤未访问且未删除的邻居
            unvisited = [e for e in neighbors
                        if e not in visited and not self.deleted[e]]
            if not unvisited:
                continue
            visited.update(unvisited)

            # 批量距离计算（numpy向量化，减少Python循环开销）
            neighbor_vecs = np.array([self.vectors[e] for e in unvisited])
            diffs = neighbor_vecs - query
            dists = np.sum(diffs * diffs, axis=1)

            for e, d_e in zip(unvisited, dists):
                d_e = float(d_e)
                d_f = -results[0][0] if results else float('inf')
                if len(results) < ef or d_e < d_f:
                    heapq.heappush(candidates, (d_e, e))
                    heapq.heappush(results, (-d_e, e))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(-neg_d, nid) for neg_d, nid in results])

    def _select_neighbors(self, candidates: List[Tuple[float, int]],
                          M: int) -> List[int]:
        """
        邻居选择启发式（Algorithm 4）
        优先选择距离近且能改善多样性的候选：
        若候选e到某已选邻居r的距离 < e到query的距离，则跳过e
        （e被r"遮挡"，添加e不增加新方向）

        优化：
        1. 快速路径：所有候选距离相同时（如全相同向量），直接取前M个
        2. 预计算候选间两两距离矩阵，避免O(M²)次_dist_sq调用
        """
        if len(candidates) <= M:
            return [e for _, e in candidates]

        # 快速路径：所有候选距离相同 → 多样性启发式无效，直接取前M个
        first_dist = candidates[0][0]
        if all(abs(d - first_dist) < 1e-12 for d, _ in candidates):
            return [e for _, e in candidates[:M]]

        n = len(candidates)
        cand_ids = [e for _, e in candidates]
        cand_vecs = np.array([self.vectors[e] for e in cand_ids])
        cand_dists = np.array([d for d, _ in candidates])

        # 预计算两两距离矩阵：||v_i - v_j||² = ||v_i||² + ||v_j||² - 2*v_i·v_j
        norms_sq = np.sum(cand_vecs ** 2, axis=1)
        dot_products = cand_vecs @ cand_vecs.T
        pairwise_dists = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dot_products
        np.maximum(pairwise_dists, 0.0, out=pairwise_dists)  # 防止浮点误差导致负值

        result_indices = []
        for i in range(n):
            if len(result_indices) >= M:
                break
            good = True
            for r in result_indices:
                if pairwise_dists[i, r] < cand_dists[i]:
                    good = False
                    break
            if good:
                result_indices.append(i)

        return [cand_ids[i] for i in result_indices]

    def _greedy_search_from(self, query: np.ndarray, entry: int,
                            layer: int) -> Tuple[int, float]:
        """贪心搜索（ef=1）：在指定层从entry开始找最近节点"""
        cur = entry
        cur_d = self._dist_sq(query, self.vectors[cur])
        changed = True
        while changed:
            changed = False
            if layer < len(self.graph[cur]):
                for e in self.graph[cur][layer]:
                    if self.deleted[e]:
                        continue
                    d_e = self._dist_sq(query, self.vectors[e])
                    if d_e < cur_d:
                        cur_d = d_e
                        cur = e
                        changed = True
        return cur, cur_d

    def _evict_oldest(self):
        """驱逐最旧的非删除节点（软删除/墓碑机制）"""
        while self._eviction_ptr < self._next_id:
            idx = self._eviction_ptr
            self._eviction_ptr += 1
            if not self.deleted[idx]:
                self.deleted[idx] = True
                self._size -= 1
                self._deleted_count += 1
                if idx == self.entry_point:
                    self._update_entry_point()
                return

    def _update_entry_point(self):
        """更新入口点为存活的最高层节点"""
        best = None
        best_level = -1
        for i in range(self._next_id):
            if not self.deleted[i] and self.levels[i] > best_level:
                best = i
                best_level = self.levels[i]
        self.entry_point = best
        self.max_level = best_level if best is not None else -1

    def _rebuild(self):
        """重建索引（清除墓碑，恢复搜索效率）"""
        alive = [(self.vectors[i], self.contents[i])
                 for i in range(self._next_id) if not self.deleted[i]]
        logger.debug(f"[HNSWIndex] 重建 {len(alive)} 个存活节点")
        self.vectors = []
        self.contents = []
        self.levels = []
        self.graph = []
        self.deleted = []
        self.entry_point = None
        self.max_level = -1
        self._size = 0
        self._next_id = 0
        self._deleted_count = 0
        self._eviction_ptr = 0
        for vec, content in alive:
            self._insert(vec, content)

    def _insert(self, vector: np.ndarray, content: Any):
        """实际HNSW插入逻辑（不含容量管理）"""
        level = self._random_level()
        node_id = self._next_id
        self._next_id += 1

        self.vectors.append(vector)
        self.contents.append(content)
        self.levels.append(level)
        self.deleted.append(False)
        self.graph.append([[] for _ in range(level + 1)])
        self._size += 1

        # 首个节点：直接设为入口点
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        ep = self.entry_point
        L = self.max_level

        # 从顶层到 level+1 贪心搜索（缩小搜索范围）
        cur = ep
        for lc in range(L, level, -1):
            cur, _ = self._greedy_search_from(vector, cur, lc)

        # 从 min(L, level) 到 0 插入连接
        for lc in range(min(L, level), -1, -1):
            # 束搜索找候选
            candidates = self._search_layer(vector, [cur],
                                            self.ef_construction, lc)
            # 过滤已删除
            candidates = [(d, n) for d, n in candidates if not self.deleted[n]]

            # 选择M个邻居（使用启发式）
            selected = self._select_neighbors(candidates, self.M)

            # 双向连接
            self.graph[node_id][lc] = list(selected)
            max_conn = self.M_max0 if lc == 0 else self.M_max
            for n in selected:
                self.graph[n][lc].append(node_id)
                # 检查n的邻居数是否超限，超限则修剪
                if len(self.graph[n][lc]) > max_conn:
                    # 向量化距离计算（避免逐个_dist_sq调用）
                    n_vec = self.vectors[n]
                    neighbor_ids = [x for x in self.graph[n][lc] if not self.deleted[x]]
                    neighbor_vecs = np.array([self.vectors[x] for x in neighbor_ids])
                    diffs = neighbor_vecs - n_vec
                    dists = np.sum(diffs * diffs, axis=1)
                    n_cands = sorted(zip(dists.tolist(), neighbor_ids))
                    self.graph[n][lc] = self._select_neighbors(n_cands, max_conn)

            # 更新当前节点为最近的候选，用于下一层搜索
            if selected:
                cur = selected[0]
            elif candidates:
                cur = candidates[0][1]

        # 更新入口点（新节点层级更高时）
        if level > L:
            self.entry_point = node_id
            self.max_level = level

    def add(self, vector: np.ndarray, content: Any):
        """添加向量到索引"""
        vector = self._normalize_vector(vector)

        # 容量管理：软删除最旧节点
        if self._size >= self.capacity:
            self._evict_oldest()

        self._insert(vector, content)

        # 周期性重建（墓碑过多时，避免搜索效率退化）
        if self._deleted_count > self.capacity and self._deleted_count > self._size * 2:
            self._rebuild()

        logger.debug(f"[HNSWIndex] 添加 idx={self._next_id-1} size={self._size}")

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Any]]:
        """搜索k个最近邻"""
        if self.entry_point is None or self._size == 0:
            return []
        if k <= 0:
            return []

        query = self._normalize_vector(query)
        ef = max(k, self.M, 16)  # 搜索时的ef参数

        # 从顶层到1贪心搜索（快速定位到目标区域）
        cur = self.entry_point
        for lc in range(self.max_level, 0, -1):
            cur, _ = self._greedy_search_from(query, cur, lc)

        # 在layer 0进行束搜索（精确搜索）
        candidates = self._search_layer(query, [cur], ef, 0)
        # 过滤已删除
        candidates = [(d, n) for d, n in candidates if not self.deleted[n]]

        # 返回top-k（距离从平方转回欧氏距离）
        results = [(np.sqrt(max(d, 0.0)), self.contents[n])
                   for d, n in candidates[:k]]
        logger.debug(f"[HNSWIndex] 搜索 k={k} 返回{len(results)}个结果")
        return results

    def size(self) -> int:
        return self._size

    def clear(self):
        """清空索引"""
        self.vectors = []
        self.contents = []
        self.levels = []
        self.graph = []
        self.deleted = []
        self.entry_point = None
        self.max_level = -1
        self._size = 0
        self._next_id = 0
        self._deleted_count = 0
        self._eviction_ptr = 0
        logger.debug("[HNSWIndex] 清空完成")


# 向后兼容别名（旧代码可能直接导入 SimpleHNSW）
SimpleHNSW = HNSWIndex


class SumTree:
    """
    Sum Tree数据结构，用于O(log N)按比例采样
    基于 Schaul et al. (2015) "Prioritized Experience Replay" 的实现

    核心原理：
    - 完全二叉树，叶节点存储优先级，内部节点存储子节点之和
    - 采样：从根开始，根据随机值选择左/右子树，O(log N)
    - 更新：修改叶节点优先级后向上传播差值，O(log N)
    - 总和：直接读取根节点，O(1)

    复杂度：采样O(log N)，更新O(log N)，对比numpy.choice的O(N)
    """

    def __init__(self, capacity: int):
        self.capacity = max(capacity, 1)
        # 完全二叉树：capacity个叶节点 + capacity-1个内部节点
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        # 数据索引数组：叶节点对应的record在episodic_records中的位置
        self.data_indices: List[Optional[int]] = [None] * self.capacity
        self.write_ptr = 0       # 下一个写入位置（环形）
        self.n_entries = 0       # 当前有效条目数
        self._oldest_ptr = 0     # 最旧条目位置（用于FIFO淘汰）

    def _propagate(self, tree_idx: int, delta: float):
        """向上传播优先级变化"""
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, tree_idx: int, value: float) -> int:
        """从根向下查找：给定value找到对应叶节点"""
        left = 2 * tree_idx + 1
        right = left + 1
        while left < len(self.tree):
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
            left = 2 * tree_idx + 1
            right = left + 1
        return tree_idx

    def total(self) -> float:
        """返回所有优先级之和（根节点值）"""
        return float(self.tree[0])

    def add(self, priority: float, data_idx: int):
        """添加新条目（环形覆盖最旧条目）"""
        tree_idx = self.write_ptr + self.capacity - 1
        self.data_indices[self.write_ptr] = data_idx
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float):
        """更新叶节点优先级并向上传播"""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def update_by_data_idx(self, data_idx: int, priority: float):
        """通过数据索引更新优先级"""
        # 查找对应的tree_idx
        for i in range(self.n_entries):
            if self.data_indices[i] == data_idx:
                tree_idx = i + self.capacity - 1
                self.update(tree_idx, priority)
                return
        # 未找到则忽略（可能已被淘汰）

    def get(self, value: float) -> Tuple[int, float, Optional[int]]:
        """
        按值采样：返回(tree_idx, priority, data_idx)
        value应在[0, total())范围内
        """
        if self.n_entries == 0 or self.total() <= 0:
            return 0, 0.0, None
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, float(self.tree[tree_idx]), self.data_indices[data_idx]

    def clear(self):
        """清空SumTree"""
        self.tree.fill(0.0)
        self.data_indices = [None] * self.capacity
        self.write_ptr = 0
        self.n_entries = 0


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    基于 Schaul et al. (2015) "Prioritized Experience Replay"

    核心特性：
    - 按比例优先级采样：P(i) = p_i^α / Σ p_j^α
    - 重要性采样权重：w_i = (1/(N·P(i)))^β，修正非均匀采样偏差
    - β退火：从β₀线性退火到1.0
    - SumTree实现O(log N)采样

    优先级计算：
    p_i = (error_magnitude + λ_s·surprise + λ_n·is_novel + ε)^α
    - error_magnitude: 预测误差幅度（主信号，类似TD error）
    - surprise: 惊讶度（负对数证据，新颖性信号）
    - is_novel: 新颖性奖励（鼓励探索）
    - ε: 小常数避免零优先级
    - α: 优先级强度（0=均匀，1=完全优先）
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 10000,
                 eps: float = 1e-3,
                 lambda_surprise: float = 0.3,
                 lambda_novel: float = 0.1):
        self.capacity = capacity
        self.alpha = alpha          # 优先级强度
        self.beta = beta            # 重要性采样权重（当前）
        self.beta_start = beta
        self.beta_end = beta_end    # β退火终点
        self.beta_anneal_steps = beta_anneal_steps
        self.eps = eps              # 避免零优先级
        self.lambda_surprise = lambda_surprise
        self.lambda_novel = lambda_novel
        self.sum_tree = SumTree(capacity)
        # 自维护的环形记录缓冲区（与SumTree同步）
        self._records: List[Optional[ExperienceRecord]] = [None] * capacity
        self._write_ptr = 0
        self._size = 0
        self._update_count = 0      # 用于β退火

    def compute_priority(self, record: ExperienceRecord) -> float:
        """计算单条经验的优先级"""
        # 基础优先级：预测误差幅度
        base = 0.0
        if record.error is not None:
            base = abs(record.error.magnitude)
            # 加入惊讶度（新颖性信号）
            base += self.lambda_surprise * record.error.surprise
            # 新颖性奖励
            if record.error.is_novel:
                base += self.lambda_novel
        # 加入奖励幅度（高奖励经验也有价值）
        base += 0.05 * abs(record.reward)
        # (priority + ε)^α
        return (base + self.eps) ** self.alpha

    def add(self, record: ExperienceRecord):
        """添加经验到缓冲区（环形覆盖最旧条目）"""
        priority = self.compute_priority(record)
        idx = self._write_ptr
        self._records[idx] = record
        self.sum_tree.add(priority, idx)
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int) -> Tuple[List[ExperienceRecord], np.ndarray, List[int]]:
        """
        优先级采样
        返回：(records, is_weights, internal_indices)
        - records: 采样到的经验记录列表
        - is_weights: 重要性采样权重数组，shape=(batch_size,)
        - internal_indices: 内部索引（用于update_priorities）
        """
        n = self.sum_tree.n_entries
        if n == 0:
            return [], np.array([]), []
        batch_size = min(batch_size, n)
        total = self.sum_tree.total()
        if total <= 0:
            # 全零优先级 → 均匀采样
            indices = list(np.random.choice(n, batch_size, replace=False))
            records = [self._records[i] for i in indices if self._records[i] is not None]
            weights = np.ones(len(records))
            return records, weights, indices

        # 分段采样：将[0, total)分成batch_size段，每段内随机取一点
        segment = total / batch_size
        records = []
        internal_indices = []
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.sum_tree.get(value)
            if data_idx is not None and self._records[data_idx] is not None:
                records.append(self._records[data_idx])
                internal_indices.append(data_idx)
                priorities.append(priority)

        if len(records) == 0:
            return [], np.array([]), []

        # 计算重要性采样权重: w_i = (1/(N·P(i)))^β
        # P(i) = priority_i / total
        priorities_arr = np.array(priorities)
        probs = priorities_arr / total
        weights = (n * probs) ** (-self.beta)
        # 归一化：除以最大权重，使最大权重为1
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight

        return records, weights, internal_indices

    def update_priorities(self, internal_indices: List[int],
                          new_errors: List[float]):
        """回放后更新优先级（基于新TD error）"""
        for idx, error in zip(internal_indices, new_errors):
            priority = (abs(error) + self.eps) ** self.alpha
            self.sum_tree.update_by_data_idx(idx, priority)
        # β退火
        self._anneal_beta()

    def _anneal_beta(self):
        """β退火：从beta_start线性退火到beta_end"""
        self._update_count += 1
        if self._update_count >= self.beta_anneal_steps:
            self.beta = self.beta_end
        else:
            fraction = self._update_count / self.beta_anneal_steps
            self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def size(self) -> int:
        return self._size

    def clear(self):
        self.sum_tree.clear()
        self._records = [None] * self.capacity
        self._write_ptr = 0
        self._size = 0
        self.beta = self.beta_start
        self._update_count = 0


class MemoryAndReview:
    """
    记忆与离线复盘模块
    - 情景记忆（Episodic）：具体经验
    - 语义记忆（Semantic）：抽象知识
    - 工作记忆（Working）：当前上下文
    - 程序记忆（Procedural）：技能
    - 情绪记忆（Emotional）：情感标记
    """

    def __init__(self, config: Config):
        self.config = config
        self.episodic_capacity = config.get("memory", "episodic_capacity")
        self.semantic_capacity = config.get("memory", "semantic_capacity")
        self.working_capacity = config.get("memory", "working_capacity")
        self.replay_batch = config.get("memory", "replay_batch_size")
        self.consolidation_interval = config.get("memory", "consolidation_interval")

        feature_dim = config.get("perception", "feature_dim")
        self.episodic_index = HNSWIndex(feature_dim, self.episodic_capacity)
        self.episodic_records: deque = deque(maxlen=self.episodic_capacity)
        # 阶段18-步骤2: 优先级经验回放缓冲区（独立于episodic_records）
        # episodic_records用于巩固（相似度去重），PER缓冲区用于回放（优先级采样）
        self.replay_buffer = PrioritizedReplayBuffer(self.episodic_capacity)
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: deque = deque(maxlen=self.working_capacity)
        self.procedural_memory: Dict[str, Any] = {}
        self.emotional_memory: List[Tuple[np.ndarray, float]] = []

        self.consolidation_count = 0
        self.last_consolidation_step = 0
        self._last_replay_weights = np.array([])
        self._last_replay_indices = []
        logger.info(f"[MemoryAndReview] 初始化 episodic_cap={self.episodic_capacity}")

    def store_experience(self, record: ExperienceRecord):
        """存储经验记录"""
        self.episodic_records.append(record)
        # 索引到HNSW
        if record.state is not None:
            self.episodic_index.add(record.state.internal, record)
        # 阶段18-步骤2: 添加到优先级经验回放缓冲区
        self.replay_buffer.add(record)
        # 工作记忆
        self.working_memory.append(record)
        # 情绪记忆（基于奖励）
        if record.state is not None:
            self.emotional_memory.append((record.state.internal.copy(), record.reward))
            if len(self.emotional_memory) > 1000:
                self.emotional_memory.pop(0)
        logger.debug(f"[MemoryAndReview] 存储经验 id={record.record_id} "
                     f"episodic={len(self.episodic_records)}")

    def retrieve_episodic(self, query: np.ndarray, k: int = 5) -> List[ExperienceRecord]:
        """检索情景记忆"""
        results = self.episodic_index.search(query, k)
        records = [content for _, content in results]
        logger.debug(f"[MemoryAndReview] 检索返回{len(records)}条")
        return records

    def store_episodic(self, key: np.ndarray, content: Any):
        """直接存储情景记忆（用于测试）"""
        self.episodic_index.add(key, content)
        logger.debug(f"[MemoryAndReview] 直接存储 episodic_size={self.episodic_index.size()}")

    def replay(self, batch_size: Optional[int] = None) -> List[ExperienceRecord]:
        """
        优先级经验回放（PER）：基于误差幅度的按比例采样
        返回采样的经验记录列表（向后兼容接口）
        """
        n = batch_size if batch_size is not None else self.replay_batch
        if self.replay_buffer.size() == 0:
            logger.debug("[MemoryAndReview] 无记忆可回放")
            return []
        records, weights, indices = self.replay_buffer.sample(n)
        # 缓存最近一次采样的权重和索引，供update_replay_priorities使用
        self._last_replay_weights = weights
        self._last_replay_indices = indices
        logger.info(f"[MemoryAndReview] PER回放{len(records)}条记忆 "
                    f"buffer_size={self.replay_buffer.size()}")
        return records

    def replay_with_weights(self, batch_size: Optional[int] = None
                            ) -> Tuple[List[ExperienceRecord], np.ndarray, List[int]]:
        """
        优先级经验回放（PER）：返回记录+重要性采样权重+内部索引
        用于需要IS权重修正梯度的在线学习场景
        """
        n = batch_size if batch_size is not None else self.replay_batch
        if self.replay_buffer.size() == 0:
            return [], np.array([]), []
        records, weights, indices = self.replay_buffer.sample(n)
        self._last_replay_weights = weights
        self._last_replay_indices = indices
        logger.info(f"[MemoryAndReview] PER回放(含权重){len(records)}条 "
                    f"buffer_size={self.replay_buffer.size()}")
        return records, weights, indices

    def update_replay_priorities(self, internal_indices: List[int],
                                  new_errors: List[float]):
        """回放后更新优先级（基于新预测误差）"""
        self.replay_buffer.update_priorities(internal_indices, new_errors)

    def _kmeans(self, vectors: np.ndarray, k: int,
                max_iters: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-means聚类（numpy向量化实现）
        返回：(centroids, labels)
        - centroids: shape=(k, dim)
        - labels: shape=(n,) 每个点的簇索引

        算法：
        1. K-means++初始化： spread初始中心点
        2. 迭代：分配→更新中心
        3. 收敛：中心移动 < 1e-4 或达到max_iters
        """
        n, dim = vectors.shape
        k = min(k, n)
        rng = np.random.RandomState(seed)

        # 清洗NaN/Inf（防御性编程，与consolidate()一致）
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=10.0, neginf=-10.0)

        # K-means++ 初始化
        centroids = np.zeros((k, dim))
        first_idx = rng.randint(n)
        centroids[0] = vectors[first_idx]
        # 计算到最近中心的距离
        dists = np.sum((vectors - centroids[0]) ** 2, axis=1)
        for i in range(1, k):
            # 按距离^2的概率选择下一个中心
            total_dist = dists.sum()
            if total_dist < 1e-12:
                # 所有点相同或距离为零 → 均匀采样
                probs = np.ones(n) / n
            else:
                probs = dists / total_dist
            idx = rng.choice(n, p=probs)
            centroids[i] = vectors[idx]
            new_dists = np.sum((vectors - centroids[i]) ** 2, axis=1)
            dists = np.minimum(dists, new_dists)

        # 预计算向量范数（用于向量化距离计算）
        vec_norms_sq = np.sum(vectors ** 2, axis=1)  # shape=(n,)

        labels = np.zeros(n, dtype=int)
        for iteration in range(max_iters):
            # 向量化距离计算: ||v - c||^2 = ||v||^2 + ||c||^2 - 2*v·c
            cent_norms_sq = np.sum(centroids ** 2, axis=1)  # shape=(k,)
            # dot_products[i, j] = vectors[i] · centroids[j]
            dot_products = vectors @ centroids.T  # shape=(n, k)
            all_dists = vec_norms_sq[:, None] + cent_norms_sq[None, :] - 2.0 * dot_products
            np.maximum(all_dists, 0.0, out=all_dists)

            new_labels = np.argmin(all_dists, axis=1)

            # 检查收敛
            if iteration > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # 更新中心
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(vectors[mask], axis=0)
                # 空簇保持原中心

        return centroids, labels

    def consolidate(self):
        """
        离线巩固：K-means聚类 + 簇抽象 + 情景记忆修剪

        流程：
        1. 提取所有有效记忆的状态向量
        2. K-means聚类（自适应k = max(2, n//10)）
        3. 每簇计算抽象统计（中心、平均奖励、平均误差、主导动作）
        4. 抽象存入语义记忆
        5. 每簇保留top-k高优先级记忆（~20%），其余修剪
        """
        if len(self.episodic_records) < 10:
            logger.debug("[MemoryAndReview] 记忆不足，跳过巩固")
            return

        initial_count = len(self.episodic_records)
        records_list = list(self.episodic_records)

        # 提取有效记忆（有state的）
        valid_indices = [i for i, r in enumerate(records_list)
                         if r.state is not None]
        if len(valid_indices) < 2:
            logger.debug("[MemoryAndReview] 有效记忆不足，跳过巩固")
            return

        dim = records_list[valid_indices[0]].state.internal.shape[0]
        n_valid = len(valid_indices)
        vectors = np.zeros((n_valid, dim))
        for idx, i in enumerate(valid_indices):
            vec = records_list[i].state.internal
            # 清洗NaN/Inf（与HNSWIndex._normalize_vector一致）
            vec = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
            vectors[idx] = vec

        # 自适应聚类数: 每簇平均10条记忆
        k = max(2, min(n_valid // 10, 50))
        centroids, labels = self._kmeans(vectors, k, max_iters=20, seed=42)

        # 簇抽象 + 修剪
        merged = []
        # 记录无state的记忆（直接保留）
        for i, rec in enumerate(records_list):
            if rec.state is None:
                merged.append(rec)

        for c in range(k):
            cluster_mask = labels == c
            cluster_size = int(np.sum(cluster_mask))
            if cluster_size == 0:
                continue

            cluster_records = [records_list[valid_indices[i]]
                               for i in range(n_valid) if cluster_mask[i]]

            # 簇抽象：计算统计量并存入语义记忆
            cluster_rewards = [r.reward for r in cluster_records]
            cluster_errors = [r.error.magnitude for r in cluster_records
                              if r.error is not None]
            cluster_actions = [r.action.type.value for r in cluster_records
                               if r.action is not None]
            abstraction = {
                "centroid": centroids[c].copy(),
                "size": cluster_size,
                "avg_reward": float(np.mean(cluster_rewards)),
                "avg_error": float(np.mean(cluster_errors)) if cluster_errors else 0.0,
                "dominant_action": max(set(cluster_actions), key=cluster_actions.count)
                                   if cluster_actions else "none",
                "consolidation_step": self.consolidation_count + 1,
            }
            self.update_semantic(f"cluster_{self.consolidation_count}_{c}", abstraction)

            # 修剪：保留每簇top-k高优先级记忆（按奖励+误差排序）
            keep_count = max(1, cluster_size // 5)  # 保留~20%
            cluster_records.sort(
                key=lambda r: r.reward + (abs(r.error.magnitude) if r.error else 0),
                reverse=True
            )
            merged.extend(cluster_records[:keep_count])

        # 重建episodic_records（用于replay和巩固历史）
        self.episodic_records = deque(merged, maxlen=self.episodic_capacity)

        # 阶段18: 索引与records解耦
        # 仅当合并后记忆数大幅减少（>50%）时才重建索引以释放空间
        if len(merged) < initial_count * 0.5:
            self.episodic_index.clear()
            for rec in self.episodic_records:
                if rec.state is not None:
                    self.episodic_index.add(rec.state.internal, rec)

        self.consolidation_count += 1
        logger.info(f"[MemoryAndReview] 巩固 #{self.consolidation_count} "
                    f"k-means(k={k}) {initial_count} -> {len(self.episodic_records)} "
                    f"语义簇={len(self.semantic_memory)}")

    def episodic_count(self) -> int:
        return len(self.episodic_records)

    def get_working_memory(self) -> List[ExperienceRecord]:
        return list(self.working_memory)

    def update_semantic(self, key: str, value: Any):
        """更新语义记忆"""
        if len(self.semantic_memory) >= self.semantic_capacity:
            # 简单LRU：删除第一个key
            first_key = next(iter(self.semantic_memory))
            del self.semantic_memory[first_key]
        self.semantic_memory[key] = value
        logger.debug(f"[MemoryAndReview] 语义记忆更新 key={key}")

    def get_semantic(self, key: str) -> Optional[Any]:
        return self.semantic_memory.get(key)

    def health_check(self) -> bool:
        return True

    def reset(self):
        """重置记忆模块"""
        self.episodic_records = deque(maxlen=self.episodic_capacity)
        self.episodic_index.clear()
        self.replay_buffer.clear()
        self.semantic_memory = {}
        self.working_memory = deque(maxlen=self.working_capacity)
        self.procedural_memory = {}
        self.emotional_memory = []
        self.consolidation_count = 0
        self.last_consolidation_step = 0
        self._last_replay_weights = np.array([])
        self._last_replay_indices = []
        logger.info("[MemoryAndReview] 重置完成")
