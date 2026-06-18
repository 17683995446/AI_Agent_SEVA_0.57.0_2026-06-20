"""
记忆与离线复盘模块（MemoryAndReview）
对应设计文档第14章、第30章
五类记忆 + HNSW检索(简化) + Replay巩固
"""
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from .types import ExperienceRecord, gen_id
from .config import Config

logger = logging.getLogger("seva.memory")


class SimpleHNSW:
    """
    简化HNSW近似最近邻索引
    （生产环境应使用faiss或hnswlib，此处用纯Python实现便于测试）
    """

    def __init__(self, dim: int, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        self.vectors: List[np.ndarray] = []
        self.contents: List[Any] = []
        logger.info(f"[SimpleHNSW] 初始化 dim={dim} capacity={capacity}")

    def add(self, vector: np.ndarray, content: Any):
        """添加向量"""
        vector = np.asarray(vector, dtype=np.float64)
        # 修复Bug16: NaN/Inf处理，防止污染索引
        vector = np.nan_to_num(vector, nan=0.0, posinf=10.0, neginf=-10.0)
        vector = np.clip(vector, -10.0, 10.0)
        if vector.shape != (self.dim,):
            vector = np.resize(vector, self.dim)
        if len(self.vectors) >= self.capacity:
            # 驱逐最旧的
            self.vectors.pop(0)
            self.contents.pop(0)
        self.vectors.append(vector)
        self.contents.append(content)
        logger.debug(f"[SimpleHNSW] 添加 idx={len(self.vectors)-1}")

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Any]]:
        """搜索k个最近邻"""
        if not self.vectors:
            return []
        # 修复Bug31: k<=0时返回空列表
        if k <= 0:
            return []
        query = np.asarray(query, dtype=np.float64)
        # 修复Bug16: NaN/Inf处理
        query = np.nan_to_num(query, nan=0.0, posinf=10.0, neginf=-10.0)
        query = np.clip(query, -10.0, 10.0)
        if query.shape != (self.dim,):
            query = np.resize(query, self.dim)

        # 修复Bug6: 使用向量化计算代替循环，提升性能
        vectors_arr = np.array(self.vectors)
        diffs = vectors_arr - query
        dists = np.linalg.norm(diffs, axis=1)
        # 修复Bug16: 距离NaN/Inf保护
        dists = np.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        k = min(k, len(dists))
        sorted_idx = np.argpartition(dists, k-1)[:k]
        sorted_idx = sorted_idx[np.argsort(dists[sorted_idx])]
        results = [(float(dists[i]), self.contents[i]) for i in sorted_idx]
        logger.debug(f"[SimpleHNSW] 搜索 k={k} 返回{len(results)}个结果")
        return results

    def size(self) -> int:
        return len(self.vectors)

    def clear(self):
        self.vectors = []
        self.contents = []


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
        self.episodic_index = SimpleHNSW(feature_dim, self.episodic_capacity)
        self.episodic_records: deque = deque(maxlen=self.episodic_capacity)
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: deque = deque(maxlen=self.working_capacity)
        self.procedural_memory: Dict[str, Any] = {}
        self.emotional_memory: List[Tuple[np.ndarray, float]] = []

        self.consolidation_count = 0
        self.last_consolidation_step = 0
        logger.info(f"[MemoryAndReview] 初始化 episodic_cap={self.episodic_capacity}")

    def store_experience(self, record: ExperienceRecord):
        """存储经验记录"""
        self.episodic_records.append(record)
        # 索引到HNSW
        if record.state is not None:
            self.episodic_index.add(record.state.internal, record)
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
        """Replay回放：从情景记忆中采样"""
        n = batch_size if batch_size is not None else self.replay_batch
        if len(self.episodic_records) == 0:
            logger.debug("[MemoryAndReview] 无记忆可回放")
            return []
        n = min(n, len(self.episodic_records))
        # 随机采样
        indices = np.random.choice(len(self.episodic_records), n, replace=False)
        replayed = [self.episodic_records[i] for i in indices]
        logger.info(f"[MemoryAndReview] 回放{n}条记忆")
        return replayed

    def consolidate(self):
        """离线巩固：合并相似记忆"""
        if len(self.episodic_records) < 10:
            logger.debug("[MemoryAndReview] 记忆不足，跳过巩固")
            return

        initial_count = len(self.episodic_records)
        records_list = list(self.episodic_records)
        n = len(records_list)

        # 修复Bug24: 保留全部记忆参与巩固，仅限制单次比较范围避免O(n²)
        # 优化: 向量化相似度计算，预计算归一化向量
        valid_indices = [i for i, r in enumerate(records_list)
                         if r.state is not None]
        if len(valid_indices) < 2:
            logger.debug("[MemoryAndReview] 有效记忆不足，跳过巩固")
            return

        # 预计算归一化向量矩阵
        dim = records_list[valid_indices[0]].state.internal.shape[0]
        vectors = np.zeros((len(valid_indices), dim))
        for idx, i in enumerate(valid_indices):
            vec = records_list[i].state.internal
            norm = np.linalg.norm(vec) + 1e-8
            vectors[idx] = vec / norm

        merged = []
        used = set()  # 记录valid_indices中的索引
        compare_window = 100  # 每条最多与后续100条比较

        for local_idx, global_idx in enumerate(valid_indices):
            if local_idx in used:
                continue
            cluster = [records_list[global_idx]]
            used.add(local_idx)

            # 向量化计算与后续compare_window条的相似度
            end_idx = min(local_idx + compare_window + 1, len(valid_indices))
            if end_idx > local_idx + 1:
                # 批量计算相似度
                candidates = np.arange(local_idx + 1, end_idx)
                # 过滤已使用的
                candidates = [c for c in candidates if c not in used]
                if candidates:
                    cand_arr = np.array(candidates)
                    cand_vectors = vectors[cand_arr]
                    sims = cand_vectors @ vectors[local_idx]
                    # 找出相似度>阈值的
                    similar_mask = sims > 0.85
                    for c, sim in zip(candidates, sims):
                        if sim > 0.85 and c not in used:
                            cluster.append(records_list[valid_indices[c]])
                            used.add(c)

            # 保留聚类代表（奖励最高的）
            representative = max(cluster, key=lambda r: r.reward)
            merged.append(representative)

        # 修复Bug24: 未参与比较的记忆（无state的）直接保留，不丢弃
        for i, rec in enumerate(records_list):
            if rec.state is None:
                merged.append(rec)

        # 重建记忆
        self.episodic_records = deque(merged, maxlen=self.episodic_capacity)
        self.episodic_index.clear()
        for rec in self.episodic_records:
            if rec.state is not None:
                self.episodic_index.add(rec.state.internal, rec)

        self.consolidation_count += 1
        logger.info(f"[MemoryAndReview] 巩固 #{self.consolidation_count} "
                    f"{initial_count} -> {len(self.episodic_records)}")

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
        self.semantic_memory = {}
        self.working_memory = deque(maxlen=self.working_capacity)
        self.procedural_memory = {}
        self.emotional_memory = []
        self.consolidation_count = 0
        self.last_consolidation_step = 0
        logger.info("[MemoryAndReview] 重置完成")
