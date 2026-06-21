"""
LLM增强的记忆与上下文管理（阶段61）

对应设计原则：
    1. LLM = 认知引擎（做记忆摘要、上下文压缩、相关性判断）
    2. SEVA = 控制骨架（决定什么存、什么取、什么时候压缩）
    3. 记忆有"成本"——存储成本和检索成本
    4. Token预算管理——避免LLM调用时超出上下文窗口

模块组成：
    - MemoryType: 5类记忆 (episodic/semantic/procedural/working/sensory)
    - MemoryEntry: 单条记忆 (id/type/content/embedding/importance/...)
    - ContextCompressor: LLM驱动的上下文压缩器
    - TokenBudget: token预算管理 (max_total/per_section分配)
    - MemoryManager: 记忆管理器 (增删改查/检索/摘要/持久化)
    - get_memory_manager/set_memory_manager: 全局单例
"""
from __future__ import annotations

import os
import re
import json
import time
import math
import hashlib
import logging
import threading
import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque

from .llm import Message, LLMError, get_llm_client
from .prompts import (
    PromptTemplate, PromptSection, PromptVariable, FewShotExample,
    register_prompt, render_prompt, get_prompt,
)


logger = logging.getLogger("seva.memory_llm")


# ============================================================================
# 记忆类型
# ============================================================================

class MemoryType(Enum):
    """五类记忆"""
    EPISODIC = "episodic"          # 情景记忆：具体事件 (我在某时某地做了某事)
    SEMANTIC = "semantic"          # 语义记忆：抽象知识 (某概念的定义)
    PROCEDURAL = "procedural"      # 程序记忆：技能/操作 (怎么完成某任务)
    WORKING = "working"            # 工作记忆：当前任务上下文 (短期)
    SENSORY = "sensory"            # 感觉记忆：原始输入缓冲 (瞬时)


# ============================================================================
# 记忆条目
# ============================================================================

@dataclass
class MemoryEntry:
    """单条记忆"""
    id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[List[float]] = None  # 简单词频向量(无numpy依赖)
    summary: Optional[str] = None           # LLM生成的摘要
    keywords: List[str] = field(default_factory=list)  # 关键词
    importance: float = 0.5                 # 重要性 [0, 1]
    access_count: int = 0                   # 访问次数
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.01                # 记忆衰减率(每次访问)
    related_ids: List[str] = field(default_factory=list)  # 关联记忆ID

    def access(self):
        """访问记忆(更新访问计数和时间)"""
        self.access_count += 1
        self.last_accessed = time.time()

    def current_importance(self) -> float:
        """计算当前重要性(随时间衰减)"""
        elapsed_hours = (time.time() - self.created_at) / 3600
        decay = math.exp(-self.decay_rate * elapsed_hours * 10)
        # 频繁访问提升重要性
        boost = math.log(1 + self.access_count) * 0.05
        return min(1.0, self.importance * decay + boost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "embedding": self.embedding,
            "summary": self.summary,
            "keywords": self.keywords,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
            "decay_rate": self.decay_rate,
            "related_ids": self.related_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            embedding=data.get("embedding"),
            summary=data.get("summary"),
            keywords=data.get("keywords", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            metadata=data.get("metadata", {}),
            decay_rate=data.get("decay_rate", 0.01),
            related_ids=data.get("related_ids", []),
        )


# ============================================================================
# Token预算
# ============================================================================

class TokenBudget:
    """
    Token预算管理

    用于避免LLM调用时超出上下文窗口。分配预算给：
        - 系统消息 (固定)
        - 记忆上下文 (可压缩)
        - 用户消息 (固定)
        - LLM输出 (固定)
    """

    def __init__(self, total_budget: int = 4096,
                 system_reserve: int = 200,
                 output_reserve: int = 512,
                 user_reserve: int = 500,
                 memory_ratio: float = 0.6):
        """
        Args:
            total_budget: 总token预算
            system_reserve: 系统消息保留
            output_reserve: 输出保留
            user_reserve: 用户消息保留
            memory_ratio: 记忆占剩余预算的比例
        """
        self.total_budget = total_budget
        self.system_reserve = system_reserve
        self.output_reserve = output_reserve
        self.user_reserve = user_reserve
        self.memory_ratio = memory_ratio

    @property
    def memory_budget(self) -> int:
        available = self.total_budget - self.system_reserve - \
                    self.output_reserve - self.user_reserve
        return max(0, int(available * self.memory_ratio))

    def estimate_tokens(self, text: str) -> int:
        """粗略估计token数(中英文混合, ~1.5 tokens/字)"""
        if not text:
            return 0
        # 英文按4字符≈1token, 中文按1.5字符≈1token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars + other_chars // 4 + 1

    def fit_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数"""
        if self.estimate_tokens(text) <= max_tokens:
            return text
        # 简单按字符数截断
        char_budget = max_tokens * 2  # 保守估计
        if len(text) > char_budget:
            return text[:char_budget] + "..."
        return text


# ============================================================================
# 上下文压缩器（LLM驱动）
# ============================================================================

class ContextCompressor:
    """
    LLM驱动的上下文压缩器

    核心方法:
        compress_with_llm(): 用LLM压缩长文本
        compress_simple(): 不调用LLM的简单截断
        compress_multiple(): 压缩多条记忆成单一摘要
    """

    def __init__(self, llm_client=None, default_max_tokens: int = 200):
        """
        Args:
            llm_client: LLM客户端,None时自动获取
            default_max_tokens: 压缩后目标token数
        """
        self._client = llm_client
        self.default_max_tokens = default_max_tokens
        self._lock = threading.RLock()

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def compress_simple(self, text: str, max_tokens: int = 200) -> str:
        """
        简单截断压缩(无LLM)

        策略:
            1. 保留前30%和后20%(开头结尾通常重要)
            2. 中间部分提取关键句
        """
        if not text:
            return ""

        budget = TokenBudget()
        if budget.estimate_tokens(text) <= max_tokens:
            return text

        char_budget = max_tokens * 3
        if len(text) <= char_budget:
            return text

        # 保留开头和结尾
        head_size = int(char_budget * 0.3)
        tail_size = int(char_budget * 0.2)
        head = text[:head_size]
        tail = text[-tail_size:] if tail_size < len(text) else ""
        middle_omitted = f"\n[...省略{len(text) - head_size - tail_size}字符...]\n"
        return head + middle_omitted + tail

    def compress_with_llm(self, text: str, max_tokens: int = 200,
                          context: Optional[str] = None,
                          use_cache: bool = True) -> str:
        """
        LLM驱动的压缩

        Args:
            text: 要压缩的文本
            max_tokens: 目标最大token数
            context: 上下文信息(帮助LLM保留关键信息)
            use_cache: 是否使用LLM客户端缓存

        Returns:
            压缩后的文本
        """
        if not text:
            return ""

        # 短文本不需压缩
        budget = TokenBudget()
        if budget.estimate_tokens(text) <= max_tokens:
            return text

        # 构造压缩prompt
        ctx_text = f"\n上下文:{context}" if context else ""
        system_content = (
            "你是一个文本压缩助手。请将用户提供的文本压缩到目标长度,"
            "保留所有关键信息和事实,删除冗余表述。"
        )
        user_content = (
            f"请将以下文本压缩到约{max_tokens} tokens以内,"
            f"保留关键事实和数据。{ctx_text}\n\n"
            f"原文:\n{text}\n\n"
            f"压缩后:"
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = use_cache
            try:
                response = client.chat(
                    [Message.system(system_content), Message.user(user_content)],
                    temperature=0.1,
                    max_tokens=max_tokens + 50,
                )
            finally:
                client.use_cache = original_cache
            return response.content.strip()
        except LLMError as e:
            logger.warning(f"[Compressor] LLM压缩失败,降级到简单截断: {e}")
            return self.compress_simple(text, max_tokens)
        except Exception as e:
            logger.error(f"[Compressor] 异常: {e}")
            return self.compress_simple(text, max_tokens)

    def compress_multiple(self, entries: List[MemoryEntry],
                          max_tokens: int = 300,
                          query: Optional[str] = None) -> str:
        """
        压缩多条记忆为单一摘要

        Args:
            entries: 记忆列表
            max_tokens: 目标token数
            query: 当前查询(用于保持相关性)
        """
        if not entries:
            return ""

        if len(entries) == 1:
            return self.compress_with_llm(
                entries[0].content, max_tokens, context=query
            )

        # 合并内容
        combined = "\n---\n".join(
            f"[{e.memory_type.value}] {e.content}" for e in entries
        )
        return self.compress_with_llm(combined, max_tokens, context=query)


# ============================================================================
# 记忆管理器
# ============================================================================

class MemoryManager:
    """
    记忆管理器

    职责:
        1. 增删改查
        2. 检索(基于embedding相似度或关键词匹配)
        3. LLM摘要
        4. Token预算控制
        5. 持久化

    不使用重型向量库,采用简单TF-IDF式embedding + 余弦相似度。
    """

    PERSIST_DIR = Path.home() / ".seva" / "memory"

    def __init__(self, capacity: int = 10000,
                 embedding_dim: int = 128,
                 token_budget: Optional[TokenBudget] = None,
                 compressor: Optional[ContextCompressor] = None,
                 enable_llm_summary: bool = True,
                 persist: bool = True):
        """
        Args:
            capacity: 最大记忆容量
            embedding_dim: embedding维度
            token_budget: token预算
            compressor: 上下文压缩器
            enable_llm_summary: 是否启用LLM摘要
            persist: 是否持久化
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.token_budget = token_budget or TokenBudget()
        self.compressor = compressor or ContextCompressor()
        self.enable_llm_summary = enable_llm_summary
        self.persist = persist

        # 存储
        self._memories: Dict[str, MemoryEntry] = {}
        self._type_index: Dict[MemoryType, set] = defaultdict(set)
        self._keyword_index: Dict[str, set] = defaultdict(set)
        self._lock = threading.RLock()

        # 词表(用于embedding)
        self._vocab: Dict[str, int] = {}
        self._vocab_lock = threading.Lock()

        # 统计
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_retrieved": 0,
            "llm_summary_calls": 0,
        }
        self._stats_lock = threading.Lock()

        # 持久化路径
        if persist:
            self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._load()

        logger.info(f"[Memory] 初始化 capacity={capacity} "
                   f"enable_llm_summary={enable_llm_summary}")

    # ========================================================================
    # CRUD
    # ========================================================================

    def add(self, content: str, memory_type: MemoryType = MemoryType.EPISODIC,
            importance: float = 0.5, metadata: Optional[Dict] = None,
            keywords: Optional[List[str]] = None,
            auto_summarize: bool = True) -> MemoryEntry:
        """
        添加记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 初始重要性
            metadata: 元数据
            keywords: 关键词(自动从内容提取 if None)
            auto_summarize: 是否自动生成LLM摘要
        """
        if not content:
            raise ValueError("content不能为空")

        entry_id = self._gen_id(content)
        embedding = self._embed(content)
        kws = keywords or self._extract_keywords(content)
        now = time.time()

        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            keywords=kws,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )

        with self._lock:
            # 容量管理:淘汰最不重要的记忆
            if len(self._memories) >= self.capacity:
                self._evict_one()
            self._memories[entry_id] = entry
            self._type_index[memory_type].add(entry_id)
            for kw in kws:
                self._keyword_index[kw].add(entry_id)

        with self._stats_lock:
            self.stats["total_added"] += 1

        # LLM摘要(可选)
        if auto_summarize and self.enable_llm_summary and len(content) > 200:
            self._summarize_entry(entry)

        if self.persist:
            self._save()
        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取记忆(更新访问统计)"""
        with self._lock:
            entry = self._memories.get(entry_id)
        if entry is not None:
            entry.access()
            with self._stats_lock:
                self.stats["total_retrieved"] += 1
        return entry

    def remove(self, entry_id: str) -> bool:
        """删除记忆"""
        with self._lock:
            entry = self._memories.pop(entry_id, None)
            if entry is None:
                return False
            self._type_index[entry.memory_type].discard(entry_id)
            for kw in entry.keywords:
                self._keyword_index[kw].discard(entry_id)
        with self._stats_lock:
            self.stats["total_removed"] += 1
        if self.persist:
            self._save()
        return True

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """清空记忆(可指定类型)"""
        with self._lock:
            if memory_type is None:
                count = len(self._memories)
                self._memories.clear()
                self._type_index.clear()
                self._keyword_index.clear()
            else:
                ids = list(self._type_index.get(memory_type, set()))
                for eid in ids:
                    self._memories.pop(eid, None)
                self._type_index[memory_type].clear()
                count = len(ids)
        if self.persist:
            self._save()
        return count

    def count(self, memory_type: Optional[MemoryType] = None) -> int:
        """统计记忆数"""
        with self._lock:
            if memory_type is None:
                return len(self._memories)
            return len(self._type_index.get(memory_type, set()))

    # ========================================================================
    # 检索
    # ========================================================================

    def retrieve_by_keywords(self, keywords: List[str], limit: int = 5,
                              memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """按关键词检索"""
        if not keywords:
            return []
        # 关键词统一小写以匹配索引
        normalized_kws = [kw.lower() for kw in keywords]
        scores: Dict[str, float] = defaultdict(float)
        with self._lock:
            for kw in normalized_kws:
                for eid in self._keyword_index.get(kw, set()):
                    entry = self._memories.get(eid)
                    if entry is None:
                        continue
                    if memory_type and entry.memory_type != memory_type:
                        continue
                    scores[eid] += entry.current_importance()
        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:limit]
        results = []
        for eid in sorted_ids:
            entry = self.get(eid)
            if entry is not None:
                results.append(entry)
        return results

    def retrieve_by_similarity(self, query: str, limit: int = 5,
                                memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """按embedding相似度检索"""
        query_embedding = self._embed(query)
        scored: List[Tuple[float, str]] = []
        with self._lock:
            for eid, entry in self._memories.items():
                if memory_type and entry.memory_type != memory_type:
                    continue
                if not entry.embedding:
                    continue
                sim = self._cosine_similarity(query_embedding, entry.embedding)
                # 重要性加权
                weighted_sim = sim * 0.7 + entry.current_importance() * 0.3
                scored.append((weighted_sim, eid))
        scored.sort(key=lambda x: -x[0])
        results = []
        for _, eid in scored[:limit]:
            entry = self.get(eid)
            if entry is not None:
                results.append(entry)
        return results

    def retrieve(self, query: str, limit: int = 5,
                  memory_type: Optional[MemoryType] = None,
                  strategy: str = "hybrid") -> List[MemoryEntry]:
        """
        统一检索接口

        Args:
            query: 查询
            limit: 返回数量
            memory_type: 类型过滤
            strategy: 检索策略: keywords/similarity/hybrid
        """
        keywords = self._extract_keywords(query)
        if strategy == "keywords":
            return self.retrieve_by_keywords(keywords, limit, memory_type)
        elif strategy == "similarity":
            return self.retrieve_by_similarity(query, limit, memory_type)
        else:  # hybrid
            kw_results = self.retrieve_by_keywords(keywords, limit, memory_type)
            sim_results = self.retrieve_by_similarity(query, limit, memory_type)
            # 合并去重,按重要性排序
            seen = set()
            merged = []
            for entry in kw_results + sim_results:
                if entry.id in seen:
                    continue
                seen.add(entry.id)
                merged.append(entry)
            merged.sort(key=lambda e: -e.current_importance())
            return merged[:limit]

    # ========================================================================
    # 上下文组装
    # ========================================================================

    def build_context(self, query: str, max_tokens: Optional[int] = None,
                      memory_types: Optional[List[MemoryType]] = None,
                      use_compression: bool = True) -> str:
        """
        构造LLM上下文

        Args:
            query: 当前查询
            max_tokens: 上下文最大token数
            memory_types: 使用的记忆类型
            use_compression: 是否使用LLM压缩
        """
        max_tokens = max_tokens or self.token_budget.memory_budget
        memory_types = memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC]

        # 检索相关记忆
        all_results: List[MemoryEntry] = []
        for mt in memory_types:
            results = self.retrieve(query, limit=5, memory_type=mt, strategy="hybrid")
            all_results.extend(results)

        # 去重
        seen = set()
        unique_results = []
        for e in all_results:
            if e.id in seen:
                continue
            seen.add(e.id)
            unique_results.append(e)

        # 按重要性排序
        unique_results.sort(key=lambda e: -e.current_importance())

        # token预算控制
        budget = TokenBudget()
        selected = []
        current_tokens = 0
        for entry in unique_results:
            entry_tokens = budget.estimate_tokens(entry.content)
            if current_tokens + entry_tokens > max_tokens:
                break
            selected.append(entry)
            current_tokens += entry_tokens

        if not selected:
            return ""

        # 构造上下文
        if use_compression and self.compressor and len(selected) > 2:
            compressed = self.compressor.compress_multiple(
                selected, max_tokens=max_tokens, query=query
            )
            return f"相关记忆:\n{compressed}"
        else:
            context = "\n".join(
                f"- [{e.memory_type.value}] {e.content[:200]}"
                for e in selected
            )
            return f"相关记忆:\n{context}"

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _gen_id(self, content: str) -> str:
        """生成记忆ID"""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"mem_{h}"

    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """简单关键词提取(基于词频和中文字符n-gram)"""
        if not text:
            return []
        # 中文n-gram
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        cn_ngrams = []
        for segment in chinese_chars:
            for n in [2]:
                for i in range(len(segment) - n + 1):
                    cn_ngrams.append(segment[i:i+n])
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        # 统计频率
        freq: Dict[str, int] = defaultdict(int)
        for ng in cn_ngrams:
            freq[ng] += 1
        for w in english_words:
            freq[w] += 1
        # 排序取top
        sorted_kws = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_kws[:top_k]]

    def _embed(self, text: str) -> List[float]:
        """简单embedding(基于词频的稀疏向量)"""
        if not text:
            return [0.0] * self.embedding_dim
        # 提取tokens
        tokens = []
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for seg in chinese_chars:
            for n in [2]:
                for i in range(len(seg) - n + 1):
                    tokens.append(seg[i:i+n])
        tokens.extend(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        if not tokens:
            return [0.0] * self.embedding_dim
        # 构建/查询词表
        with self._vocab_lock:
            for token in tokens:
                if token not in self._vocab:
                    if len(self._vocab) < self.embedding_dim * 100:
                        self._vocab[token] = len(self._vocab) % self.embedding_dim
        # 向量化:每个token映射到一个维度,累加计数
        vector = [0.0] * self.embedding_dim
        for token in tokens:
            idx = self._vocab.get(token, -1)
            if idx >= 0:
                vector[idx] += 1.0
        # L2归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """余弦相似度"""
        if not vec_a or not vec_b:
            return 0.0
        min_len = min(len(vec_a), len(vec_b))
        a, b = vec_a[:min_len], vec_b[:min_len]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _evict_one(self):
        """淘汰最不重要的记忆"""
        if not self._memories:
            return
        min_importance = float("inf")
        min_id = None
        for eid, entry in self._memories.items():
            ci = entry.current_importance()
            if ci < min_importance:
                min_importance = ci
                min_id = eid
        if min_id:
            self.remove(min_id)

    def _summarize_entry(self, entry: MemoryEntry):
        """使用LLM为记忆生成摘要"""
        if not self.compressor:
            return
        try:
            with self._stats_lock:
                self.stats["llm_summary_calls"] += 1
            summary = self.compressor.compress_with_llm(
                entry.content, max_tokens=50, use_cache=True
            )
            entry.summary = summary
            entry.updated_at = time.time()
        except Exception as e:
            logger.warning(f"[Memory] 摘要失败: {e}")

    # ========================================================================
    # 持久化
    # ========================================================================

    def _save(self):
        """保存到磁盘"""
        if not self.persist:
            return
        try:
            data = {
                "memories": {eid: e.to_dict()
                            for eid, e in self._memories.items()},
                "vocab": self._vocab,
            }
            fp = self.PERSIST_DIR / "memories.json"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Memory] 持久化失败: {e}")

    def _load(self):
        """从磁盘加载"""
        if not self.persist:
            return
        try:
            fp = self.PERSIST_DIR / "memories.json"
            if not fp.exists():
                return
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                for eid, ed in data.get("memories", {}).items():
                    entry = MemoryEntry.from_dict(ed)
                    self._memories[eid] = entry
                    self._type_index[entry.memory_type].add(eid)
                    for kw in entry.keywords:
                        self._keyword_index[kw].add(eid)
            with self._vocab_lock:
                self._vocab.update(data.get("vocab", {}))
            logger.info(f"[Memory] 加载 {len(self._memories)} 条记忆")
        except Exception as e:
            logger.warning(f"[Memory] 加载失败: {e}")

    # ========================================================================
    # 报告
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self.stats)
        with self._lock:
            stats["current_count"] = len(self._memories)
            stats["vocab_size"] = len(self._vocab)
            type_counts = {mt.value: len(ids) for mt, ids in self._type_index.items()}
            stats["type_distribution"] = type_counts
        return stats

    def get_report(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "memory_budget": self.token_budget.memory_budget,
            "enable_llm_summary": self.enable_llm_summary,
        }


# ============================================================================
# 全局单例
# ============================================================================

_global_manager: Optional[MemoryManager] = None
_global_lock = threading.Lock()


def get_memory_manager(persist: bool = True) -> MemoryManager:
    """获取全局记忆管理器"""
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = MemoryManager(persist=persist)
        return _global_manager


def set_memory_manager(manager: MemoryManager):
    """设置全局记忆管理器"""
    global _global_manager
    with _global_lock:
        _global_manager = manager


def reset_memory_manager():
    """重置全局记忆管理器"""
    global _global_manager
    with _global_lock:
        _global_manager = None
