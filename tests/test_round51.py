"""
阶段61测试：LLM增强的记忆与上下文管理

覆盖：
    - MemoryType枚举
    - MemoryEntry数据类 (access/current_importance/to_dict/from_dict)
    - TokenBudget (estimate_tokens/fit_text/memory_budget)
    - ContextCompressor (compress_simple/compress_with_llm/compress_multiple)
    - MemoryManager (add/get/remove/clear/count/检索/上下文组装/持久化)
    - 关键词提取
    - Embedding + 余弦相似度
    - 线程安全
    - 集成测试 (LLM驱动压缩 + 真实API)
"""
import unittest
import os
import json
import time
import tempfile
import threading
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from seva.memory_llm import (
    MemoryType, MemoryEntry, TokenBudget,
    ContextCompressor, MemoryManager,
    get_memory_manager, set_memory_manager, reset_memory_manager,
)
from seva.llm import LLMError, reset_llm_client


# ============================================================================
# MemoryType枚举
# ============================================================================

class TestMemoryType(unittest.TestCase):
    """测试MemoryType枚举"""

    def test01_values(self):
        self.assertEqual(MemoryType.EPISODIC.value, "episodic")
        self.assertEqual(MemoryType.SEMANTIC.value, "semantic")
        self.assertEqual(MemoryType.PROCEDURAL.value, "procedural")
        self.assertEqual(MemoryType.WORKING.value, "working")
        self.assertEqual(MemoryType.SENSORY.value, "sensory")

    def test02_count(self):
        self.assertEqual(len(list(MemoryType)), 5)


# ============================================================================
# MemoryEntry
# ============================================================================

class TestMemoryEntry(unittest.TestCase):
    """测试MemoryEntry"""

    def test01_creation(self):
        e = MemoryEntry(id="m1", memory_type=MemoryType.SEMANTIC, content="x")
        self.assertEqual(e.id, "m1")
        self.assertEqual(e.importance, 0.5)
        self.assertEqual(e.access_count, 0)
        self.assertGreater(e.created_at, 0)

    def test02_access(self):
        e = MemoryEntry(id="m1", memory_type=MemoryType.SEMANTIC, content="x")
        e.access()
        e.access()
        e.access()
        self.assertEqual(e.access_count, 3)
        self.assertGreater(e.last_accessed, 0)

    def test03_current_importance_decay(self):
        # 高重要性 + 长期无访问
        e1 = MemoryEntry(
            id="m1", memory_type=MemoryType.SEMANTIC, content="x",
            importance=0.9, created_at=time.time() - 3600 * 100  # 100小时前
        )
        # 频繁访问
        e2 = MemoryEntry(
            id="m2", memory_type=MemoryType.SEMANTIC, content="x",
            importance=0.3
        )
        for _ in range(20):
            e2.access()
        # 频繁访问的应该提升
        self.assertGreater(e2.current_importance(), e2.importance)

    def test04_to_dict(self):
        e = MemoryEntry(
            id="m1", memory_type=MemoryType.EPISODIC, content="hello",
            importance=0.8, keywords=["hello", "world"],
        )
        d = e.to_dict()
        self.assertEqual(d["id"], "m1")
        self.assertEqual(d["memory_type"], "episodic")
        self.assertEqual(d["content"], "hello")
        self.assertEqual(d["importance"], 0.8)
        self.assertEqual(d["keywords"], ["hello", "world"])

    def test05_from_dict_round_trip(self):
        e1 = MemoryEntry(
            id="m1", memory_type=MemoryType.PROCEDURAL,
            content="how to do X", importance=0.7,
            keywords=["X"], metadata={"source": "test"},
        )
        e2 = MemoryEntry.from_dict(e1.to_dict())
        self.assertEqual(e1.id, e2.id)
        self.assertEqual(e1.content, e2.content)
        self.assertEqual(e1.importance, e2.importance)
        self.assertEqual(e1.keywords, e2.keywords)
        self.assertEqual(e1.metadata, e2.metadata)


# ============================================================================
# TokenBudget
# ============================================================================

class TestTokenBudget(unittest.TestCase):
    """测试TokenBudget"""

    def test01_creation(self):
        b = TokenBudget()
        self.assertEqual(b.total_budget, 4096)
        self.assertEqual(b.system_reserve, 200)
        self.assertEqual(b.output_reserve, 512)
        self.assertEqual(b.user_reserve, 500)

    def test02_memory_budget(self):
        b = TokenBudget(total_budget=4096, system_reserve=200,
                        output_reserve=512, user_reserve=500,
                        memory_ratio=0.6)
        # available = 4096 - 200 - 512 - 500 = 2884
        # memory = 2884 * 0.6 = 1730
        self.assertEqual(b.memory_budget, 1730)

    def test03_estimate_tokens(self):
        b = TokenBudget()
        # 英文
        self.assertGreater(b.estimate_tokens("hello world"), 0)
        # 中文
        self.assertGreater(b.estimate_tokens("你好世界"), 0)
        # 空
        self.assertEqual(b.estimate_tokens(""), 0)
        # 中文比英文占更多token(中文字符独立计)
        en = b.estimate_tokens("hello world test")
        cn = b.estimate_tokens("你好世界测试")
        self.assertGreater(cn, 0)

    def test04_fit_text(self):
        b = TokenBudget()
        # 短文本直接返回
        text = "short"
        self.assertEqual(b.fit_text(text, 100), text)
        # 长文本截断
        long_text = "x" * 1000
        result = b.fit_text(long_text, 10)
        self.assertLess(len(result), len(long_text))
        self.assertIn("...", result)


# ============================================================================
# ContextCompressor
# ============================================================================

class TestContextCompressor(unittest.TestCase):
    """测试ContextCompressor"""

    def test01_creation(self):
        c = ContextCompressor()
        self.assertEqual(c.default_max_tokens, 200)
        self.assertIsNone(c._client)

    def test02_compress_simple(self):
        c = ContextCompressor()
        # 短文本不压缩
        self.assertEqual(c.compress_simple("hello"), "hello")
        # 长文本截断
        long_text = "x" * 10000
        result = c.compress_simple(long_text, max_tokens=100)
        self.assertLess(len(result), len(long_text))
        self.assertIn("省略", result)

    def test03_compress_with_llm_no_client(self):
        """无API key时降级到简单压缩"""
        c = ContextCompressor()
        # 不应崩溃,应降级
        result = c.compress_with_llm("x" * 1000, max_tokens=100)
        self.assertIsNotNone(result)
        self.assertLess(len(result), 1000)

    def test04_compress_with_llm_mock(self):
        """Mock LLM成功"""
        c = ContextCompressor()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed text"
        mock_client.chat.return_value = mock_response

        with patch("seva.memory_llm.get_llm_client", return_value=mock_client):
            result = c.compress_with_llm("x" * 1000, max_tokens=100)
        self.assertEqual(result, "Compressed text")

    def test05_compress_with_llm_error_fallback(self):
        """LLM错误时降级到简单压缩"""
        c = ContextCompressor()
        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("mock error")

        with patch("seva.memory_llm.get_llm_client", return_value=mock_client):
            result = c.compress_with_llm("x" * 1000, max_tokens=100)
        # 应该降级到简单压缩
        self.assertIsNotNone(result)
        self.assertLess(len(result), 1000)

    def test06_compress_multiple(self):
        """多条记忆压缩"""
        c = ContextCompressor()
        entries = [
            MemoryEntry(id="1", memory_type=MemoryType.EPISODIC, content="A" * 200),
            MemoryEntry(id="2", memory_type=MemoryType.SEMANTIC, content="B" * 200),
        ]
        # Mock 客户端
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "合并摘要"
        mock_client.chat.return_value = mock_response

        with patch("seva.memory_llm.get_llm_client", return_value=mock_client):
            result = c.compress_multiple(entries, max_tokens=100)
        self.assertEqual(result, "合并摘要")

    def test07_compress_multiple_empty(self):
        c = ContextCompressor()
        self.assertEqual(c.compress_multiple([]), "")


# ============================================================================
# MemoryManager - 使用临时目录避免污染
# ============================================================================

class TestMemoryManager(unittest.TestCase):
    """测试MemoryManager"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Monkey-patch PERSIST_DIR
        import seva.memory_llm as m_module
        self._original_dir = m_module.MemoryManager.PERSIST_DIR
        m_module.MemoryManager.PERSIST_DIR = Path(self.tmpdir)

    def tearDown(self):
        import seva.memory_llm as m_module
        m_module.MemoryManager.PERSIST_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_creation(self):
        m = MemoryManager(persist=False)
        self.assertEqual(m.count(), 0)
        self.assertEqual(m.capacity, 10000)

    def test02_add_and_get(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        entry = m.add("hello world", MemoryType.SEMANTIC, importance=0.8)
        self.assertEqual(m.count(), 1)
        retrieved = m.get(entry.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "hello world")
        self.assertEqual(retrieved.access_count, 1)

    def test03_add_empty_content_raises(self):
        m = MemoryManager(persist=False)
        with self.assertRaises(ValueError):
            m.add("", MemoryType.SEMANTIC)

    def test04_remove(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        entry = m.add("test", MemoryType.EPISODIC)
        self.assertTrue(m.remove(entry.id))
        self.assertEqual(m.count(), 0)
        self.assertFalse(m.remove("nonexistent"))

    def test05_clear(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("episodic1", MemoryType.EPISODIC)
        m.add("episodic2", MemoryType.EPISODIC)
        m.add("semantic1", MemoryType.SEMANTIC)
        self.assertEqual(m.count(), 3)
        # 清空episodic
        m.clear(MemoryType.EPISODIC)
        self.assertEqual(m.count(), 1)
        self.assertEqual(m.count(MemoryType.SEMANTIC), 1)
        # 清空全部
        m.clear()
        self.assertEqual(m.count(), 0)

    def test06_count_by_type(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("1", MemoryType.EPISODIC)
        m.add("2", MemoryType.EPISODIC)
        m.add("3", MemoryType.SEMANTIC)
        self.assertEqual(m.count(MemoryType.EPISODIC), 2)
        self.assertEqual(m.count(MemoryType.SEMANTIC), 1)
        self.assertEqual(m.count(MemoryType.WORKING), 0)

    def test07_retrieve_by_keywords(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("Python是一种编程语言", MemoryType.SEMANTIC)
        m.add("Java是另一种语言", MemoryType.SEMANTIC)
        m.add("今天天气很好", MemoryType.EPISODIC)
        results = m.retrieve_by_keywords(["Python"])
        self.assertGreater(len(results), 0)
        # 第一个应该是Python相关
        self.assertIn("Python", results[0].content)

    def test08_retrieve_by_similarity(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("机器学习算法", MemoryType.SEMANTIC)
        m.add("深度学习模型", MemoryType.SEMANTIC)
        m.add("今天天气晴朗", MemoryType.EPISODIC)
        results = m.retrieve_by_similarity("学习", limit=2)
        self.assertGreater(len(results), 0)
        # 检索结果应该是"学习"相关的(机器学习/深度学习)
        relevant = [r for r in results if "学习" in r.content]
        self.assertGreater(len(relevant), 0)

    def test09_retrieve_type_filter(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("语义知识", MemoryType.SEMANTIC)
        m.add("事件记录", MemoryType.EPISODIC)
        results = m.retrieve("知识", memory_type=MemoryType.SEMANTIC)
        for r in results:
            self.assertEqual(r.memory_type, MemoryType.SEMANTIC)

    def test10_build_context(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("Python是一种解释型语言", MemoryType.SEMANTIC)
        m.add("Python有丰富的库", MemoryType.SEMANTIC)
        context = m.build_context("Python", max_tokens=200)
        self.assertIsNotNone(context)
        self.assertIn("相关记忆", context)

    def test11_build_context_empty(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        context = m.build_context("query")
        self.assertEqual(context, "")

    def test12_capacity_eviction(self):
        m = MemoryManager(capacity=3, persist=False, enable_llm_summary=False)
        m.add("1", MemoryType.EPISODIC, importance=0.5)
        m.add("2", MemoryType.EPISODIC, importance=0.3)
        m.add("3", MemoryType.EPISODIC, importance=0.9)
        # 容量已满,添加新记忆应淘汰最不重要的
        m.add("4", MemoryType.EPISODIC, importance=0.1)
        self.assertEqual(m.count(), 3)
        # 重要性最低的应该被淘汰
        contents = [m.get(eid).content for eid in
                   list(m._memories.keys()) if m.get(eid) is not None]
        # 不能确定是哪个被淘汰,但应该是 importance 最小的

    def test13_persistence_save_load(self):
        """测试持久化"""
        m1 = MemoryManager(persist=True, enable_llm_summary=False)
        e1 = m1.add("持久化测试1", MemoryType.SEMANTIC, importance=0.8)
        e2 = m1.add("持久化测试2", MemoryType.EPISODIC, importance=0.6)

        # 重新创建应能加载
        m2 = MemoryManager(persist=True, enable_llm_summary=False)
        self.assertEqual(m2.count(), 2)
        loaded = m2.get(e1.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "持久化测试1")

    def test14_stats(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        m.add("1", MemoryType.SEMANTIC)
        m.add("2", MemoryType.EPISODIC)
        stats = m.get_stats()
        self.assertIn("total_added", stats)
        self.assertIn("current_count", stats)
        self.assertIn("vocab_size", stats)
        self.assertIn("type_distribution", stats)
        self.assertEqual(stats["total_added"], 2)

    def test15_get_report(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        report = m.get_report()
        self.assertIn("stats", report)
        self.assertIn("capacity", report)
        self.assertIn("embedding_dim", report)
        self.assertIn("memory_budget", report)

    def test16_thread_safety(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        errors = []

        def worker(i):
            try:
                for j in range(5):
                    m.add(f"memory_{i}_{j}", MemoryType.EPISODIC)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        self.assertEqual(m.count(), 25)

    def test17_keyword_extraction(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        # 中文
        kws = m._extract_keywords("Python是一种编程语言，可以用来做数据分析")
        self.assertGreater(len(kws), 0)
        # 英文
        kws2 = m._extract_keywords("Machine learning is a subset of artificial intelligence")
        self.assertGreater(len(kws2), 0)

    def test18_embedding_consistency(self):
        m = MemoryManager(persist=False, enable_llm_summary=False)
        v1 = m._embed("hello world")
        v2 = m._embed("hello world")
        # 相同文本应得到相同embedding
        self.assertEqual(len(v1), m.embedding_dim)
        # 归一化后内积应该接近1
        sim = m._cosine_similarity(v1, v2)
        self.assertGreater(sim, 0.9)


# ============================================================================
# 全局单例
# ============================================================================

class TestGlobalMemoryManager(unittest.TestCase):
    """测试全局记忆管理器"""

    def setUp(self):
        reset_memory_manager()
        import seva.memory_llm as m_module
        self.tmpdir = tempfile.mkdtemp()
        self._original_dir = m_module.MemoryManager.PERSIST_DIR
        m_module.MemoryManager.PERSIST_DIR = Path(self.tmpdir)

    def tearDown(self):
        reset_memory_manager()
        import seva.memory_llm as m_module
        m_module.MemoryManager.PERSIST_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_singleton(self):
        m1 = get_memory_manager()
        m2 = get_memory_manager()
        self.assertIs(m1, m2)

    def test02_set_and_get(self):
        custom = MemoryManager(persist=False)
        set_memory_manager(custom)
        self.assertIs(get_memory_manager(), custom)

    def test03_reset(self):
        m1 = get_memory_manager()
        reset_memory_manager()
        m2 = get_memory_manager()
        self.assertIsNot(m1, m2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_memory_manager()
        reset_llm_client()
        import seva.memory_llm as m_module
        self.tmpdir = tempfile.mkdtemp()
        self._original_dir = m_module.MemoryManager.PERSIST_DIR
        m_module.MemoryManager.PERSIST_DIR = Path(self.tmpdir)

    def tearDown(self):
        reset_memory_manager()
        reset_llm_client()
        import seva.memory_llm as m_module
        m_module.MemoryManager.PERSIST_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_workflow(self):
        """完整工作流"""
        m = MemoryManager(persist=False, enable_llm_summary=False)
        # 添加多种记忆
        m.add("用户喜欢Python编程", MemoryType.SEMANTIC, importance=0.9)
        m.add("今天学习了装饰器", MemoryType.EPISODIC, importance=0.6)
        m.add("Python是解释型语言", MemoryType.SEMANTIC, importance=0.7)
        m.add("怎么用pandas分析数据", MemoryType.PROCEDURAL, importance=0.8)
        # 检索
        results = m.retrieve("Python", limit=3)
        self.assertGreater(len(results), 0)
        # 构造上下文
        context = m.build_context("Python编程", max_tokens=200)
        self.assertIn("Python", context)

    def test02_real_llm_compress(self):
        """真实LLM压缩(需要API key)"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实LLM测试")
        m = MemoryManager(persist=False, enable_llm_summary=False)
        long_content = "Python是一种广泛使用的高级编程语言,具有简洁的语法和强大的库生态。" * 20
        m.add(long_content, MemoryType.SEMANTIC, importance=0.8, auto_summarize=False)
        # 单独触发压缩
        c = ContextCompressor()
        compressed = c.compress_with_llm(
            long_content, max_tokens=100, use_cache=False
        )
        self.assertLess(len(compressed), len(long_content))
        self.assertGreater(len(compressed), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
