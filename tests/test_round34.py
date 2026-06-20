"""
阶段44测试：性能优化与启动加速
验证 PerformanceMonitor / LazyLoader / StartupProfiler / LRUCache / MemoryTracker
及便捷函数和CLI集成
"""
import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


# 测试前重置单例
import seva
from seva.performance import (PerformanceMonitor, LazyLoader, LazyAttribute,
                              StartupProfiler, LRUCache, MemoryTracker,
                              profile_operation, get_performance_report,
                              get_performance_stats, save_performance,
                              reset_performance, optimize_agent,
                              benchmark, benchmark_report,
                              startup_benchmark, _MISSING,
                              DEFAULT_LRU_SIZE, PERF_FILE)


def _reset_monitor():
    """重置PerformanceMonitor单例"""
    PerformanceMonitor.reset_instance()


class TestPerformanceMonitor(unittest.TestCase):
    """测试PerformanceMonitor"""

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test01_singleton(self):
        """单例模式"""
        m1 = PerformanceMonitor()
        m2 = PerformanceMonitor()
        self.assertIs(m1, m2)

    def test02_start_stop(self):
        """start/stop计时"""
        m = PerformanceMonitor()
        m.start("op1")
        time.sleep(0.01)
        elapsed = m.stop("op1")
        self.assertGreater(elapsed, 0)
        stats = m.get_stats("op1")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["total"], 0)

    def test03_measure_context(self):
        """measure上下文管理器"""
        m = PerformanceMonitor()
        with m.measure("ctx_op"):
            time.sleep(0.01)
        stats = m.get_stats("ctx_op")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg"], 0)

    def test04_multiple_calls(self):
        """多次调用统计"""
        m = PerformanceMonitor()
        for _ in range(5):
            with m.measure("multi"):
                time.sleep(0.001)
        stats = m.get_stats("multi")
        self.assertEqual(stats["count"], 5)
        self.assertGreater(stats["total"], 0)
        self.assertGreater(stats["avg"], 0)

    def test05_stats_empty(self):
        """空统计"""
        m = PerformanceMonitor()
        stats = m.get_stats("nonexistent")
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["total"], 0.0)

    def test06_stats_fields(self):
        """统计字段完整性"""
        m = PerformanceMonitor()
        with m.measure("op"):
            pass
        stats = m.get_stats("op")
        for field in ["count", "total", "avg", "min", "max", "last"]:
            self.assertIn(field, stats)

    def test07_get_all_stats(self):
        """获取所有统计"""
        m = PerformanceMonitor()
        with m.measure("op1"):
            pass
        with m.measure("op2"):
            pass
        all_stats = m.get_all_stats()
        self.assertIn("op1", all_stats)
        self.assertIn("op2", all_stats)

    def test08_reset(self):
        """重置"""
        m = PerformanceMonitor()
        with m.measure("op"):
            pass
        m.reset()
        stats = m.get_stats("op")
        self.assertEqual(stats["count"], 0)

    def test09_get_report(self):
        """获取报告"""
        m = PerformanceMonitor()
        with m.measure("op"):
            pass
        report = m.get_report()
        self.assertIn("性能监控报告", report)
        self.assertIn("op", report)

    def test10_get_report_empty(self):
        """空报告"""
        m = PerformanceMonitor()
        report = m.get_report()
        self.assertIn("暂无数据", report)

    def test11_stop_without_start(self):
        """未start就stop"""
        m = PerformanceMonitor()
        elapsed = m.stop("not_started")
        self.assertEqual(elapsed, 0.0)

    def test12_save_to_file(self):
        """保存到文件"""
        m = PerformanceMonitor()
        with m.measure("op"):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "perf.json")
            saved_path = m.save(path)
            self.assertEqual(saved_path, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("operations", data)
            self.assertIn("op", data["operations"])

    def test13_save_default_path(self):
        """保存到默认路径"""
        m = PerformanceMonitor()
        with m.measure("op"):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("seva.performance.PERF_FILE",
                       Path(tmpdir) / "perf.json"):
                saved_path = m.save()
                self.assertTrue(os.path.exists(saved_path))


class TestProfileOperation(unittest.TestCase):
    """测试profile_operation装饰器"""

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test14_decorator_basic(self):
        """装饰器基本功能"""
        @profile_operation
        def my_func():
            return 42

        result = my_func()
        self.assertEqual(result, 42)
        # 应该记录了性能数据
        stats = PerformanceMonitor().get_all_stats()
        # 找到包含my_func的操作
        found = any("my_func" in name for name in stats.keys())
        self.assertTrue(found)

    def test15_decorator_preserves_metadata(self):
        """装饰器保留函数元数据"""
        @profile_operation
        def my_func():
            """docstring"""
            return 0

        self.assertEqual(my_func.__name__, "my_func")
        self.assertEqual(my_func.__doc__, "docstring")

    def test16_decorator_with_args(self):
        """带参数的函数"""
        @profile_operation
        def add(a, b):
            return a + b

        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(3, 4), 7)


class TestLazyLoader(unittest.TestCase):
    """测试LazyLoader"""

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test17_lazy_load(self):
        """延迟加载"""
        call_count = [0]

        def loader():
            call_count[0] += 1
            return "loaded"

        lz = LazyLoader("test", loader)
        self.assertFalse(lz.is_loaded)
        self.assertEqual(call_count[0], 0)

        # 第一次访问触发加载
        value = lz.value
        self.assertEqual(value, "loaded")
        self.assertTrue(lz.is_loaded)
        self.assertEqual(call_count[0], 1)

        # 第二次访问不重新加载
        value = lz.value
        self.assertEqual(call_count[0], 1)

    def test18_preload(self):
        """预加载"""
        loaded = [False]

        def loader():
            loaded[0] = True
            return "data"

        lz = LazyLoader("test", loader)
        self.assertFalse(loaded[0])
        lz.preload()
        self.assertTrue(loaded[0])
        self.assertTrue(lz.is_loaded)

    def test19_reset(self):
        """重置后重新加载"""
        call_count = [0]

        def loader():
            call_count[0] += 1
            return call_count[0]

        lz = LazyLoader("test", loader)
        self.assertEqual(lz.value, 1)
        lz.reset()
        self.assertFalse(lz.is_loaded)
        self.assertEqual(lz.value, 2)

    def test20_name_property(self):
        """name属性"""
        lz = LazyLoader("myname", lambda: None)
        self.assertEqual(lz.name, "myname")

    def test21_lazy_attribute_descriptor(self):
        """LazyAttribute描述符"""
        class MyClass:
            def __init__(self):
                self._compute_count = 0

            heavy = LazyAttribute("heavy",
                                  lambda self: self._compute())

            def _compute(self):
                self._compute_count += 1
                return f"computed_{self._compute_count}"

        obj = MyClass()
        self.assertEqual(obj.heavy, "computed_1")
        # 第二次访问不重新计算
        self.assertEqual(obj.heavy, "computed_1")
        self.assertEqual(obj._compute_count, 1)


class TestStartupProfiler(unittest.TestCase):
    """测试StartupProfiler"""

    def test22_start_end(self):
        """start/end"""
        profiler = StartupProfiler()
        profiler.start()
        time.sleep(0.01)
        total = profiler.end()
        self.assertGreater(total, 0)

    def test23_checkpoint(self):
        """checkpoint"""
        profiler = StartupProfiler()
        profiler.start()
        profiler.checkpoint("step1")
        time.sleep(0.01)
        profiler.checkpoint("step2")
        profiler.end()
        steps = profiler.get_steps()
        self.assertEqual(len(steps), 3)  # step1, step2, _end
        self.assertEqual(steps[0]["name"], "step1")
        self.assertEqual(steps[1]["name"], "step2")

    def test24_total_time_property(self):
        """total_time属性"""
        profiler = StartupProfiler()
        self.assertEqual(profiler.total_time, 0.0)
        profiler.start()
        time.sleep(0.01)
        t1 = profiler.total_time
        profiler.end()
        t2 = profiler.total_time
        self.assertGreater(t1, 0)
        self.assertGreaterEqual(t2, t1)

    def test25_get_slowest(self):
        """get_slowest"""
        profiler = StartupProfiler()
        profiler.start()
        profiler.checkpoint("fast")
        time.sleep(0.02)
        profiler.checkpoint("slow")
        profiler.end()
        slowest = profiler.get_slowest(2)
        self.assertEqual(len(slowest), 2)
        # 最慢的应该在前
        self.assertGreaterEqual(slowest[0]["duration"],
                                slowest[1]["duration"])

    def test26_report(self):
        """report"""
        profiler = StartupProfiler()
        profiler.start()
        profiler.checkpoint("step1")
        profiler.end()
        report = profiler.report()
        self.assertIn("启动剖析报告", report)
        self.assertIn("总耗时", report)
        self.assertIn("step1", report)

    def test27_report_not_started(self):
        """未开始的报告"""
        profiler = StartupProfiler()
        report = profiler.report()
        self.assertIn("尚未开始", report)

    def test28_checkpoint_before_start(self):
        """start之前调用checkpoint"""
        profiler = StartupProfiler()
        with self.assertRaises(RuntimeError):
            profiler.checkpoint("step")

    def test29_end_before_start(self):
        """start之前调用end"""
        profiler = StartupProfiler()
        with self.assertRaises(RuntimeError):
            profiler.end()


class TestLRUCache(unittest.TestCase):
    """测试LRUCache"""

    def test30_basic_get_set(self):
        """基本get/set"""
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)
        self.assertIsNone(cache.get("c"))

    def test31_default_maxsize(self):
        """默认大小"""
        cache = LRUCache()
        self.assertEqual(cache.maxsize, DEFAULT_LRU_SIZE)

    def test32_lru_eviction(self):
        """LRU淘汰"""
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        # 访问a，使b成为最久未使用
        cache.get("a")
        cache.set("c", 3)
        # b应该被淘汰
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    def test33_get_or_compute(self):
        """get_or_compute"""
        cache = LRUCache(maxsize=10)
        call_count = [0]

        def factory():
            call_count[0] += 1
            return "computed"

        # 第一次计算
        v1 = cache.get_or_compute("key", factory)
        self.assertEqual(v1, "computed")
        self.assertEqual(call_count[0], 1)
        # 第二次从缓存取
        v2 = cache.get_or_compute("key", factory)
        self.assertEqual(v2, "computed")
        self.assertEqual(call_count[0], 1)

    def test34_stats(self):
        """stats"""
        cache = LRUCache(maxsize=10)
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["maxsize"], 10)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test35_clear(self):
        """clear"""
        cache = LRUCache(maxsize=10)
        cache.set("a", 1)
        cache.clear()
        self.assertEqual(len(cache), 0)
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

    def test36_contains(self):
        """__contains__"""
        cache = LRUCache(maxsize=10)
        cache.set("a", 1)
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)

    def test37_len(self):
        """__len__"""
        cache = LRUCache(maxsize=10)
        self.assertEqual(len(cache), 0)
        cache.set("a", 1)
        self.assertEqual(len(cache), 1)

    def test38_overwrite_existing(self):
        """覆盖已存在的key"""
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("a", 2)
        self.assertEqual(cache.get("a"), 2)
        self.assertEqual(len(cache), 1)


class TestMemoryTracker(unittest.TestCase):
    """测试MemoryTracker"""

    def test39_snapshot(self):
        """snapshot"""
        tracker = MemoryTracker()
        snap = tracker.snapshot("before")
        self.assertEqual(snap["name"], "before")
        self.assertIn("timestamp", snap)
        self.assertIn("process_rss", snap)

    def test40_get_snapshot(self):
        """get_snapshot"""
        tracker = MemoryTracker()
        tracker.snapshot("s1")
        snap = tracker.get_snapshot("s1")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["name"], "s1")
        self.assertIsNone(tracker.get_snapshot("nonexistent"))

    def test41_diff(self):
        """diff"""
        tracker = MemoryTracker()
        tracker.snapshot("a")
        time.sleep(0.01)
        tracker.snapshot("b")
        diff = tracker.diff("a", "b")
        self.assertEqual(diff["from"], "a")
        self.assertEqual(diff["to"], "b")
        self.assertGreater(diff["time_diff"], 0)

    def test42_diff_missing(self):
        """diff快照不存在"""
        tracker = MemoryTracker()
        diff = tracker.diff("x", "y")
        self.assertIn("error", diff)

    def test43_diff_report(self):
        """diff_report"""
        tracker = MemoryTracker()
        tracker.snapshot("a")
        tracker.snapshot("b")
        report = tracker.diff_report("a", "b")
        self.assertIn("内存差异", report)
        self.assertIn("时间差", report)

    def test44_clear(self):
        """clear"""
        tracker = MemoryTracker()
        tracker.snapshot("a")
        tracker.clear()
        self.assertIsNone(tracker.get_snapshot("a"))


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test45_get_performance_report(self):
        """get_performance_report"""
        monitor = PerformanceMonitor()
        with monitor.measure("op"):
            pass
        report = get_performance_report()
        self.assertIn("性能监控报告", report)

    def test46_get_performance_stats(self):
        """get_performance_stats"""
        monitor = PerformanceMonitor()
        with monitor.measure("op"):
            pass
        stats = get_performance_stats()
        self.assertIn("op", stats)

    def test47_save_performance(self):
        """save_performance"""
        monitor = PerformanceMonitor()
        with monitor.measure("op"):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_performance(os.path.join(tmpdir, "p.json"))
            self.assertTrue(os.path.exists(path))

    def test48_reset_performance(self):
        """reset_performance"""
        monitor = PerformanceMonitor()
        with monitor.measure("op"):
            pass
        reset_performance()
        stats = get_performance_stats()
        self.assertEqual(len(stats), 0)

    def test49_optimize_agent(self):
        """optimize_agent"""
        # 使用真实对象而非MagicMock，避免auto-attribute干扰hasattr检查
        class FakeConfig:
            log_level = "DEBUG"
            steps = 10
            enable_metacognition = True

        class FakeMemory:
            episodes = list(range(2000))
            max_episodes = 10000

        class FakeAgent:
            config = FakeConfig()
            memory = FakeMemory()

        agent = FakeAgent()
        result = optimize_agent(agent)
        self.assertIn("suggestions", result)
        self.assertIn("applied", result)
        self.assertGreater(result["suggestion_count"], 0)
        self.assertGreater(result["applied_count"], 0)
        # 应该添加了缓存和监控
        self.assertTrue(hasattr(agent, "_perf_cache"))
        self.assertTrue(hasattr(agent, "_perf_monitor"))

    def test50_optimize_agent_no_config(self):
        """optimize_agent无config"""
        agent = MagicMock(spec=[])  # 没有任何属性
        result = optimize_agent(agent)
        self.assertEqual(result["suggestion_count"], 0)
        self.assertGreaterEqual(result["applied_count"], 0)

    def test51_benchmark(self):
        """benchmark"""
        def func():
            return sum(range(100))

        result = benchmark(func, iterations=10, warmup=2)
        self.assertEqual(result["iterations"], 10)
        self.assertGreater(result["total_time"], 0)
        self.assertGreater(result["avg_time"], 0)
        self.assertGreater(result["throughput"], 0)

    def test52_benchmark_report(self):
        """benchmark_report"""
        def func():
            return 1

        report = benchmark_report(func, "test_func", iterations=5, warmup=1)
        self.assertIn("基准测试", report)
        self.assertIn("test_func", report)
        self.assertIn("吞吐量", report)

    def test53_startup_benchmark(self):
        """startup_benchmark"""
        result = startup_benchmark(preset="easy", env_type="mixed", seed=42)
        self.assertIn("total_time", result)
        self.assertGreater(result["total_time"], 0)
        self.assertIn("steps", result)
        self.assertEqual(result["preset"], "easy")
        self.assertEqual(result["env_type"], "mixed")


class TestPackageExports(unittest.TestCase):
    """测试包导出"""

    def test54_performance_monitor_exported(self):
        self.assertTrue(hasattr(seva, "PerformanceMonitor"))

    def test55_lazy_loader_exported(self):
        self.assertTrue(hasattr(seva, "LazyLoader"))

    def test56_startup_profiler_exported(self):
        self.assertTrue(hasattr(seva, "StartupProfiler"))

    def test57_lru_cache_exported(self):
        self.assertTrue(hasattr(seva, "LRUCache"))

    def test58_memory_tracker_exported(self):
        self.assertTrue(hasattr(seva, "MemoryTracker"))

    def test59_version_updated(self):
        self.assertGreaterEqual(seva.__version__, "0.44.0")

    def test60_all_exports(self):
        for name in ["PerformanceMonitor", "LazyLoader", "LazyAttribute",
                     "StartupProfiler", "LRUCache", "MemoryTracker",
                     "profile_operation", "get_performance_report",
                     "get_performance_stats", "save_performance",
                     "reset_performance", "optimize_agent",
                     "benchmark", "benchmark_report",
                     "startup_benchmark"]:
            self.assertIn(name, seva.__all__,
                          f"{name} 未在__all__中")


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def _run_cli(self, *args):
        """运行CLI并返回退出码"""
        from seva.cli import main
        old_argv = sys.argv
        sys.argv = ["seva"] + list(args)
        try:
            return main()
        finally:
            sys.argv = old_argv

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test61_cli_perf_default(self):
        """seva perf 默认显示报告"""
        code = self._run_cli("perf")
        self.assertEqual(code, 0)

    def test62_cli_perf_report(self):
        """seva perf --report"""
        code = self._run_cli("perf", "--report")
        self.assertEqual(code, 0)

    def test63_cli_perf_reset(self):
        """seva perf --reset"""
        code = self._run_cli("perf", "--reset")
        self.assertEqual(code, 0)

    def test64_cli_perf_stats_empty(self):
        """seva perf --stats 空数据"""
        code = self._run_cli("perf", "--stats")
        self.assertEqual(code, 0)

    def test65_cli_perf_save(self):
        """seva perf --save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "perf.json")
            code = self._run_cli("perf", "--save", path)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path))

    def test66_cli_perf_startup(self):
        """seva perf --startup"""
        code = self._run_cli("perf", "--startup", "--preset", "easy")
        self.assertEqual(code, 0)

    def test67_cli_perf_optimize(self):
        """seva perf --optimize"""
        code = self._run_cli("perf", "--optimize", "--preset", "easy")
        self.assertEqual(code, 0)


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def setUp(self):
        _reset_monitor()

    def tearDown(self):
        _reset_monitor()

    def test68_full_monitoring_flow(self):
        """完整监控流程"""
        monitor = PerformanceMonitor()

        # 模拟多个操作
        for i in range(3):
            with monitor.measure(f"operation_{i}"):
                time.sleep(0.001)

        # 获取统计
        all_stats = monitor.get_all_stats()
        self.assertEqual(len(all_stats), 3)

        # 生成报告
        report = monitor.get_report()
        self.assertIn("operation_0", report)
        self.assertIn("operation_1", report)
        self.assertIn("operation_2", report)

        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            path = monitor.save(os.path.join(tmpdir, "perf.json"))
            self.assertTrue(os.path.exists(path))

    def test69_lazy_loader_with_monitor(self):
        """LazyLoader与监控集成"""
        lz = LazyLoader("expensive", lambda: [1, 2, 3])
        value = lz.value
        self.assertEqual(value, [1, 2, 3])

        # 应该记录了加载时间
        stats = PerformanceMonitor().get_all_stats()
        lazy_ops = [k for k in stats.keys() if "lazy_load" in k]
        self.assertGreater(len(lazy_ops), 0)

    def test70_cache_performance(self):
        """缓存提升性能"""
        call_count = [0]

        def expensive_compute():
            call_count[0] += 1
            time.sleep(0.01)
            return call_count[0]

        cache = LRUCache(maxsize=10)

        # 不用缓存
        start = time.perf_counter()
        for _ in range(5):
            expensive_compute()
        no_cache_time = time.perf_counter() - start
        self.assertEqual(call_count[0], 5)

        # 用缓存
        call_count[0] = 0
        start = time.perf_counter()
        for _ in range(5):
            cache.get_or_compute("key", expensive_compute)
        cache_time = time.perf_counter() - start
        self.assertEqual(call_count[0], 1)  # 只计算一次

        # 缓存应该更快
        self.assertLess(cache_time, no_cache_time)

    def test71_startup_profiler_full(self):
        """完整启动剖析"""
        profiler = StartupProfiler()
        profiler.start()

        # 模拟初始化步骤
        time.sleep(0.005)
        profiler.checkpoint("config_loaded")

        time.sleep(0.005)
        profiler.checkpoint("modules_imported")

        time.sleep(0.005)
        profiler.checkpoint("agent_created")

        total = profiler.end()
        self.assertGreater(total, 0.01)

        steps = profiler.get_steps()
        self.assertEqual(len(steps), 4)  # 3个checkpoint + _end

        report = profiler.report()
        self.assertIn("config_loaded", report)
        self.assertIn("modules_imported", report)
        self.assertIn("agent_created", report)
        self.assertIn("最慢步骤", report)

    def test72_optimize_with_real_agent(self):
        """用真实Agent测试优化"""
        from seva import SEVAFactory
        agent = SEVAFactory.create(preset="easy")
        result = optimize_agent(agent)
        self.assertIsInstance(result, dict)
        self.assertIn("suggestions", result)
        self.assertIn("applied", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
