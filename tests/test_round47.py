"""
阶段57测试：性能基准与自动调优
"""
import unittest
import time
import threading

from seva.benchmark import (
    Benchmark, BenchmarkSuite, BenchmarkResult,
    TuningStrategy, TuningResult, ParamTuner,
    ABTest, ABTestResult,
    get_benchmark_suite, benchmark, add_benchmark,
    run_benchmarks, tune, ab_test,
    get_benchmark_report, get_benchmark_status,
    reset_benchmarks,
)


# ============================================================================
# 测试BenchmarkResult
# ============================================================================

class TestBenchmarkResult(unittest.TestCase):
    """测试基准测试结果"""

    def test01_creation_with_times(self):
        result = BenchmarkResult(name="test", iterations=3, times=[0.1, 0.2, 0.15])
        self.assertEqual(result.min_time, 0.1)
        self.assertEqual(result.max_time, 0.2)
        self.assertAlmostEqual(result.avg_time, 0.15)
        self.assertAlmostEqual(result.median_time, 0.15)
        self.assertGreater(result.std_dev, 0)
        self.assertAlmostEqual(result.total_time, 0.45)
        self.assertGreater(result.operations_per_second, 0)

    def test02_creation_empty(self):
        result = BenchmarkResult(name="test")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.min_time, 0.0)
        self.assertEqual(result.avg_time, 0.0)

    def test03_to_dict(self):
        result = BenchmarkResult(name="test", iterations=1, times=[0.1])
        d = result.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["iterations"], 1)

    def test04_summary(self):
        result = BenchmarkResult(name="test", iterations=3, times=[0.1, 0.2, 0.15])
        s = result.summary()
        self.assertIn("test", s)
        self.assertIn("3次", s)

    def test05_summary_with_error(self):
        result = BenchmarkResult(name="test", error="测试错误")
        s = result.summary()
        self.assertIn("错误", s)

    def test06_repr(self):
        result = BenchmarkResult(name="test", iterations=1, times=[0.1])
        r = repr(result)
        self.assertIn("BenchmarkResult", r)
        self.assertIn("test", r)


# ============================================================================
# 测试Benchmark
# ============================================================================

class TestBenchmark(unittest.TestCase):
    """测试基准测试"""

    def test01_run_success(self):
        bench = Benchmark("test", iterations=5)
        result = bench.run(lambda: sum(range(100)))
        self.assertEqual(result.iterations, 5)
        self.assertIsNone(result.error)
        self.assertGreater(result.avg_time, 0)

    def test02_run_with_args(self):
        bench = Benchmark("test", iterations=3)
        result = bench.run(lambda x, y: x + y, 1, 2)
        self.assertEqual(result.iterations, 3)
        self.assertIsNone(result.error)

    def test03_run_with_error(self):
        def fail():
            raise ValueError("失败")

        bench = Benchmark("test", iterations=3)
        result = bench.run(fail)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.iterations, 0)

    def test04_warmup(self):
        calls = [0]

        def func():
            calls[0] += 1
            return calls[0]

        bench = Benchmark("test", iterations=3, warmup=2)
        result = bench.run(func)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(calls[0], 5)  # 2 warmup + 3 iterations

    def test05_repr(self):
        bench = Benchmark("test", iterations=10)
        r = repr(bench)
        self.assertIn("Benchmark", r)
        self.assertIn("test", r)
        self.assertIn("10", r)


# ============================================================================
# 测试BenchmarkSuite
# ============================================================================

class TestBenchmarkSuite(unittest.TestCase):
    """测试基准测试套件"""

    def setUp(self):
        self.suite = BenchmarkSuite("test_suite")

    def test01_add_and_list(self):
        self.suite.add("test1", lambda: None)
        self.suite.add("test2", lambda: None)
        self.assertEqual(set(self.suite.list_tests()), {"test1", "test2"})

    def test02_remove(self):
        self.suite.add("test1", lambda: None)
        self.assertTrue(self.suite.remove("test1"))
        self.assertNotIn("test1", self.suite.list_tests())

    def test03_remove_nonexistent(self):
        self.assertFalse(self.suite.remove("nonexistent"))

    def test04_run_single(self):
        self.suite.add("test1", lambda: sum(range(100)))
        result = self.suite.run("test1")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test1")

    def test05_run_nonexistent(self):
        result = self.suite.run("nonexistent")
        self.assertIsNone(result)

    def test06_run_all(self):
        self.suite.add("test1", lambda: sum(range(100)))
        self.suite.add("test2", lambda: sorted([3, 1, 2]))
        results = self.suite.run_all()
        self.assertEqual(len(results), 2)
        self.assertIn("test1", results)
        self.assertIn("test2", results)

    def test07_chain_add(self):
        self.suite.add("test1", lambda: None).add("test2", lambda: None)
        self.assertEqual(len(self.suite.list_tests()), 2)

    def test08_set_baseline_and_check_regression(self):
        self.suite.add("test1", lambda: None)
        baseline = self.suite.run_all()
        self.suite.set_baseline(baseline)

        # 模拟回归（使用更慢的函数）
        self.suite.remove("test1")
        self.suite.add("test1", lambda: sum(range(10000)))

        current = self.suite.run_all()
        report = self.suite.check_regression(current, threshold=0.0)

        self.assertIn("test1", report)
        self.assertTrue(report["test1"]["is_regression"])

    def test09_check_regression_no_baseline(self):
        report = self.suite.check_regression({})
        self.assertEqual(report, {})

    def test10_compare(self):
        self.suite.add("test1", lambda: sum(range(100)))
        results = self.suite.run_all()
        output = self.suite.compare(results)
        self.assertIsInstance(output, str)
        self.assertIn("test1", output)
        self.assertIn("test_suite", output)

    def test11_get_report(self):
        self.suite.add("test1", lambda: None)
        report = self.suite.get_report()
        self.assertEqual(report["name"], "test_suite")
        self.assertEqual(report["test_count"], 1)

    def test12_clear(self):
        self.suite.add("test1", lambda: None)
        self.suite.clear()
        self.assertEqual(len(self.suite.list_tests()), 0)

    def test13_repr(self):
        r = repr(self.suite)
        self.assertIn("BenchmarkSuite", r)
        self.assertIn("test_suite", r)


# ============================================================================
# 测试TuningStrategy
# ============================================================================

class TestTuningStrategy(unittest.TestCase):
    """测试调优策略枚举"""

    def test01_values(self):
        self.assertEqual(TuningStrategy.GRID.value, "grid")
        self.assertEqual(TuningStrategy.RANDOM.value, "random")


# ============================================================================
# 测试TuningResult
# ============================================================================

class TestTuningResult(unittest.TestCase):
    """测试调优结果"""

    def test01_creation(self):
        result = TuningResult(
            best_params={"x": 1, "y": 2},
            best_score=0.5,
            total_trials=10,
            strategy="grid",
        )
        self.assertEqual(result.best_params, {"x": 1, "y": 2})
        self.assertEqual(result.best_score, 0.5)

    def test02_to_dict(self):
        result = TuningResult(best_params={"x": 1}, best_score=0.5, total_trials=5)
        d = result.to_dict()
        self.assertEqual(d["best_params"], {"x": 1})
        self.assertEqual(d["best_score"], 0.5)

    def test03_summary(self):
        result = TuningResult(best_params={"x": 1}, best_score=0.5,
                              total_trials=10, strategy="grid")
        s = result.summary()
        self.assertIn("grid", s)
        self.assertIn("10", s)

    def test04_repr(self):
        result = TuningResult(best_params={"x": 1}, best_score=0.5)
        r = repr(result)
        self.assertIn("TuningResult", r)


# ============================================================================
# 测试ParamTuner
# ============================================================================

class TestParamTuner(unittest.TestCase):
    """测试参数调优器"""

    def test01_grid_search(self):
        def objective(x, y):
            return -(x**2 + y**2)  # 最大化，最佳是x=0, y=0

        tuner = ParamTuner(objective, {"x": (0, 3), "y": (0, 3)}, maximize=True)
        result = tuner.tune(strategy="grid", step=1)

        self.assertEqual(result.best_params, {"x": 0, "y": 0})
        self.assertEqual(result.best_score, 0)
        self.assertEqual(result.total_trials, 16)  # 4x4

    def test02_grid_search_minimize(self):
        def objective(x):
            return x**2  # 最小化，最佳是x=0

        tuner = ParamTuner(objective, {"x": (0, 5)}, maximize=False)
        result = tuner.tune(strategy="grid", step=1)

        self.assertEqual(result.best_params, {"x": 0})
        self.assertEqual(result.best_score, 0)

    def test03_random_search(self):
        def objective(x, y):
            return -(x**2 + y**2)

        tuner = ParamTuner(objective, {"x": (0, 5), "y": (0, 5)}, maximize=True)
        result = tuner.tune(strategy="random", n_trials=20)

        self.assertEqual(result.total_trials, 20)
        self.assertLessEqual(result.best_score, 0)

    def test04_grid_with_list_values(self):
        def objective(x):
            return x

        tuner = ParamTuner(objective, {"x": [1, 3, 5, 7]}, maximize=True)
        result = tuner.tune(strategy="grid")

        self.assertEqual(result.best_params, {"x": 7})
        self.assertEqual(result.best_score, 7)

    def test05_strategy_from_string(self):
        def objective(x):
            return x

        tuner = ParamTuner(objective, {"x": (0, 2)})
        result = tuner.tune(strategy="grid", step=1)
        self.assertEqual(result.strategy, "grid")

    def test06_all_results_recorded(self):
        def objective(x):
            return x

        tuner = ParamTuner(objective, {"x": (0, 3)}, maximize=True)
        result = tuner.tune(strategy="grid", step=1)

        self.assertEqual(len(result.all_results), 4)
        for r in result.all_results:
            self.assertIn("params", r)
            self.assertIn("score", r)

    def test07_objective_error_handled(self):
        def objective(x):
            if x == 2:
                raise ValueError("错误")
            return x

        tuner = ParamTuner(objective, {"x": (0, 3)}, maximize=True)
        result = tuner.tune(strategy="grid", step=1)

        # x=2应该得到-inf分数
        self.assertEqual(result.total_trials, 4)

    def test08_repr(self):
        tuner = ParamTuner(lambda x: x, {"x": (0, 1)})
        r = repr(tuner)
        self.assertIn("ParamTuner", r)


# ============================================================================
# 测试ABTest
# ============================================================================

class TestABTest(unittest.TestCase):
    """测试A/B测试"""

    def test01_run(self):
        ab = ABTest("test_ab", iterations=5)
        result = ab.run(
            func_a=lambda: sum(range(100)),
            func_b=lambda: sum(range(1000)),
        )
        self.assertEqual(result.name, "test_ab")
        self.assertIsNotNone(result.a_result)
        self.assertIsNotNone(result.b_result)

    def test02_winner(self):
        ab = ABTest("test_ab", iterations=5)
        result = ab.run(
            func_a=lambda: sum(range(10000)),  # 慢
            func_b=lambda: sum(range(10)),     # 快
        )
        self.assertEqual(result.winner, "B")
        self.assertGreater(result.improvement, 0)
        self.assertGreater(result.improvement_percent, 0)

    def test03_to_dict(self):
        ab = ABTest("test_ab", iterations=3)
        result = ab.run(func_a=lambda: None, func_b=lambda: None)
        d = result.to_dict()
        self.assertEqual(d["name"], "test_ab")
        self.assertIn("winner", d)

    def test04_summary(self):
        ab = ABTest("test_ab", iterations=3)
        result = ab.run(func_a=lambda: None, func_b=lambda: None)
        s = result.summary()
        self.assertIn("test_ab", s)
        self.assertIn("A", s)
        self.assertIn("B", s)

    def test05_repr(self):
        ab = ABTest("test_ab", iterations=5)
        r = repr(ab)
        self.assertIn("ABTest", r)
        self.assertIn("test_ab", r)


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        reset_benchmarks()

    def tearDown(self):
        reset_benchmarks()

    def test01_get_benchmark_suite(self):
        suite = get_benchmark_suite()
        self.assertIsInstance(suite, BenchmarkSuite)

    def test02_benchmark_function(self):
        result = benchmark("test", lambda: sum(range(100)), iterations=3)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.name, "test")

    def test03_add_benchmark(self):
        add_benchmark("test", lambda: None)
        suite = get_benchmark_suite()
        self.assertIn("test", suite.list_tests())

    def test04_run_benchmarks(self):
        add_benchmark("test1", lambda: sum(range(100)))
        add_benchmark("test2", lambda: sorted([3, 1, 2]))
        results = run_benchmarks()
        self.assertEqual(len(results), 2)

    def test05_tune_function(self):
        def objective(x):
            return -x**2

        result = tune(objective, {"x": (0, 3)}, strategy="grid", step=1)
        self.assertIsInstance(result, TuningResult)
        self.assertEqual(result.best_params, {"x": 0})

    def test06_ab_test_function(self):
        result = ab_test("test", lambda: None, lambda: None, iterations=3)
        self.assertIsInstance(result, ABTestResult)

    def test07_get_benchmark_report(self):
        add_benchmark("test", lambda: sum(range(100)))
        report = get_benchmark_report()
        self.assertIsInstance(report, str)

    def test08_get_benchmark_status(self):
        status = get_benchmark_status()
        self.assertIn("name", status)

    def test09_reset_benchmarks(self):
        add_benchmark("test", lambda: None)
        reset_benchmarks()
        suite = get_benchmark_suite()
        self.assertEqual(len(suite.list_tests()), 0)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_benchmarks()

    def tearDown(self):
        reset_benchmarks()

    def test01_full_benchmark_workflow(self):
        """完整基准测试工作流"""
        suite = get_benchmark_suite()
        suite.add("sort", lambda: sorted([3, 1, 4, 1, 5, 9, 2, 6] * 100))
        suite.add("reverse", lambda: list(reversed([1, 2, 3, 4, 5] * 100)))

        # 运行并设置基线
        results = suite.run_all()
        suite.set_baseline(results)

        # 再次运行检查回归
        current = suite.run_all()
        report = suite.check_regression(current, threshold=0.5)

        self.assertEqual(len(results), 2)
        for name in ["sort", "reverse"]:
            self.assertIn(name, report)

    def test02_tuning_optimization(self):
        """调优优化"""
        # 寻找二次函数的最小值
        def objective(x, y):
            return (x - 3)**2 + (y - 5)**2

        tuner = ParamTuner(objective, {"x": (0, 6), "y": (0, 10)}, maximize=False)
        result = tuner.tune(strategy="grid", step=1)

        self.assertEqual(result.best_params, {"x": 3, "y": 5})
        self.assertEqual(result.best_score, 0)

    def test03_ab_test_comparison(self):
        """A/B测试对比"""
        result = ab_test(
            "list_vs_set",
            func_a=lambda: [i for i in range(100) if i in list(range(50))],
            func_b=lambda: [i for i in range(100) if i in set(range(50))],
            iterations=10,
        )

        self.assertIn(result.winner, ["A", "B", "tie"])
        self.assertIsNotNone(result.a_result)
        self.assertIsNotNone(result.b_result)

    def test04_thread_safety(self):
        """线程安全测试"""
        suite = BenchmarkSuite("thread_test")
        errors = []

        def worker(i):
            try:
                suite.add(f"test_{i}", lambda: sum(range(100)))
                suite.run(f"test_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(suite.list_tests()), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
