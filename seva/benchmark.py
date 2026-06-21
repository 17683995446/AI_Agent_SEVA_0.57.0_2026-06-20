"""
性能基准与自动调优（阶段57）
对应第一性原理：基准 = 量化测量性能；调优 = 系统化搜索最优参数组合

提供能力:
    - Benchmark: 基准测试（测量函数执行时间、内存、统计）
    - BenchmarkSuite: 基准测试套件（批量运行、对比、回归检测）
    - BenchmarkResult: 基准测试结果（统计信息）
    - ParamTuner: 参数调优器（网格搜索、随机搜索）
    - TuningResult: 调优结果
    - ABTest: A/B测试（对比两个方案）
    - 便捷函数: benchmark / tune / ab_test / get_benchmark_report 等

使用示例:
    from seva import Benchmark, ParamTuner

    # 基准测试
    bench = Benchmark("sort_test")
    result = bench.run(lambda: sorted([3, 1, 4, 1, 5, 9, 2, 6]))

    # 参数调优
    def objective(x, y):
        return -(x**2 + y**2)  # 最大化

    tuner = ParamTuner(objective, {"x": (0, 10), "y": (0, 10)})
    result = tuner.tune(strategy="grid")
"""
from __future__ import annotations

import os
import sys
import time
import json
import statistics
import threading
import functools
import random
import itertools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
BENCHMARK_FILE = SEVA_HOME / "benchmarks.json"


# ============================================================================
# BenchmarkResult - 基准测试结果
# ============================================================================

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int = 0
    times: List[float] = field(default_factory=list)
    min_time: float = 0.0
    max_time: float = 0.0
    avg_time: float = 0.0
    median_time: float = 0.0
    std_dev: float = 0.0
    total_time: float = 0.0
    operations_per_second: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.times:
            self.min_time = min(self.times)
            self.max_time = max(self.times)
            self.avg_time = statistics.mean(self.times)
            self.median_time = statistics.median(self.times)
            self.std_dev = statistics.stdev(self.times) if len(self.times) > 1 else 0.0
            self.total_time = sum(self.times)
            if self.avg_time > 0:
                self.operations_per_second = 1.0 / self.avg_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "total_time": self.total_time,
            "operations_per_second": self.operations_per_second,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """摘要字符串"""
        if self.error:
            return f"[{self.name}] 错误: {self.error}"
        return (f"[{self.name}] {self.iterations}次, "
                f"平均{self.avg_time*1000:.3f}ms, "
                f"最小{self.min_time*1000:.3f}ms, "
                f"最大{self.max_time*1000:.3f}ms, "
                f"{self.operations_per_second:.1f} ops/s")

    def __repr__(self) -> str:
        return f"BenchmarkResult(name={self.name!r}, avg={self.avg_time*1000:.3f}ms)"


# ============================================================================
# Benchmark - 基准测试
# ============================================================================

class Benchmark:
    """
    基准测试：测量函数执行性能

    使用示例:
        bench = Benchmark("my_test", iterations=100)
        result = bench.run(lambda: sorted([3, 1, 4, 1, 5]))
        print(result.summary())
    """

    def __init__(self, name: str, iterations: int = 10, warmup: int = 0):
        """
        Args:
            name: 基准测试名称
            iterations: 迭代次数
            warmup: 预热次数（不计入结果）
        """
        self.name = name
        self.iterations = max(1, iterations)
        self.warmup = max(0, warmup)

    def run(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        运行基准测试

        Args:
            func: 要测试的函数
            *args, **kwargs: 函数参数

        Returns:
            基准测试结果
        """
        # 预热
        for _ in range(self.warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass

        # 正式测试
        times: List[float] = []
        error: Optional[str] = None

        for _ in range(self.iterations):
            try:
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                error = str(e)
                break

        return BenchmarkResult(
            name=self.name,
            iterations=len(times),
            times=times,
            error=error,
        )

    def run_async(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """运行异步函数的基准测试"""
        import asyncio

        async def run_once():
            start = time.perf_counter()
            await func(*args, **kwargs)
            return time.perf_counter() - start

        # 预热
        for _ in range(self.warmup):
            try:
                asyncio.run(run_once())
            except Exception:
                pass

        times: List[float] = []
        error: Optional[str] = None

        for _ in range(self.iterations):
            try:
                elapsed = asyncio.run(run_once())
                times.append(elapsed)
            except Exception as e:
                error = str(e)
                break

        return BenchmarkResult(
            name=self.name,
            iterations=len(times),
            times=times,
            error=error,
        )

    def __repr__(self) -> str:
        return f"Benchmark(name={self.name!r}, iterations={self.iterations})"


# ============================================================================
# BenchmarkSuite - 基准测试套件
# ============================================================================

class BenchmarkSuite:
    """
    基准测试套件：管理多个基准测试，批量运行、对比、回归检测

    使用示例:
        suite = BenchmarkSuite("my_suite")
        suite.add("sort", lambda: sorted([3, 1, 4]))
        suite.add("reverse", lambda: list(reversed([1, 2, 3])))
        results = suite.run_all()
        suite.compare(results)
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._benchmarks: Dict[str, Tuple[Callable, tuple, dict, int, int]] = {}
        self._baseline: Optional[Dict[str, BenchmarkResult]] = None
        self._lock = threading.RLock()

    def add(self, name: str, func: Callable, *args,
            iterations: int = 10, warmup: int = 0, **kwargs) -> "BenchmarkSuite":
        """添加基准测试"""
        with self._lock:
            self._benchmarks[name] = (func, args, kwargs, iterations, warmup)
        return self

    def remove(self, name: str) -> bool:
        """移除基准测试"""
        with self._lock:
            if name in self._benchmarks:
                del self._benchmarks[name]
                return True
            return False

    def list_tests(self) -> List[str]:
        """列出所有测试名"""
        with self._lock:
            return list(self._benchmarks.keys())

    def run(self, name: str) -> Optional[BenchmarkResult]:
        """运行单个测试"""
        with self._lock:
            if name not in self._benchmarks:
                return None
            func, args, kwargs, iterations, warmup = self._benchmarks[name]

        bench = Benchmark(name, iterations=iterations, warmup=warmup)
        return bench.run(func, *args, **kwargs)

    def run_all(self) -> Dict[str, BenchmarkResult]:
        """运行所有测试"""
        results: Dict[str, BenchmarkResult] = {}
        with self._lock:
            names = list(self._benchmarks.keys())

        for name in names:
            result = self.run(name)
            if result:
                results[name] = result

        return results

    def set_baseline(self, results: Dict[str, BenchmarkResult]) -> None:
        """设置基线结果（用于回归检测）"""
        with self._lock:
            self._baseline = dict(results)

    def check_regression(self, results: Dict[str, BenchmarkResult],
                         threshold: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        检查性能回归

        Args:
            results: 当前结果
            threshold: 回归阈值（如0.2表示慢20%算回归）

        Returns:
            回归报告 {test_name: {baseline, current, change, is_regression}}
        """
        if self._baseline is None:
            return {}

        report: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            baseline = dict(self._baseline)

        for name, result in results.items():
            if name not in baseline:
                continue

            base_avg = baseline[name].avg_time
            curr_avg = result.avg_time

            if base_avg > 0:
                change = (curr_avg - base_avg) / base_avg
            else:
                change = 0.0

            report[name] = {
                "baseline_avg": base_avg,
                "current_avg": curr_avg,
                "change": change,
                "change_percent": change * 100,
                "is_regression": change > threshold,
                "is_improvement": change < -threshold,
            }

        return report

    def compare(self, results: Dict[str, BenchmarkResult]) -> str:
        """对比结果，返回格式化字符串"""
        lines = []
        lines.append(f"基准测试套件: {self.name}")
        lines.append("=" * 70)
        lines.append(f"{'名称':<20} {'迭代':<6} {'平均(ms)':<12} {'最小(ms)':<12} {'ops/s':<12}")
        lines.append("-" * 70)

        for name, result in results.items():
            lines.append(
                f"{name:<20} {result.iterations:<6} "
                f"{result.avg_time*1000:<12.3f} "
                f"{result.min_time*1000:<12.3f} "
                f"{result.operations_per_second:<12.1f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_report(self) -> Dict[str, Any]:
        """获取套件报告"""
        with self._lock:
            return {
                "name": self.name,
                "test_count": len(self._benchmarks),
                "tests": list(self._benchmarks.keys()),
                "has_baseline": self._baseline is not None,
            }

    def clear(self) -> None:
        """清空所有测试"""
        with self._lock:
            self._benchmarks.clear()
            self._baseline = None

    def __repr__(self) -> str:
        return f"BenchmarkSuite(name={self.name!r}, tests={len(self._benchmarks)})"


# ============================================================================
# TuningStrategy - 调优策略枚举
# ============================================================================

class TuningStrategy(Enum):
    """调优策略"""
    GRID = "grid"        # 网格搜索
    RANDOM = "random"    # 随机搜索


# ============================================================================
# TuningResult - 调优结果
# ============================================================================

@dataclass
class TuningResult:
    """调优结果"""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    total_trials: int = 0
    strategy: str = ""
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "total_trials": self.total_trials,
            "strategy": self.strategy,
            "duration": self.duration,
            "all_results": self.all_results,
        }

    def summary(self) -> str:
        """摘要"""
        return (f"调优结果 ({self.strategy}): {self.total_trials}次试验, "
                f"最佳分数={self.best_score:.4f}, "
                f"最佳参数={self.best_params}, "
                f"耗时{self.duration:.2f}s")

    def __repr__(self) -> str:
        return f"TuningResult(best_score={self.best_score:.4f}, params={self.best_params})"


# ============================================================================
# ParamTuner - 参数调优器
# ============================================================================

class ParamTuner:
    """
    参数调优器：搜索最优参数组合

    支持策略:
        - GRID: 网格搜索（穷举所有组合）
        - RANDOM: 随机搜索（随机采样）

    使用示例:
        def objective(x, y):
            return -(x**2 + y**2)  # 最大化（越接近0越好）

        tuner = ParamTuner(objective, {"x": (0, 10), "y": (0, 10)})
        result = tuner.tune(strategy="grid", step=2)
        print(result.best_params)  # {"x": 0, "y": 0}
    """

    def __init__(self, objective: Callable, param_space: Dict[str, Tuple],
                 maximize: bool = True):
        """
        Args:
            objective: 目标函数，签名为 objective(**params) -> float
            param_space: 参数空间 {param_name: (min, max)} 或 {param_name: [values]}
            maximize: 是否最大化目标函数
        """
        self.objective = objective
        self.param_space = param_space
        self.maximize = maximize

    def tune(self, strategy: Union[TuningStrategy, str] = TuningStrategy.GRID,
             step: int = 1, n_trials: int = 10) -> TuningResult:
        """
        执行调优

        Args:
            strategy: 调优策略
            step: 网格搜索的步长（仅GRID模式）
            n_trials: 随机搜索的试验次数（仅RANDOM模式）

        Returns:
            调优结果
        """
        if isinstance(strategy, str):
            strategy = TuningStrategy(strategy)

        start_time = time.time()

        if strategy == TuningStrategy.GRID:
            result = self._grid_search(step=step)
        else:
            result = self._random_search(n_trials=n_trials)

        result.strategy = strategy.value
        result.duration = time.time() - start_time
        return result

    def _grid_search(self, step: int = 1) -> TuningResult:
        """网格搜索"""
        # 生成参数网格
        param_names = list(self.param_space.keys())
        param_values: List[List[Any]] = []

        for name in param_names:
            space = self.param_space[name]
            if isinstance(space, tuple) and len(space) == 2:
                # (min, max) 范围
                values = list(range(space[0], space[1] + 1, max(1, step)))
                param_values.append(values)
            elif isinstance(space, list):
                param_values.append(space)
            else:
                param_values.append([space])

        all_results: List[Dict[str, Any]] = []
        best_score = float("-inf") if self.maximize else float("inf")
        best_params: Dict[str, Any] = {}

        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            try:
                score = self.objective(**params)
            except Exception:
                score = float("-inf") if self.maximize else float("inf")

            all_results.append({"params": params, "score": score})

            if self._is_better(score, best_score):
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            total_trials=len(all_results),
        )

    def _random_search(self, n_trials: int = 10) -> TuningResult:
        """随机搜索"""
        param_names = list(self.param_space.keys())
        all_results: List[Dict[str, Any]] = []
        best_score = float("-inf") if self.maximize else float("inf")
        best_params: Dict[str, Any] = {}

        for _ in range(n_trials):
            params: Dict[str, Any] = {}
            for name in param_names:
                space = self.param_space[name]
                if isinstance(space, tuple) and len(space) == 2:
                    params[name] = random.randint(space[0], space[1])
                elif isinstance(space, list):
                    params[name] = random.choice(space)
                else:
                    params[name] = space

            try:
                score = self.objective(**params)
            except Exception:
                score = float("-inf") if self.maximize else float("inf")

            all_results.append({"params": params, "score": score})

            if self._is_better(score, best_score):
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            total_trials=len(all_results),
        )

    def _is_better(self, score: float, best: float) -> bool:
        """判断score是否比best更好"""
        if self.maximize:
            return score > best
        return score < best

    def __repr__(self) -> str:
        return (f"ParamTuner(params={list(self.param_space.keys())}, "
                f"maximize={self.maximize})")


# ============================================================================
# ABTest - A/B测试
# ============================================================================

@dataclass
class ABTestResult:
    """A/B测试结果"""
    name: str
    a_result: BenchmarkResult
    b_result: BenchmarkResult
    winner: str = ""
    improvement: float = 0.0
    improvement_percent: float = 0.0

    def __post_init__(self):
        a_avg = self.a_result.avg_time
        b_avg = self.b_result.avg_time

        if a_avg > 0:
            self.improvement = a_avg - b_avg
            self.improvement_percent = (self.improvement / a_avg) * 100

        if b_avg < a_avg:
            self.winner = "B"
        elif a_avg < b_avg:
            self.winner = "A"
        else:
            self.winner = "tie"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "a_avg": self.a_result.avg_time,
            "b_avg": self.b_result.avg_time,
            "winner": self.winner,
            "improvement": self.improvement,
            "improvement_percent": self.improvement_percent,
        }

    def summary(self) -> str:
        return (f"A/B测试 '{self.name}': A={self.a_result.avg_time*1000:.3f}ms, "
                f"B={self.b_result.avg_time*1000:.3f}ms, "
                f"胜者={self.winner}, "
                f"提升={self.improvement_percent:.1f}%")


class ABTest:
    """
    A/B测试：对比两个方案的性能

    使用示例:
        ab = ABTest("sort_vs_sorted")
        result = ab.run(
            func_a=lambda: sorted([3, 1, 4]),
            func_b=lambda: list(reversed(sorted([3, 1, 4]))),
        )
        print(result.summary())
    """

    def __init__(self, name: str, iterations: int = 10):
        self.name = name
        self.iterations = iterations

    def run(self, func_a: Callable, func_b: Callable,
            *args, **kwargs) -> ABTestResult:
        """运行A/B测试"""
        bench_a = Benchmark(f"{self.name}_A", iterations=self.iterations)
        bench_b = Benchmark(f"{self.name}_B", iterations=self.iterations)

        result_a = bench_a.run(func_a, *args, **kwargs)
        result_b = bench_b.run(func_b, *args, **kwargs)

        return ABTestResult(
            name=self.name,
            a_result=result_a,
            b_result=result_b,
        )

    def __repr__(self) -> str:
        return f"ABTest(name={self.name!r}, iterations={self.iterations})"


# ============================================================================
# 便捷函数
# ============================================================================

_benchmark_suite: Optional[BenchmarkSuite] = None
_suite_lock = threading.Lock()


def get_benchmark_suite() -> BenchmarkSuite:
    """获取全局基准测试套件"""
    global _benchmark_suite
    if _benchmark_suite is None:
        with _suite_lock:
            if _benchmark_suite is None:
                _benchmark_suite = BenchmarkSuite("global")
    return _benchmark_suite


def benchmark(name: str, func: Callable, *args,
              iterations: int = 10, **kwargs) -> BenchmarkResult:
    """运行基准测试"""
    bench = Benchmark(name, iterations=iterations)
    return bench.run(func, *args, **kwargs)


def add_benchmark(name: str, func: Callable, *args,
                  iterations: int = 10, **kwargs) -> "BenchmarkSuite":
    """添加基准测试到全局套件"""
    suite = get_benchmark_suite()
    suite.add(name, func, *args, iterations=iterations, **kwargs)
    return suite


def run_benchmarks() -> Dict[str, BenchmarkResult]:
    """运行全局套件中的所有基准测试"""
    return get_benchmark_suite().run_all()


def tune(objective: Callable, param_space: Dict[str, Tuple],
         strategy: Union[TuningStrategy, str] = TuningStrategy.GRID,
         maximize: bool = True, **kwargs) -> TuningResult:
    """参数调优"""
    tuner = ParamTuner(objective, param_space, maximize=maximize)
    return tuner.tune(strategy=strategy, **kwargs)


def ab_test(name: str, func_a: Callable, func_b: Callable,
            iterations: int = 10, **kwargs) -> ABTestResult:
    """A/B测试"""
    test = ABTest(name, iterations=iterations)
    return test.run(func_a, func_b, **kwargs)


def get_benchmark_report() -> str:
    """获取基准测试报告"""
    suite = get_benchmark_suite()
    results = suite.run_all()
    if results:
        return suite.compare(results)
    return f"基准测试套件: {suite.name}\n(无注册测试)"


def get_benchmark_status() -> Dict[str, Any]:
    """获取基准测试状态"""
    return get_benchmark_suite().get_report()


def reset_benchmarks() -> None:
    """重置基准测试状态"""
    global _benchmark_suite
    with _suite_lock:
        if _benchmark_suite is not None:
            _benchmark_suite.clear()
        _benchmark_suite = None
