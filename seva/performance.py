"""
性能优化与启动加速（阶段44）
对应第一性原理：高效 = 快速响应、低资源占用、可见的性能指标

提供能力:
    - PerformanceMonitor: 操作耗时监控（上下文管理器 + 持久化）
    - LazyLoader: 延迟加载，按需初始化重型组件
    - StartupProfiler: 启动过程剖析，定位慢初始化
    - LRUCache: LRU缓存，避免重复计算
    - MemoryTracker: 内存占用跟踪
    - 便捷函数: profile_operation / get_performance_report / optimize_agent / benchmark

使用示例:
    from seva import PerformanceMonitor, profile_operation

    # 监控操作耗时
    with PerformanceMonitor().measure("my_operation"):
        do_something()

    # 函数装饰器
    @profile_operation
    def my_function():
        ...

    # 启动剖析
    from seva import StartupProfiler
    profiler = StartupProfiler()
    profiler.start()
    # ... 初始化代码 ...
    profiler.checkpoint("config_loaded")
    profiler.checkpoint("agent_created")
    profiler.end()
    print(profiler.report())
"""
from __future__ import annotations

import os
import sys
import json
import time
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
PERF_FILE = SEVA_HOME / "performance.json"
DEFAULT_LRU_SIZE = 128


# ============================================================================
# PerformanceMonitor - 操作耗时监控
# ============================================================================

class PerformanceMonitor:
    """
    操作耗时监控器（线程安全）

    用法:
        monitor = PerformanceMonitor()
        with monitor.measure("operation_name"):
            do_something()
        stats = monitor.get_stats("operation_name")
    """

    _instance: Optional["PerformanceMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PerformanceMonitor":
        """单例模式，全局共享一个监控器"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._operations: Dict[str, List[float]] = {}
        self._active: Dict[str, float] = {}
        self._data_lock = threading.Lock()

    def start(self, name: str) -> None:
        """开始计时"""
        with self._data_lock:
            self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """停止计时并返回耗时（秒）"""
        end_time = time.perf_counter()
        with self._data_lock:
            start_time = self._active.pop(name, None)
            if start_time is None:
                return 0.0
            elapsed = end_time - start_time
            if name not in self._operations:
                self._operations[name] = []
            self._operations[name].append(elapsed)
            return elapsed

    def measure(self, name: str) -> "_MeasureContext":
        """上下文管理器，自动计时

        用法:
            with monitor.measure("op"):
                do_something()
        """
        return _MeasureContext(self, name)

    def get_stats(self, name: str) -> Dict[str, Any]:
        """获取指定操作的统计信息

        返回:
            {
                "count": 调用次数,
                "total": 总耗时,
                "avg": 平均耗时,
                "min": 最小耗时,
                "max": 最大耗时,
                "last": 最近一次耗时
            }
        """
        with self._data_lock:
            times = self._operations.get(name, [])
        if not times:
            return {
                "count": 0,
                "total": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "last": 0.0,
            }
        return {
            "count": len(times),
            "total": sum(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "last": times[-1],
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有操作的统计信息"""
        with self._data_lock:
            names = list(self._operations.keys())
        return {name: self.get_stats(name) for name in names}

    def reset(self) -> None:
        """重置所有统计"""
        with self._data_lock:
            self._operations.clear()
            self._active.clear()

    def get_report(self) -> str:
        """生成文本报告"""
        stats = self.get_all_stats()
        if not stats:
            return "性能监控: 暂无数据"
        lines = ["=" * 60, "性能监控报告", "=" * 60]
        header = f"{'操作':<30} {'次数':>6} {'总耗时':>10} {'平均':>10} {'最大':>10}"
        lines.append(header)
        lines.append("-" * 60)
        # 按总耗时降序
        sorted_items = sorted(
            stats.items(),
            key=lambda x: x[1]["total"],
            reverse=True,
        )
        for name, s in sorted_items:
            lines.append(
                f"{name:<30} {s['count']:>6} "
                f"{s['total']*1000:>8.2f}ms "
                f"{s['avg']*1000:>8.2f}ms "
                f"{s['max']*1000:>8.2f}ms"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Optional[str] = None) -> str:
        """保存性能数据到文件"""
        path = path or str(PERF_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "timestamp": time.time(),
            "operations": self.get_all_stats(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            cls._instance = None


class _MeasureContext:
    """measure() 返回的上下文管理器"""

    def __init__(self, monitor: PerformanceMonitor, name: str) -> None:
        self._monitor = monitor
        self._name = name

    def __enter__(self) -> "_MeasureContext":
        self._monitor.start(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._monitor.stop(self._name)


# ============================================================================
# profile_operation 装饰器
# ============================================================================

def profile_operation(func: Callable) -> Callable:
    """
    函数装饰器：自动监控函数耗时

    用法:
        @profile_operation
        def my_function():
            ...
    """
    op_name = f"{func.__module__}.{func.__name__}"
    monitor = PerformanceMonitor()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with monitor.measure(op_name):
            return func(*args, **kwargs)

    return wrapper


# ============================================================================
# LazyLoader - 延迟加载
# ============================================================================

class LazyLoader:
    """
    延迟加载器，按需初始化重型组件

    用法:
        loader = LazyLoader("numpy", lambda: __import__("numpy"))
        # 第一次访问时才加载
        arr = loader.value

    或作为类属性:
        class MyClass:
            def __init__(self):
                self._heavy = LazyLoader("heavy", self._init_heavy)

            @property
            def heavy(self):
                return self._heavy.value
    """

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self._name = name
        self._loader = loader
        self._value: Any = None
        self._loaded: bool = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def value(self) -> Any:
        """获取值（首次访问时加载）"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    monitor = PerformanceMonitor()
                    with monitor.measure(f"lazy_load:{self._name}"):
                        self._value = self._loader()
                    self._loaded = True
        return self._value

    def reset(self) -> None:
        """重置，下次访问会重新加载"""
        with self._lock:
            self._value = None
            self._loaded = False

    def preload(self) -> None:
        """预加载"""
        _ = self.value


class LazyAttribute:
    """
    类属性延迟加载描述符

    用法:
        class MyClass:
            heavy = LazyAttribute("heavy", lambda self: self._compute_heavy())

            def _compute_heavy(self):
                return [...]
    """

    def __init__(self, name: str, factory: Callable[[Any], Any]) -> None:
        self._name = name
        self._factory = factory
        self._attr_name = f"_lazy_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, self._attr_name):
            value = self._factory(instance)
            setattr(instance, self._attr_name, value)
        return getattr(instance, self._attr_name)

    def __set__(self, instance, value):
        setattr(instance, self._attr_name, value)


# ============================================================================
# StartupProfiler - 启动剖析
# ============================================================================

class StartupProfiler:
    """
    启动过程剖析器，定位慢初始化步骤

    用法:
        profiler = StartupProfiler()
        profiler.start()
        # ... 初始化代码 ...
        profiler.checkpoint("config_loaded")
        profiler.checkpoint("agent_created")
        profiler.end()
        print(profiler.report())
    """

    def __init__(self) -> None:
        self._checkpoints: List[Tuple[str, float]] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> None:
        """开始剖析"""
        self._checkpoints = []
        self._start_time = time.perf_counter()
        self._end_time = None

    def checkpoint(self, name: str) -> float:
        """记录检查点，返回距上一个检查点的耗时（秒）"""
        if self._start_time is None:
            raise RuntimeError("必须先调用 start()")
        now = time.perf_counter()
        prev = self._checkpoints[-1][1] if self._checkpoints else self._start_time
        self._checkpoints.append((name, now))
        return now - prev

    def end(self) -> float:
        """结束剖析，返回总耗时（秒）"""
        if self._start_time is None:
            raise RuntimeError("必须先调用 start()")
        self._end_time = time.perf_counter()
        return self._end_time - self._start_time

    @property
    def total_time(self) -> float:
        """总耗时"""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.perf_counter()
        return end - self._start_time

    def get_steps(self) -> List[Dict[str, Any]]:
        """获取各步骤耗时"""
        steps = []
        prev_time = self._start_time or 0.0
        for name, t in self._checkpoints:
            steps.append({
                "name": name,
                "duration": t - prev_time,
                "timestamp": t,
            })
            prev_time = t
        if self._end_time is not None and self._checkpoints:
            steps.append({
                "name": "_end",
                "duration": self._end_time - prev_time,
                "timestamp": self._end_time,
            })
        return steps

    def get_slowest(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """获取最慢的N个步骤"""
        steps = self.get_steps()
        return sorted(steps, key=lambda x: x["duration"], reverse=True)[:top_n]

    def report(self) -> str:
        """生成文本报告"""
        if self._start_time is None:
            return "启动剖析: 尚未开始"
        lines = ["=" * 60, "启动剖析报告", "=" * 60]
        lines.append(f"总耗时: {self.total_time*1000:.2f}ms")
        lines.append("-" * 60)
        steps = self.get_steps()
        if steps:
            lines.append(f"{'步骤':<30} {'耗时':>10} {'占比':>8}")
            lines.append("-" * 60)
            total = self.total_time
            for step in steps:
                pct = (step["duration"] / total * 100) if total > 0 else 0
                lines.append(
                    f"{step['name']:<30} "
                    f"{step['duration']*1000:>8.2f}ms "
                    f"{pct:>6.1f}%"
                )
            lines.append("-" * 60)
            slowest = self.get_slowest(3)
            lines.append("最慢步骤:")
            for s in slowest:
                lines.append(f"  - {s['name']}: {s['duration']*1000:.2f}ms")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# LRUCache - LRU缓存
# ============================================================================

class LRUCache:
    """
    简单的LRU缓存

    用法:
        cache = LRUCache(maxsize=100)
        value = cache.get_or_compute("key", expensive_func)
    """

    def __init__(self, maxsize: int = DEFAULT_LRU_SIZE) -> None:
        self._maxsize = maxsize
        self._cache: Dict[Any, Any] = {}
        self._order: List[Any] = []
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def size(self) -> int:
        return len(self._cache)

    def get(self, key: Any, default: Any = None) -> Any:
        """获取值"""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # 移到末尾（最近使用）
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            self._misses += 1
            return default

    def set(self, key: Any, value: Any) -> None:
        """设置值"""
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._maxsize:
                # 淘汰最久未使用
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[key] = value
            self._order.append(key)

    def get_or_compute(self, key: Any, factory: Callable[[], Any]) -> Any:
        """获取或计算（如果不存在）"""
        value = self.get(key, _MISSING)
        if value is _MISSING:
            value = factory()
            self.set(key, value)
        return value

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


_MISSING = object()


# ============================================================================
# MemoryTracker - 内存跟踪
# ============================================================================

class MemoryTracker:
    """
    内存占用跟踪器（基于sys.getsizeof，可选psutil）

    用法:
        tracker = MemoryTracker()
        tracker.snapshot("before")
        do_something()
        tracker.snapshot("after")
        print(tracker.diff_report("before", "after"))
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _get_process_memory() -> Optional[int]:
        """获取进程内存占用（字节）"""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            return None

    def snapshot(self, name: str) -> Dict[str, Any]:
        """拍摄内存快照"""
        mem = self._get_process_memory()
        snap = {
            "name": name,
            "timestamp": time.time(),
            "process_rss": mem,
            "object_count": 0,
        }
        self._snapshots[name] = snap
        return snap

    def get_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        return self._snapshots.get(name)

    def diff(self, name1: str, name2: str) -> Dict[str, Any]:
        """计算两个快照的差异"""
        s1 = self._snapshots.get(name1)
        s2 = self._snapshots.get(name2)
        if s1 is None or s2 is None:
            return {"error": "快照不存在"}
        mem_diff = None
        if s1["process_rss"] is not None and s2["process_rss"] is not None:
            mem_diff = s2["process_rss"] - s1["process_rss"]
        return {
            "from": name1,
            "to": name2,
            "memory_diff": mem_diff,
            "time_diff": s2["timestamp"] - s1["timestamp"],
        }

    def diff_report(self, name1: str, name2: str) -> str:
        """生成差异报告"""
        d = self.diff(name1, name2)
        if "error" in d:
            return d["error"]
        lines = [
            f"内存差异 ({name1} -> {name2}):",
            f"  时间差: {d['time_diff']*1000:.2f}ms",
        ]
        if d["memory_diff"] is not None:
            diff = d["memory_diff"]
            sign = "+" if diff >= 0 else ""
            lines.append(f"  内存差: {sign}{diff/1024:.2f}KB")
        else:
            lines.append("  内存差: (psutil未安装，无法测量)")
        return "\n".join(lines)

    def clear(self) -> None:
        self._snapshots.clear()


# ============================================================================
# 便捷函数
# ============================================================================

def get_performance_report() -> str:
    """获取性能监控报告"""
    return PerformanceMonitor().get_report()


def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """获取所有性能统计"""
    return PerformanceMonitor().get_all_stats()


def save_performance(path: Optional[str] = None) -> str:
    """保存性能数据到文件"""
    return PerformanceMonitor().save(path)


def reset_performance() -> None:
    """重置性能监控"""
    PerformanceMonitor().reset()


def optimize_agent(agent: Any) -> Dict[str, Any]:
    """
    优化Agent配置以提升性能

    返回优化建议和应用的优化项
    """
    suggestions: List[str] = []
    applied: List[str] = []

    # 检查配置
    config = getattr(agent, "config", None)
    if config is not None:
        # 日志级别优化
        log_level = getattr(config, "log_level", "INFO")
        if log_level == "DEBUG":
            suggestions.append("将日志级别从DEBUG改为INFO可提升性能")
        # 检查是否启用了不必要的特性
        if hasattr(config, "enable_metacognition"):
            if config.enable_metacognition and getattr(config, "steps", 100) < 50:
                suggestions.append("少量步数时可考虑关闭元认知以加速")

    # 检查内存
    memory = getattr(agent, "memory", None)
    if memory is not None:
        # 检查记忆大小
        if hasattr(memory, "episodes") and len(memory.episodes) > 1000:
            suggestions.append("记忆episodes超过1000条，建议清理旧记忆")
        if hasattr(memory, "max_episodes"):
            if memory.max_episodes > 5000:
                suggestions.append(
                    f"max_episodes={memory.max_episodes}较大，"
                    "考虑减小以降低内存占用"
                )

    # 检查技能库
    skills = getattr(agent, "skill_library", None) or getattr(agent, "skills", None)
    if skills is not None:
        if hasattr(skills, "skills") and len(skills.skills) > 100:
            suggestions.append("技能数超过100，建议清理低频技能")

    # 检查是否使用了缓存
    if not hasattr(agent, "_perf_cache"):
        agent._perf_cache = LRUCache(maxsize=64)
        applied.append("添加了LRU缓存(_perf_cache)")

    # 添加性能监控
    if not hasattr(agent, "_perf_monitor"):
        agent._perf_monitor = PerformanceMonitor()
        applied.append("添加了性能监控器(_perf_monitor)")

    return {
        "suggestions": suggestions,
        "applied": applied,
        "suggestion_count": len(suggestions),
        "applied_count": len(applied),
    }


def benchmark(func: Callable, iterations: int = 100,
              warmup: int = 10) -> Dict[str, Any]:
    """
    基准测试函数性能

    Args:
        func: 要测试的函数
        iterations: 测试迭代次数
        warmup: 预热次数（不计入统计）

    Returns:
        {
            "iterations": 迭代次数,
            "total_time": 总耗时,
            "avg_time": 平均耗时,
            "min_time": 最小耗时,
            "max_time": 最大耗时,
            "throughput": 吞吐量（次/秒）
        }
    """
    # 预热
    for _ in range(warmup):
        func()

    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    total = sum(times)
    return {
        "iterations": iterations,
        "total_time": total,
        "avg_time": total / iterations,
        "min_time": min(times),
        "max_time": max(times),
        "throughput": iterations / total if total > 0 else 0.0,
    }


def benchmark_report(func: Callable, name: str = "function",
                     iterations: int = 100, warmup: int = 10) -> str:
    """生成基准测试报告"""
    result = benchmark(func, iterations, warmup)
    lines = [
        "=" * 60,
        f"基准测试: {name}",
        "=" * 60,
        f"  迭代次数:   {result['iterations']}",
        f"  总耗时:     {result['total_time']*1000:.2f}ms",
        f"  平均耗时:   {result['avg_time']*1000:.4f}ms",
        f"  最小耗时:   {result['min_time']*1000:.4f}ms",
        f"  最大耗时:   {result['max_time']*1000:.4f}ms",
        f"  吞吐量:     {result['throughput']:.0f} 次/秒",
        "=" * 60,
    ]
    return "\n".join(lines)


def startup_benchmark(preset: str = "easy", env_type: str = "mixed",
                      seed: int = 42) -> Dict[str, Any]:
    """
    测量Agent启动时间

    Returns:
        {
            "total_time": 总启动时间,
            "steps": 各阶段耗时,
            "preset": 预设,
            "env_type": 环境类型
        }
    """
    profiler = StartupProfiler()
    profiler.start()

    # 延迟导入避免循环
    from .api import SEVAFactory
    profiler.checkpoint("import_api")

    agent = SEVAFactory.create(preset=preset, env_type=env_type, seed=seed)
    profiler.checkpoint("agent_created")

    # 初始化运行环境
    if hasattr(agent, "reset"):
        agent.reset()
    profiler.checkpoint("agent_reset")

    total = profiler.end()
    return {
        "total_time": total,
        "steps": profiler.get_steps(),
        "preset": preset,
        "env_type": env_type,
    }


__all__ = [
    "PerformanceMonitor",
    "LazyLoader",
    "LazyAttribute",
    "StartupProfiler",
    "LRUCache",
    "MemoryTracker",
    "profile_operation",
    "get_performance_report",
    "get_performance_stats",
    "save_performance",
    "reset_performance",
    "optimize_agent",
    "benchmark",
    "benchmark_report",
    "startup_benchmark",
]
