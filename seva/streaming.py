"""
实时流处理与事件驱动（阶段56）
对应第一性原理：流处理 = 数据像水流一样持续通过处理管道，每个元素被逐步变换、过滤、聚合

提供能力:
    - Event: 事件数据结构（类型、数据、时间戳、来源）
    - EventBus: 事件总线（发布/订阅，同步/异步分发，通配符订阅）
    - Stream: 数据流（惰性求值，操作符链：map/filter/reduce/foreach等）
    - StreamProcessor: 流处理器（窗口聚合、滑动窗口、会话窗口）
    - ReactiveStream: 反应式流（背压、订阅者模式）
    - 便捷函数: publish / subscribe / stream / get_event_bus 等

使用示例:
    from seva import EventBus, Stream

    # 事件总线
    bus = EventBus()
    bus.subscribe("user.*", lambda event: print(event.data))
    bus.publish("user.login", {"user": "alice"})

    # 数据流
    result = Stream([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 4).to_list()
    # [6, 8, 10]
"""
from __future__ import annotations

import os
import sys
import time
import json
import threading
import queue
import uuid
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, Iterator
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = os.path.expanduser("~/.seva")


# ============================================================================
# Event - 事件
# ============================================================================

@dataclass
class Event:
    """
    事件数据结构

    Args:
        event_type: 事件类型
        data: 事件数据
        source: 事件来源
        timestamp: 时间戳（默认当前时间）
        event_id: 事件ID（默认自动生成）
    """
    event_type: str
    data: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            event_type=data["event_type"],
            data=data.get("data"),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", time.time()),
            event_id=data.get("event_id", str(uuid.uuid4())[:8]),
        )

    def __repr__(self) -> str:
        return f"Event(type={self.event_type!r}, source={self.source!r})"


# ============================================================================
# EventBus - 事件总线
# ============================================================================

class EventBus:
    """
    事件总线：发布/订阅模式

    支持:
        - 精确匹配订阅（"user.login"）
        - 通配符订阅（"user.*" 匹配 "user.login", "user.logout"等）
        - 同步/异步分发
        - 多订阅者
    """

    def __init__(self, async_dispatch: bool = False, max_workers: int = 4):
        """
        Args:
            async_dispatch: 是否异步分发
            max_workers: 异步分发的线程池大小
        """
        self._subscribers: Dict[str, List[Tuple[Callable, int]]] = {}
        self._wildcard_subscribers: List[Tuple[str, Callable, int]] = []
        self._lock = threading.RLock()
        self._async_dispatch = async_dispatch
        self._executor: Optional[ThreadPoolExecutor] = None
        if async_dispatch:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # 统计
        self._total_published = 0
        self._total_delivered = 0
        self._total_errors = 0

    def subscribe(self, pattern: str, callback: Callable,
                  priority: int = 0) -> Callable:
        """
        订阅事件

        Args:
            pattern: 事件类型模式（支持*通配符）
            callback: 回调函数，签名为 callback(event: Event)
            priority: 优先级（数字小的先调用）

        Returns:
            取消订阅函数
        """
        with self._lock:
            if "*" in pattern:
                self._wildcard_subscribers.append((pattern, callback, priority))
                self._wildcard_subscribers.sort(key=lambda x: x[2])
            else:
                if pattern not in self._subscribers:
                    self._subscribers[pattern] = []
                self._subscribers[pattern].append((callback, priority))
                self._subscribers[pattern].sort(key=lambda x: x[1])

        def unsubscribe():
            self.unsubscribe(pattern, callback)

        return unsubscribe

    def unsubscribe(self, pattern: str, callback: Callable) -> bool:
        """取消订阅"""
        with self._lock:
            if "*" in pattern:
                before = len(self._wildcard_subscribers)
                self._wildcard_subscribers = [
                    (p, c, pri) for p, c, pri in self._wildcard_subscribers
                    if not (p == pattern and c == callback)
                ]
                return len(self._wildcard_subscribers) < before
            else:
                if pattern not in self._subscribers:
                    return False
                before = len(self._subscribers[pattern])
                self._subscribers[pattern] = [
                    (c, p) for c, p in self._subscribers[pattern] if c != callback
                ]
                return len(self._subscribers[pattern]) < before

    def publish(self, event_type: str, data: Any = None,
                source: str = "") -> Event:
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源

        Returns:
            发布的事件
        """
        event = Event(event_type=event_type, data=data, source=source)
        self._publish_event(event)
        return event

    def publish_event(self, event: Event) -> None:
        """发布已构造的事件"""
        self._publish_event(event)

    def _publish_event(self, event: Event) -> None:
        """内部发布逻辑"""
        with self._lock:
            self._total_published += 1
            # 收集匹配的订阅者
            callbacks: List[Callable] = []

            # 精确匹配
            for callback, _ in self._subscribers.get(event.event_type, []):
                callbacks.append(callback)

            # 通配符匹配
            for pattern, callback, _ in self._wildcard_subscribers:
                if self._match_pattern(pattern, event.event_type):
                    callbacks.append(callback)

        # 分发
        for callback in callbacks:
            if self._async_dispatch and self._executor:
                self._executor.submit(self._safe_call, callback, event)
            else:
                self._safe_call(callback, event)

    def _safe_call(self, callback: Callable, event: Event) -> None:
        """安全调用回调"""
        try:
            callback(event)
            with self._lock:
                self._total_delivered += 1
        except Exception:
            with self._lock:
                self._total_errors += 1

    @staticmethod
    def _match_pattern(pattern: str, event_type: str) -> bool:
        """匹配通配符模式"""
        if "*" not in pattern:
            return pattern == event_type

        # 将通配符模式转为正则
        import re
        regex = "^" + pattern.replace(".", "\\.").replace("*", ".*") + "$"
        return bool(re.match(regex, event_type))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_published": self._total_published,
                "total_delivered": self._total_delivered,
                "total_errors": self._total_errors,
                "subscriber_count": (
                    sum(len(subs) for subs in self._subscribers.values()) +
                    len(self._wildcard_subscribers)
                ),
                "event_types": list(self._subscribers.keys()),
            }

    def clear(self) -> None:
        """清空所有订阅"""
        with self._lock:
            self._subscribers.clear()
            self._wildcard_subscribers.clear()

    def shutdown(self) -> None:
        """关闭事件总线"""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __repr__(self) -> str:
        return (f"EventBus(subscribers={self.get_stats()['subscriber_count']}, "
                f"published={self._total_published})")


# ============================================================================
# Stream - 数据流
# ============================================================================

class Stream:
    """
    数据流：惰性求值的操作符链

    支持函数式操作：map / filter / reduce / flat_map / distinct / take / skip 等

    使用示例:
        result = Stream([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 4).to_list()
        # [6, 8, 10]

        total = Stream([1, 2, 3]).reduce(lambda a, b: a + b, 0)
        # 6
    """

    def __init__(self, source: Union[Iterator, List, Tuple, range, "Stream"]):
        """
        Args:
            source: 数据源（迭代器、列表、元组、range、或另一个Stream）
        """
        if isinstance(source, Stream):
            self._source = source._source
        elif isinstance(source, (list, tuple, range)):
            self._source = iter(source)
        else:
            self._source = source

    def __iter__(self) -> Iterator:
        return self._source

    # ---- 中间操作（返回新Stream，惰性求值）----

    def map(self, func: Callable) -> "Stream":
        """映射：对每个元素应用函数"""
        return Stream(map(func, self._source))

    def filter(self, predicate: Callable) -> "Stream":
        """过滤：保留满足条件的元素"""
        return Stream(filter(predicate, self._source))

    def flat_map(self, func: Callable) -> "Stream":
        """扁平化映射：对每个元素应用函数并展平结果"""
        def gen():
            for item in self._source:
                yield from func(item)
        return Stream(gen())

    def take(self, n: int) -> "Stream":
        """取前n个元素"""
        def gen():
            for i, item in enumerate(self._source):
                if i >= n:
                    break
                yield item
        return Stream(gen())

    def skip(self, n: int) -> "Stream":
        """跳过前n个元素"""
        def gen():
            for i, item in enumerate(self._source):
                if i >= n:
                    yield item
        return Stream(gen())

    def distinct(self) -> "Stream":
        """去重"""
        def gen():
            seen: Set = set()
            for item in self._source:
                key = item if isinstance(item, (int, float, str, bool)) else str(item)
                if key not in seen:
                    seen.add(key)
                    yield item
        return Stream(gen())

    def sorted(self, key: Optional[Callable] = None, reverse: bool = False) -> "Stream":
        """排序（注意：会消费整个流）"""
        return Stream(sorted(self._source, key=key, reverse=reverse))

    def limit(self, n: int) -> "Stream":
        """限制元素数量（take的别名）"""
        return self.take(n)

    def enumerate(self, start: int = 0) -> "Stream":
        """枚举（返回(索引, 元素)对）"""
        return Stream(enumerate(self._source, start))

    def zip_with(self, other: Union[List, Iterator]) -> "Stream":
        """与另一个序列配对"""
        return Stream(zip(self._source, other))

    def chain(self, other: Union[List, Iterator]) -> "Stream":
        """连接另一个序列"""
        import itertools
        return Stream(itertools.chain(self._source, other))

    def peek(self, action: Callable) -> "Stream":
        """对每个元素执行操作但不改变元素（用于调试）"""
        def gen():
            for item in self._source:
                action(item)
                yield item
        return Stream(gen())

    # ---- 终端操作（消费流，返回结果）----

    def to_list(self) -> List:
        """转为列表"""
        return list(self._source)

    def to_set(self) -> Set:
        """转为集合"""
        return set(self._source)

    def to_dict(self, key_func: Callable, value_func: Optional[Callable] = None) -> Dict:
        """转为字典"""
        result = {}
        for item in self._source:
            key = key_func(item)
            value = value_func(item) if value_func else item
            result[key] = value
        return result

    def reduce(self, func: Callable, initial: Any = None) -> Any:
        """归约"""
        import functools as ft
        if initial is not None:
            return ft.reduce(func, self._source, initial)
        return ft.reduce(func, self._source)

    def sum(self) -> Union[int, float]:
        """求和"""
        return sum(self._source)

    def count(self) -> int:
        """计数"""
        count = 0
        for _ in self._source:
            count += 1
        return count

    def min(self, key: Optional[Callable] = None) -> Any:
        """最小值"""
        return min(self._source, key=key) if key else min(self._source)

    def max(self, key: Optional[Callable] = None) -> Any:
        """最大值"""
        return max(self._source, key=key) if key else max(self._source)

    def first(self, default: Any = None) -> Any:
        """第一个元素"""
        for item in self._source:
            return item
        return default

    def last(self, default: Any = None) -> Any:
        """最后一个元素"""
        result = default
        for item in self._source:
            result = item
        return result

    def any(self, predicate: Callable) -> bool:
        """是否存在满足条件的元素"""
        return any(predicate(item) for item in self._source)

    def all(self, predicate: Callable) -> bool:
        """是否所有元素都满足条件"""
        return all(predicate(item) for item in self._source)

    def for_each(self, action: Callable) -> None:
        """对每个元素执行操作"""
        for item in self._source:
            action(item)

    def group_by(self, key_func: Callable) -> Dict[Any, List]:
        """分组"""
        result: Dict[Any, List] = {}
        for item in self._source:
            key = key_func(item)
            if key not in result:
                result[key] = []
            result[key].append(item)
        return result

    def partition(self, predicate: Callable) -> Tuple[List, List]:
        """分区（返回满足和不满足的两个列表）"""
        true_list: List = []
        false_list: List = []
        for item in self._source:
            if predicate(item):
                true_list.append(item)
            else:
                false_list.append(item)
        return true_list, false_list

    def __repr__(self) -> str:
        return "Stream(<lazy>)"


# ============================================================================
# WindowType - 窗口类型枚举
# ============================================================================

class WindowType(Enum):
    """窗口类型"""
    TUMBLING = "tumbling"  # 翻滚窗口（不重叠）
    SLIDING = "sliding"    # 滑动窗口（可重叠）
    SESSION = "session"    # 会话窗口（基于间隔）


# ============================================================================
# StreamProcessor - 流处理器
# ============================================================================

class StreamProcessor:
    """
    流处理器：窗口聚合操作

    支持窗口类型:
        - 翻滚窗口：固定大小，不重叠
        - 滑动窗口：固定大小，可重叠
        - 会话窗口：基于元素间隔

    使用示例:
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        results = processor.process([1, 2, 3, 4, 5, 6], lambda w: sum(w))
        # [6, 15]  (1+2+3=6, 4+5+6=15)
    """

    def __init__(self, window_size: int = 10, window_type: WindowType = WindowType.TUMBLING,
                 slide_size: int = 1, session_timeout: float = 1.0):
        """
        Args:
            window_size: 窗口大小（元素数）
            window_type: 窗口类型
            slide_size: 滑动步长（仅SLIDING模式）
            session_timeout: 会话超时秒数（仅SESSION模式）
        """
        self.window_size = max(1, window_size)
        self.window_type = window_type
        self.slide_size = max(1, slide_size)
        self.session_timeout = max(0.1, session_timeout)

    def process(self, data: List[Any],
                aggregate_func: Callable[[List[Any]], Any]) -> List[Any]:
        """
        处理数据流

        Args:
            data: 输入数据列表
            aggregate_func: 聚合函数，接收窗口内元素列表，返回聚合结果

        Returns:
            聚合结果列表
        """
        if self.window_type == WindowType.TUMBLING:
            return self._process_tumbling(data, aggregate_func)
        elif self.window_type == WindowType.SLIDING:
            return self._process_sliding(data, aggregate_func)
        elif self.window_type == WindowType.SESSION:
            return self._process_session(data, aggregate_func)
        return []

    def _process_tumbling(self, data: List[Any],
                          aggregate_func: Callable) -> List[Any]:
        """翻滚窗口"""
        results = []
        for i in range(0, len(data), self.window_size):
            window = data[i:i + self.window_size]
            if window:
                results.append(aggregate_func(window))
        return results

    def _process_sliding(self, data: List[Any],
                         aggregate_func: Callable) -> List[Any]:
        """滑动窗口"""
        results = []
        for i in range(0, len(data), self.slide_size):
            window = data[i:i + self.window_size]
            if window:
                results.append(aggregate_func(window))
        return results

    def _process_session(self, data: List[Any],
                         aggregate_func: Callable) -> List[Any]:
        """会话窗口（基于时间间隔）"""
        if not data:
            return []

        results = []
        current_session: List[Any] = [data[0]]
        last_time = self._get_timestamp(data[0])

        for item in data[1:]:
            current_time = self._get_timestamp(item)
            if current_time - last_time > self.session_timeout:
                # 会话结束
                results.append(aggregate_func(current_session))
                current_session = [item]
            else:
                current_session.append(item)
            last_time = current_time

        # 最后一个会话
        if current_session:
            results.append(aggregate_func(current_session))

        return results

    @staticmethod
    def _get_timestamp(item: Any) -> float:
        """获取元素的时间戳"""
        if isinstance(item, (int, float)):
            return float(item)
        elif isinstance(item, dict) and "timestamp" in item:
            return float(item["timestamp"])
        elif isinstance(item, Event):
            return item.timestamp
        else:
            return time.time()

    def get_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        return {
            "window_size": self.window_size,
            "window_type": self.window_type.value,
            "slide_size": self.slide_size,
            "session_timeout": self.session_timeout,
        }

    def __repr__(self) -> str:
        return (f"StreamProcessor(size={self.window_size}, "
                f"type={self.window_type.value})")


# ============================================================================
# ReactiveStream - 反应式流
# ============================================================================

class ReactiveStream:
    """
    反应式流：支持背压的发布-订阅模式

    使用示例:
        stream = ReactiveStream()
        stream.subscribe(lambda x: print(x))
        stream.on_next(1)
        stream.on_next(2)
        stream.on_complete()
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Args:
            buffer_size: 缓冲区大小（背压控制）
        """
        self._subscribers: List[Callable] = []
        self._buffer: queue.Queue = queue.Queue(maxsize=max(1, buffer_size))
        self._completed = False
        self._error: Optional[Exception] = None
        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._item_count = 0
        self._delivered_count = 0

    def subscribe(self, callback: Callable) -> Callable:
        """
        订阅流

        Args:
            callback: 回调函数，签名为 callback(item)

        Returns:
            取消订阅函数
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe():
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def on_next(self, item: Any) -> bool:
        """
        发射下一个元素

        Returns:
            是否成功加入缓冲区（False表示缓冲区满，背压）
        """
        if self._completed:
            return False

        try:
            self._buffer.put_nowait(item)
            self._item_count += 1
            self._ensure_worker()
            return True
        except queue.Full:
            return False

    def on_error(self, error: Exception) -> None:
        """发射错误"""
        with self._lock:
            self._error = error
            self._completed = True

    def on_complete(self) -> None:
        """完成流"""
        with self._lock:
            self._completed = True
            # 发送结束信号
            try:
                self._buffer.put_nowait(None)
            except queue.Full:
                pass

    def _ensure_worker(self) -> None:
        """确保工作线程运行"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _worker(self) -> None:
        """工作线程：从缓冲区取元素并分发给订阅者"""
        while True:
            try:
                item = self._buffer.get(timeout=0.1)
            except queue.Empty:
                if self._completed:
                    break
                continue

            if item is None and self._completed:
                break

            # 分发给所有订阅者
            with self._lock:
                subscribers = list(self._subscribers)

            for callback in subscribers:
                try:
                    callback(item)
                    self._delivered_count += 1
                except Exception:
                    pass

    def wait_completion(self, timeout: float = 5.0) -> bool:
        """等待流完成"""
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            return not self._worker_thread.is_alive()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "item_count": self._item_count,
                "delivered_count": self._delivered_count,
                "subscriber_count": len(self._subscribers),
                "completed": self._completed,
                "has_error": self._error is not None,
                "buffer_size": self._buffer.qsize(),
            }

    def __repr__(self) -> str:
        return (f"ReactiveStream(subscribers={len(self._subscribers)}, "
                f"items={self._item_count})")


# ============================================================================
# 全局事件总线（单例）
# ============================================================================

_global_bus: Optional[EventBus] = None
_global_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _global_bus
    if _global_bus is None:
        with _global_bus_lock:
            if _global_bus is None:
                _global_bus = EventBus()
    return _global_bus


def set_event_bus(bus: EventBus) -> None:
    """设置全局事件总线"""
    global _global_bus
    with _global_bus_lock:
        _global_bus = bus


# ============================================================================
# 便捷函数
# ============================================================================

def publish(event_type: str, data: Any = None, source: str = "") -> Event:
    """发布事件到全局事件总线"""
    return get_event_bus().publish(event_type, data, source)


def subscribe(pattern: str, callback: Callable, priority: int = 0) -> Callable:
    """订阅全局事件总线"""
    return get_event_bus().subscribe(pattern, callback, priority)


def unsubscribe(pattern: str, callback: Callable) -> bool:
    """取消订阅全局事件总线"""
    return get_event_bus().unsubscribe(pattern, callback)


def stream(source: Union[Iterator, List, Tuple, range]) -> Stream:
    """创建数据流"""
    return Stream(source)


def create_stream_processor(window_size: int = 10,
                            window_type: WindowType = WindowType.TUMBLING,
                            **kwargs) -> StreamProcessor:
    """创建流处理器"""
    return StreamProcessor(window_size=window_size, window_type=window_type, **kwargs)


def create_reactive_stream(buffer_size: int = 1000) -> ReactiveStream:
    """创建反应式流"""
    return ReactiveStream(buffer_size=buffer_size)


def get_streaming_stats() -> Dict[str, Any]:
    """获取流处理统计信息"""
    return get_event_bus().get_stats()


def reset_streaming() -> None:
    """重置流处理状态"""
    global _global_bus
    with _global_bus_lock:
        if _global_bus is not None:
            _global_bus.clear()
            _global_bus.shutdown()
        _global_bus = None
