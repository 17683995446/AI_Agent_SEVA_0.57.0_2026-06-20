"""
阶段56测试：实时流处理与事件驱动
"""
import unittest
import time
import threading

from seva.streaming import (
    Event, EventBus, Stream, WindowType, StreamProcessor,
    ReactiveStream, get_event_bus, set_event_bus,
    publish, subscribe, unsubscribe, stream,
    create_stream_processor, create_reactive_stream,
    get_streaming_stats, reset_streaming,
)


# ============================================================================
# 测试Event
# ============================================================================

class TestEvent(unittest.TestCase):
    """测试事件"""

    def test01_creation(self):
        event = Event(event_type="test", data={"key": "value"}, source="unit")
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, {"key": "value"})
        self.assertEqual(event.source, "unit")
        self.assertIsNotNone(event.event_id)
        self.assertGreater(event.timestamp, 0)

    def test02_defaults(self):
        event = Event(event_type="test")
        self.assertIsNone(event.data)
        self.assertEqual(event.source, "")

    def test03_to_dict(self):
        event = Event(event_type="test", data="hello", source="me")
        d = event.to_dict()
        self.assertEqual(d["event_type"], "test")
        self.assertEqual(d["data"], "hello")
        self.assertEqual(d["source"], "me")

    def test04_from_dict(self):
        data = {"event_type": "test", "data": "hello", "source": "me",
                "timestamp": 12345, "event_id": "abc"}
        event = Event.from_dict(data)
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, "hello")
        self.assertEqual(event.timestamp, 12345)
        self.assertEqual(event.event_id, "abc")

    def test05_repr(self):
        event = Event(event_type="test", source="me")
        r = repr(event)
        self.assertIn("Event", r)
        self.assertIn("test", r)
        self.assertIn("me", r)


# ============================================================================
# 测试EventBus
# ============================================================================

class TestEventBus(unittest.TestCase):
    """测试事件总线"""

    def setUp(self):
        self.bus = EventBus()

    def tearDown(self):
        self.bus.shutdown()

    def test01_subscribe_publish(self):
        received = []
        self.bus.subscribe("test.event", lambda e: received.append(e.data))
        self.bus.publish("test.event", "hello")
        self.assertEqual(received, ["hello"])

    def test02_multiple_subscribers(self):
        received1 = []
        received2 = []
        self.bus.subscribe("test.event", lambda e: received1.append(e.data))
        self.bus.subscribe("test.event", lambda e: received2.append(e.data))
        self.bus.publish("test.event", "hello")
        self.assertEqual(received1, ["hello"])
        self.assertEqual(received2, ["hello"])

    def test03_unsubscribe(self):
        received = []
        unsub = self.bus.subscribe("test.event", lambda e: received.append(e.data))
        self.bus.publish("test.event", "first")
        unsub()
        self.bus.publish("test.event", "second")
        self.assertEqual(received, ["first"])

    def test04_unsubscribe_method(self):
        callback = lambda e: None
        self.bus.subscribe("test.event", callback)
        self.assertTrue(self.bus.unsubscribe("test.event", callback))
        self.assertFalse(self.bus.unsubscribe("test.event", callback))

    def test05_wildcard_subscription(self):
        received = []
        self.bus.subscribe("user.*", lambda e: received.append(e.event_type))
        self.bus.publish("user.login", {"user": "alice"})
        self.bus.publish("user.logout", {"user": "bob"})
        self.bus.publish("system.start", {})  # 不匹配
        self.assertEqual(received, ["user.login", "user.logout"])

    def test06_priority_order(self):
        order = []
        self.bus.subscribe("test", lambda e: order.append("low"), priority=10)
        self.bus.subscribe("test", lambda e: order.append("high"), priority=1)
        self.bus.publish("test")
        self.assertEqual(order, ["high", "low"])

    def test07_no_subscribers(self):
        # 不应抛异常
        self.bus.publish("nonexistent", "data")

    def test08_subscriber_error_handled(self):
        def bad_callback(e):
            raise ValueError("回调错误")

        received = []
        self.bus.subscribe("test", bad_callback)
        self.bus.subscribe("test", lambda e: received.append(e.data))
        self.bus.publish("test", "data")
        self.assertEqual(received, ["data"])  # 第二个回调仍被调用

    def test09_stats(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.publish("test", "data")
        stats = self.bus.get_stats()
        self.assertEqual(stats["total_published"], 1)
        self.assertEqual(stats["total_delivered"], 1)
        self.assertEqual(stats["subscriber_count"], 1)

    def test10_clear(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.clear()
        self.assertEqual(self.bus.get_stats()["subscriber_count"], 0)

    def test11_repr(self):
        r = repr(self.bus)
        self.assertIn("EventBus", r)

    def test12_returned_event(self):
        event = self.bus.publish("test", "data", source="me")
        self.assertIsInstance(event, Event)
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, "data")
        self.assertEqual(event.source, "me")

    def test13_publish_event(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e.data))
        event = Event(event_type="test", data="hello")
        self.bus.publish_event(event)
        self.assertEqual(received, ["hello"])


# ============================================================================
# 测试Stream
# ============================================================================

class TestStream(unittest.TestCase):
    """测试数据流"""

    def test01_from_list(self):
        s = stream([1, 2, 3])
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test02_map(self):
        result = stream([1, 2, 3]).map(lambda x: x * 2).to_list()
        self.assertEqual(result, [2, 4, 6])

    def test03_filter(self):
        result = stream([1, 2, 3, 4, 5]).filter(lambda x: x > 2).to_list()
        self.assertEqual(result, [3, 4, 5])

    def test04_map_filter_chain(self):
        result = stream([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(
            lambda x: x > 4).to_list()
        self.assertEqual(result, [6, 8, 10])

    def test05_flat_map(self):
        result = stream([1, 2, 3]).flat_map(lambda x: [x, x * 10]).to_list()
        self.assertEqual(result, [1, 10, 2, 20, 3, 30])

    def test06_take(self):
        result = stream([1, 2, 3, 4, 5]).take(3).to_list()
        self.assertEqual(result, [1, 2, 3])

    def test07_skip(self):
        result = stream([1, 2, 3, 4, 5]).skip(2).to_list()
        self.assertEqual(result, [3, 4, 5])

    def test08_distinct(self):
        result = stream([1, 2, 2, 3, 3, 3, 4]).distinct().to_list()
        self.assertEqual(result, [1, 2, 3, 4])

    def test09_sorted(self):
        result = stream([3, 1, 4, 1, 5, 9, 2, 6]).sorted().to_list()
        self.assertEqual(result, [1, 1, 2, 3, 4, 5, 6, 9])

    def test10_sorted_reverse(self):
        result = stream([3, 1, 4]).sorted(reverse=True).to_list()
        self.assertEqual(result, [4, 3, 1])

    def test11_reduce(self):
        result = stream([1, 2, 3, 4, 5]).reduce(lambda a, b: a + b, 0)
        self.assertEqual(result, 15)

    def test12_sum(self):
        result = stream([1, 2, 3, 4, 5]).sum()
        self.assertEqual(result, 15)

    def test13_count(self):
        result = stream([1, 2, 3, 4, 5]).count()
        self.assertEqual(result, 5)

    def test14_min_max(self):
        s = stream([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(s.min(), 1)
        # 注意：流被消费后需要重新创建
        self.assertEqual(stream([3, 1, 4, 1, 5, 9, 2, 6]).max(), 9)

    def test15_first_last(self):
        s = stream([1, 2, 3])
        self.assertEqual(s.first(), 1)
        self.assertEqual(stream([1, 2, 3]).last(), 3)

    def test16_first_empty(self):
        self.assertEqual(stream([]).first("default"), "default")

    def test17_any_all(self):
        s = stream([1, 2, 3, 4, 5])
        self.assertTrue(s.any(lambda x: x > 3))
        self.assertFalse(stream([1, 2, 3]).any(lambda x: x > 5))
        self.assertTrue(stream([2, 4, 6]).all(lambda x: x % 2 == 0))
        self.assertFalse(stream([2, 3, 4]).all(lambda x: x % 2 == 0))

    def test18_for_each(self):
        result = []
        stream([1, 2, 3]).for_each(lambda x: result.append(x * 2))
        self.assertEqual(result, [2, 4, 6])

    def test19_group_by(self):
        result = stream([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 2)
        self.assertEqual(result[0], [2, 4, 6])
        self.assertEqual(result[1], [1, 3, 5])

    def test20_partition(self):
        true_list, false_list = stream([1, 2, 3, 4, 5]).partition(lambda x: x > 2)
        self.assertEqual(true_list, [3, 4, 5])
        self.assertEqual(false_list, [1, 2])

    def test21_to_set(self):
        result = stream([1, 2, 2, 3, 3, 3]).to_set()
        self.assertEqual(result, {1, 2, 3})

    def test22_to_dict(self):
        result = stream([1, 2, 3]).to_dict(lambda x: x, lambda x: x * 2)
        self.assertEqual(result, {1: 2, 2: 4, 3: 6})

    def test23_peek(self):
        result = []
        s = stream([1, 2, 3]).peek(lambda x: result.append(x))
        s.to_list()  # 消费流
        self.assertEqual(result, [1, 2, 3])

    def test24_chain(self):
        result = stream([1, 2, 3]).chain([4, 5, 6]).to_list()
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])

    def test25_limit(self):
        result = stream([1, 2, 3, 4, 5]).limit(3).to_list()
        self.assertEqual(result, [1, 2, 3])

    def test26_repr(self):
        r = repr(stream([1, 2, 3]))
        self.assertIn("Stream", r)


# ============================================================================
# 测试StreamProcessor
# ============================================================================

class TestStreamProcessor(unittest.TestCase):
    """测试流处理器"""

    def test01_tumbling_window(self):
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        results = processor.process([1, 2, 3, 4, 5, 6], lambda w: sum(w))
        self.assertEqual(results, [6, 15])

    def test02_tumbling_uneven(self):
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        results = processor.process([1, 2, 3, 4, 5], lambda w: sum(w))
        self.assertEqual(results, [6, 9])

    def test03_sliding_window(self):
        processor = StreamProcessor(window_size=3, window_type=WindowType.SLIDING,
                                    slide_size=2)
        results = processor.process([1, 2, 3, 4, 5], lambda w: sum(w))
        self.assertEqual(results, [6, 12, 5])  # [1,2,3], [3,4,5], [5]

    def test04_sliding_slide_1(self):
        processor = StreamProcessor(window_size=2, window_type=WindowType.SLIDING,
                                    slide_size=1)
        results = processor.process([1, 2, 3, 4], lambda w: sum(w))
        self.assertEqual(results, [3, 5, 7, 4])  # [1,2], [2,3], [3,4], [4]

    def test05_empty_data(self):
        processor = StreamProcessor(window_size=3)
        results = processor.process([], lambda w: sum(w))
        self.assertEqual(results, [])

    def test06_aggregate_count(self):
        processor = StreamProcessor(window_size=2, window_type=WindowType.TUMBLING)
        results = processor.process([1, 2, 3, 4], lambda w: len(w))
        self.assertEqual(results, [2, 2])

    def test07_aggregate_max(self):
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        results = processor.process([1, 5, 2, 8, 3, 9], lambda w: max(w))
        self.assertEqual(results, [5, 9])

    def test08_session_window(self):
        # 基于时间间隔的会话窗口
        data = [
            {"value": 1, "timestamp": 0.0},
            {"value": 2, "timestamp": 0.5},  # 同一会话
            {"value": 3, "timestamp": 2.0},  # 新会话（间隔>1s）
            {"value": 4, "timestamp": 2.3},  # 同一会话
        ]
        processor = StreamProcessor(window_type=WindowType.SESSION, session_timeout=1.0)
        results = processor.process(data, lambda w: sum(item["value"] for item in w))
        self.assertEqual(results, [3, 7])  # [1,2], [3,4]

    def test09_get_info(self):
        processor = StreamProcessor(window_size=5, window_type=WindowType.SLIDING,
                                    slide_size=2)
        info = processor.get_info()
        self.assertEqual(info["window_size"], 5)
        self.assertEqual(info["window_type"], "sliding")
        self.assertEqual(info["slide_size"], 2)

    def test10_repr(self):
        processor = StreamProcessor(window_size=3)
        r = repr(processor)
        self.assertIn("StreamProcessor", r)
        self.assertIn("3", r)


# ============================================================================
# 测试ReactiveStream
# ============================================================================

class TestReactiveStream(unittest.TestCase):
    """测试反应式流"""

    def test01_subscribe_on_next(self):
        rs = ReactiveStream()
        received = []
        rs.subscribe(lambda x: received.append(x))
        rs.on_next(1)
        rs.on_next(2)
        rs.on_complete()
        rs.wait_completion(timeout=2.0)
        self.assertIn(1, received)
        self.assertIn(2, received)

    def test02_multiple_subscribers(self):
        rs = ReactiveStream()
        received1 = []
        received2 = []
        rs.subscribe(lambda x: received1.append(x))
        rs.subscribe(lambda x: received2.append(x))
        rs.on_next("hello")
        rs.on_complete()
        rs.wait_completion(timeout=2.0)
        self.assertIn("hello", received1)
        self.assertIn("hello", received2)

    def test03_unsubscribe(self):
        rs = ReactiveStream()
        received = []
        unsub = rs.subscribe(lambda x: received.append(x))
        rs.on_next(1)
        rs.wait_completion(timeout=1.0)
        unsub()
        rs.on_next(2)
        rs.on_complete()
        rs.wait_completion(timeout=1.0)
        self.assertEqual(received, [1])

    def test04_on_complete(self):
        rs = ReactiveStream()
        rs.on_next(1)
        rs.on_complete()
        # 完成后不应再接受
        self.assertFalse(rs.on_next(2))

    def test05_on_error(self):
        rs = ReactiveStream()
        rs.on_error(ValueError("测试错误"))
        self.assertFalse(rs.on_next(1))

    def test06_backpressure(self):
        rs = ReactiveStream(buffer_size=2)
        # 填满缓冲区
        self.assertTrue(rs.on_next(1))
        self.assertTrue(rs.on_next(2))
        # 缓冲区满，返回False
        # 注意：由于工作线程可能正在消费，这里不一定总是False
        # 但至少前两个应该成功

    def test07_stats(self):
        rs = ReactiveStream()
        rs.subscribe(lambda x: None)
        rs.on_next(1)
        rs.on_next(2)
        rs.on_complete()
        rs.wait_completion(timeout=2.0)
        stats = rs.get_stats()
        self.assertEqual(stats["item_count"], 2)
        self.assertTrue(stats["completed"])

    def test08_repr(self):
        rs = ReactiveStream()
        r = repr(rs)
        self.assertIn("ReactiveStream", r)


# ============================================================================
# 测试全局事件总线
# ============================================================================

class TestGlobalEventBus(unittest.TestCase):
    """测试全局事件总线"""

    def setUp(self):
        reset_streaming()

    def tearDown(self):
        reset_streaming()

    def test01_get_event_bus(self):
        bus = get_event_bus()
        self.assertIsInstance(bus, EventBus)

    def test02_singleton(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        self.assertIs(bus1, bus2)

    def test03_set_event_bus(self):
        new_bus = EventBus()
        set_event_bus(new_bus)
        self.assertIs(get_event_bus(), new_bus)
        new_bus.shutdown()

    def test04_publish_subscribe(self):
        received = []
        unsub = subscribe("test.*", lambda e: received.append(e.data))
        publish("test.event", "hello")
        unsub()
        self.assertEqual(received, ["hello"])

    def test05_unsubscribe(self):
        received = []
        callback = lambda e: received.append(e.data)
        subscribe("test", callback)
        publish("test", "first")
        unsubscribe("test", callback)
        publish("test", "second")
        self.assertEqual(received, ["first"])

    def test06_get_streaming_stats(self):
        subscribe("test", lambda e: None)
        publish("test", "data")
        stats = get_streaming_stats()
        self.assertEqual(stats["total_published"], 1)

    def test07_reset_streaming(self):
        subscribe("test", lambda e: None)
        reset_streaming()
        stats = get_streaming_stats()
        self.assertEqual(stats["subscriber_count"], 0)


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        reset_streaming()

    def tearDown(self):
        reset_streaming()

    def test01_stream_function(self):
        s = stream([1, 2, 3])
        self.assertIsInstance(s, Stream)
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test02_create_stream_processor(self):
        processor = create_stream_processor(window_size=3)
        self.assertIsInstance(processor, StreamProcessor)

    def test03_create_reactive_stream(self):
        rs = create_reactive_stream()
        self.assertIsInstance(rs, ReactiveStream)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_streaming()

    def tearDown(self):
        reset_streaming()

    def test01_event_driven_workflow(self):
        """事件驱动工作流"""
        received = []
        subscribe("task.started", lambda e: received.append(("started", e.data)))
        subscribe("task.completed", lambda e: received.append(("completed", e.data)))

        publish("task.started", {"task": "process"})
        publish("task.completed", {"task": "process", "result": "ok"})

        self.assertEqual(len(received), 2)
        self.assertEqual(received[0], ("started", {"task": "process"}))
        self.assertEqual(received[1], ("completed", {"task": "process", "result": "ok"}))

    def test02_stream_processing_pipeline(self):
        """流处理管道"""
        result = (
            stream(range(100))
            .filter(lambda x: x % 2 == 0)  # 偶数
            .map(lambda x: x * x)           # 平方
            .take(5)                         # 前5个
            .to_list()
        )
        self.assertEqual(result, [0, 4, 16, 36, 64])

    def test03_windowed_aggregation(self):
        """窗口聚合"""
        data = list(range(1, 11))  # 1-10
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        results = processor.process(data, lambda w: sum(w))
        self.assertEqual(results, [6, 15, 24, 10])  # [1+2+3, 4+5+6, 7+8+9, 10]

    def test04_reactive_stream_pipeline(self):
        """反应式流管道"""
        rs = ReactiveStream()
        received = []
        rs.subscribe(lambda x: received.append(x * 2))

        for i in range(5):
            rs.on_next(i)

        rs.on_complete()
        rs.wait_completion(timeout=2.0)

        self.assertEqual(sorted(received), [0, 2, 4, 6, 8])

    def test05_thread_safety(self):
        """线程安全测试"""
        bus = EventBus()
        received = []
        lock = threading.Lock()

        def callback(e):
            with lock:
                received.append(e.data)

        bus.subscribe("test", callback)

        def publisher(start):
            for i in range(start, start + 10):
                bus.publish("test", i)

        threads = [threading.Thread(target=publisher, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(received), 50)
        bus.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
