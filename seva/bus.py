"""
统一消息格式与消息总线实现
对应设计文档第31章：系统整合架构与消息总线

设计要点：
1. AgentMessage: 模块间通信
2. SystemMessage: 系统级消息（含trace_id用于分布式追踪）
3. MessageBus: 唯一耦合点，支持优先级、TTL、广播
4. 每句关键代码后打印日志，便于测试验证
"""
import time
import uuid
import threading
import logging
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("seva.bus")


class MessageType(Enum):
    """消息类型枚举"""
    DATA = "DATA"          # 数据消息
    COMMAND = "COMMAND"    # 命令消息
    EVENT = "EVENT"        # 事件消息
    ERROR = "ERROR"        # 错误消息
    STATE = "STATE"        # 状态消息
    PREDICTION = "PREDICTION"  # 预测消息
    ACTION = "ACTION"      # 动作消息
    LEARN = "LEARN"        # 学习消息


@dataclass
class AgentMessage:
    """模块间通信消息（对应设计文档第31章）"""
    timestamp: float
    source_module: str
    target_module: str
    msg_type: MessageType
    payload: Dict[str, Any]
    priority: int = 5  # 0-10，用于中断机制

    def __post_init__(self):
        logger.debug(f"[AgentMessage] 创建: {self.source_module}->{self.target_module} "
                     f"type={self.msg_type} priority={self.priority}")


@dataclass
class SystemMessage:
    """系统级消息（对应设计文档第31章）"""
    msg_id: str
    timestamp: float
    source_module: str
    target_module: Optional[str]  # None表示广播
    msg_type: MessageType
    priority: int  # 0-10
    payload: Dict[str, Any]
    ttl: int = 100  # 生存时间（跳数）
    trace_id: str = ""

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())[:8]
        logger.debug(f"[SystemMessage] 创建 id={self.msg_id} trace={self.trace_id} "
                     f"{self.source_module}->{self.target_module} type={self.msg_type}")


class MessageBus:
    """
    消息总线：系统唯一耦合点
    - 支持点对点与广播
    - 支持优先级队列（高优先级抢占）
    - 支持TTL过期
    - 线程安全
    - 支持消息录制与回放（用于测试）
    """
    def __init__(self, capacity: int = 10000, enable_recording: bool = False):
        self.capacity = capacity
        self.enable_recording = enable_recording
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._broadcast_subscribers: List[Callable] = []
        self._queue: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._dropped_count = 0
        self._delivered_count = 0
        self._recording: List[SystemMessage] = []
        logger.info(f"[MessageBus] 初始化 capacity={capacity} recording={enable_recording}")

    def subscribe(self, target_module: str, callback: Callable):
        """订阅发往target_module的消息"""
        with self._lock:
            self._subscribers[target_module].append(callback)
            logger.info(f"[MessageBus] 订阅: {callback.__qualname__} -> {target_module}")

    def subscribe_broadcast(self, callback: Callable):
        """订阅广播消息"""
        with self._lock:
            self._broadcast_subscribers.append(callback)
            logger.info(f"[MessageBus] 广播订阅: {callback.__qualname__}")

    def publish(self, msg: SystemMessage) -> bool:
        """发布消息，返回是否成功入队"""
        with self._lock:
            # TTL检查
            if msg.ttl <= 0:
                logger.warning(f"[MessageBus] 消息TTL过期丢弃 id={msg.msg_id}")
                return False
            # 容量检查
            if len(self._queue) >= self.capacity:
                # 驱逐最低优先级的消息
                self._evict_lowest_priority()
                self._dropped_count += 1
                logger.warning(f"[MessageBus] 队列满，驱逐低优先级消息 "
                               f"dropped_total={self._dropped_count}")
            self._queue.append(msg)
            if self.enable_recording:
                self._recording.append(msg)
            logger.debug(f"[MessageBus] 入队 id={msg.msg_id} prio={msg.priority} "
                         f"size={len(self._queue)}")
            return True

    def _evict_lowest_priority(self):
        """驱逐最低优先级消息（优先级数值越大优先级越高）"""
        if not self._queue:
            return
        # 找到最低优先级（数值最小）
        min_idx = 0
        min_prio = self._queue[0].priority
        for i, msg in enumerate(self._queue):
            if msg.priority < min_prio:
                min_prio = msg.priority
                min_idx = i
        evicted = self._queue[min_idx]
        # 转为list删除再转回（deque不支持按索引删除高效）
        lst = list(self._queue)
        lst.pop(min_idx)
        self._queue = deque(lst, maxlen=self.capacity)
        logger.debug(f"[MessageBus] 驱逐 id={evicted.msg_id} prio={evicted.priority}")

    def drain(self, max_msgs: int = None) -> int:
        """排空队列，派发所有消息。返回派发数量"""
        delivered = 0
        with self._lock:
            # 按优先级排序（高优先级先派发）
            msgs = sorted(self._queue, key=lambda m: -m.priority)
            if max_msgs:
                msgs = msgs[:max_msgs]
            self._queue.clear()
        # 在锁外派发（避免回调中再次publish导致死锁）
        for msg in msgs:
            self._dispatch(msg)
            delivered += 1
        logger.info(f"[MessageBus] drain完成 delivered={delivered}")
        self._delivered_count += delivered
        return delivered

    def _dispatch(self, msg: SystemMessage):
        """派发单条消息"""
        logger.debug(f"[MessageBus] 派发 id={msg.msg_id} "
                     f"{msg.source_module}->{msg.target_module}")
        if msg.target_module is None:
            # 广播
            for cb in list(self._broadcast_subscribers):
                try:
                    cb(msg)
                except Exception as e:
                    logger.error(f"[MessageBus] 广播回调异常 cb={cb.__qualname__} "
                                 f"err={e}", exc_info=True)
        else:
            # 点对点
            for cb in list(self._subscribers.get(msg.target_module, [])):
                try:
                    cb(msg)
                except Exception as e:
                    logger.error(f"[MessageBus] 点对点回调异常 target={msg.target_module} "
                                 f"cb={cb.__qualname__} err={e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取总线统计"""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "capacity": self.capacity,
                "dropped": self._dropped_count,
                "delivered": self._delivered_count,
                "subscribers": {k: len(v) for k, v in self._subscribers.items()},
                "broadcast_subscribers": len(self._broadcast_subscribers),
            }

    def get_recording(self) -> List[SystemMessage]:
        """获取录制消息（用于测试回放）"""
        return list(self._recording)

    def clear(self):
        """清空总线"""
        with self._lock:
            self._queue.clear()
            self._recording.clear()
            logger.info("[MessageBus] 已清空")
