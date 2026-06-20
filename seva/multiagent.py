"""
多Agent协作系统（MultiAgentCoordinator）
阶段27: 多Agent协作系统
- Agent通信协议: 结构化消息(REQUEST/RESPONSE/BROADCAST/COORDINATE)
- 任务分配: 基于能力匹配+负载均衡+历史表现
- 协调机制: 中央协调器管理注册、分配、收集、聚合

第一性原理:
- 多Agent协作 = 将复杂任务分解为子任务，分配给最合适的Agent
- 通信 = Agent间通过结构化消息交换信息
- 协调 = 中央协调器管理Agent生命周期和任务流转
- 聚合 = 收集各Agent结果，合并为最终输出
"""
import time
import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from .types import Goal, GoalType, TaskState, gen_id
from .task import Task
from .config import Config

logger = logging.getLogger("seva.multiagent")


class AgentMessageType(Enum):
    """Agent间消息类型"""
    REQUEST = "request"        # 请求（任务分配）
    RESPONSE = "response"      # 响应（任务结果）
    BROADCAST = "broadcast"    # 广播（状态更新）
    COORDINATE = "coordinate"  # 协调（同步信号）


class AgentStatus(Enum):
    """Agent状态"""
    IDLE = "idle"          # 空闲
    BUSY = "busy"          # 忙碌（执行任务中）
    DONE = "done"          # 完成
    FAILED = "failed"      # 失败
    OFFLINE = "offline"    # 离线


@dataclass
class AgentInfo:
    """Agent能力与状态信息"""
    agent_id: str
    name: str
    capabilities: Set[GoalType] = field(default_factory=set)  # 支持的目标类型
    skill_q_values: Dict[str, float] = field(default_factory=dict)  # 技能名->Q值
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    total_tasks_assigned: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """任务成功率"""
        total = self.total_tasks_completed + self.total_tasks_failed
        return self.total_tasks_completed / max(total, 1)

    @property
    def is_available(self) -> bool:
        """是否可用（空闲且在线）"""
        return self.status == AgentStatus.IDLE

    @property
    def load(self) -> float:
        """当前负载（0=空闲, 1=忙碌）"""
        return 0.0 if self.is_available else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": [c.value for c in self.capabilities],
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "total_assigned": self.total_tasks_assigned,
            "total_completed": self.total_tasks_completed,
            "total_failed": self.total_tasks_failed,
            "success_rate": round(self.success_rate, 4),
            "is_available": self.is_available,
        }


@dataclass
class AgentMessage:
    """Agent间通信消息"""
    message_id: str
    msg_type: AgentMessageType
    sender_id: str
    receiver_id: Optional[str]  # None = 广播
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    in_reply_to: Optional[str] = None  # 回复的消息ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class MultiAgentCoordinator:
    """
    多Agent协调器
    - 注册/注销Agent
    - 任务分配（能力匹配+负载均衡+历史表现）
    - 消息路由
    - 结果聚合
    - 协调轮次执行
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self.messages: List[AgentMessage] = []
        self.message_inbox: Dict[str, List[AgentMessage]] = {}  # agent_id -> 消息列表
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_results: Dict[str, Dict[str, Any]] = {}  # task_id -> 结果
        self.max_messages = 1000
        # 统计
        self.total_tasks_distributed = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_messages_sent = 0
        logger.info("[MultiAgentCoordinator] 初始化")

    def register_agent(self, agent_info: AgentInfo) -> bool:
        """注册Agent"""
        if agent_info.agent_id in self.agents:
            logger.warning(f"[Coordinator] Agent已存在: {agent_info.agent_id}")
            return False
        self.agents[agent_info.agent_id] = agent_info
        self.message_inbox[agent_info.agent_id] = []
        logger.info(f"[Coordinator] 注册Agent {agent_info.name} "
                    f"caps={[c.value for c in agent_info.capabilities]}")
        return True

    def unregister_agent(self, agent_id: str):
        """注销Agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = AgentStatus.OFFLINE
            del self.agents[agent_id]
            if agent_id in self.message_inbox:
                del self.message_inbox[agent_id]
            logger.info(f"[Coordinator] 注销Agent {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """获取Agent信息"""
        return self.agents.get(agent_id)

    def get_available_agents(self) -> List[AgentInfo]:
        """获取所有可用Agent"""
        return [a for a in self.agents.values() if a.is_available]

    def assign_task(self, task: Task) -> Optional[str]:
        """
        分配任务给最合适的Agent
        策略: 能力匹配 → 负载均衡 → 历史表现
        返回: agent_id，无可用Agent返回None
        """
        goal_type = task.goal.type

        # 1. 筛选有能力且可用的Agent
        candidates = [a for a in self.agents.values()
                      if a.is_available and goal_type in a.capabilities]
        if not candidates:
            # 无能力匹配的Agent，尝试任意可用Agent
            candidates = self.get_available_agents()
        if not candidates:
            logger.warning(f"[Coordinator] 无可用Agent分配任务 {task.task_id}")
            return None

        # 2. 按综合得分排序: 成功率(0.5) + 负载(0.3) + 技能匹配(0.2)
        def score(agent: AgentInfo) -> float:
            success_score = agent.success_rate * 0.5
            load_score = (1.0 - agent.load) * 0.3
            # 技能匹配度
            skill_score = 0.0
            if agent.skill_q_values:
                relevant = [q for name, q in agent.skill_q_values.items()
                           if goal_type.value in name]
                skill_score = (sum(relevant) / len(relevant)) * 0.2 \
                    if relevant else 0.0
            return success_score + load_score + skill_score

        candidates.sort(key=score, reverse=True)
        best_agent = candidates[0]

        # 3. 分配任务
        best_agent.status = AgentStatus.BUSY
        best_agent.current_task_id = task.task_id
        best_agent.total_tasks_assigned += 1
        best_agent.last_heartbeat = time.time()
        self.task_assignments[task.task_id] = best_agent.agent_id
        self.total_tasks_distributed += 1

        # 4. 发送任务分配消息
        self.send_message(AgentMessage(
            message_id=gen_id("msg"),
            msg_type=AgentMessageType.REQUEST,
            sender_id="coordinator",
            receiver_id=best_agent.agent_id,
            content={
                "task_id": task.task_id,
                "goal_type": goal_type.value,
                "goal_params": task.goal.parameters,
                "max_steps": task.max_steps,
            }
        ))

        logger.info(f"[Coordinator] 分配任务 {task.task_id} -> "
                    f"Agent {best_agent.name} (score={score(best_agent):.3f})")
        return best_agent.agent_id

    def report_result(self, agent_id: str, task_id: str,
                      success: bool, result: Dict[str, Any]):
        """Agent报告任务结果"""
        agent = self.agents.get(agent_id)
        if agent is None:
            logger.warning(f"[Coordinator] 未知Agent: {agent_id}")
            return

        agent.current_task_id = None
        agent.status = AgentStatus.IDLE if success else AgentStatus.IDLE
        agent.last_heartbeat = time.time()

        if success:
            agent.total_tasks_completed += 1
            self.total_tasks_completed += 1
        else:
            agent.total_tasks_failed += 1
            self.total_tasks_failed += 1

        self.task_results[task_id] = {
            "agent_id": agent_id,
            "success": success,
            "result": result,
            "timestamp": time.time(),
        }

        # 发送确认消息
        self.send_message(AgentMessage(
            message_id=gen_id("msg"),
            msg_type=AgentMessageType.RESPONSE,
            sender_id="coordinator",
            receiver_id=agent_id,
            content={"task_id": task_id, "acknowledged": True}
        ))

        logger.info(f"[Coordinator] Agent {agent.name} 报告任务 {task_id} "
                    f"{'成功' if success else '失败'}")

    def send_message(self, msg: AgentMessage):
        """发送消息"""
        self.messages.append(msg)
        self.total_messages_sent += 1
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages//2:]

        if msg.receiver_id is None:
            # 广播：发给所有Agent
            for aid in self.message_inbox:
                self.message_inbox[aid].append(msg)
        else:
            # 点对点
            if msg.receiver_id in self.message_inbox:
                self.message_inbox[msg.receiver_id].append(msg)
            if msg.sender_id in self.message_inbox and msg.sender_id != msg.receiver_id:
                # 发送者也保留一份
                pass

    def get_messages(self, agent_id: str) -> List[AgentMessage]:
        """获取Agent的消息"""
        msgs = self.message_inbox.get(agent_id, [])
        # 取出并清空
        self.message_inbox[agent_id] = []
        return msgs

    def broadcast(self, sender_id: str, content: Dict[str, Any]):
        """广播消息"""
        self.send_message(AgentMessage(
            message_id=gen_id("msg"),
            msg_type=AgentMessageType.BROADCAST,
            sender_id=sender_id,
            receiver_id=None,
            content=content
        ))

    def coordinate(self) -> Dict[str, Any]:
        """
        执行一轮协调
        - 检查Agent心跳
        - 清理超时任务
        - 返回当前状态
        """
        now = time.time()
        timeout = 300  # 5分钟超时

        # 检查心跳超时
        for agent in self.agents.values():
            if agent.status == AgentStatus.BUSY:
                if now - agent.last_heartbeat > timeout:
                    logger.warning(f"[Coordinator] Agent {agent.name} 心跳超时")
                    agent.status = AgentStatus.FAILED
                    if agent.current_task_id:
                        self.total_tasks_failed += 1
                        agent.total_tasks_failed += 1
                        agent.current_task_id = None

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        available = sum(1 for a in self.agents.values() if a.is_available)
        busy = sum(1 for a in self.agents.values()
                   if a.status == AgentStatus.BUSY)
        return {
            "total_agents": len(self.agents),
            "available_agents": available,
            "busy_agents": busy,
            "total_tasks_distributed": self.total_tasks_distributed,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "total_messages_sent": self.total_messages_sent,
            "pending_results": len(self.task_results),
            "agents": [a.to_dict() for a in self.agents.values()],
        }

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        return self.task_results.get(task_id)

    def reset(self):
        """重置协调器"""
        self.agents.clear()
        self.messages.clear()
        self.message_inbox.clear()
        self.task_assignments.clear()
        self.task_results.clear()
        self.total_tasks_distributed = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_messages_sent = 0
        logger.info("[Coordinator] 重置完成")

    def health_check(self) -> bool:
        return True
