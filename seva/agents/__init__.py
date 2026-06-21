"""
SEVA Agent应用层（阶段64-69）

阶段64: Agent基类与生命周期管理
阶段65: 对话Agent
阶段66: 编程Agent
阶段67: 研究Agent
阶段68: 工作流Agent
阶段69: Agent注册中心与工厂模式

快速启动:
    from seva.agents import BaseAgent, AgentConfig, AgentRunner
    from seva.agents import get_agent_runner

    agent = BaseAgent(config=AgentConfig(name="my_agent"))
    runner = get_agent_runner()
    runner.register(agent)
    agent.start()
    result = agent.run("hello")
"""
from .base import (
    AgentState, AgentEventType, AgentEvent,
    AgentConfig, AgentStats, AgentContext, AgentRunResult,
    BaseAgent, AgentRunner,
    get_agent_runner, reset_agent_runner,
)

__all__ = [
    "AgentState",
    "AgentEventType",
    "AgentEvent",
    "AgentConfig",
    "AgentStats",
    "AgentContext",
    "AgentRunResult",
    "BaseAgent",
    "AgentRunner",
    "get_agent_runner",
    "reset_agent_runner",
]
