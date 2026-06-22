"""
Agent注册中心与工厂模式（阶段69）

对应第一性原理：
    1. 注册中心 = Agent类型的单一真相源(Single Source of Truth)
    2. 工厂 = 解耦创建: 消费者只需知道类型名,无需知道具体类
    3. 加载器 = 持久化: Agent可从配置保存/加载,保证可复现
    4. 开闭原则: 新增Agent类型无需修改工厂代码
    5. 类型安全: 工厂验证配置类型与注册的Agent类型匹配

模块组成：
    - AgentTypeInfo: Agent类型信息
    - AgentRegistry: Agent注册中心
    - AgentFactory: Agent工厂
    - AgentLoader: Agent加载器
    - 全局单例: get_agent_registry/get_agent_factory/get_agent_loader
"""
from __future__ import annotations

import os
import json
import uuid
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from pathlib import Path

from .base import (
    BaseAgent, AgentConfig, AgentState, AgentRunResult,
)
from .conversation import ConversationAgent, ConversationConfig
from .code import CodeAgent, CodeAgentConfig
from .research import ResearchAgent, ResearchAgentConfig
from .workflow import WorkflowAgent, WorkflowAgentConfig


logger = logging.getLogger("seva.agents.registry")


# ============================================================================
# Agent类型信息
# ============================================================================

@dataclass
class AgentTypeInfo:
    """Agent类型信息"""
    type_name: str                                    # 类型名(唯一标识)
    agent_class: Type[BaseAgent]                      # Agent类
    config_class: Type[AgentConfig]                   # 配置类
    description: str = ""                             # 描述
    version: str = "1.0.0"                            # 版本
    category: str = "custom"                          # 分类(builtin/custom)
    aliases: List[str] = field(default_factory=list)  # 别名
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转为字典(不含类引用)"""
        return {
            "type_name": self.type_name,
            "agent_class": self.agent_class.__name__,
            "config_class": self.config_class.__name__,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "aliases": list(self.aliases),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Agent注册中心
# ============================================================================

class AgentRegistry:
    """
    Agent注册中心

    职责:
        1. 注册/注销Agent类型
        2. 按类型名查询Agent类和配置类
        3. 支持别名机制
        4. 列出所有已注册类型

    第一性原理:
        - 注册中心是Agent类型的唯一真相源
        - 所有Agent创建必须通过注册中心查询类型
        - 注册操作幂等(重复注册同类型会覆盖)
    """

    def __init__(self):
        self._types: Dict[str, AgentTypeInfo] = {}
        self._aliases: Dict[str, str] = {}  # 别名 -> 类型名
        self._lock = threading.RLock()
        # 注册内置类型
        self._register_builtins()

    def _register_builtins(self):
        """注册内置Agent类型"""
        builtins = [
            AgentTypeInfo(
                type_name="base",
                agent_class=BaseAgent,
                config_class=AgentConfig,
                description="基础Agent(需子类化使用)",
                category="builtin",
                aliases=["basic"],
            ),
            AgentTypeInfo(
                type_name="conversation",
                agent_class=ConversationAgent,
                config_class=ConversationConfig,
                description="对话Agent: 多轮对话、会话管理、上下文压缩",
                category="builtin",
                aliases=["chat", "dialogue"],
            ),
            AgentTypeInfo(
                type_name="code",
                agent_class=CodeAgent,
                config_class=CodeAgentConfig,
                description="编程Agent: 代码生成、审查、执行、调试",
                category="builtin",
                aliases=["coder", "programming"],
            ),
            AgentTypeInfo(
                type_name="research",
                agent_class=ResearchAgent,
                config_class=ResearchAgentConfig,
                description="研究Agent: 信息检索、分析、报告生成",
                category="builtin",
                aliases=["researcher", "investigate"],
            ),
            AgentTypeInfo(
                type_name="workflow",
                agent_class=WorkflowAgent,
                config_class=WorkflowAgentConfig,
                description="工作流Agent: DAG步骤编排、状态机驱动",
                category="builtin",
                aliases=["orchestrator", "pipeline"],
            ),
        ]
        for info in builtins:
            self._types[info.type_name] = info
            for alias in info.aliases:
                self._aliases[alias] = info.type_name

    def register(self, type_name: str,
                 agent_class: Type[BaseAgent],
                 config_class: Type[AgentConfig],
                 description: str = "",
                 version: str = "1.0.0",
                 category: str = "custom",
                 aliases: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册Agent类型

        Args:
            type_name: 类型名(唯一标识)
            agent_class: Agent类(必须是BaseAgent的子类)
            config_class: 配置类(必须是AgentConfig的子类)
            description: 描述
            version: 版本
            category: 分类(builtin/custom)
            aliases: 别名列表
            metadata: 额外元数据

        Returns:
            是否注册成功
        """
        if not type_name:
            raise ValueError("type_name不能为空")

        # 验证类继承关系
        if not (isinstance(agent_class, type) and issubclass(agent_class, BaseAgent)):
            raise TypeError(f"agent_class必须是BaseAgent的子类,得到: {agent_class}")
        if not (isinstance(config_class, type) and issubclass(config_class, AgentConfig)):
            raise TypeError(f"config_class必须是AgentConfig的子类,得到: {config_class}")

        with self._lock:
            info = AgentTypeInfo(
                type_name=type_name,
                agent_class=agent_class,
                config_class=config_class,
                description=description,
                version=version,
                category=category,
                aliases=list(aliases or []),
                metadata=dict(metadata or {}),
            )
            self._types[type_name] = info
            # 注册别名
            for alias in (aliases or []):
                self._aliases[alias] = type_name

            logger.info(f"[Registry] 注册Agent类型: {type_name} "
                       f"({agent_class.__name__})")
            return True

    def unregister(self, type_name: str) -> bool:
        """注销Agent类型"""
        with self._lock:
            if type_name not in self._types:
                return False
            info = self._types.pop(type_name)
            # 移除别名
            for alias in info.aliases:
                self._aliases.pop(alias, None)
            logger.info(f"[Registry] 注销Agent类型: {type_name}")
            return True

    def get(self, type_name: str) -> Optional[AgentTypeInfo]:
        """获取类型信息(支持别名)"""
        resolved = self.resolve_alias(type_name)
        with self._lock:
            return self._types.get(resolved)

    def get_class(self, type_name: str) -> Optional[Type[BaseAgent]]:
        """获取Agent类"""
        info = self.get(type_name)
        return info.agent_class if info else None

    def get_config_class(self, type_name: str) -> Optional[Type[AgentConfig]]:
        """获取配置类"""
        info = self.get(type_name)
        return info.config_class if info else None

    def resolve_alias(self, name: str) -> str:
        """解析别名为类型名"""
        with self._lock:
            if name in self._types:
                return name
            if name in self._aliases:
                return self._aliases[name]
            return name

    def is_registered(self, type_name: str) -> bool:
        """检查是否已注册"""
        return self.get(type_name) is not None

    def list_types(self) -> List[str]:
        """列出所有类型名"""
        with self._lock:
            return list(self._types.keys())

    def list_aliases(self) -> Dict[str, str]:
        """列出所有别名"""
        with self._lock:
            return dict(self._aliases)

    def list_detailed(self) -> List[AgentTypeInfo]:
        """列出所有类型详细信息"""
        with self._lock:
            return list(self._types.values())

    def list_by_category(self, category: str) -> List[AgentTypeInfo]:
        """按分类列出类型"""
        with self._lock:
            return [info for info in self._types.values()
                    if info.category == category]

    def clear(self):
        """清空所有注册(含内置)"""
        with self._lock:
            self._types.clear()
            self._aliases.clear()

    def reset(self):
        """重置为初始状态(重新注册内置)"""
        with self._lock:
            self._types.clear()
            self._aliases.clear()
            self._register_builtins()

    def count(self) -> int:
        """已注册类型数"""
        with self._lock:
            return len(self._types)

    def get_report(self) -> Dict[str, Any]:
        """获取注册中心报告"""
        with self._lock:
            return {
                "total_types": len(self._types),
                "total_aliases": len(self._aliases),
                "types": [info.to_dict() for info in self._types.values()],
                "categories": {
                    cat: len([t for t in self._types.values()
                             if t.category == cat])
                    for cat in set(t.category for t in self._types.values())
                },
            }


# ============================================================================
# Agent工厂
# ============================================================================

class AgentFactory:
    """
    Agent工厂

    职责:
        1. 按类型名创建Agent实例
        2. 从配置字典创建Agent
        3. 验证配置
        4. 创建默认配置的Agent

    第一性原理:
        - 工厂解耦了创建者与具体类
        - 消费者只需知道类型名和配置
        - 工厂确保类型安全(配置类匹配)
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        self._registry = registry or AgentRegistry()
        self._lock = threading.RLock()
        self.stats = {
            "agents_created": 0,
            "creation_errors": 0,
        }

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def create(self, type_name: str,
               config: Optional[AgentConfig] = None,
               agent_id: Optional[str] = None,
               **kwargs) -> BaseAgent:
        """
        创建Agent实例

        Args:
            type_name: 类型名(或别名)
            config: 配置对象(如不提供则用默认配置)
            agent_id: Agent ID
            **kwargs: 额外构造参数

        Returns:
            Agent实例

        Raises:
            ValueError: 类型未注册或配置不匹配
        """
        with self._lock:
            try:
                info = self._registry.get(type_name)
                if info is None:
                    raise ValueError(f"未注册的Agent类型: {type_name}")

                # 验证配置类型
                if config is not None:
                    if not isinstance(config, info.config_class):
                        raise TypeError(
                            f"配置类型不匹配: 期望 {info.config_class.__name__}, "
                            f"得到 {type(config).__name__}"
                        )

                if config is None:
                    # 使用默认配置
                    agent = info.agent_class(
                        agent_id=agent_id,
                        **kwargs,
                    )
                else:
                    agent = info.agent_class(
                        agent_id=agent_id,
                        config=config,
                        **kwargs,
                    )
                self.stats["agents_created"] += 1
                logger.info(f"[Factory] 创建Agent: type={type_name}, "
                           f"id={agent.agent_id}")
                return agent
            except Exception as e:
                self.stats["creation_errors"] += 1
                logger.error(f"[Factory] 创建Agent失败: {e}")
                raise

    def create_default(self, type_name: str) -> BaseAgent:
        """创建默认配置的Agent"""
        return self.create(type_name)

    def create_from_dict(self, data: Dict[str, Any]) -> BaseAgent:
        """
        从字典创建Agent

        字典格式:
            {
                "type": "conversation",  # Agent类型
                "config": {              # 配置(可选)
                    "name": "my_agent",
                    ...
                }
            }
        """
        if "type" not in data:
            raise ValueError("字典必须包含'type'字段")

        type_name = data["type"]
        info = self._registry.get(type_name)
        if info is None:
            raise ValueError(f"未注册的Agent类型: {type_name}")

        # 从字典构造配置
        config = None
        if "config" in data and data["config"]:
            config_dict = data["config"]
            try:
                config = info.config_class(**config_dict)
            except TypeError as e:
                raise ValueError(f"配置参数错误: {e}")

        agent_id = data.get("agent_id")

        return self.create(type_name, config=config, agent_id=agent_id)

    def validate_config(self, type_name: str,
                        config: AgentConfig) -> bool:
        """验证配置是否匹配类型"""
        info = self._registry.get(type_name)
        if info is None:
            return False
        return isinstance(config, info.config_class)

    def get_supported_types(self) -> List[str]:
        """获取支持的类型列表"""
        return self._registry.list_types()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return dict(self.stats)


# ============================================================================
# Agent加载器
# ============================================================================

class AgentLoader:
    """
    Agent加载器

    职责:
        1. 从文件/字典加载Agent配置并创建实例
        2. 将Agent配置保存到文件/字典
        3. 批量加载多个Agent

    第一性原理:
        - 加载器实现Agent的持久化与可复现性
        - 配置格式为JSON,便于跨语言交互
        - 保存的是配置而非实例状态,保证可重建
    """

    def __init__(self, factory: Optional[AgentFactory] = None):
        self._factory = factory or AgentFactory()
        self._lock = threading.RLock()

    @property
    def factory(self) -> AgentFactory:
        return self._factory

    def load_from_dict(self, data: Dict[str, Any]) -> BaseAgent:
        """从字典加载Agent"""
        return self._factory.create_from_dict(data)

    def load_from_file(self, path: Union[str, Path]) -> BaseAgent:
        """
        从JSON文件加载Agent

        文件格式:
            {
                "type": "conversation",
                "config": {
                    "name": "my_agent",
                    ...
                }
            }
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.load_from_dict(data)

    def load_batch(self, configs: List[Dict[str, Any]]) -> List[BaseAgent]:
        """批量加载多个Agent"""
        agents = []
        for cfg in configs:
            try:
                agent = self.load_from_dict(cfg)
                agents.append(agent)
            except Exception as e:
                logger.error(f"[Loader] 加载Agent失败: {e}, 配置: {cfg}")
                agents.append(None)
        return agents

    def load_batch_from_file(self, path: Union[str, Path]) -> List[BaseAgent]:
        """
        从JSON文件批量加载Agent

        文件格式:
            [
                {"type": "conversation", "config": {...}},
                {"type": "code", "config": {...}}
            ]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("批量加载文件必须包含JSON数组")

        return self.load_batch(data)

    def save_to_dict(self, agent: BaseAgent) -> Dict[str, Any]:
        """
        将Agent配置保存为字典

        注意: 保存的是配置,不是运行时状态
        """
        # 查找Agent类型
        type_name = self._find_type_name(agent)
        if type_name is None:
            raise ValueError(f"无法确定Agent类型: {type(agent).__name__}")

        config_dict = {}
        if agent.config is not None:
            # 获取配置的所有字段
            try:
                config_dict = agent.config.__dict__
                # 移除私有字段
                config_dict = {
                    k: v for k, v in config_dict.items()
                    if not k.startswith("_")
                }
            except Exception:
                config_dict = {}

        return {
            "type": type_name,
            "agent_id": agent.agent_id,
            "config": config_dict,
        }

    def save_to_file(self, agent: BaseAgent,
                     path: Union[str, Path]) -> bool:
        """将Agent配置保存到JSON文件"""
        data = self.save_to_dict(agent)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    def _find_type_name(self, agent: BaseAgent) -> Optional[str]:
        """根据Agent实例查找注册类型名(精确匹配类)"""
        agent_class = type(agent)
        for info in self._factory.registry.list_detailed():
            if info.agent_class is agent_class:
                return info.type_name
        return None


# ============================================================================
# 全局单例
# ============================================================================

_global_registry: Optional[AgentRegistry] = None
_global_factory: Optional[AgentFactory] = None
_global_loader: Optional[AgentLoader] = None
_global_lock = threading.RLock()


def get_agent_registry() -> AgentRegistry:
    """获取全局Agent注册中心"""
    global _global_registry
    with _global_lock:
        if _global_registry is None:
            _global_registry = AgentRegistry()
        return _global_registry


def reset_agent_registry():
    """重置全局Agent注册中心"""
    global _global_registry
    with _global_lock:
        _global_registry = None


def get_agent_factory() -> AgentFactory:
    """获取全局Agent工厂"""
    global _global_factory
    with _global_lock:
        if _global_factory is None:
            _global_factory = AgentFactory(get_agent_registry())
        return _global_factory


def reset_agent_factory():
    """重置全局Agent工厂"""
    global _global_factory
    with _global_lock:
        _global_factory = None


def get_agent_loader() -> AgentLoader:
    """获取全局Agent加载器"""
    global _global_loader
    with _global_lock:
        if _global_loader is None:
            _global_loader = AgentLoader(get_agent_factory())
        return _global_loader


def reset_agent_loader():
    """重置全局Agent加载器"""
    global _global_loader
    with _global_lock:
        _global_loader = None


def reset_all_registry():
    """重置所有全局单例"""
    reset_agent_loader()
    reset_agent_factory()
    reset_agent_registry()


# ============================================================================
# 便捷函数
# ============================================================================

def create_agent(type_name: str,
                 config: Optional[AgentConfig] = None,
                 **kwargs) -> BaseAgent:
    """便捷函数: 创建Agent"""
    return get_agent_factory().create(type_name, config=config, **kwargs)


def list_agent_types() -> List[str]:
    """便捷函数: 列出所有Agent类型"""
    return get_agent_registry().list_types()


def register_agent_type(type_name: str,
                        agent_class: Type[BaseAgent],
                        config_class: Type[AgentConfig],
                        **kwargs) -> bool:
    """便捷函数: 注册Agent类型"""
    return get_agent_registry().register(
        type_name, agent_class, config_class, **kwargs
    )
