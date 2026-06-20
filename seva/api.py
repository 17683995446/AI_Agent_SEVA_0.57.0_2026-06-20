"""
统一API与快速启动模块（阶段29）
对应第一性原理：易用 = 减少必须的样板代码，一行启动

提供：
- quickstart(): 一行代码创建并启动Agent
- SEVAFactory: 工厂模式创建Agent
- 预设配置模板: easy/standard/research/production
- run(): 一行运行N步并返回统计
- PresetConfigs: 预设配置集合

使用示例:
    from seva.api import quickstart
    agent, stats = quickstart(steps=100)  # 一行启动

    from seva.api import SEVAFactory, PresetConfigs
    agent = SEVAFactory.create(preset="standard")
    stats = agent.run(steps=500)
"""
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple
from .config import Config
from .core import SelfEvolvingAgentCore
from .environment import SimulationEnvironment

logger = logging.getLogger("seva.api")


class PresetConfigs:
    """
    预设配置模板集合
    每个模板针对不同使用场景优化
    """
    # 易用模式：小规模、快速、低资源占用
    EASY = {
        "core": {"max_cycles": 1000, "cycle_timeout_ms": 2000,
                 "metacognition_interval": 200},
        "perception": {"input_dim": 16, "feature_dim": 64},
        "memory": {"episodic_capacity": 1000, "semantic_capacity": 100,
                   "consolidation_interval": 200},
        "environment": {"grid_size": 20, "num_resources": 5,
                        "num_threats": 2, "max_steps": 1000},
        "skill": {"max_skills": 30, "acquisition_interval": 100},
        "evolution": {"evaluation_interval": 200},
    }

    # 标准模式：平衡性能与资源，日常使用推荐
    STANDARD = {
        "core": {"max_cycles": 10000, "cycle_timeout_ms": 1000,
                 "metacognition_interval": 100},
        "perception": {"input_dim": 32, "feature_dim": 128},
        "memory": {"episodic_capacity": 10000, "semantic_capacity": 1000,
                   "consolidation_interval": 1000},
        "environment": {"grid_size": 50, "num_resources": 10,
                        "num_threats": 5, "max_steps": 10000},
        "skill": {"max_skills": 100, "acquisition_interval": 50},
        "evolution": {"evaluation_interval": 100},
    }

    # 研究模式：大规模、高精度、长时运行
    RESEARCH = {
        "core": {"max_cycles": 100000, "cycle_timeout_ms": 2000,
                 "metacognition_interval": 50},
        "perception": {"input_dim": 64, "feature_dim": 256},
        "memory": {"episodic_capacity": 50000, "semantic_capacity": 5000,
                   "consolidation_interval": 500},
        "environment": {"grid_size": 100, "num_resources": 20,
                        "num_threats": 10, "max_steps": 50000},
        "skill": {"max_skills": 500, "acquisition_interval": 25},
        "evolution": {"evaluation_interval": 50},
    }

    # 生产模式：稳定优先、资源受限、监控密集
    PRODUCTION = {
        "core": {"max_cycles": 1000000, "cycle_timeout_ms": 500,
                 "metacognition_interval": 100},
        "perception": {"input_dim": 32, "feature_dim": 128},
        "memory": {"episodic_capacity": 20000, "semantic_capacity": 2000,
                   "consolidation_interval": 2000},
        "environment": {"grid_size": 50, "num_resources": 10,
                        "num_threats": 5, "max_steps": 100000},
        "skill": {"max_skills": 200, "acquisition_interval": 100},
        "evolution": {"evaluation_interval": 200},
        "safety": {"circuit_breaker_threshold": 5},  # 生产环境更严格
    }

    @classmethod
    def get(cls, preset: str) -> Dict:
        """获取预设配置"""
        presets = {
            "easy": cls.EASY,
            "standard": cls.STANDARD,
            "research": cls.RESEARCH,
            "production": cls.PRODUCTION,
        }
        if preset not in presets:
            raise ValueError(
                f"未知预设: {preset}。可用预设: {list(presets.keys())}"
            )
        return presets[preset]

    @classmethod
    def list_presets(cls) -> list:
        """列出所有可用预设"""
        return ["easy", "standard", "research", "production"]


class SEVAFactory:
    """
    Agent工厂：简化Agent创建流程
    """
    @staticmethod
    def create(preset: str = "standard",
               env_type: str = "mixed",
               seed: int = 42,
               overrides: Optional[Dict] = None) -> SelfEvolvingAgentCore:
        """
        创建Agent实例

        参数:
            preset: 预设模板 (easy/standard/research/production)
            env_type: 环境类型 (blank/mixed/drift/adversarial)
            seed: 随机种子
            overrides: 配置覆盖（最高优先级）

        返回:
            已初始化的SelfEvolvingAgentCore实例（已附加环境）

        示例:
            agent = SEVAFactory.create("standard")
            agent = SEVAFactory.create("easy", env_type="blank")
        """
        preset_config = PresetConfigs.get(preset)
        if overrides:
            # 深度合并overrides
            for k, v in overrides.items():
                if k in preset_config and isinstance(preset_config[k], dict) \
                   and isinstance(v, dict):
                    preset_config[k].update(v)
                else:
                    preset_config[k] = v

        config = Config(instance_overrides=preset_config)
        core = SelfEvolvingAgentCore(config)
        core.initialize()

        # 附加环境
        env = SimulationEnvironment(config, env_type=env_type, seed=seed)
        core.attach_environment(env)
        # 注入初始观测
        obs = env.reset()
        core.inject_input(obs)

        logger.info(f"[SEVAFactory] 创建Agent preset={preset} "
                    f"env={env_type} seed={seed}")
        return core

    @staticmethod
    def create_without_env(preset: str = "standard",
                           overrides: Optional[Dict] = None
                           ) -> SelfEvolvingAgentCore:
        """
        创建不带环境的Agent（用于纯算法测试）
        """
        preset_config = PresetConfigs.get(preset)
        if overrides:
            for k, v in overrides.items():
                if k in preset_config and isinstance(preset_config[k], dict) \
                   and isinstance(v, dict):
                    preset_config[k].update(v)
                else:
                    preset_config[k] = v

        config = Config(instance_overrides=preset_config)
        core = SelfEvolvingAgentCore(config)
        core.initialize()
        logger.info(f"[SEVAFactory] 创建Agent(无环境) preset={preset}")
        return core


def quickstart(steps: int = 100,
               preset: str = "easy",
               env_type: str = "mixed",
               seed: int = 42,
               verbose: bool = False
               ) -> Tuple[SelfEvolvingAgentCore, Dict[str, Any]]:
    """
    一行代码快速启动Agent并运行

    参数:
        steps: 运行步数
        preset: 预设模板 (easy/standard/research/production)
        env_type: 环境类型 (blank/mixed/drift/adversarial)
        seed: 随机种子
        verbose: 是否打印进度

    返回:
        (agent, stats) 元组

    示例:
        from seva.api import quickstart
        agent, stats = quickstart(steps=100)
        print(stats["total_reward"])
    """
    agent = SEVAFactory.create(preset=preset, env_type=env_type, seed=seed)
    stats = run(agent, steps=steps, verbose=verbose)
    logger.info(f"[quickstart] 完成 steps={steps} "
                f"reward={stats['total_real_reward']:.2f}")
    return agent, stats


def run(agent: SelfEvolvingAgentCore,
        steps: int = 100,
        verbose: bool = False
        ) -> Dict[str, Any]:
    """
    运行Agent指定步数

    参数:
        agent: 已初始化的Agent实例
        steps: 运行步数
        verbose: 是否打印每步进度

    返回:
        统计信息字典
    """
    if agent.environment is None:
        raise RuntimeError("Agent未附加环境，请使用SEVAFactory.create()或"
                          "agent.attach_environment()")

    input_dim = agent.config.get("perception", "input_dim")
    report_interval = max(steps // 10, 1) if verbose else 0

    for i in range(steps):
        if agent._env_done:
            if verbose:
                logger.info(f"[run] 环境结束于第{i}步")
            break
        # 若无pending_input，注入随机输入
        if not hasattr(agent, "_pending_input") or agent._pending_input is None:
            agent.inject_input(np.random.randn(input_dim) * 0.3)
        agent.run_cycle()
        if verbose and report_interval > 0 and (i + 1) % report_interval == 0:
            metrics = agent.get_metrics()
            logger.info(f"[run] 步骤 {i+1}/{steps} "
                       f"reward={metrics['total_reward']:.2f} "
                       f"mode={metrics['mode']}")

    # 从executor获取真实reward统计
    exec_stats = agent.modules["executor"].get_stats()
    metrics = agent.get_metrics()
    return {
        "steps_completed": agent.cycle_count,
        "total_real_reward": exec_stats.get("total_real_reward", 0.0),
        "total_cycles": agent.cycle_count,
        "env_done": agent._env_done,
        "mode": metrics["mode"],
        "avg_cycle_time": metrics["avg_cycle_time"],
        "avg_prediction_error": metrics["avg_prediction_error"],
        "task_stats": metrics.get("task_stats", {}),
        "skill_stats": metrics.get("skill_stats", {}),
        "evolution_stats": metrics.get("evolution_stats", {}),
    }


def create_config(preset: str = "standard",
                  overrides: Optional[Dict] = None) -> Config:
    """
    创建配置对象（便于用户自定义）

    示例:
        from seva.api import create_config
        config = create_config("easy", overrides={"core": {"max_cycles": 500}})
    """
    preset_config = PresetConfigs.get(preset)
    if overrides:
        for k, v in overrides.items():
            if k in preset_config and isinstance(preset_config[k], dict) \
               and isinstance(v, dict):
                preset_config[k].update(v)
            else:
                preset_config[k] = v
    return Config(instance_overrides=preset_config)


def get_status(agent: SelfEvolvingAgentCore) -> Dict[str, Any]:
    """
    获取Agent运行状态摘要（易读格式）

    返回:
        状态摘要字典，含cycle/mode/health/关键指标
    """
    metrics = agent.get_metrics()
    return {
        "cycle_count": metrics["cycle_count"],
        "mode": metrics["mode"],
        "healthy": agent.health_check(),
        "total_reward": round(metrics["total_reward"], 2),
        "avg_cycle_time_ms": round(metrics["avg_cycle_time"] * 1000, 2),
        "avg_prediction_error": round(metrics["avg_prediction_error"], 4),
        "env_attached": agent.environment is not None,
        "env_done": getattr(agent, "_env_done", False),
        "tasks_completed": metrics.get("task_stats", {}).get(
            "total_completed", 0),
        "skills_total": metrics.get("skill_stats", {}).get("total_skills", 0),
        "evolution_evaluations": metrics.get("evolution_stats", {}).get(
            "total_evaluations", 0),
        "current_difficulty": metrics.get("evolution_stats", {}).get(
            "current_difficulty", "EASY"),
    }
