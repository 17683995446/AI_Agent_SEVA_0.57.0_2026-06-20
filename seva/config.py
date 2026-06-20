"""
配置管理（对应设计文档第32章）
分层配置：默认 -> 环境 -> 实例 -> 运行时
"""
import os
import json
import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("seva.config")


class Config:
    """分层配置管理"""
    DEFAULTS = {
        # 核心循环
        "core": {
            "max_cycles": 100000,
            "cycle_timeout_ms": 1000,
            "enable_metacognition": True,
            "metacognition_interval": 100,
        },
        # 感知模块
        "perception": {
            "input_dim": 32,
            "feature_dim": 128,
            "sensor_noise_std": 0.1,
            "enable_predictive_coding": True,
        },
        # 驱力引擎
        "drive_engine": {
            "safety_weight": 1.0,
            "gain_weight": 0.8,
            "efficiency_weight": 0.6,
            "neuromodulator_baseline": {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "norepinephrine": 0.3,
                "acetylcholine": 0.4,
            },
        },
        # 目标生成
        "goal_generator": {
            "max_goals": 5,
            "htn_max_depth": 3,
            "exploration_rate": 0.2,
        },
        # 预测模型
        "predictor": {
            "bottom_layer_dim": 32,
            "middle_layer_dim": 64,
            "top_layer_dim": 64,
            "prediction_horizon": 5,
        },
        # 误差评估
        "error_evaluator": {
            "surprise_threshold": 5.0,
            "novelty_threshold": 0.8,
        },
        # 优化器
        "optimizer": {
            "online_learning_rate": 0.01,
            "offline_learning_rate": 0.001,
            "fusion_alpha": 0.3,
            "kalman_q": 0.01,  # 过程噪声
            "kalman_r": 0.1,   # 观测噪声
        },
        # 记忆
        "memory": {
            "episodic_capacity": 10000,
            "semantic_capacity": 1000,
            "working_capacity": 100,
            "hnsw_M": 16,
            "hnsw_ef": 50,
            "consolidation_interval": 1000,
            "replay_batch_size": 32,
        },
        # 安全
        "safety": {
            "instinct_veto_enabled": True,
            "value_alignment_enabled": True,
            "meta_ethics_enabled": True,
            "dangerous_action_types": ["self_harm", "harm_others", "destroy_env"],
            "circuit_breaker_threshold": 10,
        },
        # 消息总线
        "bus": {
            "capacity": 10000,
            "enable_recording": False,
        },
        # 仿真环境
        "environment": {
            "grid_size": 50,
            "num_resources": 10,
            "num_threats": 5,
            "sensor_noise_std": 0.1,
            "action_delay_steps": 1,
            "reward_resource": 1.0,
            "reward_threat": -1.0,
            "reward_step": -0.01,
            "max_steps": 10000,
        },
        # 阶段26: 技能学习与增强
        "skill": {
            "max_skills": 100,                # 技能库最大容量
            "learning_rate": 0.1,             # TD学习率
            "prune_threshold": 0.1,           # 价值修剪阈值
            "min_q_value": -2.0,              # Q值下限
            "pattern_threshold": 3,           # 模式重复阈值
            "reward_threshold": 0.0,          # 非负reward即视为成功（步进成本-0.01环境下）
            "acquisition_interval": 50,       # 技能获取检测间隔（周期）
        },
        # 阶段27: 多Agent协作
        "multiagent": {
            "max_agents": 20,                 # 最大Agent数
            "max_messages": 1000,             # 消息队列上限
            "heartbeat_timeout": 300,         # 心跳超时（秒）
            "enable_coordination": True,      # 是否启用协调
        },
        # 阶段28: 自主进化闭环
        "evolution": {
            "evaluation_interval": 100,       # 进化评估间隔（周期）
            "max_history": 50,                # 进化历史上限
            "success_threshold": 0.7,         # 课程提升成功率阈值
            "failure_threshold": 0.3,         # 课程回退失败率阈值
            "evaluation_window": 50,          # 课程评估窗口
        },
    }

    def __init__(self, env_overrides: Optional[Dict] = None,
                 instance_overrides: Optional[Dict] = None):
        self._config = copy.deepcopy(self.DEFAULTS)
        # 环境层覆盖
        if env_overrides:
            self._deep_update(self._config, env_overrides)
            logger.info(f"[Config] 应用环境层覆盖: {list(env_overrides.keys())}")
        # 实例层覆盖
        if instance_overrides:
            self._deep_update(self._config, instance_overrides)
            logger.info(f"[Config] 应用实例层覆盖: {list(instance_overrides.keys())}")
        # 运行时覆盖（环境变量）
        self._apply_env_overrides()
        logger.info("[Config] 配置初始化完成")

    def _deep_update(self, base: Dict, override: Dict):
        """深度更新字典"""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def _apply_env_overrides(self):
        """从环境变量SEVA_<SECTION>_<KEY>读取覆盖"""
        for env_key, env_val in os.environ.items():
            if env_key.startswith("SEVA_"):
                parts = env_key[5:].lower().split("__", 1)
                if len(parts) == 2:
                    section, key = parts
                    if section in self._config and key in self._config[section]:
                        old_val = self._config[section][key]
                        try:
                            # 类型转换
                            if isinstance(old_val, bool):
                                self._config[section][key] = env_val.lower() in ("true", "1", "yes")
                            elif isinstance(old_val, int):
                                self._config[section][key] = int(env_val)
                            elif isinstance(old_val, float):
                                self._config[section][key] = float(env_val)
                            else:
                                self._config[section][key] = env_val
                            logger.debug(f"[Config] 环境变量覆盖 {section}.{key}={env_val}")
                        except ValueError as e:
                            logger.warning(f"[Config] 环境变量类型转换失败 "
                                         f"{env_key}={env_val} err={e}")

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """获取配置值"""
        if section not in self._config:
            logger.warning(f"[Config] 未知section: {section}")
            return default
        if key is None:
            return self._config[section]
        return self._config[section].get(key, default)

    def set_runtime(self, section: str, key: str, value: Any):
        """运行时设置（最高优先级）"""
        if section not in self._config:
            self._config[section] = {}
        old = self._config[section].get(key)
        self._config[section][key] = value
        logger.info(f"[Config] 运行时设置 {section}.{key}: {old} -> {value}")

    def to_dict(self) -> Dict:
        return copy.deepcopy(self._config)

    def validate(self) -> List[str]:
        """配置校验，返回错误列表"""
        errors = []
        if self._config["core"]["max_cycles"] <= 0:
            errors.append("core.max_cycles必须>0")
        if self._config["perception"]["input_dim"] <= 0:
            errors.append("perception.input_dim必须>0")
        if not (0 <= self._config["goal_generator"]["exploration_rate"] <= 1):
            errors.append("goal_generator.exploration_rate必须在[0,1]")
        if not (0 <= self._config["optimizer"]["fusion_alpha"] <= 1):
            errors.append("optimizer.fusion_alpha必须在[0,1]")
        if self._config["memory"]["episodic_capacity"] <= 0:
            errors.append("memory.episodic_capacity必须>0")
        if errors:
            logger.error(f"[Config] 校验失败: {errors}")
        else:
            logger.info("[Config] 校验通过")
        return errors
