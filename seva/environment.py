"""
仿真环境（SimulationEnvironment）
对应设计文档第35章
四种环境：空白/混合/漂移/对抗
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from .config import Config

logger = logging.getLogger("seva.env")


class SimulationEnvironment:
    """
    仿真环境
    - 2D网格世界
    - 资源点、威胁源、其他Agent
    - 传感器噪声、动作延迟、随机事件
    """

    def __init__(self, config: Config, env_type: str = "mixed", seed: int = 42):
        self.config = config
        self.env_type = env_type
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.grid_size = config.get("environment", "grid_size")
        self.num_resources = config.get("environment", "num_resources")
        self.num_threats = config.get("environment", "num_threats")
        self.sensor_noise = config.get("environment", "sensor_noise_std")
        self.action_delay = config.get("environment", "action_delay_steps")
        self.reward_resource = config.get("environment", "reward_resource")
        self.reward_threat = config.get("environment", "reward_threat")
        self.reward_step = config.get("environment", "reward_step")
        self.max_steps = config.get("environment", "max_steps")

        self.input_dim = config.get("perception", "input_dim")

        self.step_count = 0
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.resources: List[np.ndarray] = []
        self.threats: List[np.ndarray] = []
        self.other_agents: List[np.ndarray] = []
        self.pending_actions: List = []
        self.events: List[Dict] = []
        self.distribution_shifted = False

        self._init_environment()
        logger.info(f"[SimEnv] 初始化 type={env_type} grid={self.grid_size} "
                    f"seed={seed}")

    def _init_environment(self):
        """初始化环境元素"""
        if self.env_type == "blank":
            # 空白环境：无资源无威胁
            pass
        else:
            # 混合/漂移/对抗环境
            for _ in range(self.num_resources):
                pos = self.rng.integers(0, self.grid_size, size=2)
                self.resources.append(pos)
            for _ in range(self.num_threats):
                pos = self.rng.integers(0, self.grid_size, size=2)
                self.threats.append(pos)
            # 其他Agent
            for _ in range(2):
                pos = self.rng.integers(0, self.grid_size, size=2)
                self.other_agents.append(pos)

        # 漂移环境的事件
        if self.env_type == "drift":
            self.events.append({"time": 5000, "type": "distribution_shift"})
        # 对抗环境的事件
        if self.env_type == "adversarial":
            self.events.append({"time": 100, "type": "adversarial_start", "interval": 100})

    def reset(self) -> np.ndarray:
        """重置环境，返回初始观测"""
        self.step_count = 0
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.rng = np.random.default_rng(self.seed)
        self.resources = []
        self.threats = []
        self.other_agents = []
        self.pending_actions = []
        self.distribution_shifted = False
        self._init_environment()
        logger.info("[SimEnv] 重置完成")
        return self._generate_observation()

    def _generate_observation(self) -> np.ndarray:
        """生成观测向量"""
        # 观测：自身位置 + 最近资源 + 最近威胁 + 其他Agent
        obs_parts = []

        # 自身位置（归一化）
        obs_parts.append(self.agent_pos / self.grid_size)

        # 最近资源
        if self.resources:
            dists = [np.linalg.norm(self.agent_pos - r) for r in self.resources]
            nearest_idx = np.argmin(dists)
            obs_parts.append(self.resources[nearest_idx] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))

        # 最近威胁
        if self.threats:
            dists = [np.linalg.norm(self.agent_pos - t) for t in self.threats]
            nearest_idx = np.argmin(dists)
            obs_parts.append(self.threats[nearest_idx] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))

        # 其他Agent
        if self.other_agents:
            obs_parts.append(self.other_agents[0] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))

        obs = np.concatenate(obs_parts)

        # 对抗环境：注入对抗噪声
        if self.env_type == "adversarial" and self.step_count > 0:
            if self.step_count % 100 == 0:
                # FGSM风格扰动
                noise = self.rng.normal(0, 0.3, obs.shape)
                obs = obs + noise
                logger.debug(f"[SimEnv] 对抗扰动 step={self.step_count}")

        # 传感器噪声
        noise = self.rng.normal(0, self.sensor_noise, obs.shape)
        obs = obs + noise

        # 对齐到input_dim
        if len(obs) < self.input_dim:
            obs = np.pad(obs, (0, self.input_dim - len(obs)))
        elif len(obs) > self.input_dim:
            obs = obs[:self.input_dim]

        return obs.astype(np.float64)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """推进一步"""
        self.step_count += 1
        reward = self.reward_step
        info: Dict[str, Any] = {}

        # 处理事件
        self._process_events()

        # 执行动作
        from .types import ActionType
        if hasattr(action, "type"):
            if action.type == ActionType.MOVE:
                direction = action.parameters.get("direction", "up")
                if direction == "up":
                    self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
                elif direction == "down":
                    self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
                elif direction == "left":
                    self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
                elif direction == "right":
                    self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
                elif direction == "away_from_threat":
                    # 远离最近威胁
                    if self.threats:
                        dists = [np.linalg.norm(self.agent_pos - t) for t in self.threats]
                        nearest = self.threats[np.argmin(dists)]
                        diff = self.agent_pos - nearest
                        if np.linalg.norm(diff) > 0:
                            move = np.sign(diff).astype(int)
                            self.agent_pos = np.clip(
                                self.agent_pos + move, 0, self.grid_size - 1)
            elif action.type == ActionType.COLLECT:
                # 检查是否在资源附近
                for i, r in enumerate(self.resources):
                    if np.linalg.norm(self.agent_pos - r) < 2:
                        reward += self.reward_resource
                        self.resources.pop(i)
                        info["collected"] = True
                        logger.debug(f"[SimEnv] 收集资源 reward+={self.reward_resource}")
                        break
            elif action.type == ActionType.WAIT:
                pass
            elif action.type == ActionType.OBSERVE:
                reward += 0.05  # 观察有小奖励

        # 威胁移动（追踪Agent）
        for i, t in enumerate(self.threats):
            if self.env_type == "drift" and self.distribution_shifted:
                # 漂移后威胁追踪
                diff = self.agent_pos - t
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < 10:
                    move = np.sign(diff).astype(int)
                    self.threats[i] = np.clip(t + move, 0, self.grid_size - 1)
            else:
                # 随机移动
                move = self.rng.integers(-1, 2, size=2)
                self.threats[i] = np.clip(t + move, 0, self.grid_size - 1)

        # 检查威胁碰撞
        for t in self.threats:
            if np.array_equal(self.agent_pos, t):
                reward += self.reward_threat
                info["hit_threat"] = True
                logger.debug(f"[SimEnv] 撞击威胁 reward+={self.reward_threat}")
                break

        # 其他Agent随机移动
        for i, a in enumerate(self.other_agents):
            move = self.rng.integers(-1, 2, size=2)
            self.other_agents[i] = np.clip(a + move, 0, self.grid_size - 1)

        # 补充资源（保持数量）
        if len(self.resources) < self.num_resources and self.rng.random() < 0.01:
            pos = self.rng.integers(0, self.grid_size, size=2)
            self.resources.append(pos)

        observation = self._generate_observation()
        done = self.step_count >= self.max_steps
        info["step"] = self.step_count
        info["agent_pos"] = self.agent_pos.tolist()

        logger.debug(f"[SimEnv] step={self.step_count} reward={reward:.3f} done={done}")
        return observation, reward, done, info

    def _process_events(self):
        """处理随机事件"""
        for event in self.events:
            if event.get("triggered"):
                continue
            if event["type"] == "distribution_shift" and self.step_count >= event["time"]:
                # 分布漂移
                self.resources = []
                self.threats = []
                for _ in range(self.num_resources):
                    self.resources.append(self.rng.integers(0, self.grid_size, size=2))
                for _ in range(self.num_threats):
                    self.threats.append(self.rng.integers(0, self.grid_size, size=2))
                self.distribution_shifted = True
                event["triggered"] = True
                logger.warning(f"[SimEnv] 分布漂移触发 step={self.step_count}")

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态"""
        return {
            "step": self.step_count,
            "agent_pos": self.agent_pos.tolist(),
            "num_resources": len(self.resources),
            "num_threats": len(self.threats),
            "distribution_shifted": self.distribution_shifted,
        }
