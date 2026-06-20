"""
执行器模块（Executor）
执行动作并返回结果
阶段24: 接入SimulationEnvironment，使用真实环境reward
"""
import logging
from typing import Any, Dict, Optional
from .types import Action, ActionType
from .config import Config

logger = logging.getLogger("seva.executor")


class Executor:
    """执行器：执行动作，接入真实环境获取reward"""

    def __init__(self, config: Config):
        self.config = config
        self.execute_count = 0
        self.fail_count = 0
        self.total_real_reward = 0.0
        logger.info("[Executor] 初始化")

    def execute(self, action: Action,
                environment: Optional[Any] = None) -> Dict[str, Any]:
        """
        执行动作
        - 若environment提供：调用env.step()获取真实reward和下一观测
        - 若environment为None：使用伪reward（向后兼容）
        """
        logger.debug(f"[Executor] 执行 type={action.type} params={action.parameters} "
                     f"env={'有' if environment else '无'}")
        self.execute_count += 1

        result = {
            "action_id": action.action_id,
            "type": action.type.value,
            "success": True,
            "reward": action.expected_reward,
            "side_effects": [],
            "error": None,
            "next_observation": None,
            "env_done": False,
            "env_info": {},
        }

        try:
            if environment is not None and hasattr(environment, "step"):
                # 阶段24: 真实环境闭环
                obs, reward, done, info = environment.step(action)
                result["reward"] = float(reward)
                result["next_observation"] = obs
                result["env_done"] = done
                result["env_info"] = info
                result["effect"] = info.get("effect", "env_executed")
                self.total_real_reward += float(reward)
                if info.get("hit_threat"):
                    result["side_effects"].append("hit_threat")
                if info.get("collected"):
                    result["side_effects"].append("collected_resource")
                logger.debug(f"[Executor] 环境执行 reward={reward:.3f} "
                             f"done={done} info_keys={list(info.keys())}")
            else:
                # 向后兼容：无环境时使用伪reward
                self._execute_without_env(action, result)
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.fail_count += 1
            logger.error(f"[Executor] 执行异常: {e}", exc_info=True)

        logger.info(f"[Executor] 完成 success={result['success']} "
                    f"reward={result['reward']:.3f}")
        return result

    def _execute_without_env(self, action: Action, result: Dict[str, Any]):
        """无环境时的伪执行（向后兼容）"""
        if action.type == ActionType.MOVE:
            direction = action.parameters.get("direction", "up")
            result["effect"] = f"moved_{direction}"
        elif action.type == ActionType.COLLECT:
            result["effect"] = "collected_resource"
            result["reward"] = 1.0
        elif action.type == ActionType.WAIT:
            result["effect"] = "waited"
            result["reward"] = 0.0
        elif action.type == ActionType.OBSERVE:
            result["effect"] = "observed"
            result["reward"] = 0.1
        elif action.type == ActionType.SELF_HARM:
            result["success"] = False
            result["error"] = "dangerous_action_reached_executor"
            self.fail_count += 1
            logger.error("[Executor] 危险动作到达执行器！")
        else:
            result["effect"] = "unknown"
            result["reward"] = 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "execute_count": self.execute_count,
            "fail_count": self.fail_count,
            "total_real_reward": self.total_real_reward,
        }

    def health_check(self) -> bool:
        return True
