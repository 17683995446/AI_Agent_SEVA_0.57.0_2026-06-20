"""
执行器模块（Executor）
执行动作并返回结果
"""
import logging
from typing import Any, Dict, Optional
from .types import Action, ActionType
from .config import Config

logger = logging.getLogger("seva.executor")


class Executor:
    """执行器：执行动作"""

    def __init__(self, config: Config):
        self.config = config
        self.execute_count = 0
        self.fail_count = 0
        logger.info("[Executor] 初始化")

    def execute(self, action: Action,
                environment: Optional[Any] = None) -> Dict[str, Any]:
        """执行动作"""
        logger.debug(f"[Executor] 执行 type={action.type} params={action.parameters}")
        self.execute_count += 1

        result = {
            "action_id": action.action_id,
            "type": action.type.value,
            "success": True,
            "reward": action.expected_reward,
            "side_effects": [],
            "error": None
        }

        try:
            if action.type == ActionType.MOVE:
                direction = action.parameters.get("direction", "up")
                result["effect"] = f"moved_{direction}"
                logger.debug(f"[Executor] 移动方向={direction}")
            elif action.type == ActionType.COLLECT:
                result["effect"] = "collected_resource"
                result["reward"] = 1.0
                logger.debug("[Executor] 收集资源")
            elif action.type == ActionType.WAIT:
                result["effect"] = "waited"
                result["reward"] = 0.0
                logger.debug("[Executor] 等待")
            elif action.type == ActionType.OBSERVE:
                result["effect"] = "observed"
                result["reward"] = 0.1
                logger.debug("[Executor] 观察")
            elif action.type == ActionType.SELF_HARM:
                # 不应执行到这里（应被安全模块拦截）
                result["success"] = False
                result["error"] = "dangerous_action_reached_executor"
                self.fail_count += 1
                logger.error("[Executor] 危险动作到达执行器！")
            else:
                result["effect"] = "unknown"
                result["reward"] = 0.0
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.fail_count += 1
            logger.error(f"[Executor] 执行异常: {e}", exc_info=True)

        logger.info(f"[Executor] 完成 success={result['success']} "
                    f"reward={result['reward']}")
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "execute_count": self.execute_count,
            "fail_count": self.fail_count,
        }

    def health_check(self) -> bool:
        return True
