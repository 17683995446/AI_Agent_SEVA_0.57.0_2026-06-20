"""
决策模块（DecisionMaker）
根据驱力、目标、预测选择动作
"""
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from .types import Action, ActionType, Drives, Goal, GoalType, PerceptionState, gen_id
from .config import Config

logger = logging.getLogger("seva.decision")


class DecisionMaker:
    """决策模块：根据目标与预测选择动作"""

    def __init__(self, config: Config):
        self.config = config
        self.last_action: Optional[Action] = None
        self.candidate_goals: List[Goal] = []
        logger.info("[DecisionMaker] 初始化")

    def select_action(self, drives: Drives, goals: List[Goal],
                      state: PerceptionState,
                      prediction: Optional[Any] = None) -> Action:
        """选择动作"""
        self.candidate_goals = goals
        logger.debug(f"[DecisionMaker] 选择动作 goals={len(goals)} "
                     f"drives=({drives.safety:.2f},{drives.gain:.2f},{drives.efficiency:.2f})")

        if not goals:
            # 无目标时等待
            action = Action(
                action_id=gen_id("act"),
                type=ActionType.WAIT,
                parameters={},
                expected_reward=0.0,
                source="decision"
            )
            self.last_action = action
            return action

        # 选择最高优先级目标
        primary_goal = max(goals, key=lambda g: g.priority)
        logger.debug(f"[DecisionMaker] 主目标: {primary_goal.type.value} "
                     f"prio={primary_goal.priority:.2f}")

        # 根据目标类型生成动作
        action = self._goal_to_action(primary_goal, drives, state)
        self.last_action = action
        logger.info(f"[DecisionMaker] 选择动作: {action.type.value}")
        return action

    def _goal_to_action(self, goal: Goal, drives: Drives,
                        state: PerceptionState) -> Action:
        """将目标转化为具体动作"""
        if goal.type == GoalType.COLLECT:
            return Action(
                action_id=gen_id("act"),
                type=ActionType.COLLECT,
                parameters={"target": goal.parameters.get("target", "resource")},
                expected_reward=1.0 * goal.priority,
                source="decision"
            )
        elif goal.type == GoalType.AVOID:
            return Action(
                action_id=gen_id("act"),
                type=ActionType.MOVE,
                parameters={"direction": "away_from_threat"},
                expected_reward=0.5 * goal.priority,
                source="decision"
            )
        elif goal.type == GoalType.EXPLORE:
            direction = goal.parameters.get("direction", "random")
            if direction == "random":
                direction = np.random.choice(["up", "down", "left", "right"])
            return Action(
                action_id=gen_id("act"),
                type=ActionType.MOVE,
                parameters={"direction": direction},
                expected_reward=0.2,
                source="decision"
            )
        elif goal.type == GoalType.MAINTAIN:
            return Action(
                action_id=gen_id("act"),
                type=ActionType.WAIT,
                parameters={},
                expected_reward=0.1,
                source="decision"
            )
        elif goal.type == GoalType.LEARN:
            return Action(
                action_id=gen_id("act"),
                type=ActionType.OBSERVE,
                parameters={"focus": goal.parameters.get("focus", "all")},
                expected_reward=0.3,
                source="decision"
            )
        else:
            return Action(
                action_id=gen_id("act"),
                type=ActionType.WAIT,
                parameters={},
                expected_reward=0.0,
                source="decision"
            )

    def get_candidate_goals(self) -> List[Goal]:
        return self.candidate_goals

    def get_last_action(self) -> Optional[Action]:
        return self.last_action

    def health_check(self) -> bool:
        return True
