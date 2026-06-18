"""
自主目标生成模块（GoalGenerator）
对应设计文档第10章、第26章
HTN分解 + 主动推断选择
"""
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from .types import Goal, GoalType, Drives, PerceptionState, gen_id
from .config import Config

logger = logging.getLogger("seva.goal")


class GoalGenerator:
    """
    自主目标生成模块
    - 输入：驱力、感知状态、上下文
    - 输出：候选目标列表
    - 含HTN分解与主动推断选择
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_goals = config.get("goal_generator", "max_goals")
        self.htn_max_depth = config.get("goal_generator", "htn_max_depth")
        self.exploration_rate = config.get("goal_generator", "exploration_rate")
        self.last_goals: List[Goal] = []
        self.goal_history: List[GoalType] = []
        logger.info(f"[GoalGenerator] 初始化 max_goals={self.max_goals} "
                    f"explore_rate={self.exploration_rate}")

    def generate(self, drives: Drives, state: PerceptionState,
                 context: Optional[Dict] = None) -> List[Goal]:
        """生成候选目标"""
        logger.debug(f"[GoalGenerator] 生成目标 drives=({drives.safety:.2f},"
                     f"{drives.gain:.2f},{drives.efficiency:.2f})")
        goals: List[Goal] = []

        # 基于驱力生成目标
        # 安全驱力低 -> 生成规避目标
        if drives.safety < 0.4:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.AVOID,
                parameters={"target": "nearest_threat"},
                priority=1.0 - drives.safety,
                expected_free_energy=5.0 * (1.0 - drives.safety)
            ))
            logger.debug(f"[GoalGenerator] 生成AVOID目标 prio={1.0-drives.safety:.2f}")

        # 收益驱力高 -> 生成收集目标
        if drives.gain > 0.5:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.COLLECT,
                parameters={"target": "nearest_resource"},
                priority=drives.gain,
                expected_free_energy=3.0 * (1.0 - drives.gain)
            ))
            logger.debug(f"[GoalGenerator] 生成COLLECT目标 prio={drives.gain:.2f}")

        # 能效驱力高 -> 生成维持目标
        if drives.efficiency > 0.6:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.MAINTAIN,
                parameters={"target": "current_state"},
                priority=drives.efficiency * 0.5,
                expected_free_energy=1.0
            ))
            logger.debug(f"[GoalGenerator] 生成MAINTAIN目标")

        # 探索目标（基于探索率）
        if np.random.random() < self.exploration_rate:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.EXPLORE,
                parameters={"direction": "random"},
                priority=0.3,
                expected_free_energy=2.0
            ))
            logger.debug(f"[GoalGenerator] 生成EXPLORE目标（探索）")

        # 学习目标（当精度低时）
        if state is not None and np.mean(state.precision) < 0.3:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.LEARN,
                parameters={"focus": "low_precision_dimensions"},
                priority=0.6,
                expected_free_energy=1.5
            ))
            logger.debug(f"[GoalGenerator] 生成LEARN目标（低精度）")

        # 如果没有目标，生成默认探索目标
        if not goals:
            goals.append(Goal(
                goal_id=gen_id("goal"),
                type=GoalType.EXPLORE,
                parameters={"direction": "random"},
                priority=0.2,
                expected_free_energy=2.0
            ))
            logger.debug(f"[GoalGenerator] 生成默认EXPLORE目标")

        # 限制数量
        goals = goals[:self.max_goals]
        self.last_goals = goals
        for g in goals:
            self.goal_history.append(g.type)

        logger.info(f"[GoalGenerator] 生成{len(goals)}个目标: "
                    f"{[g.type.value for g in goals]}")
        return goals

    def decompose(self, goal: Goal, depth: int = 0) -> List[Goal]:
        """HTN分解：将复合目标分解为子目标"""
        if depth >= self.htn_max_depth:
            logger.debug(f"[GoalGenerator] HTN达到最大深度 {depth}")
            return [goal]

        sub_goals: List[Goal] = []

        if goal.type == GoalType.EXPLORE:
            sub_goals.append(Goal(
                goal_id=gen_id("subgoal"),
                type=GoalType.MAINTAIN,
                parameters={"target": "energy"},
                priority=goal.priority * 0.8,
                parent_id=goal.goal_id,
                expected_free_energy=0.5
            ))
            sub_goals.append(Goal(
                goal_id=gen_id("subgoal"),
                type=GoalType.COLLECT,
                parameters={"target": "information"},
                priority=goal.priority * 0.6,
                parent_id=goal.goal_id,
                expected_free_energy=1.0
            ))
        elif goal.type == GoalType.COLLECT:
            sub_goals.append(Goal(
                goal_id=gen_id("subgoal"),
                type=GoalType.MAINTAIN,
                parameters={"target": "path_to_resource"},
                priority=goal.priority * 0.9,
                parent_id=goal.goal_id,
                expected_free_energy=0.3
            ))
        else:
            # 不可分解的目标
            sub_goals = [goal]

        logger.debug(f"[GoalGenerator] HTN分解 {goal.type.value} -> "
                     f"{[sg.type.value for sg in sub_goals]}")
        return sub_goals

    def select_by_active_inference(self, goals: List[Goal]) -> Goal:
        """主动推断选择：选择预期自由能最小的目标"""
        if not goals:
            logger.warning("[GoalGenerator] 无目标可选")
            return Goal(
                goal_id=gen_id("goal"),
                type=GoalType.MAINTAIN,
                parameters={},
                priority=0.1
            )
        # 预期自由能越小越好
        selected = min(goals, key=lambda g: g.expected_free_energy)
        logger.info(f"[GoalGenerator] 主动推断选择: {selected.type.value} "
                    f"EFE={selected.expected_free_energy:.2f}")
        return selected

    def get_last_generated_goals(self) -> List[Goal]:
        return self.last_goals

    def get_recent_goals(self, n: int = 10) -> List[GoalType]:
        return self.goal_history[-n:]

    def health_check(self) -> bool:
        return True
