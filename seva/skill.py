"""
技能学习与增强模块（SkillLibrary + SkillAcquisition）
阶段26: 技能学习与增强
- 技能: 可复用的策略片段（前置条件→动作→后置效果）
- 技能库: 存储、检索、Q值更新、修剪
- 技能获取: 从成功经验中检测重复模式，抽象为技能
- 接入决策: 决策时优先使用高价值技能

第一性原理:
- 技能 = "在状态S下执行动作A能获得好结果"的抽象
- 技能学习 = 从交互经验中提取重复成功模式
- 技能增强 = 通过TD学习更新Q值，通过修剪淘汰低价值技能
- 技能应用 = 决策时根据当前目标检索最优技能
"""
import time
import logging
import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from .types import (Goal, GoalType, ActionType, Action, Drives,
                    PerceptionState, ExperienceRecord, gen_id)
from .config import Config

logger = logging.getLogger("seva.skill")


@dataclass
class Skill:
    """
    技能：可复用的策略片段
    - precondition: 触发条件（goal_type + drive_pattern）
    - action_template: 动作模板
    - q_value: Q值估计（TD学习更新）
    - success_rate: 历史成功率
    """
    skill_id: str
    name: str
    goal_type: GoalType                              # 适用的目标类型
    action_type: ActionType                          # 执行的动作类型
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    q_value: float = 0.0                             # Q值估计
    success_count: int = 0                           # 成功次数
    fail_count: int = 0                              # 失败次数
    use_count: int = 0                               # 总使用次数
    total_reward: float = 0.0                        # 累积reward
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    avg_drives: Optional[Tuple[float, float, float]] = None  # (safety, gain, efficiency)

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_count + self.fail_count
        return self.success_count / max(total, 1)

    @property
    def value(self) -> float:
        """综合价值 = Q值 × 成功率"""
        return self.q_value * self.success_rate

    @property
    def is_reliable(self) -> bool:
        """是否可靠（至少使用3次，有足够经验）"""
        total = self.success_count + self.fail_count
        return total >= 3

    def update(self, reward: float, success: bool,
               learning_rate: float = 0.1):
        """
        TD学习更新Q值
        Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        """
        self.use_count += 1
        self.total_reward += reward
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        # TD更新
        td_error = reward - self.q_value
        self.q_value += learning_rate * td_error
        self.last_used_at = time.time()

    def to_action(self) -> Action:
        """从技能生成动作"""
        return Action(
            action_id=gen_id("act"),
            type=self.action_type,
            parameters=dict(self.action_parameters),
            expected_reward=self.q_value,
            source="skill"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "goal_type": self.goal_type.value,
            "action_type": self.action_type.value,
            "q_value": round(self.q_value, 4),
            "success_rate": round(self.success_rate, 4),
            "value": round(self.value, 4),
            "use_count": self.use_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "is_reliable": self.is_reliable,
        }


class SkillLibrary:
    """
    技能库：存储、检索、更新、修剪
    - 检索: 根据目标类型匹配，按价值排序
    - 更新: TD学习更新Q值
    - 修剪: 淘汰低价值技能（使用次数不足且Q值低）
    """

    def __init__(self, config: Config):
        self.config = config
        self.skills: Dict[str, Skill] = {}
        self.max_skills = config.get("skill", "max_skills")
        self.prune_threshold = config.get("skill", "prune_threshold")
        self.min_q_value = config.get("skill", "min_q_value")
        self.learning_rate = config.get("skill", "learning_rate")
        logger.info("[SkillLibrary] 初始化")

    def add(self, skill: Skill) -> str:
        """添加技能"""
        # 去重：相同goal_type + action_type + parameters的技能合并
        existing = self._find_duplicate(skill)
        if existing is not None:
            logger.debug(f"[SkillLibrary] 技能已存在，合并: {skill.name}")
            return existing

        if len(self.skills) >= self.max_skills:
            self.prune()
            # 如果prune后仍满，移除最低价值技能
            if len(self.skills) >= self.max_skills:
                worst_sid = min(self.skills, key=lambda sid: self.skills[sid].value)
                del self.skills[worst_sid]
                logger.debug(f"[SkillLibrary] 移除最低价值技能以腾出空间")

        self.skills[skill.skill_id] = skill
        logger.info(f"[SkillLibrary] 新增技能 {skill.name} "
                    f"goal={skill.goal_type.value} action={skill.action_type.value}")
        return skill.skill_id

    def _find_duplicate(self, skill: Skill) -> Optional[str]:
        """查找重复技能"""
        for sid, existing in self.skills.items():
            if (existing.goal_type == skill.goal_type and
                existing.action_type == skill.action_type and
                existing.action_parameters == skill.action_parameters):
                return sid
        return None

    def retrieve(self, goal: Goal,
                 drives: Optional[Drives] = None) -> Optional[Skill]:
        """
        检索最优技能
        - 按goal_type匹配
        - 按value排序，选择价值最高的可靠技能
        - 无可靠技能时返回None（回退到默认决策）
        """
        candidates = [s for s in self.skills.values()
                      if s.goal_type == goal.type and s.is_reliable]
        if not candidates:
            # 无可靠技能，尝试有使用经验的技能
            candidates = [s for s in self.skills.values()
                          if s.goal_type == goal.type and s.use_count > 0]
        if not candidates:
            return None

        # 按价值排序
        candidates.sort(key=lambda s: s.value, reverse=True)
        best = candidates[0]
        logger.debug(f"[SkillLibrary] 检索到技能 {best.name} "
                     f"value={best.value:.3f} q={best.q_value:.3f}")
        return best

    def update(self, skill_id: str, reward: float, success: bool):
        """更新技能Q值"""
        skill = self.skills.get(skill_id)
        if skill is not None:
            skill.update(reward, success, self.learning_rate)
            logger.debug(f"[SkillLibrary] 更新技能 {skill.name} "
                        f"reward={reward:.3f} q={skill.q_value:.3f}")

    def prune(self):
        """修剪低价值技能"""
        before = len(self.skills)
        to_remove = []
        for sid, skill in self.skills.items():
            total_exp = skill.success_count + skill.fail_count
            # 使用次数>5且价值极低的技能被修剪
            if total_exp > 5 and skill.value < self.prune_threshold:
                to_remove.append(sid)
            # Q值极低且使用次数>3的技能被修剪
            elif total_exp > 3 and skill.q_value < self.min_q_value:
                to_remove.append(sid)
        for sid in to_remove:
            del self.skills[sid]
        if to_remove:
            logger.info(f"[SkillLibrary] 修剪 {len(to_remove)} 个低价值技能 "
                       f"({before}→{len(self.skills)})")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        reliable = [s for s in self.skills.values() if s.is_reliable]
        return {
            "total_skills": len(self.skills),
            "reliable_skills": len(reliable),
            "total_uses": sum(s.use_count for s in self.skills.values()),
            "total_successes": sum(s.success_count for s in self.skills.values()),
            "avg_q_value": (sum(s.q_value for s in self.skills.values()) /
                           max(len(self.skills), 1)),
            "best_skill": max(self.skills.values(), key=lambda s: s.value).to_dict()
                          if self.skills else None,
        }

    def get_all_skills(self) -> List[Skill]:
        """获取所有技能（按价值排序）"""
        return sorted(self.skills.values(), key=lambda s: s.value, reverse=True)

    def reset(self):
        """重置技能库"""
        self.skills.clear()
        logger.info("[SkillLibrary] 重置完成")

    def health_check(self) -> bool:
        return True


class SkillAcquisition:
    """
    技能获取：从成功经验中学习新技能
    - 观察经验序列
    - 检测重复的成功模式（相同goal_type + action_type + 正reward）
    - 当模式重复>=阈值次时，抽象为技能
    """

    def __init__(self, config: Config, library: SkillLibrary):
        self.config = config
        self.library = library
        self.pattern_buffer: List[ExperienceRecord] = []
        self.max_buffer = 200
        self.pattern_threshold = config.get("skill", "pattern_threshold")
        self.reward_threshold = config.get("skill", "reward_threshold")
        # 模式计数: (goal_type, action_type, params_key) -> count
        self._pattern_counts: Dict[Tuple, int] = {}
        logger.info("[SkillAcquisition] 初始化")

    def observe(self, record: ExperienceRecord):
        """观察一条经验，追踪所有模式（不限于正reward）"""
        if record.goal is None or record.action is None:
            return

        self.pattern_buffer.append(record)
        if len(self.pattern_buffer) > self.max_buffer:
            self.pattern_buffer = self.pattern_buffer[-self.max_buffer//2:]

        # 阶段26: 追踪所有模式，用相对reward判断价值
        # 即使在负reward环境（如步进成本-0.01），也能识别相对较好的动作
        pattern_key = self._make_pattern_key(record)
        self._pattern_counts[pattern_key] = \
            self._pattern_counts.get(pattern_key, 0) + 1

    def _make_pattern_key(self, record: ExperienceRecord) -> Tuple:
        """
        生成模式键: (goal_type, action_type)
        不含parameters：MOVE的direction等参数变化不应阻止模式检测
        技能创建时使用最常见参数
        """
        return (record.goal.type, record.action.type)

    def detect_and_create(self) -> Optional[str]:
        """
        检测重复模式并创建技能
        返回新创建的skill_id，无新技能返回None
        """
        new_skill_id = None
        for pattern_key, count in list(self._pattern_counts.items()):
            if count >= self.pattern_threshold:
                goal_type, action_type = pattern_key
                # 收集所有相关经验
                relevant = [r for r in self.pattern_buffer
                           if r.goal is not None and r.action is not None and
                           r.goal.type == goal_type and
                           r.action.type == action_type]
                if not relevant:
                    self._pattern_counts[pattern_key] = 0
                    continue

                # 使用最常见的参数
                params_counter: Dict[Tuple, Dict[str, Any]] = {}
                for r in relevant:
                    params_tuple = tuple(sorted(r.action.parameters.items())) \
                        if r.action.parameters else ()
                    if params_tuple not in params_counter:
                        params_counter[params_tuple] = r.action.parameters
                # 选最常见的参数
                best_params = (max(params_counter.values(),
                                  key=lambda p: sum(1 for r in relevant
                                                  if r.action.parameters == p))
                              if params_counter else {})

                # 检查是否已有此技能
                existing = self._find_existing_skill(goal_type, action_type, best_params)
                if existing is not None:
                    self._pattern_counts[pattern_key] = 0
                    continue

                # 计算平均reward
                avg_reward = sum(r.reward for r in relevant) / len(relevant)

                # 计算平均drives
                drives_list = [(r.drives.safety, r.drives.gain, r.drives.efficiency)
                              for r in relevant if r.drives is not None]
                avg_drives = None
                if drives_list:
                    avg_drives = (
                        sum(d[0] for d in drives_list) / len(drives_list),
                        sum(d[1] for d in drives_list) / len(drives_list),
                        sum(d[2] for d in drives_list) / len(drives_list),
                    )

                # 创建技能
                skill = Skill(
                    skill_id=gen_id("skill"),
                    name=f"{goal_type.value}_{action_type.value}_{count}",
                    goal_type=goal_type,
                    action_type=action_type,
                    action_parameters=best_params,
                    q_value=avg_reward,
                    avg_drives=avg_drives,
                )
                # success = reward >= avg_reward（相对成功）
                success_count = sum(1 for r in relevant
                                   if r.reward >= avg_reward)
                skill.success_count = success_count
                skill.fail_count = len(relevant) - success_count
                skill.use_count = count
                skill.total_reward = sum(r.reward for r in relevant)

                new_skill_id = self.library.add(skill)
                logger.info(f"[SkillAcquisition] 从{count}次重复模式创建技能: "
                           f"{skill.name} q={skill.q_value:.3f} "
                           f"success_rate={skill.success_rate:.2f}")

                # 重置计数
                self._pattern_counts[pattern_key] = 0

        return new_skill_id

    def _find_existing_skill(self, goal_type: GoalType,
                             action_type: ActionType,
                             params: Dict) -> Optional[str]:
        """查找是否已有匹配技能"""
        for sid, skill in self.library.skills.items():
            if (skill.goal_type == goal_type and
                skill.action_type == action_type and
                skill.action_parameters == params):
                return sid
        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "buffer_size": len(self.pattern_buffer),
            "tracked_patterns": len(self._pattern_counts),
            "patterns_above_threshold": sum(
                1 for c in self._pattern_counts.values()
                if c >= self.pattern_threshold
            ),
        }

    def reset(self):
        """重置"""
        self.pattern_buffer.clear()
        self._pattern_counts.clear()
        logger.info("[SkillAcquisition] 重置完成")

    def health_check(self) -> bool:
        return True
