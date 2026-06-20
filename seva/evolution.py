"""
自主进化闭环模块（EvolutionLoop）
阶段28: 自主进化闭环
- 元学习(MetaLearner): 监控学习指标，评估学习策略
- 自我调参(SelfTuner): 自适应调整超参数(学习率/探索率/阈值)
- 课程学习(CurriculumScheduler): 由易到难的任务调度
- 长程自驱(EvolutionLoop): 周期性自我评估与改进

第一性原理:
- 元学习 = 监控"学习过程"本身，找出哪些学习策略有效
- 自我调参 = 根据性能反馈自动调整超参数，无需人工干预
- 课程学习 = 从简单任务开始，掌握后逐步提升难度
- 进化闭环 = 评估→调整→执行→再评估的持续改进循环
"""
import time
import logging
import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from .config import Config

logger = logging.getLogger("seva.evolution")


class DifficultyLevel(Enum):
    """任务难度等级"""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class LearningMetrics:
    """学习指标快照"""
    cycle: int
    avg_error: float
    avg_reward: float
    skill_count: int
    task_success_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TunableParameter:
    """可调参数"""
    name: str
    value: float
    min_value: float
    max_value: float
    adjust_step: float
    history: List[float] = field(default_factory=list)

    def adjust(self, direction: float):
        """调整参数值，direction>0增大，<0减小"""
        old_value = self.value
        self.value = max(self.min_value,
                        min(self.max_value,
                            self.value + direction * self.adjust_step))
        self.history.append(self.value)
        if len(self.history) > 100:
            self.history = self.history[-50:]
        return self.value != old_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "min": self.min_value,
            "max": self.max_value,
            "adjustments": len(self.history),
        }


class MetaLearner:
    """
    元学习器：监控学习过程，评估学习策略
    - 追踪误差趋势（下降/震荡/停滞）
    - 评估收敛速度
    - 计算技能获取率
    - 提供元反馈给自我调参器
    """

    def __init__(self, config: Config):
        self.config = config
        self.metrics_history: List[LearningMetrics] = []
        self.max_history = 100
        self.evaluation_interval = 50  # 评估间隔
        # 元学习状态
        self.error_trend: str = "stable"  # decreasing/stable/increasing/oscillating
        self.convergence_rate: float = 0.0
        self.skill_acquisition_rate: float = 0.0
        self.last_skill_count: int = 0
        logger.info("[MetaLearner] 初始化")

    def record(self, cycle: int, avg_error: float, avg_reward: float,
               skill_count: int, task_success_rate: float):
        """记录学习指标"""
        metrics = LearningMetrics(
            cycle=cycle, avg_error=avg_error, avg_reward=avg_reward,
            skill_count=skill_count, task_success_rate=task_success_rate
        )
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history//2:]

    def evaluate(self) -> Dict[str, Any]:
        """
        评估学习过程
        返回元反馈：误差趋势、收敛速度、技能获取率
        """
        if len(self.metrics_history) < 2:
            return {"error_trend": "insufficient_data",
                    "convergence_rate": 0.0,
                    "skill_acquisition_rate": 0.0}

        # 误差趋势分析
        recent = self.metrics_history[-10:] if len(self.metrics_history) >= 10 \
            else self.metrics_history
        errors = [m.avg_error for m in recent]
        first_half = errors[:len(errors)//2]
        second_half = errors[len(errors)//2:]
        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)

        if avg_second < avg_first * 0.9:
            self.error_trend = "decreasing"
        elif avg_second > avg_first * 1.1:
            self.error_trend = "increasing"
        else:
            # 检查震荡
            variance = sum((e - avg_second) ** 2 for e in errors) / len(errors)
            if variance > avg_second * 0.1:
                self.error_trend = "oscillating"
            else:
                self.error_trend = "stable"

        # 收敛速度
        if len(self.metrics_history) >= 2:
            first = self.metrics_history[0].avg_error
            last = self.metrics_history[-1].avg_error
            cycles = self.metrics_history[-1].cycle - self.metrics_history[0].cycle
            if cycles > 0 and first > 0:
                self.convergence_rate = (first - last) / (first * cycles)
            else:
                self.convergence_rate = 0.0

        # 技能获取率
        if self.metrics_history:
            current_skills = self.metrics_history[-1].skill_count
            skill_delta = current_skills - self.last_skill_count
            cycles_elapsed = (self.metrics_history[-1].cycle -
                             (self.metrics_history[-2].cycle if len(self.metrics_history) >= 2
                              else self.metrics_history[-1].cycle))
            if cycles_elapsed > 0:
                self.skill_acquisition_rate = skill_delta / cycles_elapsed
            self.last_skill_count = current_skills

        return {
            "error_trend": self.error_trend,
            "convergence_rate": round(self.convergence_rate, 6),
            "skill_acquisition_rate": round(self.skill_acquisition_rate, 6),
            "avg_error": round(self.metrics_history[-1].avg_error, 4) if self.metrics_history else 0,
            "avg_reward": round(self.metrics_history[-1].avg_reward, 4) if self.metrics_history else 0,
            "skill_count": self.metrics_history[-1].skill_count if self.metrics_history else 0,
        }

    def get_feedback(self) -> Dict[str, float]:
        """
        获取元反馈（给SelfTuner用）
        - error_direction: 误差方向（>0=增大需减小学习率, <0=减小可增大学习率）
        - exploration_need: 探索需求（>0=需要更多探索）
        - skill_acquisition_need: 技能获取需求（>0=需要降低阈值）
        """
        feedback = {
            "error_direction": 0.0,
            "exploration_need": 0.0,
            "skill_acquisition_need": 0.0,
        }

        if self.error_trend == "oscillating":
            feedback["error_direction"] = -1.0  # 减小学习率
        elif self.error_trend == "decreasing":
            feedback["error_direction"] = 0.5  # 可适当增大学习率
        elif self.error_trend == "increasing":
            feedback["error_direction"] = -0.5  # 减小学习率
        elif self.error_trend == "stable":
            feedback["exploration_need"] = 0.5  # 需要更多探索

        if self.skill_acquisition_rate < 0.01:
            feedback["skill_acquisition_need"] = 0.5  # 需要降低阈值

        return feedback

    def reset(self):
        self.metrics_history.clear()
        self.error_trend = "stable"
        self.convergence_rate = 0.0
        self.skill_acquisition_rate = 0.0
        self.last_skill_count = 0

    def health_check(self) -> bool:
        return True


class SelfTuner:
    """
    自我调参器：根据元学习反馈自动调整超参数
    - 学习率：误差震荡时减小，持续下降时增大
    - 探索率：成功率高时减小，低时增大
    - 模式阈值：技能获取太少时降低
    """

    def __init__(self, config: Config):
        self.config = config
        self.params: Dict[str, TunableParameter] = {
            "learning_rate": TunableParameter(
                name="learning_rate",
                value=config.get("skill", "learning_rate"),
                min_value=0.001, max_value=0.5, adjust_step=0.01
            ),
            "exploration_rate": TunableParameter(
                name="exploration_rate",
                value=config.get("goal_generator", "exploration_rate"),
                min_value=0.05, max_value=0.5, adjust_step=0.02
            ),
            "pattern_threshold": TunableParameter(
                name="pattern_threshold",
                value=float(config.get("skill", "pattern_threshold")),
                min_value=2.0, max_value=10.0, adjust_step=1.0
            ),
        }
        self.adjustment_count = 0
        logger.info("[SelfTuner] 初始化")

    def tune(self, feedback: Dict[str, float]):
        """根据元反馈调整参数"""
        adjustments = []

        # 学习率调整
        error_dir = feedback.get("error_direction", 0.0)
        if error_dir != 0:
            lr = self.params["learning_rate"]
            old = lr.value
            lr.adjust(error_dir)
            if lr.value != old:
                adjustments.append(f"lr:{old:.3f}->{lr.value:.3f}")

        # 探索率调整
        explore_need = feedback.get("exploration_need", 0.0)
        if explore_need > 0:
            er = self.params["exploration_rate"]
            old = er.value
            er.adjust(explore_need)
            if er.value != old:
                adjustments.append(f"explore:{old:.3f}->{er.value:.3f}")

        # 模式阈值调整
        skill_need = feedback.get("skill_acquisition_need", 0.0)
        if skill_need > 0:
            pt = self.params["pattern_threshold"]
            old = pt.value
            pt.adjust(-skill_need)  # 降低阈值
            if pt.value != old:
                adjustments.append(f"threshold:{old:.1f}->{pt.value:.1f}")

        if adjustments:
            self.adjustment_count += 1
            logger.info(f"[SelfTuner] 调整#{self.adjustment_count}: "
                       f"{', '.join(adjustments)}")

        return adjustments

    def get_param(self, name: str) -> float:
        """获取参数值"""
        param = self.params.get(name)
        return param.value if param else 0.0

    def get_all_params(self) -> Dict[str, Any]:
        """获取所有参数"""
        return {name: p.to_dict() for name, p in self.params.items()}

    def reset(self):
        for p in self.params.values():
            p.history.clear()
        self.adjustment_count = 0

    def health_check(self) -> bool:
        return True


class CurriculumScheduler:
    """
    课程调度器：由易到难的任务调度
    - 根据当前能力评估难度等级
    - 掌握当前难度后提升
    - 能力不足时回退
    """

    def __init__(self, config: Config):
        self.config = config
        self.current_difficulty: DifficultyLevel = DifficultyLevel.EASY
        self.difficulty_history: List[Tuple[int, DifficultyLevel, float]] = []
        self.success_threshold = 0.7  # 提升难度的成功率阈值
        self.failure_threshold = 0.3  # 回退难度的失败率阈值
        self.evaluation_window = 50  # 评估窗口（周期数）
        self.recent_results: List[bool] = []  # 最近的任务结果
        self.max_results = 100
        logger.info("[CurriculumScheduler] 初始化 难度=EASY")

    def record_result(self, success: bool):
        """记录任务结果"""
        self.recent_results.append(success)
        if len(self.recent_results) > self.max_results:
            self.recent_results = self.recent_results[-self.max_results//2:]

    def evaluate_and_adjust(self, cycle: int) -> Optional[DifficultyLevel]:
        """
        评估当前表现并调整难度
        返回新的难度等级（如果变化），否则None
        """
        if len(self.recent_results) < 10:
            return None

        success_rate = sum(self.recent_results) / len(self.recent_results)
        old_difficulty = self.current_difficulty

        # 提升难度
        if success_rate >= self.success_threshold:
            if self.current_difficulty.value < DifficultyLevel.EXPERT.value:
                self.current_difficulty = DifficultyLevel(
                    self.current_difficulty.value + 1
                )
                logger.info(f"[Curriculum] 难度提升 "
                           f"{old_difficulty.name}->{self.current_difficulty.name} "
                           f"success_rate={success_rate:.2f}")
        # 回退难度
        elif success_rate < self.failure_threshold:
            if self.current_difficulty.value > DifficultyLevel.EASY.value:
                self.current_difficulty = DifficultyLevel(
                    self.current_difficulty.value - 1
                )
                logger.info(f"[Curriculum] 难度回退 "
                           f"{old_difficulty.name}->{self.current_difficulty.name} "
                           f"success_rate={success_rate:.2f}")

        if self.current_difficulty != old_difficulty:
            self.difficulty_history.append(
                (cycle, self.current_difficulty, success_rate)
            )
            return self.current_difficulty
        return None

    def get_difficulty(self) -> DifficultyLevel:
        return self.current_difficulty

    def get_success_rate(self) -> float:
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_difficulty": self.current_difficulty.name,
            "current_level": self.current_difficulty.value,
            "success_rate": round(self.get_success_rate(), 4),
            "total_results": len(self.recent_results),
            "difficulty_changes": len(self.difficulty_history),
        }

    def reset(self):
        self.current_difficulty = DifficultyLevel.EASY
        self.difficulty_history.clear()
        self.recent_results.clear()

    def health_check(self) -> bool:
        return True


class EvolutionLoop:
    """
    自主进化闭环：整合元学习、自我调参、课程学习
    - 周期性自我评估（每evaluation_interval周期）
    - 根据评估结果调整参数和策略
    - 记录进化历史
    - 长程自驱：持续运行、持续改进
    """

    def __init__(self, config: Config):
        self.config = config
        self.meta_learner = MetaLearner(config)
        self.self_tuner = SelfTuner(config)
        self.curriculum = CurriculumScheduler(config)
        self.evolution_interval = 100  # 进化评估间隔
        self.evolution_history: List[Dict[str, Any]] = []
        self.max_history = 50
        self.total_evaluations = 0
        self.total_adjustments = 0
        logger.info("[EvolutionLoop] 初始化")

    def step(self, cycle: int, avg_error: float, avg_reward: float,
             skill_count: int, task_success_rate: float,
             task_result: Optional[bool] = None) -> Dict[str, Any]:
        """
        进化闭环一步：
        1. 记录学习指标
        2. 记录任务结果（用于课程调度）
        3. 周期性评估和调整
        """
        # 1. 记录学习指标
        self.meta_learner.record(cycle, avg_error, avg_reward,
                                 skill_count, task_success_rate)

        # 2. 记录任务结果
        if task_result is not None:
            self.curriculum.record_result(task_result)

        # 3. 周期性评估和调整
        result = {
            "evaluated": False,
            "adjustments": [],
            "difficulty_change": None,
        }

        if cycle > 0 and cycle % self.evolution_interval == 0:
            result["evaluated"] = True
            self.total_evaluations += 1

            # 元学习评估
            meta_feedback = self.meta_learner.evaluate()
            feedback = self.meta_learner.get_feedback()

            # 自我调参
            adjustments = self.self_tuner.tune(feedback)
            if adjustments:
                self.total_adjustments += len(adjustments)
            result["adjustments"] = adjustments

            # 课程调度
            new_difficulty = self.curriculum.evaluate_and_adjust(cycle)
            if new_difficulty is not None:
                result["difficulty_change"] = new_difficulty.name

            # 记录进化历史
            evolution_entry = {
                "cycle": cycle,
                "meta": meta_feedback,
                "adjustments": adjustments,
                "difficulty": self.curriculum.get_difficulty().name,
                "params": self.self_tuner.get_all_params(),
            }
            self.evolution_history.append(evolution_entry)
            if len(self.evolution_history) > self.max_history:
                self.evolution_history = self.evolution_history[-self.max_history//2:]

            logger.info(f"[EvolutionLoop] 评估#{self.total_evaluations} "
                       f"cycle={cycle} trend={meta_feedback.get('error_trend')} "
                       f"difficulty={self.curriculum.get_difficulty().name}")

        return result

    def get_evolution_stats(self) -> Dict[str, Any]:
        """获取进化统计"""
        return {
            "total_evaluations": self.total_evaluations,
            "total_adjustments": self.total_adjustments,
            "current_difficulty": self.curriculum.get_difficulty().name,
            "success_rate": self.curriculum.get_success_rate(),
            "error_trend": self.meta_learner.error_trend,
            "convergence_rate": self.meta_learner.convergence_rate,
            "skill_acquisition_rate": self.meta_learner.skill_acquisition_rate,
            "params": self.self_tuner.get_all_params(),
            "evolution_history_size": len(self.evolution_history),
        }

    def get_latest_evolution(self) -> Optional[Dict[str, Any]]:
        """获取最近的进化记录"""
        return self.evolution_history[-1] if self.evolution_history else None

    def reset(self):
        self.meta_learner.reset()
        self.self_tuner.reset()
        self.curriculum.reset()
        self.evolution_history.clear()
        self.total_evaluations = 0
        self.total_adjustments = 0
        logger.info("[EvolutionLoop] 重置完成")

    def health_check(self) -> bool:
        return all([
            self.meta_learner.health_check(),
            self.self_tuner.health_check(),
            self.curriculum.health_check(),
        ])
