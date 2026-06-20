"""
核心数据结构定义
对应设计文档中的统一状态表示、动作、目标等
"""
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger("seva.types")


class GoalType(Enum):
    """目标类型"""
    EXPLORE = "explore"
    COLLECT = "collect"
    AVOID = "avoid"
    MAINTAIN = "maintain"
    COMMUNICATE = "communicate"
    LEARN = "learn"


class TaskState(Enum):
    """任务状态（阶段25: 长任务自驱状态机）"""
    TODO = "todo"          # 待执行
    DOING = "doing"        # 执行中
    DONE = "done"          # 已完成
    FAILED = "failed"      # 失败
    CANCELLED = "cancelled"  # 已取消


class ActionType(Enum):
    """动作类型"""
    MOVE = "move"
    COLLECT = "collect"
    WAIT = "wait"
    COMMUNICATE = "communicate"
    OBSERVE = "observe"
    SELF_HARM = "self_harm"          # 危险动作（测试安全模块）
    HARM_OTHERS = "harm_others"      # 危险动作
    DESTROY_ENV = "destroy_env"      # 危险动作


@dataclass
class PerceptionState:
    """感知状态：统一内部表示"""
    internal: np.ndarray          # 内部状态向量
    external: np.ndarray          # 外部环境观测
    precision: np.ndarray         # 精度（不确定性倒数）
    timestamp: float = field(default_factory=time.time)
    is_fallback: bool = False     # 是否为fallback状态

    def __post_init__(self):
        logger.debug(f"[PerceptionState] 创建 shape_internal={np.shape(self.internal)} "
                     f"fallback={self.is_fallback}")


@dataclass
class Drives:
    """三大驱力"""
    safety: float       # [0,1]，越高越安全
    gain: float         # [0,1]，越高收益越大
    efficiency: float   # [0,1]，越高越能效

    def __post_init__(self):
        # 归一化校验
        for name, val in [("safety", self.safety), ("gain", self.gain),
                          ("efficiency", self.efficiency)]:
            # 修复Bug10: NaN/Inf处理（max/min对NaN无效）
            if np.isnan(val) or np.isinf(val):
                logger.warning(f"[Drives] {name}={val} 非有限值，置为0.5")
                val = 0.5
            elif not (0.0 <= val <= 1.0):
                logger.warning(f"[Drives] {name}={val} 超出[0,1]，将截断")
                val = max(0.0, min(1.0, val))
            setattr(self, name, val)
        logger.debug(f"[Drives] safety={self.safety:.3f} gain={self.gain:.3f} "
                     f"eff={self.efficiency:.3f}")


@dataclass
class Neuromodulators:
    """虚拟神经调质"""
    dopamine: float = 0.5          # 多巴胺：与收益驱力相关
    serotonin: float = 0.5         # 血清素：与安全驱力相关
    norepinephrine: float = 0.3    # 去甲肾上腺素：与惊奇/威胁相关
    acetylcholine: float = 0.4     # 乙酰胆碱：与注意力相关
    dopamine_baseline: float = 0.5
    serotonin_baseline: float = 0.5
    norepinephrine_baseline: float = 0.3
    acetylcholine_baseline: float = 0.4

    def reset_to_baseline(self):
        self.dopamine = self.dopamine_baseline
        self.serotonin = self.serotonin_baseline
        self.norepinephrine = self.norepinephrine_baseline
        self.acetylcholine = self.acetylcholine_baseline
        logger.debug("[Neuromodulators] 重置到基线")


@dataclass
class Goal:
    """目标"""
    goal_id: str
    type: GoalType
    parameters: Dict[str, Any]
    priority: float  # [0,1]
    deadline: float = float('inf')  # 截止时间
    parent_id: Optional[str] = None
    expected_free_energy: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.priority <= 1.0):
            logger.warning(f"[Goal] priority={self.priority} 超出[0,1]，截断")
            self.priority = max(0.0, min(1.0, self.priority))
        logger.debug(f"[Goal] 创建 id={self.goal_id} type={self.type} prio={self.priority}")


@dataclass
class Prediction:
    """预测结果"""
    predicted_state: np.ndarray
    confidence: float  # [0,1]
    variance: np.ndarray  # 预测方差
    horizon: int  # 预测步长
    source_layer: str = "bottom"  # bottom/middle/top/fused

    def __post_init__(self):
        logger.debug(f"[Prediction] layer={self.source_layer} horizon={self.horizon} "
                     f"conf={self.confidence:.3f}")


@dataclass
class Action:
    """动作"""
    action_id: str
    type: ActionType
    parameters: Dict[str, Any]
    expected_reward: float = 0.0
    source: str = "decision"  # decision/safe_fallback/emergency

    def __post_init__(self):
        logger.debug(f"[Action] 创建 id={self.action_id} type={self.type} src={self.source}")


@dataclass
class ErrorSignal:
    """误差信号"""
    magnitude: float
    perception_error: np.ndarray
    prediction_error: np.ndarray
    action_error: float
    surprise: float  # 负对数证据
    is_novel: bool = False

    def __post_init__(self):
        logger.debug(f"[ErrorSignal] mag={self.magnitude:.3f} surprise={self.surprise:.3f} "
                     f"novel={self.is_novel}")


@dataclass
class ErrorAttribution:
    """四源贝叶斯归因"""
    perception: float  # 感知源
    model: float       # 模型源
    action: float      # 动作源
    environment: float # 环境源

    def __post_init__(self):
        total = self.perception + self.model + self.action + self.environment
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"[ErrorAttribution] 归因和={total} != 1.0，将归一化")
            if total > 0:
                self.perception /= total
                self.model /= total
                self.action /= total
                self.environment /= total
        logger.debug(f"[ErrorAttribution] perc={self.perception:.3f} "
                     f"model={self.model:.3f} act={self.action:.3f} "
                     f"env={self.environment:.3f}")


@dataclass
class ExperienceRecord:
    """经验记录（用于记忆存储）"""
    record_id: str
    timestamp: float
    state: PerceptionState
    drives: Drives
    goal: Optional[Goal]
    prediction: Optional[Prediction]
    action: Optional[Action]
    error: Optional[ErrorSignal]
    attribution: Optional[ErrorAttribution]
    reward: float = 0.0

    def __post_init__(self):
        logger.debug(f"[ExperienceRecord] 创建 id={self.record_id} reward={self.reward}")


@dataclass
class SafetyReviewResult:
    """安全审查结果"""
    is_approved: bool
    veto_reason: str = ""
    explanation: str = ""
    veto_layer: str = ""  # instinct/value/meta

    def __post_init__(self):
        logger.debug(f"[SafetyReview] approved={self.is_approved} "
                     f"layer={self.veto_layer} reason={self.veto_reason}")


def gen_id(prefix: str = "") -> str:
    """生成唯一ID"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
