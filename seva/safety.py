"""
安全与伦理模块（SafetyEthicsModule）
对应设计文档第21章
三层架构：本能安全层 / 价值对齐层 / 元伦理层
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from .types import Action, ActionType, SafetyReviewResult, Drives
from .config import Config

logger = logging.getLogger("seva.safety")


class SafetyEthicsModule:
    """
    安全与伦理模块
    - 本能安全层：硬否决危险动作
    - 价值对齐层：软约束，可解释否决
    - 元伦理层：价值冲突时的可解释权衡
    """

    def __init__(self, config: Config):
        self.config = config
        self.instinct_enabled = config.get("safety", "instinct_veto_enabled")
        self.value_enabled = config.get("safety", "value_alignment_enabled")
        self.meta_enabled = config.get("safety", "meta_ethics_enabled")
        self.dangerous_types = set(config.get("safety", "dangerous_action_types"))
        self.cb_threshold = config.get("safety", "circuit_breaker_threshold")

        self.veto_count = 0
        self.approve_count = 0
        self.recent_vetoes: List[float] = []
        self.circuit_breaker_active = False
        self.violation_history: List[str] = []
        logger.info(f"[SafetyEthics] 初始化 instinct={self.instinct_enabled} "
                    f"value={self.value_enabled} meta={self.meta_enabled}")

    def review_action(self, action: Action,
                      context: Optional[Dict] = None) -> SafetyReviewResult:
        """审查动作"""
        logger.debug(f"[SafetyEthics] 审查动作 type={action.type} src={action.source}")

        # 熔断检查
        if self.circuit_breaker_active:
            logger.warning("[SafetyEthics] 熔断激活，否决所有动作")
            return SafetyReviewResult(
                is_approved=False,
                veto_reason="circuit_breaker_active",
                explanation="系统处于熔断状态，所有非紧急动作被否决",
                veto_layer="instinct"
            )

        # 第一层：本能安全（硬否决）
        if self.instinct_enabled:
            result = self._instinct_safety_check(action)
            if not result.is_approved:
                self.veto_count += 1
                self.recent_vetoes.append(1.0)
                self._check_circuit_breaker()
                logger.warning(f"[SafetyEthics] 本能层否决: {result.veto_reason}")
                return result

        # 第二层：价值对齐（软约束）
        if self.value_enabled:
            result = self._value_alignment_check(action, context)
            if not result.is_approved:
                self.veto_count += 1
                self.recent_vetoes.append(1.0)
                self._check_circuit_breaker()
                logger.warning(f"[SafetyEthics] 价值层否决: {result.veto_reason}")
                return result

        # 第三层：元伦理（价值冲突权衡）
        if self.meta_enabled:
            result = self._meta_ethics_check(action, context)
            if not result.is_approved:
                self.veto_count += 1
                self.recent_vetoes.append(1.0)
                self._check_circuit_breaker()
                logger.warning(f"[SafetyEthics] 元伦理层否决: {result.veto_reason}")
                return result

        self.approve_count += 1
        self.recent_vetoes.append(0.0)
        if len(self.recent_vetoes) > 100:
            self.recent_vetoes.pop(0)
        logger.info(f"[SafetyEthics] 动作批准 type={action.type}")
        return SafetyReviewResult(
            is_approved=True,
            veto_layer="none"
        )

    def _instinct_safety_check(self, action: Action) -> SafetyReviewResult:
        """本能安全层检查"""
        if action.type.value in self.dangerous_types:
            self.violation_history.append(f"instinct:{action.type.value}")
            return SafetyReviewResult(
                is_approved=False,
                veto_reason="instinct_safety_hard_veto",
                explanation=f"动作{action.type.value}违反本能安全底线",
                veto_layer="instinct"
            )
        return SafetyReviewResult(is_approved=True, veto_layer="instinct")

    def _value_alignment_check(self, action: Action,
                                context: Optional[Dict]) -> SafetyReviewResult:
        """价值对齐层检查"""
        # 检查参数中是否有欺骗性标志
        if action.parameters.get("deceptive", False):
            self.violation_history.append(f"value:deceptive:{action.type.value}")
            return SafetyReviewResult(
                is_approved=False,
                veto_reason="value_alignment_deception",
                explanation="动作包含欺骗性参数，违反诚实价值对齐",
                veto_layer="value"
            )
        # 检查是否在极端驱力下做出高风险动作
        if context and "drives" in context:
            drives: Drives = context["drives"]
            if drives.gain > 0.95 and action.expected_reward > 2.0:
                self.violation_history.append(f"value:extreme_gain:{action.type.value}")
                return SafetyReviewResult(
                    is_approved=False,
                    veto_reason="value_alignment_extreme_gain_seeking",
                    explanation="在极端收益驱力下追求高风险奖励，可能违反审慎价值",
                    veto_layer="value"
                )
        return SafetyReviewResult(is_approved=True, veto_layer="value")

    def _meta_ethics_check(self, action: Action,
                           context: Optional[Dict]) -> SafetyReviewResult:
        """元伦理层检查"""
        # 检查动作是否可解释
        if not action.parameters:
            # 无参数的动作可能缺乏可解释性，但不直接否决
            pass
        return SafetyReviewResult(is_approved=True, veto_layer="meta")

    def _check_circuit_breaker(self):
        """检查是否应触发熔断"""
        if len(self.recent_vetoes) >= 20:
            recent_veto_rate = float(np.mean(self.recent_vetoes[-20:]))
            if recent_veto_rate > 0.5:  # 最近20次超过一半被否决
                self.circuit_breaker_active = True
                logger.error(f"[SafetyEthics] 熔断触发！近期否决率={recent_veto_rate:.2f}")

    def reset_circuit_breaker(self):
        """重置熔断"""
        self.circuit_breaker_active = False
        self.recent_vetoes = []
        logger.info("[SafetyEthics] 熔断重置")

    def judge(self, dilemma: Dict) -> Dict:
        """伦理困境判断（用于价值漂移检测）"""
        action_type = dilemma.get("action_type", "unknown")
        is_safe = action_type not in self.dangerous_types
        return {
            "action_type": action_type,
            "judgment": "safe" if is_safe else "unsafe",
            "explanation": f"基于{'本能' if not is_safe else '价值'}层判断"
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "veto_count": self.veto_count,
            "approve_count": self.approve_count,
            "circuit_breaker": self.circuit_breaker_active,
            "violations": len(self.violation_history),
        }

    def health_check(self) -> bool:
        return not self.circuit_breaker_active
