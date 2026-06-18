"""
误差评估反馈模块（ErrorEvaluator）
对应设计文档第12章、第28章
四源贝叶斯归因 + 惊奇计算 + 意外检测
"""
import numpy as np
import logging
from typing import Optional, Dict, Any
from .types import (ErrorSignal, ErrorAttribution, PerceptionState,
                    Prediction, Action)
from .config import Config

logger = logging.getLogger("seva.error")


class ErrorEvaluator:
    """
    误差评估反馈模块
    - 输入：感知状态、预测、动作、实际结果
    - 输出：误差信号 + 四源归因 + 惊奇
    """

    def __init__(self, config: Config):
        self.config = config
        self.surprise_threshold = config.get("error_evaluator", "surprise_threshold")
        self.novelty_threshold = config.get("error_evaluator", "novelty_threshold")
        self.surprise_history: list = []
        self.last_error: Optional[ErrorSignal] = None
        self.last_attribution: Optional[ErrorAttribution] = None
        logger.info(f"[ErrorEvaluator] 初始化 surprise_thresh={self.surprise_threshold}")

    def evaluate(self, state: PerceptionState,
                 prediction: Optional[Prediction],
                 action: Optional[Action],
                 actual_state: Optional[PerceptionState] = None,
                 context: Optional[Dict] = None) -> ErrorSignal:
        """评估误差"""
        logger.debug("[ErrorEvaluator] 评估误差")

        # 修复Bug8: 输入数值保护
        state_internal = np.nan_to_num(state.internal, nan=0.0, posinf=10.0, neginf=-10.0)
        state_internal = np.clip(state_internal, -10.0, 10.0)
        state_precision = np.nan_to_num(state.precision, nan=0.5, posinf=1.0, neginf=0.0)
        state_precision = np.clip(state_precision, 0.0, 1.0)

        # 感知误差：状态内部的方差
        perception_error = state_internal * (1.0 - state_precision)
        perception_error = np.clip(perception_error, -10.0, 10.0)
        perception_mag = float(min(np.linalg.norm(perception_error), 100.0))

        # 预测误差：预测与实际的差
        if prediction is not None and actual_state is not None:
            # 对齐维度
            pred = np.nan_to_num(prediction.predicted_state, nan=0.0, posinf=10.0, neginf=-10.0)
            pred = np.clip(pred, -10.0, 10.0)
            actual = np.nan_to_num(actual_state.internal, nan=0.0, posinf=10.0, neginf=-10.0)
            actual = np.clip(actual, -10.0, 10.0)
            if pred.shape != actual.shape:
                pred = np.resize(pred, actual.shape)
            prediction_error = actual - pred
            prediction_error = np.clip(prediction_error, -20.0, 20.0)
            prediction_mag = float(min(np.linalg.norm(prediction_error), 200.0))
        else:
            prediction_error = np.zeros_like(state_internal)
            prediction_mag = 0.0

        # 动作误差：预期奖励与实际奖励的差（简化）
        if action is not None and context is not None:
            actual_reward = context.get("actual_reward", 0.0)
            action_error = float(abs(action.expected_reward - actual_reward))
        else:
            action_error = 0.0

        # 总误差
        magnitude = perception_mag + prediction_mag + action_error

        # 惊奇 = 负对数证据
        # 修复Bug12: compute_surprise接收处理后的state_internal，避免NaN传播
        surprise = self.compute_surprise(state_internal, prediction)

        # 意外检测
        is_novel = self.detect_novelty_from_surprise(surprise)

        error = ErrorSignal(
            magnitude=magnitude,
            perception_error=perception_error,
            prediction_error=prediction_error,
            action_error=action_error,
            surprise=surprise,
            is_novel=is_novel
        )
        self.last_error = error
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > 1000:
            self.surprise_history.pop(0)

        logger.info(f"[ErrorEvaluator] 误差 mag={magnitude:.4f} surprise={surprise:.4f} "
                    f"novel={is_novel}")
        return error

    def compute_surprise(self, observation: np.ndarray,
                         prediction: Optional[Prediction]) -> float:
        """计算惊奇 = 负对数证据"""
        # 修复Bug12: 输入NaN/Inf保护
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = np.clip(observation, -10.0, 10.0)
        if prediction is None:
            # 无预测时，基于观测范数
            return float(min(np.linalg.norm(observation), 100.0))

        pred_mean = np.nan_to_num(prediction.predicted_state, nan=0.0, posinf=10.0, neginf=-10.0)
        pred_mean = np.clip(pred_mean, -10.0, 10.0)
        # 修复Bug5: 方差下界提高，防止diff²/var爆炸
        pred_var = np.nan_to_num(prediction.variance, nan=0.01, posinf=1.0, neginf=0.01)
        pred_var = np.clip(pred_var, 0.01, 10.0)  # 方差下界0.01

        # 对齐维度
        if pred_mean.shape != observation.shape:
            pred_mean = np.resize(pred_mean, observation.shape)
            pred_var = np.resize(pred_var, observation.shape)

        # 高斯负对数似然
        diff = observation - pred_mean
        # 修复Bug5: diff裁剪防止爆炸
        diff = np.clip(diff, -10.0, 10.0)
        surprise = float(0.5 * np.sum(
            np.log(2 * np.pi * pred_var) + diff ** 2 / pred_var
        ))
        # 修复Bug5: surprise上限; 修复Bug12: NaN保护
        if np.isnan(surprise) or np.isinf(surprise):
            surprise = 100.0
        surprise = min(surprise, 100.0)
        logger.debug(f"[ErrorEvaluator] 惊奇={surprise:.4f}")
        return surprise

    def attribute_error(self, error: ErrorSignal,
                        context: Optional[Dict] = None) -> ErrorAttribution:
        """四源贝叶斯归因"""
        # 修复Bug26: 各权重NaN/Inf保护
        perc_norm = float(np.linalg.norm(error.perception_error))
        model_norm = float(np.linalg.norm(error.prediction_error))
        if not np.isfinite(perc_norm):
            perc_norm = 0.0
        if not np.isfinite(model_norm):
            model_norm = 0.0
        if not np.isfinite(error.action_error):
            action_err = 0.0
        else:
            action_err = error.action_error

        # 简化贝叶斯归因：基于各源误差的相对大小
        perc_weight = perc_norm + 1e-6
        model_weight = model_norm + 1e-6
        action_weight = action_err + 1e-6

        # 环境源：基于惊奇超出阈值的部分
        surprise_val = error.surprise if np.isfinite(error.surprise) else 0.0
        env_weight = max(0.0, surprise_val - self.surprise_threshold) + 1e-6

        total = perc_weight + model_weight + action_weight + env_weight
        # 修复Bug26: total为NaN/0时兜底
        if not np.isfinite(total) or total <= 0:
            total = 4e-6
            perc_weight = model_weight = action_weight = env_weight = 1e-6
        attribution = ErrorAttribution(
            perception=perc_weight / total,
            model=model_weight / total,
            action=action_weight / total,
            environment=env_weight / total
        )
        self.last_attribution = attribution
        logger.debug(f"[ErrorEvaluator] 归因 perc={attribution.perception:.3f} "
                     f"model={attribution.model:.3f} act={attribution.action:.3f} "
                     f"env={attribution.environment:.3f}")
        return attribution

    def detect_novelty_from_surprise(self, surprise: float) -> bool:
        """基于惊奇检测意外"""
        is_novel = surprise > self.surprise_threshold
        logger.debug(f"[ErrorEvaluator] 意外检测 surprise={surprise:.3f} "
                     f"thresh={self.surprise_threshold} novel={is_novel}")
        return is_novel

    def detect_novelty(self, context: Dict) -> bool:
        """从上下文检测意外（用于测试接口）"""
        surprise = context.get("surprise", 0.0)
        return self.detect_novelty_from_surprise(surprise)

    def get_surprise_trend(self) -> float:
        """获取惊奇趋势（正=上升，负=下降）"""
        if len(self.surprise_history) < 10:
            return 0.0
        recent = self.surprise_history[-10:]
        mid = len(recent) // 2
        return float(np.mean(recent[mid:]) - np.mean(recent[:mid]))

    def health_check(self) -> bool:
        return True
