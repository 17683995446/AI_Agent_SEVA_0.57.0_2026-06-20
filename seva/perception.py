"""
环境感知模块（PerceptionModule）
对应设计文档第8章、第24章
CNN+预测编码混合架构（简化为MLP+预测编码，便于纯Python运行）
"""
import time
import numpy as np
import logging
from typing import Any, Dict, Optional
from .types import PerceptionState
from .config import Config

logger = logging.getLogger("seva.perception")


class PredictiveCodingLayer:
    """预测编码层：预测-比较-修正"""

    def __init__(self, dim: int, learning_rate: float = 0.01):
        self.dim = dim
        self.lr = learning_rate
        # 简单线性预测器：next = W * current + b
        self.W = np.eye(dim) * 0.9  # 初始为衰减矩阵
        self.b = np.zeros(dim)
        self.last_prediction = np.zeros(dim)
        self.last_error = np.zeros(dim)
        logger.info(f"[PredictiveCodingLayer] 初始化 dim={dim} lr={learning_rate}")

    def predict(self, current: np.ndarray) -> np.ndarray:
        """预测下一步"""
        pred = self.W @ current + self.b
        self.last_prediction = pred.copy()
        logger.debug(f"[PredictiveCodingLayer] 预测 norm={np.linalg.norm(pred):.4f}")
        return pred

    def compute_error(self, prediction: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """计算预测误差 = 观测 - 预测"""
        error = observation - prediction
        self.last_error = error.copy()
        logger.debug(f"[PredictiveCodingLayer] 误差 norm={np.linalg.norm(error):.4f}")
        return error

    def update(self, current: np.ndarray, observation: np.ndarray):
        """Hebbian学习：根据误差更新参数"""
        # 修复Bug10: 输入NaN/Inf保护，防止污染W和b
        current = np.nan_to_num(current, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        prediction = self.predict(current)
        error = self.compute_error(prediction, observation)
        # 修复Bug5: 梯度裁剪，防止误差爆炸
        error_norm = np.linalg.norm(error)
        # 修复Bug10: error_norm为NaN时跳过更新
        if np.isnan(error_norm) or np.isinf(error_norm):
            logger.warning("[PredictiveCodingLayer] 误差范数异常，跳过更新")
            return
        max_error_norm = 1.0
        if error_norm > max_error_norm:
            error = error * (max_error_norm / error_norm)
            logger.debug(f"[PredictiveCodingLayer] 误差裁剪 {error_norm:.4f}->{max_error_norm}")
        # 输入裁剪
        current_norm = np.linalg.norm(current)
        if current_norm > 10.0:
            current = current * (10.0 / current_norm)
        # 梯度下降：W += lr * error * current^T
        self.W += self.lr * np.outer(error, current)
        self.b += self.lr * error
        # 修复Bug5: 更严格的数值稳定性，限制W的范数
        w_norm = np.linalg.norm(self.W)
        # 修复Bug10: w_norm为NaN时重置W
        if np.isnan(w_norm) or np.isinf(w_norm):
            logger.warning("[PredictiveCodingLayer] W范数异常，重置W")
            self.W = np.eye(self.dim) * 0.9
            self.b = np.zeros(self.dim)
            return
        max_w_norm = 2.0  # 从10.0降到2.0
        if w_norm > max_w_norm:
            self.W *= max_w_norm / w_norm
            logger.debug(f"[PredictiveCodingLayer] W范数截断 to {max_w_norm}")
        # 限制b的范围
        b_norm = np.linalg.norm(self.b)
        if np.isnan(b_norm) or np.isinf(b_norm):
            self.b = np.zeros(self.dim)
        elif b_norm > 1.0:
            self.b *= 1.0 / b_norm
        logger.debug(f"[PredictiveCodingLayer] 更新完成 W_norm={np.linalg.norm(self.W):.4f}")


class PrecisionEstimator:
    """精度估计器：估计各维度的不确定性倒数"""

    def __init__(self, dim: int, window_size: int = 100):
        self.dim = dim
        self.window_size = window_size
        self.history: list = []
        logger.info(f"[PrecisionEstimator] 初始化 dim={dim} window={window_size}")

    def estimate(self, features: np.ndarray) -> np.ndarray:
        """估计精度（基于滑动窗口方差）"""
        self.history.append(features.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)
        if len(self.history) < 2:
            precision = np.ones(self.dim)
        else:
            arr = np.array(self.history)
            var = np.var(arr, axis=0) + 1e-6  # 避免除零
            precision = 1.0 / var
            # 归一化到[0,1]
            max_p = np.max(precision) + 1e-6
            precision = precision / max_p
        logger.debug(f"[PrecisionEstimator] 估计 mean_prec={np.mean(precision):.4f}")
        return precision


class PerceptionModule:
    """
    环境感知模块
    - 输入：原始多模态观测
    - 输出：统一感知状态PerceptionState
    - 含预测编码层与精度估计
    """

    def __init__(self, config: Config):
        self.config = config
        self.input_dim = config.get("perception", "input_dim")
        self.feature_dim = config.get("perception", "feature_dim")
        self.sensor_noise = config.get("perception", "sensor_noise_std")
        self.enable_pc = config.get("perception", "enable_predictive_coding")

        # 特征提取层（简化MLP）
        self.W1 = np.random.randn(self.input_dim, self.feature_dim) * 0.1
        self.b1 = np.zeros(self.feature_dim)

        # 预测编码层
        self.pc_layer = PredictiveCodingLayer(self.feature_dim) if self.enable_pc else None

        # 精度估计器
        self.precision_estimator = PrecisionEstimator(self.feature_dim)

        # 上一次状态（用于fallback）
        self.last_state: Optional[PerceptionState] = None

        # 故障注入标志
        self._fault: Optional[str] = None

        logger.info(f"[PerceptionModule] 初始化 input_dim={self.input_dim} "
                    f"feature_dim={self.feature_dim} pc={self.enable_pc}")

    def inject_fault(self, fault: str):
        """注入故障（用于测试）"""
        self._fault = fault
        logger.warning(f"[PerceptionModule] 注入故障: {fault}")

    def clear_fault(self):
        self._fault = None
        logger.info("[PerceptionModule] 清除故障")

    def process(self, raw_input: np.ndarray) -> PerceptionState:
        """处理原始输入，返回感知状态"""
        logger.debug(f"[PerceptionModule] 处理输入 shape={np.shape(raw_input)}")
        # 故障处理
        if self._fault == "timeout":
            logger.warning("[PerceptionModule] 故障timeout，使用fallback")
            if self.last_state is not None:
                fallback = PerceptionState(
                    internal=self.last_state.internal.copy(),
                    external=self.last_state.external.copy(),
                    precision=self.last_state.precision.copy(),
                    is_fallback=True
                )
                return fallback
            else:
                # 无历史状态，返回零状态
                logger.error("[PerceptionModule] 无历史状态可用作fallback")
                return PerceptionState(
                    internal=np.zeros(self.feature_dim),
                    external=np.zeros(self.input_dim),
                    precision=np.ones(self.feature_dim) * 0.5,
                    is_fallback=True
                )

        # 输入校验
        raw_input = np.asarray(raw_input, dtype=np.float64)
        # 修复Bug10: NaN/Inf处理（np.clip对NaN无效）
        raw_input = np.nan_to_num(raw_input, nan=0.0, posinf=10.0, neginf=-10.0)
        if raw_input.shape != (self.input_dim,):
            # 尝试reshape或填充
            if raw_input.size >= self.input_dim:
                raw_input = raw_input.flatten()[:self.input_dim]
            else:
                padded = np.zeros(self.input_dim)
                padded[:raw_input.size] = raw_input.flatten()
                raw_input = padded
            logger.debug(f"[PerceptionModule] 输入reshape到 {raw_input.shape}")

        # 添加传感器噪声
        noise = np.random.randn(self.input_dim) * self.sensor_noise
        noisy_input = raw_input + noise
        # 修复Bug5: 输入裁剪，防止数值爆炸
        noisy_input = np.clip(noisy_input, -10.0, 10.0)
        logger.debug(f"[PerceptionModule] 加噪 norm={np.linalg.norm(noisy_input):.4f}")

        # 特征提取（ReLU激活）
        features = self.W1.T @ noisy_input + self.b1
        features = np.maximum(0, features)  # ReLU
        # 修复Bug5: 特征裁剪
        features = np.clip(features, 0.0, 10.0)
        logger.debug(f"[PerceptionModule] 特征提取 norm={np.linalg.norm(features):.4f}")

        # 预测编码
        if self.pc_layer is not None and self.last_state is not None:
            prediction = self.pc_layer.predict(self.last_state.internal)
            error = self.pc_layer.compute_error(prediction, features)
            # 误差修正：features += precision * error
            precision = self.precision_estimator.estimate(features)
            features = features + precision * error * 0.1  # 小幅修正
            # 修复Bug5: 修正后再次裁剪
            features = np.clip(features, 0.0, 10.0)
            # 在线学习
            self.pc_layer.update(self.last_state.internal, features)
        else:
            precision = self.precision_estimator.estimate(features)

        # 精度正定性检查
        if np.any(precision < 0):
            logger.warning("[PerceptionModule] 精度有负值，取绝对值")
            precision = np.abs(precision)

        state = PerceptionState(
            internal=features,
            external=noisy_input,
            precision=precision,
            is_fallback=False
        )
        self.last_state = state
        logger.debug(f"[PerceptionModule] 输出 state norm={np.linalg.norm(features):.4f} "
                     f"prec_mean={np.mean(precision):.4f}")
        return state

    def get_current_perception(self) -> Optional[PerceptionState]:
        return self.last_state

    def health_check(self) -> bool:
        """健康检查"""
        healthy = self._fault is None
        logger.debug(f"[PerceptionModule] health_check={healthy}")
        return healthy
