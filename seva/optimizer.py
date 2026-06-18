"""
双模式优化器（DualModeOptimizer）
对应设计文档第13章、第29章
在线卡尔曼更新 + 离线深度训练 + 渐进参数融合
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from .config import Config

logger = logging.getLogger("seva.optimizer")


class DualModeOptimizer:
    """
    双模式优化器
    - 在线模式：卡尔曼滤波风格参数更新
    - 离线模式：梯度下降深度训练
    - 渐进融合：新旧参数凸组合
    """

    def __init__(self, config: Config):
        self.config = config
        self.online_lr = config.get("optimizer", "online_learning_rate")
        self.offline_lr = config.get("optimizer", "offline_learning_rate")
        self.fusion_alpha = config.get("optimizer", "fusion_alpha")
        self.kalman_q = config.get("optimizer", "kalman_q")  # 过程噪声
        self.kalman_r = config.get("optimizer", "kalman_r")  # 观测噪声

        self.online_update_count = 0
        self.offline_update_count = 0
        self.loss_history: List[float] = []
        logger.info(f"[DualModeOptimizer] 初始化 online_lr={self.online_lr} "
                    f"offline_lr={self.offline_lr} alpha={self.fusion_alpha}")

    def update_online(self, params: np.ndarray, observation: np.ndarray,
                      learning_rate: Optional[float] = None) -> np.ndarray:
        """
        在线卡尔曼风格更新
        params: 当前参数
        observation: 观测（目标值）
        """
        lr = learning_rate if learning_rate is not None else self.online_lr
        logger.debug(f"[DualModeOptimizer] 在线更新 lr={lr} "
                     f"params_norm={np.linalg.norm(params):.4f}")

        # 修复Bug13: NaN/Inf处理（np.clip对NaN无效）
        params = np.nan_to_num(params, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        # 修复Bug5: 参数裁剪
        params = np.clip(params, -10.0, 10.0)
        observation = np.clip(observation, -10.0, 10.0)
        # 修复Bug14: 维度对齐
        if params.shape != observation.shape:
            if params.size == 0 or observation.size == 0:
                return params
            min_len = min(params.size, observation.size)
            params = np.resize(params, min_len)
            observation = np.resize(observation, min_len)

        # 卡尔曼增益 K = P / (P + R)
        # 简化：P用参数方差近似
        P = np.var(params) + self.kalman_q
        K = P / (P + self.kalman_r)

        # 更新：params = params + K * (observation - params)
        innovation = observation - params
        # 修复Bug5: innovation裁剪
        innovation = np.clip(innovation, -5.0, 5.0)
        new_params = params + K * lr * innovation
        # 修复Bug5: 更新后参数裁剪; 修复Bug13: nan_to_num
        new_params = np.nan_to_num(new_params, nan=0.0, posinf=10.0, neginf=-10.0)
        new_params = np.clip(new_params, -10.0, 10.0)

        self.online_update_count += 1
        logger.debug(f"[DualModeOptimizer] 在线更新完成 #{self.online_update_count} "
                     f"innovation_norm={np.linalg.norm(innovation):.4f}")
        return new_params

    def train_offline_epoch(self, model_params: Dict[str, np.ndarray],
                            dataset: List[Dict], learning_rate: Optional[float] = None) -> Dict[str, np.ndarray]:
        """离线训练一个epoch"""
        lr = learning_rate if learning_rate is not None else self.offline_lr
        logger.debug(f"[DualModeOptimizer] 离线训练 lr={lr} samples={len(dataset)}")

        updated = {}
        for name, params in model_params.items():
            # 简化梯度下降：对每个样本计算梯度并更新
            grad_accum = np.zeros_like(params)
            for sample in dataset:
                target = sample.get(name, np.zeros_like(params))
                if target.shape != params.shape:
                    target = np.resize(target, params.shape)
                # 梯度 = params - target（L2损失的梯度）
                grad_accum += (params - target)
            grad = grad_accum / max(len(dataset), 1)
            updated[name] = params - lr * grad

        self.offline_update_count += 1
        # 计算损失
        loss = self._compute_loss_from_params(updated, dataset)
        self.loss_history.append(loss)
        logger.info(f"[DualModeOptimizer] 离线训练 #{self.offline_update_count} "
                    f"loss={loss:.4f}")
        return updated

    def _compute_loss_from_params(self, params: Dict[str, np.ndarray],
                                  dataset: List[Dict]) -> float:
        """计算损失"""
        total_loss = 0.0
        count = 0
        for sample in dataset:
            for name, p in params.items():
                target = sample.get(name, np.zeros_like(p))
                if target.shape != p.shape:
                    target = np.resize(target, p.shape)
                total_loss += float(np.sum((p - target) ** 2))
                count += 1
        return total_loss / max(count, 1)

    def compute_loss(self, model_params: Dict[str, np.ndarray],
                     dataset: List[Dict]) -> float:
        """计算损失（对外接口）"""
        return self._compute_loss_from_params(model_params, dataset)

    def progressive_fusion(self, old_params: np.ndarray,
                          new_params: np.ndarray,
                          alpha: Optional[float] = None) -> np.ndarray:
        """渐进参数融合：凸组合"""
        a = alpha if alpha is not None else self.fusion_alpha
        if not (0.0 <= a <= 1.0):
            logger.warning(f"[DualModeOptimizer] alpha={a} 超出[0,1]，截断")
            a = max(0.0, min(1.0, a))

        # 修复Bug15: NaN/Inf处理
        old_params = np.nan_to_num(old_params, nan=0.0, posinf=10.0, neginf=-10.0)
        new_params = np.nan_to_num(new_params, nan=0.0, posinf=10.0, neginf=-10.0)

        # 对齐维度
        if old_params.shape != new_params.shape:
            new_params = np.resize(new_params, old_params.shape)

        fused = a * new_params + (1 - a) * old_params
        # 修复Bug15: 融合结果裁剪
        fused = np.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)
        fused = np.clip(fused, -10.0, 10.0)
        logger.debug(f"[DualModeOptimizer] 融合 alpha={a} "
                     f"fused_norm={np.linalg.norm(fused):.4f}")
        return fused

    def get_stats(self) -> Dict[str, Any]:
        return {
            "online_updates": self.online_update_count,
            "offline_updates": self.offline_update_count,
            "loss_history_len": len(self.loss_history),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
        }

    def health_check(self) -> bool:
        return True
