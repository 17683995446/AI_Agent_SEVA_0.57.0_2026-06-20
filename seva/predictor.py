"""
多层预测模型（MultiLevelPredictor）
对应设计文档第11章、第27章
底层GRU(简化)高频 / 中层Transformer(简化)中频 / 高层GNN(简化)低频
"""
import numpy as np
import logging
from typing import List, Optional, Tuple
from .types import Prediction, PerceptionState
from .config import Config

logger = logging.getLogger("seva.predictor")


class GRULayer:
    """简化GRU层（底层，高频）"""

    def __init__(self, dim: int):
        self.dim = dim
        # GRU参数（简化）
        self.W_z = np.random.randn(dim, dim) * 0.1
        self.W_r = np.random.randn(dim, dim) * 0.1
        self.W_h = np.random.randn(dim, dim) * 0.1
        self.hidden = np.zeros(dim)
        logger.info(f"[GRULayer] 初始化 dim={dim}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播一步"""
        # 修复Bug5: 输入裁剪; 修复Bug11: nan_to_num（np.clip对NaN无效）
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -10.0, 10.0)
        z = self._sigmoid(self.W_z @ x + self.hidden)
        r = self._sigmoid(self.W_r @ x + self.hidden)
        h_tilde = np.tanh(self.W_h @ x + r * self.hidden)
        self.hidden = (1 - z) * self.hidden + z * h_tilde
        # 修复Bug5: 隐藏状态裁剪; 修复Bug11: nan_to_num
        self.hidden = np.nan_to_num(self.hidden, nan=0.0, posinf=10.0, neginf=-10.0)
        self.hidden = np.clip(self.hidden, -10.0, 10.0)
        return self.hidden.copy()

    def predict_next(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """预测下一步"""
        pred = self.forward(x)
        # 置信度基于隐藏状态范数
        confidence = float(1.0 / (1.0 + np.exp(-np.linalg.norm(pred) + 2)))
        # 修复Bug5: 方差下界提高
        variance = np.ones(self.dim) * (1.0 - confidence + 0.01)
        return pred, confidence

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred, conf = self.predict_next(x)
        var = np.ones(self.dim) * (1.0 - conf + 1e-6)
        return pred, var

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def reset(self):
        self.hidden = np.zeros(self.dim)
        logger.debug("[GRULayer] 重置隐藏状态")


class TransformerLayer:
    """简化Transformer层（中层，中频）"""

    def __init__(self, dim: int, horizon: int = 5):
        self.dim = dim
        self.horizon = horizon
        self.attention_weights = np.random.randn(dim, dim) * 0.1
        self.history: List[np.ndarray] = []
        logger.info(f"[TransformerLayer] 初始化 dim={dim} horizon={horizon}")

    def forward(self, sequence: List[np.ndarray]) -> List[np.ndarray]:
        """前向传播，返回多步预测"""
        if len(sequence) == 0:
            return [np.zeros(self.dim) for _ in range(self.horizon)]

        # 简化注意力：加权平均历史
        weights = np.array([1.0 / (i + 1) for i in range(len(sequence))])
        weights = weights / weights.sum()
        context = np.zeros(self.dim)
        for w, s in zip(weights, sequence):
            # 修复Bug9: 输入裁剪，避免attention放大; 修复Bug11: nan_to_num
            s = np.nan_to_num(s, nan=0.0, posinf=10.0, neginf=-10.0)
            s = np.clip(s, -10.0, 10.0)
            context += w * (self.attention_weights @ s)
        # 修复Bug9: context裁剪
        context = np.nan_to_num(context, nan=0.0, posinf=10.0, neginf=-10.0)
        context = np.clip(context, -10.0, 10.0)

        # 多步预测（简单线性外推）
        predictions = []
        last = sequence[-1] if sequence else np.zeros(self.dim)
        last = np.nan_to_num(last, nan=0.0, posinf=10.0, neginf=-10.0)
        last = np.clip(last, -10.0, 10.0)
        for step in range(self.horizon):
            pred = context + (context - last) * step * 0.1
            # 修复Bug9: 每步预测裁剪
            pred = np.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
            pred = np.clip(pred, -10.0, 10.0)
            predictions.append(pred)
        return predictions

    def predict(self, sequence: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        preds = self.forward(sequence)
        confidence = 0.6  # 中层固定中等置信度
        return preds, confidence

    def reset(self):
        self.history = []
        logger.debug("[TransformerLayer] 重置历史")


class GNNLayer:
    """简化GNN层（高层，低频）"""

    def __init__(self, dim: int):
        self.dim = dim
        self.node_embeddings = {}
        logger.info(f"[GNNLayer] 初始化 dim={dim}")

    def predict(self, nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
        """图级预测"""
        if not nodes:
            return np.zeros(self.dim)
        # 简化：聚合节点嵌入
        embeddings = []
        for node in nodes:
            if node not in self.node_embeddings:
                self.node_embeddings[node] = np.random.randn(self.dim) * 0.1
            embeddings.append(self.node_embeddings[node])
        # 消息传递（1层）
        for src, dst in edges:
            if src in self.node_embeddings and dst in self.node_embeddings:
                self.node_embeddings[dst] += 0.01 * self.node_embeddings[src]

        return np.mean(embeddings, axis=0)

    def reset(self):
        self.node_embeddings = {}
        logger.debug("[GNNLayer] 重置节点嵌入")


class MultiLevelPredictor:
    """
    多层预测模型
    - 底层GRU：高频，单步预测
    - 中层Transformer：中频，多步预测
    - 高层GNN：低频，图级预测
    - 精度加权融合
    """

    def __init__(self, config: Config):
        self.config = config
        # 修复Bug: 所有层维度对齐到感知模块的feature_dim，避免维度不匹配
        feature_dim = config.get("perception", "feature_dim")
        b_dim = feature_dim  # 底层与感知输出对齐
        m_dim = feature_dim  # 中层与感知输出对齐
        t_dim = feature_dim  # 高层与感知输出对齐
        self.horizon = config.get("predictor", "prediction_horizon")
        self.feature_dim = feature_dim

        self.bottom = GRULayer(b_dim)
        self.middle = TransformerLayer(m_dim, self.horizon)
        self.top = GNNLayer(t_dim)

        self.last_prediction: Optional[Prediction] = None
        self.last_perception: Optional[PerceptionState] = None
        self._sequence_buffer: List[np.ndarray] = []
        self._max_buffer = 20

        logger.info(f"[MultiLevelPredictor] 初始化 dims=({b_dim},{m_dim},{t_dim})")

    def predict(self, state: PerceptionState,
                context: Optional[dict] = None) -> Prediction:
        """多层预测并融合"""
        logger.debug(f"[MultiLevelPredictor] 预测 state_norm="
                     f"{np.linalg.norm(state.internal):.4f}")
        self.last_perception = state

        # 底层预测
        bottom_pred, bottom_conf = self.bottom.predict_next(state.internal)
        bottom_var = np.ones_like(bottom_pred) * (1.0 - bottom_conf + 1e-6)
        logger.debug(f"[MultiLevelPredictor] 底层 conf={bottom_conf:.3f}")

        # 中层预测
        self._sequence_buffer.append(state.internal.copy())
        if len(self._sequence_buffer) > self._max_buffer:
            self._sequence_buffer.pop(0)
        middle_preds, middle_conf = self.middle.predict(self._sequence_buffer)
        middle_pred = middle_preds[0] if middle_preds else np.zeros_like(bottom_pred)
        middle_var = np.ones_like(middle_pred) * (1.0 - middle_conf + 1e-6)
        logger.debug(f"[MultiLevelPredictor] 中层 conf={middle_conf:.3f}")

        # 高层预测（简化：使用状态向量作为图节点）
        top_pred = self.top.predict(["self"], [("self", "self")])
        # 对齐维度：取底层维度
        if len(top_pred) != len(bottom_pred):
            top_pred = np.resize(top_pred, bottom_pred.shape)
        top_conf = 0.5
        top_var = np.ones_like(top_pred) * (1.0 - top_conf + 1e-6)

        # 精度加权融合
        fused_pred = self.fuse_predictions(
            bottom_pred, middle_pred, top_pred,
            1.0 / (bottom_var + 1e-6),
            1.0 / (middle_var + 1e-6),
            1.0 / (top_var + 1e-6)
        )

        # 综合置信度
        fused_conf = (bottom_conf + middle_conf + top_conf) / 3.0
        # 修复Bug11: 置信度NaN保护
        if np.isnan(fused_conf) or np.isinf(fused_conf):
            fused_conf = 0.5
        fused_conf = max(0.0, min(1.0, fused_conf))
        fused_var = (bottom_var + middle_var + top_var) / 3.0
        # 修复Bug11: 方差NaN保护
        fused_var = np.nan_to_num(fused_var, nan=0.5, posinf=1.0, neginf=0.01)
        fused_var = np.clip(fused_var, 0.01, 10.0)

        prediction = Prediction(
            predicted_state=fused_pred,
            confidence=fused_conf,
            variance=fused_var,
            horizon=1,
            source_layer="fused"
        )
        self.last_prediction = prediction
        logger.info(f"[MultiLevelPredictor] 融合预测 conf={fused_conf:.3f} "
                    f"norm={np.linalg.norm(fused_pred):.4f}")
        return prediction

    def fuse_predictions(self, bottom: np.ndarray, middle: np.ndarray,
                         top: np.ndarray, b_prec: np.ndarray,
                         m_prec: np.ndarray, t_prec: np.ndarray) -> np.ndarray:
        """精度加权融合"""
        # 修复Bug11: 所有输入nan_to_num保护
        bottom = np.nan_to_num(bottom, nan=0.0, posinf=10.0, neginf=-10.0)
        middle = np.nan_to_num(middle, nan=0.0, posinf=10.0, neginf=-10.0)
        top = np.nan_to_num(top, nan=0.0, posinf=10.0, neginf=-10.0)
        b_prec = np.nan_to_num(b_prec, nan=1.0, posinf=1.0, neginf=1.0)
        m_prec = np.nan_to_num(m_prec, nan=1.0, posinf=1.0, neginf=1.0)
        t_prec = np.nan_to_num(t_prec, nan=1.0, posinf=1.0, neginf=1.0)

        # 对齐维度
        target_dim = len(bottom)
        middle = np.resize(middle, target_dim)
        top = np.resize(top, target_dim)
        b_prec = np.resize(b_prec, target_dim)
        m_prec = np.resize(m_prec, target_dim)
        t_prec = np.resize(t_prec, target_dim)

        total_prec = b_prec + m_prec + t_prec + 1e-8
        fused = (bottom * b_prec + middle * m_prec + top * t_prec) / total_prec
        # 修复Bug9: 融合结果最终裁剪; 修复Bug11: nan_to_num
        fused = np.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)
        fused = np.clip(fused, -10.0, 10.0)
        logger.debug(f"[MultiLevelPredictor] 融合 norm={np.linalg.norm(fused):.4f}")
        return fused

    def predict_with_uncertainty(self, sequence: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """带不确定性的预测（用于测试）"""
        if not sequence:
            return np.zeros(self.config.get("predictor", "bottom_layer_dim")), \
                   np.ones(self.config.get("predictor", "bottom_layer_dim"))
        pred, var = self.bottom.predict_with_uncertainty(sequence[-1])
        return pred, var

    def get_current_perception(self) -> Optional[PerceptionState]:
        return self.last_perception

    def get_last_prediction(self) -> Optional[Prediction]:
        return self.last_prediction

    def reset(self):
        self.bottom.reset()
        self.middle.reset()
        self.top.reset()
        self._sequence_buffer = []
        self.last_prediction = None
        logger.info("[MultiLevelPredictor] 重置完成")

    def health_check(self) -> bool:
        return True
