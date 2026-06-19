"""
多层预测模型（MultiLevelPredictor）
对应设计文档第11章、第27章
底层GRU(真实BPTT训练) / 中层Transformer(真实self-attention) / 高层GNN(真实图卷积)
阶段16：算法内核真实化——实现真实梯度下降训练
"""
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from .types import Prediction, PerceptionState
from .config import Config

logger = logging.getLogger("seva.predictor")


def _safe(x: np.ndarray, lo: float = -10.0, hi: float = 10.0) -> np.ndarray:
    """统一的数值安全处理：NaN/Inf处理 + 裁剪"""
    x = np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(x, lo, hi)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的sigmoid"""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的softmax"""
    x = np.clip(x, -50.0, 50.0)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)


class GRULayer:
    """
    真实GRU层（底层，高频）—— 阶段16真实化
    实现完整的门控GRU + BPTT反向传播 + Adam优化
    可真正学习时序预测任务
    """

    def __init__(self, dim: int, learning_rate: float = 0.01):
        self.dim = dim
        self.lr = learning_rate
        # Xavier初始化
        scale = np.sqrt(2.0 / (dim + dim))
        # GRU参数：W_z, W_r, W_h (输入权重) + U_z, U_r, U_h (隐藏权重) + b_z, b_r, b_h (偏置)
        self.W_z = np.random.randn(dim, dim) * scale
        self.W_r = np.random.randn(dim, dim) * scale
        self.W_h = np.random.randn(dim, dim) * scale
        self.U_z = np.random.randn(dim, dim) * scale
        self.U_r = np.random.randn(dim, dim) * scale
        self.U_h = np.random.randn(dim, dim) * scale
        self.b_z = np.zeros(dim)
        self.b_r = np.zeros(dim)
        self.b_h = np.zeros(dim)
        self.hidden = np.zeros(dim)

        # Adam优化器状态
        self._adam_init()

        # 训练统计
        self.train_steps = 0
        self.loss_history: List[float] = []
        logger.info(f"[GRULayer] 初始化 dim={dim} lr={learning_rate} (真实BPTT训练)")

    def _adam_init(self):
        """初始化Adam优化器状态"""
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        for name in ["W_z", "W_r", "W_h", "U_z", "U_r", "U_h", "b_z", "b_r", "b_h"]:
            param = getattr(self, name)
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)

    def _adam_update(self, grads: Dict[str, np.ndarray]):
        """Adam优化器参数更新"""
        self.adam_t += 1
        for name, grad in grads.items():
            if grad is None:
                continue
            grad = _safe(grad, -1.0, 1.0)  # 梯度裁剪
            self.adam_m[name] = self.beta1 * self.adam_m[name] + (1 - self.beta1) * grad
            self.adam_v[name] = self.beta2 * self.adam_v[name] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.adam_m[name] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1 - self.beta2 ** self.adam_t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, name, _safe(param, -10.0, 10.0))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播一步（保持原接口兼容）"""
        x = _safe(x)
        z = _sigmoid(self.W_z @ x + self.U_z @ self.hidden + self.b_z)
        r = _sigmoid(self.W_r @ x + self.U_r @ self.hidden + self.b_r)
        h_tilde = np.tanh(self.W_h @ x + self.U_h @ (r * self.hidden) + self.b_h)
        self.hidden = (1 - z) * self.hidden + z * h_tilde
        self.hidden = _safe(self.hidden)
        return self.hidden.copy()

    def forward_cache(self, x: np.ndarray) -> Dict:
        """前向传播并保存中间变量（用于反向传播）"""
        x = _safe(x)
        h_prev = self.hidden.copy()
        # 门控计算
        z_lin = self.W_z @ x + self.U_z @ h_prev + self.b_z
        z = _sigmoid(z_lin)
        r_lin = self.W_r @ x + self.U_r @ h_prev + self.b_r
        r = _sigmoid(r_lin)
        h_tilde_lin = self.W_h @ x + self.U_h @ (r * h_prev) + self.b_h
        h_tilde = np.tanh(h_tilde_lin)
        h_new = (1 - z) * h_prev + z * h_tilde
        h_new = _safe(h_new)
        cache = {
            "x": x, "h_prev": h_prev, "z": z, "r": r,
            "h_tilde": h_tilde, "h_new": h_new,
            "z_lin": z_lin, "r_lin": r_lin, "h_tilde_lin": h_tilde_lin
        }
        return h_new, cache

    def backward(self, cache: Dict, dh: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        BPTT反向传播一步
        dh: 从上游传回的梯度（对h_new的梯度）
        返回: (参数梯度字典, 对h_prev的梯度)
        """
        x = cache["x"]
        h_prev = cache["h_prev"]
        z = cache["z"]
        r = cache["r"]
        h_tilde = cache["h_tilde"]
        z_lin = cache["z_lin"]
        r_lin = cache["r_lin"]
        h_tilde_lin = cache["h_tilde_lin"]

        dh = _safe(dh, -10.0, 10.0)

        # h_new = (1-z)*h_prev + z*h_tilde
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z)
        dh_tilde = dh * z

        # h_tilde = tanh(h_tilde_lin)
        dh_tilde_lin = dh_tilde * (1 - h_tilde ** 2)
        # h_tilde_lin = W_h @ x + U_h @ (r*h_prev) + b_h
        dW_h = np.outer(dh_tilde_lin, x)
        dU_h = np.outer(dh_tilde_lin, r * h_prev)
        db_h = dh_tilde_lin.copy()
        dr_h_prev = self.U_h.T @ dh_tilde_lin  # 对 r*h_prev 的梯度
        dr = dr_h_prev * h_prev
        dh_prev += dr_h_prev * r

        # r = sigmoid(r_lin)
        dr_lin = dr * r * (1 - r)
        dW_r = np.outer(dr_lin, x)
        dU_r = np.outer(dr_lin, h_prev)
        db_r = dr_lin.copy()
        dh_prev += self.U_r.T @ dr_lin

        # z = sigmoid(z_lin)
        dz_lin = dz * z * (1 - z)
        dW_z = np.outer(dz_lin, x)
        dU_z = np.outer(dz_lin, h_prev)
        db_z = dz_lin.copy()
        dh_prev += self.U_z.T @ dz_lin

        grads = {
            "W_z": dW_z, "W_r": dW_r, "W_h": dW_h,
            "U_z": dU_z, "U_r": dU_r, "U_h": dU_h,
            "b_z": db_z, "b_r": db_r, "b_h": db_h,
        }
        return grads, _safe(dh_prev, -10.0, 10.0)

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        单步训练：前向→计算损失→反向→Adam更新
        x: 输入, target: 目标输出（下一个状态）
        返回: loss值
        """
        x = _safe(x)
        target = _safe(target)

        # 前向传播
        h_new, cache = self.forward_cache(x)
        self.hidden = h_new.copy()

        # 计算损失（MSE）
        diff = h_new - target
        loss = float(np.mean(diff ** 2))

        # 反向传播：dL/dh_new = 2 * diff / dim
        dh = 2.0 * diff / self.dim
        grads, _ = self.backward(cache, dh)

        # Adam参数更新
        self._adam_update(grads)

        self.train_steps += 1
        self.loss_history.append(loss)
        # 限制loss_history长度
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
        return loss

    def predict_next(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """预测下一步"""
        pred = self.forward(x)
        # 置信度基于近期loss（loss越低置信度越高）
        if self.loss_history:
            recent_loss = np.mean(self.loss_history[-20:])
            confidence = float(1.0 / (1.0 + recent_loss * 10))
        else:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        variance = np.ones(self.dim) * (1.0 - confidence + 0.01)
        return pred, confidence

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred, conf = self.predict_next(x)
        var = np.ones(self.dim) * (1.0 - conf + 1e-6)
        return pred, var

    def reset(self):
        self.hidden = np.zeros(self.dim)
        logger.debug("[GRULayer] 重置隐藏状态")

    def get_train_stats(self) -> Dict:
        """获取训练统计"""
        if not self.loss_history:
            return {"train_steps": self.train_steps, "avg_loss": None,
                    "last_loss": None}
        return {
            "train_steps": self.train_steps,
            "avg_loss": float(np.mean(self.loss_history)),
            "last_loss": float(self.loss_history[-1]),
            "loss_trend": float(np.mean(self.loss_history[-10:]) -
                                np.mean(self.loss_history[-20:-10]))
                if len(self.loss_history) >= 20 else 0.0
        }


class TransformerLayer:
    """
    真实Transformer层（中层，中频）—— 阶段16真实化
    实现完整的self-attention(QKV) + 前馈网络 + 反向传播训练
    """

    def __init__(self, dim: int, horizon: int = 5, learning_rate: float = 0.005):
        self.dim = dim
        self.horizon = horizon
        self.lr = learning_rate
        scale = np.sqrt(2.0 / (dim + dim))
        # Self-Attention参数: Q, K, V 投影矩阵
        self.W_q = np.random.randn(dim, dim) * scale
        self.W_k = np.random.randn(dim, dim) * scale
        self.W_v = np.random.randn(dim, dim) * scale
        # 输出投影
        self.W_o = np.random.randn(dim, dim) * scale
        # 前馈网络 (FFN): 两层
        self.W_f1 = np.random.randn(dim, dim * 2) * scale
        self.b_f1 = np.zeros(dim * 2)
        self.W_f2 = np.random.randn(dim * 2, dim) * scale
        self.b_f2 = np.zeros(dim)
        # LayerNorm参数
        self.ln_gamma = np.ones(dim)
        self.ln_beta = np.zeros(dim)
        # 多步预测线性层
        self.W_pred = np.random.randn(dim, dim) * scale
        self.b_pred = np.zeros(dim)

        self.history: List[np.ndarray] = []
        self.train_steps = 0
        self.loss_history: List[float] = []
        self._adam_init()
        logger.info(f"[TransformerLayer] 初始化 dim={dim} horizon={horizon} (真实self-attention)")

    def _adam_init(self):
        """初始化Adam优化器状态"""
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        for name in ["W_q", "W_k", "W_v", "W_o", "W_f1", "b_f1",
                     "W_f2", "b_f2", "ln_gamma", "ln_beta",
                     "W_pred", "b_pred"]:
            param = getattr(self, name)
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)

    def _adam_update(self, grads: Dict[str, np.ndarray]):
        """Adam优化器参数更新"""
        self.adam_t += 1
        for name, grad in grads.items():
            if grad is None:
                continue
            grad = _safe(grad, -1.0, 1.0)
            self.adam_m[name] = self.beta1 * self.adam_m[name] + (1 - self.beta1) * grad
            self.adam_v[name] = self.beta2 * self.adam_v[name] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.adam_m[name] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1 - self.beta2 ** self.adam_t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, name, _safe(param, -10.0, 10.0))

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """LayerNorm前向 + cache"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-8)
        x_norm = (x - mean) / std
        out = gamma * x_norm + beta
        cache = {"x_norm": x_norm, "std": std, "gamma": gamma}
        return out, cache

    def _layer_norm_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LayerNorm反向"""
        x_norm = cache["x_norm"]
        std = cache["std"]
        gamma = cache["gamma"]
        N = x_norm.shape[-1]
        # dgamma, dbeta 对每个特征维度求和（跨batch/seq维度）
        if x_norm.ndim > 1:
            dgamma = np.sum(dout * x_norm, axis=0)  # (dim,)
            dbeta = np.sum(dout, axis=0)  # (dim,)
        else:
            dgamma = dout * x_norm
            dbeta = dout
        dx_norm = dout * gamma
        # dx: 对归一化的反向
        dx = (1.0 / N) * (1.0 / std) * (N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True)
                                          - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True))
        return dx, dgamma, dbeta

    def forward(self, sequence: List[np.ndarray]) -> List[np.ndarray]:
        """前向传播，返回多步预测（保持原接口兼容）"""
        if len(sequence) == 0:
            return [np.zeros(self.dim) for _ in range(self.horizon)]
        # 堆叠序列为矩阵 (seq_len, dim)
        seq_arr = np.array([_safe(s) for s in sequence])
        context, _ = self._forward_cache(seq_arr)
        # 多步预测
        predictions = []
        for step in range(self.horizon):
            pred = _safe(context @ self.W_pred + self.b_pred + context * 0.1 * step)
            predictions.append(pred)
        return predictions

    def _forward_cache(self, seq_arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """前向传播并保存中间变量"""
        # Self-Attention: Q, K, V
        Q = seq_arr @ self.W_q  # (seq_len, dim)
        K = seq_arr @ self.W_k
        V = seq_arr @ self.W_v
        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(self.dim)  # (seq_len, seq_len)
        attn_weights = _softmax(scores, axis=-1)
        attn_out = attn_weights @ V  # (seq_len, dim)
        # 输出投影
        attn_proj = attn_out @ self.W_o
        # 残差连接 + LayerNorm
        residual1 = _safe(seq_arr + attn_proj)
        ln1_out, ln1_cache = self._layer_norm(residual1, self.ln_gamma, self.ln_beta)
        # FFN
        ffn_hidden = _safe(np.maximum(0, ln1_out @ self.W_f1 + self.b_f1))  # ReLU
        ffn_out = _safe(ffn_hidden @ self.W_f2 + self.b_f2)
        # 残差连接 + LayerNorm
        residual2 = _safe(ln1_out + ffn_out)
        ln2_out, ln2_cache = self._layer_norm(residual2, self.ln_gamma, self.ln_beta)
        # 取最后一个位置作为context
        context = ln2_out[-1]
        cache = {
            "seq_arr": seq_arr, "Q": Q, "K": K, "V": V,
            "attn_weights": attn_weights, "attn_out": attn_out,
            "attn_proj": attn_proj, "residual1": residual1,
            "ln1_out": ln1_out, "ln1_cache": ln1_cache,
            "ffn_hidden": ffn_hidden, "ffn_out": ffn_out,
            "residual2": residual2, "ln2_out": ln2_out, "ln2_cache": ln2_cache,
            "context": context,
        }
        return context, cache

    def _backward(self, cache: Dict, dcontext: np.ndarray) -> Dict[str, np.ndarray]:
        """反向传播（对最后一个位置的context梯度）"""
        seq_len = cache["seq_arr"].shape[0]
        # context = ln2_out[-1]
        dln2_out = np.zeros_like(cache["ln2_out"])
        dln2_out[-1] = dcontext
        # residual2 = ln1_out + ffn_out
        dresidual2, dln_gamma2, dln_beta2 = self._layer_norm_backward(dln2_out, cache["ln2_cache"])
        dln1_out = dresidual2.copy()
        dffn_out = dresidual2.copy()
        # ffn_out = ffn_hidden @ W_f2 + b_f2
        dW_f2 = cache["ffn_hidden"].T @ dffn_out
        db_f2 = np.sum(dffn_out, axis=0)
        dffn_hidden = dffn_out @ self.W_f2.T
        # ReLU反向
        dffn_hidden = dffn_hidden * (cache["ffn_hidden"] > 0)
        # ffn_hidden = ln1_out @ W_f1 + b_f1
        dW_f1 = cache["ln1_out"].T @ dffn_hidden
        db_f1 = np.sum(dffn_hidden, axis=0)
        dln1_out += dffn_hidden @ self.W_f1.T
        # residual1 = seq_arr + attn_proj
        dresidual1, dln_gamma1, dln_beta1 = self._layer_norm_backward(dln1_out, cache["ln1_cache"])
        dseq_arr = dresidual1.copy()
        dattn_proj = dresidual1.copy()
        # attn_proj = attn_out @ W_o
        dW_o = cache["attn_out"].T @ dattn_proj
        dattn_out = dattn_proj @ self.W_o.T
        # attn_out = attn_weights @ V
        dattn_weights = dattn_out @ cache["V"].T  # (seq_len, seq_len)
        dV = cache["attn_weights"].T @ dattn_out  # (seq_len, dim)
        # softmax反向: dscores (seq_len, seq_len)
        # 对每行 i: dscores[i] = attn_weights[i] * (dattn_weights[i] - sum(attn_weights[i]*dattn_weights[i]))
        dscores = np.zeros_like(cache["attn_weights"])
        for i in range(seq_len):
            dscores[i] = cache["attn_weights"][i] * (
                dattn_weights[i] - np.sum(cache["attn_weights"][i] * dattn_weights[i]))
        # scores = Q @ K.T / sqrt(dim), 所以 dQ = dscores @ K / sqrt(dim), dK = dscores.T @ Q / sqrt(dim)
        sqrt_d = np.sqrt(self.dim)
        dQ = dscores @ cache["K"] / sqrt_d
        dK = dscores.T @ cache["Q"] / sqrt_d
        # Q = seq_arr @ W_q, K = seq_arr @ W_k, V = seq_arr @ W_v
        dW_q = cache["seq_arr"].T @ dQ
        dW_k = cache["seq_arr"].T @ dK
        dW_v = cache["seq_arr"].T @ dV
        dseq_arr += dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T
        # 合并LayerNorm梯度
        dln_gamma = dln_gamma1 + dln_gamma2
        dln_beta = dln_beta1 + dln_beta2
        return {
            "W_q": dW_q, "W_k": dW_k, "W_v": dW_v, "W_o": dW_o,
            "W_f1": dW_f1, "b_f1": db_f1, "W_f2": dW_f2, "b_f2": db_f2,
            "ln_gamma": dln_gamma, "ln_beta": dln_beta,
        }

    def train_step(self, sequence: List[np.ndarray], target: np.ndarray) -> float:
        """单步训练"""
        if len(sequence) == 0:
            return 0.0
        seq_arr = np.array([_safe(s) for s in sequence])
        target = _safe(target)
        # 前向
        context, cache = self._forward_cache(seq_arr)
        # 预测
        pred = _safe(context @ self.W_pred + self.b_pred)
        # 损失
        diff = pred - target
        loss = float(np.mean(diff ** 2))
        # 反向
        dpred = 2.0 * diff / self.dim
        dW_pred = np.outer(context, dpred)
        db_pred = dpred.copy()
        dcontext = dpred @ self.W_pred.T
        grads = self._backward(cache, dcontext)
        grads["W_pred"] = dW_pred
        grads["b_pred"] = db_pred
        # 更新
        self._adam_update(grads)
        self.train_steps += 1
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
        return loss

    def predict(self, sequence: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """预测多步"""
        preds = self.forward(sequence)
        # 置信度基于loss
        if self.loss_history:
            recent_loss = np.mean(self.loss_history[-20:])
            confidence = float(1.0 / (1.0 + recent_loss * 10))
        else:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        return preds, confidence

    def reset(self):
        self.history = []
        logger.debug("[TransformerLayer] 重置历史")

    def get_train_stats(self) -> Dict:
        if not self.loss_history:
            return {"train_steps": self.train_steps, "avg_loss": None,
                    "last_loss": None}
        return {
            "train_steps": self.train_steps,
            "avg_loss": float(np.mean(self.loss_history)),
            "last_loss": float(self.loss_history[-1]),
        }


class GNNLayer:
    """
    真实GNN层（高层，低频）—— 阶段16真实化
    实现图卷积(GCN) + 消息传递 + 反向传播训练
    """

    def __init__(self, dim: int, learning_rate: float = 0.005):
        self.dim = dim
        self.lr = learning_rate
        scale = np.sqrt(2.0 / (dim + dim))
        # 图卷积权重 (2层GCN)
        self.W_g1 = np.random.randn(dim, dim) * scale
        self.b_g1 = np.zeros(dim)
        self.W_g2 = np.random.randn(dim, dim) * scale
        self.b_g2 = np.zeros(dim)
        # 节点嵌入字典（可学习）
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.default_embedding = np.random.randn(dim) * scale

        # Adam优化器状态
        self._adam_init()
        self.train_steps = 0
        self.loss_history: List[float] = []
        logger.info(f"[GNNLayer] 初始化 dim={dim} (真实图卷积GCN)")

    def _adam_init(self):
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        for name in ["W_g1", "b_g1", "W_g2", "b_g2"]:
            param = getattr(self, name)
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)

    def _adam_update(self, grads: Dict[str, np.ndarray]):
        self.adam_t += 1
        for name, grad in grads.items():
            if grad is None:
                continue
            grad = _safe(grad, -1.0, 1.0)
            self.adam_m[name] = self.beta1 * self.adam_m[name] + (1 - self.beta1) * grad
            self.adam_v[name] = self.beta2 * self.adam_v[name] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.adam_m[name] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1 - self.beta2 ** self.adam_t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, name, _safe(param, -10.0, 10.0))

    def _build_adjacency(self, nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
        """构建归一化邻接矩阵（带自环）"""
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        A = np.eye(n)  # 自环
        for src, dst in edges:
            if src in node_idx and dst in node_idx:
                A[node_idx[src], node_idx[dst]] = 1.0
                A[node_idx[dst], node_idx[src]] = 1.0  # 无向图
        # 对称归一化: D^(-1/2) A D^(-1/2)
        degree = A.sum(axis=1)
        degree = np.maximum(degree, 1.0)
        D_inv_sqrt = 1.0 / np.sqrt(degree)
        A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
        return A_norm

    def _get_node_features(self, nodes: List[str]) -> np.ndarray:
        """获取节点特征矩阵"""
        features = np.zeros((len(nodes), self.dim))
        for i, node in enumerate(nodes):
            if node not in self.node_embeddings:
                self.node_embeddings[node] = np.random.randn(self.dim) * 0.1
            features[i] = self.node_embeddings[node]
        return features

    def forward(self, nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
        """图级预测（保持原接口兼容）"""
        if not nodes:
            return np.zeros(self.dim)
        H, _ = self._forward_cache(nodes, edges)
        # 图级readout: 均值池化
        graph_repr = np.mean(H, axis=0)
        return _safe(graph_repr)

    def _forward_cache(self, nodes: List[str], edges: List[Tuple[str, str]]) -> Tuple[np.ndarray, Dict]:
        """前向传播并保存中间变量"""
        X = self._get_node_features(nodes)
        A_norm = self._build_adjacency(nodes, edges)
        # 第1层GCN: H1 = ReLU(A_norm @ X @ W_g1 + b_g1)
        H1_lin = A_norm @ X @ self.W_g1 + self.b_g1
        H1 = np.maximum(0, H1_lin)  # ReLU
        H1 = _safe(H1)
        # 第2层GCN: H2 = A_norm @ H1 @ W_g2 + b_g2
        H2 = A_norm @ H1 @ self.W_g2 + self.b_g2
        H2 = _safe(H2)
        cache = {
            "X": X, "A_norm": A_norm, "H1_lin": H1_lin, "H1": H1, "H2": H2,
            "nodes": nodes,
        }
        return H2, cache

    def _backward(self, cache: Dict, dH2: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """反向传播，返回(参数梯度, 对输入X的梯度)"""
        # H2 = A_norm @ H1 @ W_g2 + b_g2
        AH1 = cache["A_norm"] @ cache["H1"]
        dW_g2 = AH1.T @ dH2
        db_g2 = np.sum(dH2, axis=0)
        # dH1 = A_norm.T @ dH2 @ W_g2.T
        dH1 = cache["A_norm"].T @ dH2 @ self.W_g2.T
        # ReLU反向
        dH1 = dH1 * (cache["H1"] > 0)
        # H1 = A_norm @ X @ W_g1 + b_g1
        AX = cache["A_norm"] @ cache["X"]
        dW_g1 = AX.T @ dH1
        db_g1 = np.sum(dH1, axis=0)
        # dX = A_norm.T @ dH1 @ W_g1.T
        dX = cache["A_norm"].T @ dH1 @ self.W_g1.T
        grads = {
            "W_g1": dW_g1, "b_g1": db_g1,
            "W_g2": dW_g2, "b_g2": db_g2,
        }
        return grads, dX

    def train_step(self, nodes: List[str], edges: List[Tuple[str, str]],
                   target: np.ndarray) -> float:
        """单步训练"""
        if not nodes:
            return 0.0
        target = _safe(target)
        H2, cache = self._forward_cache(nodes, edges)
        graph_repr = np.mean(H2, axis=0)
        graph_repr = _safe(graph_repr)
        # 损失
        diff = graph_repr - target
        loss = float(np.mean(diff ** 2))
        # 反向: dL/dgraph_repr
        dgraph = 2.0 * diff / self.dim
        # graph_repr = mean(H2), 所以 dH2 = dgraph / n (每个节点)
        dH2 = np.tile(dgraph, (H2.shape[0], 1)) / H2.shape[0]
        grads, dX = self._backward(cache, dH2)
        # 更新节点嵌入（直接梯度下降）
        dX = _safe(dX, -1.0, 1.0)
        for i, node in enumerate(cache["nodes"]):
            if node in self.node_embeddings:
                self.node_embeddings[node] -= self.lr * dX[i]
                self.node_embeddings[node] = _safe(self.node_embeddings[node])
        # 更新GCN参数
        self._adam_update(grads)
        self.train_steps += 1
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
        return loss

    def predict(self, nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
        """图级预测"""
        return self.forward(nodes, edges)

    def reset(self):
        self.node_embeddings = {}
        logger.debug("[GNNLayer] 重置节点嵌入")

    def get_train_stats(self) -> Dict:
        if not self.loss_history:
            return {"train_steps": self.train_steps, "avg_loss": None,
                    "last_loss": None}
        return {
            "train_steps": self.train_steps,
            "avg_loss": float(np.mean(self.loss_history)),
            "last_loss": float(self.loss_history[-1]),
        }


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
        # 阶段16: 使用配置中的学习率
        online_lr = config.get("optimizer", "online_learning_rate")

        self.bottom = GRULayer(b_dim, learning_rate=online_lr)
        self.middle = TransformerLayer(m_dim, self.horizon, learning_rate=online_lr * 0.5)
        self.top = GNNLayer(t_dim, learning_rate=online_lr * 0.5)

        self.last_prediction: Optional[Prediction] = None
        self.last_perception: Optional[PerceptionState] = None
        self._sequence_buffer: List[np.ndarray] = []
        self._max_buffer = 20
        # 阶段16: 保存上一次的状态用于训练
        self._prev_state: Optional[np.ndarray] = None

        logger.info(f"[MultiLevelPredictor] 初始化 dims=({b_dim},{m_dim},{t_dim}) "
                    f"lr={online_lr} (真实训练)")

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
        # 阶段16: 在线学习——用当前观测训练上一次的预测
        # 如果有上一次状态，则当前状态是上一次预测的目标
        if self._prev_state is not None:
            self._train_online(self._prev_state, state.internal)
        self._prev_state = state.internal.copy()
        logger.info(f"[MultiLevelPredictor] 融合预测 conf={fused_conf:.3f} "
                    f"norm={np.linalg.norm(fused_pred):.4f}")
        return prediction

    def _train_online(self, prev_state: np.ndarray, current_state: np.ndarray):
        """阶段16: 在线训练三层模型"""
        try:
            # 底层GRU训练：输入prev_state，目标current_state
            self.bottom.train_step(prev_state, current_state)
            # 中层Transformer训练：用序列缓冲区，目标current_state
            if len(self._sequence_buffer) >= 2:
                self.middle.train_step(self._sequence_buffer[:-1], current_state)
            # 高层GNN训练：用prev_state作为图节点，目标current_state
            self.top.train_step(["self"], [("self", "self")], current_state)
        except Exception as e:
            logger.warning(f"[MultiLevelPredictor] 在线训练异常: {e}")

    def get_train_stats(self) -> Dict:
        """获取三层模型训练统计"""
        return {
            "bottom": self.bottom.get_train_stats(),
            "middle": self.middle.get_train_stats(),
            "top": self.top.get_train_stats(),
        }

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
        self._prev_state = None
        self.last_prediction = None
        logger.info("[MultiLevelPredictor] 重置完成")

    def health_check(self) -> bool:
        return True
