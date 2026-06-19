"""
训练辅助模块 —— 阶段17：在线学习闭环优化
提供学习率调度、L2正则化、早停机制、梯度爆炸检测
对应设计文档第29章
"""
import numpy as np
import logging
from typing import Dict, Optional, List, Callable

logger = logging.getLogger("seva.training")


def _safe(x: np.ndarray, lo: float = -10.0, hi: float = 10.0) -> np.ndarray:
    """统一的数值安全处理"""
    x = np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(x, lo, hi)


class LRScheduler:
    """
    学习率调度器 —— 阶段17
    支持：cosine annealing、step decay、warmup+cosine
    """

    def __init__(self, initial_lr: float, schedule_type: str = "cosine",
                 total_steps: int = 10000, warmup_steps: int = 100,
                 min_lr: float = 1e-5, decay_factor: float = 0.5,
                 decay_every: int = 1000):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 1)
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_every = max(decay_every, 1)
        self.current_step = 0
        self.current_lr = initial_lr
        logger.info(f"[LRScheduler] 初始化 type={schedule_type} "
                    f"initial_lr={initial_lr} min_lr={min_lr}")

    def step(self) -> float:
        """前进一步，返回当前学习率"""
        self.current_step += 1
        if self.schedule_type == "cosine":
            self.current_lr = self._cosine_annealing()
        elif self.schedule_type == "step":
            self.current_lr = self._step_decay()
        elif self.schedule_type == "warmup_cosine":
            self.current_lr = self._warmup_cosine()
        else:  # constant
            self.current_lr = self.initial_lr
        return self.current_lr

    def _cosine_annealing(self) -> float:
        """Cosine退火"""
        progress = min(self.current_step / self.total_steps, 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine
        return float(max(self.min_lr, lr))

    def _step_decay(self) -> float:
        """阶梯衰减"""
        decay_count = self.current_step // self.decay_every
        lr = self.initial_lr * (self.decay_factor ** decay_count)
        return float(max(self.min_lr, lr))

    def _warmup_cosine(self) -> float:
        """Warmup + Cosine"""
        if self.current_step <= self.warmup_steps:
            # 线性warmup
            return float(self.initial_lr * self.current_step / self.warmup_steps)
        # cosine退火
        progress = min((self.current_step - self.warmup_steps) /
                       max(self.total_steps - self.warmup_steps, 1), 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine
        return float(max(self.min_lr, lr))

    def get_lr(self) -> float:
        return self.current_lr

    def reset(self):
        self.current_step = 0
        self.current_lr = self.initial_lr


class Regularizer:
    """
    L2正则化器 —— 阶段17
    """

    def __init__(self, l2_lambda: float = 1e-4):
        self.l2_lambda = l2_lambda
        logger.info(f"[Regularizer] L2 lambda={l2_lambda}")

    def compute_penalty(self, params: Dict[str, np.ndarray]) -> float:
        """计算L2惩罚项"""
        if self.l2_lambda <= 0:
            return 0.0
        penalty = 0.0
        for name, p in params.items():
            penalty += float(np.sum(p ** 2))
        return 0.5 * self.l2_lambda * penalty

    def compute_grad(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算L2正则化梯度"""
        grads = {}
        for name, p in params.items():
            grads[name] = self.l2_lambda * p if self.l2_lambda > 0 else np.zeros_like(p)
        return grads


class EarlyStopping:
    """
    早停机制 —— 阶段17
    监控验证loss，patience轮无改善则停止
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-5,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.wait = 0
        self.stopped = False
        logger.info(f"[EarlyStopping] patience={patience} min_delta={min_delta} mode={mode}")

    def __call__(self, score: float) -> bool:
        """返回True表示应该停止"""
        if self.stopped:
            return True
        if self.best_score is None:
            self.best_score = score
            self.wait = 0
            return False
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                logger.info(f"[EarlyStopping] 触发早停 best={self.best_score:.6f} "
                            f"wait={self.wait}")
                return True
        return False

    def reset(self):
        self.best_score = None
        self.wait = 0
        self.stopped = False


class GradientMonitor:
    """
    梯度监控器 —— 阶段17
    检测梯度爆炸/消失，自动处理
    """

    def __init__(self, max_norm: float = 5.0, min_norm: float = 1e-8,
                 explosion_threshold: float = 100.0):
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.explosion_threshold = explosion_threshold
        self.explosion_count = 0
        self.vanishing_count = 0
        self.history: List[float] = []
        logger.info(f"[GradientMonitor] max_norm={max_norm} "
                    f"explosion_threshold={explosion_threshold}")

    def clip_and_monitor(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """梯度裁剪+监控"""
        # 计算全局梯度范数
        total_norm_sq = 0.0
        for name, g in grads.items():
            if g is not None:
                total_norm_sq += float(np.sum(g ** 2))
        total_norm = float(np.sqrt(total_norm_sq))

        self.history.append(total_norm)
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        # 检测梯度爆炸
        if total_norm > self.explosion_threshold:
            self.explosion_count += 1
            logger.warning(f"[GradientMonitor] 梯度爆炸! norm={total_norm:.4f} "
                           f"count={self.explosion_count}")
            # 激进裁剪
            scale = self.max_norm / max(total_norm, 1e-8)
            clipped = {}
            for name, g in grads.items():
                clipped[name] = _safe(g * scale, -1.0, 1.0) if g is not None else None
            return clipped

        # 检测梯度消失
        if total_norm < self.min_norm:
            self.vanishing_count += 1
            if self.vanishing_count % 100 == 1:
                logger.warning(f"[GradientMonitor] 梯度消失 norm={total_norm:.8f} "
                               f"count={self.vanishing_count}")

        # 标准梯度裁剪（global norm clipping）
        if total_norm > self.max_norm:
            scale = self.max_norm / max(total_norm, 1e-8)
            clipped = {}
            for name, g in grads.items():
                clipped[name] = _safe(g * scale, -1.0, 1.0) if g is not None else None
            return clipped

        return grads

    def get_stats(self) -> Dict:
        if not self.history:
            return {"explosion_count": self.explosion_count,
                    "vanishing_count": self.vanishing_count,
                    "avg_norm": None, "max_norm_seen": None}
        return {
            "explosion_count": self.explosion_count,
            "vanishing_count": self.vanishing_count,
            "avg_norm": float(np.mean(self.history)),
            "max_norm_seen": float(np.max(self.history)),
        }

    def reset(self):
        self.explosion_count = 0
        self.vanishing_count = 0
        self.history = []


class TrainingLoop:
    """
    统一训练循环 —— 阶段17
    整合学习率调度、正则化、早停、梯度监控
    为GRU/Transformer/GNN提供统一的训练接口
    """

    def __init__(self, initial_lr: float, total_steps: int = 10000,
                 l2_lambda: float = 1e-4, patience: int = 100,
                 schedule_type: str = "warmup_cosine"):
        self.lr_scheduler = LRScheduler(
            initial_lr=initial_lr,
            schedule_type=schedule_type,
            total_steps=total_steps,
            warmup_steps=min(100, total_steps // 10),
            min_lr=initial_lr * 0.01
        )
        self.regularizer = Regularizer(l2_lambda=l2_lambda)
        self.early_stopping = EarlyStopping(patience=patience)
        self.gradient_monitor = GradientMonitor()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        logger.info(f"[TrainingLoop] 初始化 lr={initial_lr} "
                    f"schedule={schedule_type} l2={l2_lambda}")

    def get_current_lr(self) -> float:
        return self.lr_scheduler.get_lr()

    def step_lr(self) -> float:
        return self.lr_scheduler.step()

    def process_gradients(self, grads: Dict[str, np.ndarray],
                          params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """处理梯度：正则化 + 监控裁剪"""
        # 添加L2正则化梯度
        l2_grads = self.regularizer.compute_grad(params)
        for name in grads:
            if name in l2_grads and grads[name] is not None:
                grads[name] = grads[name] + l2_grads[name]
        # 梯度监控与裁剪
        grads = self.gradient_monitor.clip_and_monitor(grads)
        return grads

    def compute_loss_with_penalty(self, base_loss: float,
                                  params: Dict[str, np.ndarray]) -> float:
        """计算带正则化项的损失"""
        return base_loss + self.regularizer.compute_penalty(params)

    def record_train_loss(self, loss: float):
        self.train_losses.append(loss)
        if len(self.train_losses) > 2000:
            self.train_losses = self.train_losses[-1000:]

    def record_val_loss(self, loss: float):
        self.val_losses.append(loss)
        if len(self.val_losses) > 2000:
            self.val_losses = self.val_losses[-1000:]

    def check_early_stop(self, val_loss: float) -> bool:
        return self.early_stopping(val_loss)

    def should_stop(self) -> bool:
        return self.early_stopping.stopped

    def get_stats(self) -> Dict:
        return {
            "current_lr": self.lr_scheduler.get_lr(),
            "lr_step": self.lr_scheduler.current_step,
            "train_losses_len": len(self.train_losses),
            "val_losses_len": len(self.val_losses),
            "last_train_loss": self.train_losses[-1] if self.train_losses else None,
            "last_val_loss": self.val_losses[-1] if self.val_losses else None,
            "early_stopped": self.early_stopping.stopped,
            "best_val_loss": self.early_stopping.best_score,
            "gradient": self.gradient_monitor.get_stats(),
        }

    def reset(self):
        self.lr_scheduler.reset()
        self.early_stopping.reset()
        self.gradient_monitor.reset()
        self.train_losses = []
        self.val_losses = []
