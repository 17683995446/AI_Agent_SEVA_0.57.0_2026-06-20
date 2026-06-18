"""
驱力引擎模块（DriveEngine）
对应设计文档第9章、第25章
三大驱力：安全/收益/能效 + 虚拟神经调质
"""
import numpy as np
import logging
from typing import List, Optional
from .types import PerceptionState, Drives, Neuromodulators, ExperienceRecord
from .config import Config

logger = logging.getLogger("seva.drive")


class DriveEngine:
    """
    驱力引擎模块
    - 输入：感知状态、历史经验
    - 输出：三大驱力 + 神经调质水平
    """

    def __init__(self, config: Config):
        self.config = config
        self.safety_weight = config.get("drive_engine", "safety_weight")
        self.gain_weight = config.get("drive_engine", "gain_weight")
        self.efficiency_weight = config.get("drive_engine", "efficiency_weight")
        nm_base = config.get("drive_engine", "neuromodulator_baseline")
        self.neuromodulators = Neuromodulators(
            dopamine=nm_base["dopamine"],
            serotonin=nm_base["serotonin"],
            norepinephrine=nm_base["norepinephrine"],
            acetylcholine=nm_base["acetylcholine"],
            dopamine_baseline=nm_base["dopamine"],
            serotonin_baseline=nm_base["serotonin"],
            norepinephrine_baseline=nm_base["norepinephrine"],
            acetylcholine_baseline=nm_base["acetylcholine"],
        )
        self.last_drives: Optional[Drives] = None
        self.last_perception: Optional[PerceptionState] = None
        logger.info(f"[DriveEngine] 初始化 weights=({self.safety_weight},"
                    f"{self.gain_weight},{self.efficiency_weight})")

    def compute_drives(self, state: PerceptionState,
                       history: List[ExperienceRecord]) -> Drives:
        """计算三大驱力"""
        logger.debug(f"[DriveEngine] 计算驱力 state_norm={np.linalg.norm(state.internal):.4f} "
                     f"hist_len={len(history)}")
        self.last_perception = state

        # 安全驱力：基于感知状态中的"威胁信号"
        # 修复Bug17: internal经ReLU后全为非负，无法检测负值威胁；改用external（原始观测）
        threat_signal = float(np.abs(np.sum(np.minimum(0, state.external))))
        # 威胁越大，安全驱力越低
        safety = 1.0 / (1.0 + threat_signal * 0.5)
        logger.debug(f"[DriveEngine] threat={threat_signal:.4f} safety={safety:.4f}")

        # 收益驱力：基于近期奖励历史
        if history:
            recent_rewards = [r.reward for r in history[-10:]]
            avg_reward = float(np.mean(recent_rewards))
            # 奖励越高，收益驱力越高（但饱和）
            gain = 1.0 / (1.0 + np.exp(-(avg_reward - 0.3) * 3))
        else:
            gain = 0.5  # 无历史时中性
        logger.debug(f"[DriveEngine] gain={gain:.4f}")

        # 能效驱力：基于精度均值（高精度=高能效）
        efficiency = float(np.mean(state.precision))
        efficiency = max(0.0, min(1.0, efficiency))
        logger.debug(f"[DriveEngine] efficiency={efficiency:.4f}")

        drives = Drives(safety=safety, gain=gain, efficiency=efficiency)
        self.last_drives = drives

        # 更新神经调质
        self._update_neuromodulators(drives, history)

        return drives

    def _update_neuromodulators(self, drives: Drives,
                                 history: List[ExperienceRecord]):
        """更新虚拟神经调质水平"""
        # 多巴胺：与收益驱力上升相关
        target_dopamine = drives.gain
        self.neuromodulators.dopamine += 0.1 * (target_dopamine -
                                                self.neuromodulators.dopamine)

        # 血清素：与安全驱力相关
        target_serotonin = drives.safety
        self.neuromodulators.serotonin += 0.1 * (target_serotonin -
                                                  self.neuromodulators.serotonin)

        # 去甲肾上腺素：与惊奇/威胁相关
        if history:
            recent_errors = [r.error.magnitude for r in history[-10:]
                            if r.error is not None]
            if recent_errors:
                avg_error = float(np.mean(recent_errors))
                target_ne = min(1.0, avg_error / 5.0)
                self.neuromodulators.norepinephrine += 0.2 * (target_ne -
                    self.neuromodulators.norepinephrine)

        # 乙酰胆碱：与注意力需求相关（低精度=高注意力需求）
        if self.last_perception is not None:
            avg_precision = float(np.mean(self.last_perception.precision))
            target_ach = 1.0 - avg_precision
            self.neuromodulators.acetylcholine += 0.1 * (target_ach -
                self.neuromodulators.acetylcholine)

        logger.debug(f"[DriveEngine] NM: DA={self.neuromodulators.dopamine:.3f} "
                     f"5HT={self.neuromodulators.serotonin:.3f} "
                     f"NE={self.neuromodulators.norepinephrine:.3f} "
                     f"ACh={self.neuromodulators.acetylcholine:.3f}")

    def get_neuromodulator_levels(self) -> Neuromodulators:
        return self.neuromodulators

    def get_current_perception(self) -> Optional[PerceptionState]:
        return self.last_perception

    def get_last_drives(self) -> Optional[Drives]:
        return self.last_drives

    def health_check(self) -> bool:
        return True
