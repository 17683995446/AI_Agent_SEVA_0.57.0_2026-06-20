"""
核心循环（SelfEvolvingAgentCore）
对应设计文档第31章
12阶段核心循环：感知->驱力->目标->预测->决策->安全->执行->误差->学习->记忆->元认知->时序
"""
import time
import gc
import logging
import numpy as np
from typing import Any, Dict, List, Optional
from .config import Config
from .bus import MessageBus, SystemMessage, MessageType
from .perception import PerceptionModule
from .drive import DriveEngine
from .goal import GoalGenerator
from .predictor import MultiLevelPredictor
from .error_eval import ErrorEvaluator
from .optimizer import DualModeOptimizer
from .memory import MemoryAndReview
from .safety import SafetyEthicsModule
from .decision import DecisionMaker
from .executor import Executor
from .types import (PerceptionState, Drives, Goal, Prediction, Action,
                    ActionType, ErrorSignal, ErrorAttribution, ExperienceRecord,
                    SafetyReviewResult, gen_id)

logger = logging.getLogger("seva.core")


class SelfEvolvingAgentCore:
    """
    自驱永续进化预测修正Agent核心
    实现12阶段核心循环
    """

    def __init__(self, config: Optional[Config] = None,
                 bus: Optional[MessageBus] = None):
        self.config = config or Config()
        errors = self.config.validate()
        if errors:
            logger.error(f"[Core] 配置错误: {errors}")
            raise ValueError(f"配置错误: {errors}")

        self.bus = bus or MessageBus(
            capacity=self.config.get("bus", "capacity"),
            enable_recording=self.config.get("bus", "enable_recording")
        )

        # 初始化模块
        self.modules: Dict[str, Any] = {}
        self.modules["perception"] = PerceptionModule(self.config)
        self.modules["drive_engine"] = DriveEngine(self.config)
        self.modules["goal_generator"] = GoalGenerator(self.config)
        self.modules["predictor"] = MultiLevelPredictor(self.config)
        self.modules["error_evaluator"] = ErrorEvaluator(self.config)
        self.modules["optimizer"] = DualModeOptimizer(self.config)
        self.modules["memory"] = MemoryAndReview(self.config)
        self.modules["safety_ethics"] = SafetyEthicsModule(self.config)
        self.modules["decision_maker"] = DecisionMaker(self.config)
        self.modules["executor"] = Executor(self.config)

        # 运行状态
        self.cycle_count = 0
        self.is_running = False
        self.mode = "normal"  # normal/circuit_breaker/safe_fallback
        self.last_cycle_log: List[Dict] = []

        # 阶段24: 环境引用（可选）
        self.environment: Optional[Any] = None
        self._env_done: bool = False

        # 指标采集
        self.metrics = {
            "total_reward": 0.0,
            "prediction_errors": [],
            "surprises": [],
            "goal_types": [],
            "action_types": [],
            "cycle_times": [],
        }

        logger.info("[Core] 初始化完成，9个模块就绪")

    def attach_environment(self, environment: Any):
        """阶段24: 附加仿真环境，启用真实环境闭环"""
        self.environment = environment
        self._env_done = False
        logger.info("[Core] 环境已附加，启用真实环境闭环")

    def detach_environment(self):
        """分离环境，回退到伪reward模式"""
        self.environment = None
        logger.info("[Core] 环境已分离")

    def initialize(self):
        """初始化（注册消息总线订阅等）"""
        logger.info("[Core] initialize() 开始")
        # 模块向总线注册（简化：核心循环直接调用，总线用于跨模块事件）
        self.is_running = True
        # GC管理：提高阈值减少周期内GC触发频率，降低p99延迟尖峰
        # 阈值设高（50000），依赖_timing_control中每100周期的显式gc.collect(0)
        gc.set_threshold(50000, 50, 50)
        logger.info("[Core] initialize() 完成")

    def inject_input(self, raw_input: np.ndarray):
        """注入原始输入"""
        self._pending_input = raw_input
        logger.debug(f"[Core] 注入输入 shape={np.shape(raw_input)}")

    def run_cycle(self) -> Dict[str, Any]:
        """执行一个完整周期（12阶段）"""
        cycle_start = time.time()
        self.cycle_count += 1
        cycle_log: List[Dict] = []
        # 初始化局部变量，避免异常时未定义
        actual_reward = 0.0
        error = None
        action = None
        goals = []
        perception = None
        prediction = None
        logger.info(f"[Core] ===== 周期 {self.cycle_count} 开始 =====")

        try:
            # 修复Bug30: 周期开始时若上次异常已恢复，尝试从safe_fallback恢复
            if self.mode == "safe_fallback":
                logger.info("[Core] 检测到safe_fallback模式，尝试恢复")
                self.mode = "normal"

            # ===== 阶段1：感知 =====
            t0 = time.time()
            raw_input = getattr(self, "_pending_input", None)
            if raw_input is None:
                # 无输入时使用零向量
                raw_input = np.zeros(self.config.get("perception", "input_dim"))
            perception = self.modules["perception"].process(raw_input)
            cycle_log.append({"stage": "perception", "time": time.time()-t0,
                              "result": {"fallback": perception.is_fallback}})
            logger.debug(f"[Core] 阶段1感知完成 time={time.time()-t0:.4f}s")

            # ===== 阶段2：驱力计算 =====
            t0 = time.time()
            history = list(self.modules["memory"].episodic_records)
            drives = self.modules["drive_engine"].compute_drives(perception, history)
            cycle_log.append({"stage": "drive_engine", "time": time.time()-t0,
                              "result": {"safety": drives.safety, "gain": drives.gain,
                                         "efficiency": drives.efficiency}})
            logger.debug(f"[Core] 阶段2驱力完成 time={time.time()-t0:.4f}s")

            # ===== 阶段3：目标生成 =====
            t0 = time.time()
            goals = self.modules["goal_generator"].generate(drives, perception)
            cycle_log.append({"stage": "goal_generator", "time": time.time()-t0,
                              "result": {"num_goals": len(goals),
                                         "types": [g.type.value for g in goals]}})
            logger.debug(f"[Core] 阶段3目标完成 time={time.time()-t0:.4f}s")

            # ===== 阶段4：预测 =====
            t0 = time.time()
            prediction = self.modules["predictor"].predict(perception)
            cycle_log.append({"stage": "predictor", "time": time.time()-t0,
                              "result": {"confidence": prediction.confidence,
                                         "horizon": prediction.horizon}})
            logger.debug(f"[Core] 阶段4预测完成 time={time.time()-t0:.4f}s")

            # ===== 阶段5：动作选择 =====
            t0 = time.time()
            action = self.modules["decision_maker"].select_action(
                drives, goals, perception, prediction
            )
            cycle_log.append({"stage": "decision_maker", "time": time.time()-t0,
                              "result": {"type": action.type.value,
                                         "source": action.source}})
            logger.debug(f"[Core] 阶段5决策完成 time={time.time()-t0:.4f}s")

            # ===== 阶段6：安全审查 =====
            t0 = time.time()
            safety_context = {"drives": drives, "perception": perception}
            review = self.modules["safety_ethics"].review_action(action, safety_context)
            cycle_log.append({"stage": "safety_ethics", "time": time.time()-t0,
                              "result": {"approved": review.is_approved,
                                         "veto_layer": review.veto_layer}})
            logger.debug(f"[Core] 阶段6安全审查完成 time={time.time()-t0:.4f}s")

            # ===== 阶段7：执行 =====
            t0 = time.time()
            if review.is_approved:
                # 阶段24: 传递环境给执行器，获取真实reward和下一观测
                exec_result = self.modules["executor"].execute(
                    action, environment=self.environment)
                actual_reward = exec_result.get("reward", 0.0)
            else:
                # 被否决，执行安全fallback
                fallback_action = Action(
                    action_id=gen_id("act"),
                    type=ActionType.WAIT,
                    parameters={"reason": "safety_veto"},
                    expected_reward=0.0,
                    source="safe_fallback"
                )
                exec_result = self.modules["executor"].execute(
                    fallback_action, environment=self.environment)
                actual_reward = exec_result.get("reward", 0.0)
                action = fallback_action

            # 阶段24: 用环境观测作为下一周期输入
            next_obs = exec_result.get("next_observation")
            if next_obs is not None:
                self._pending_input = next_obs
            # 环境结束时标记
            if exec_result.get("env_done"):
                self._env_done = True

            cycle_log.append({"stage": "executor", "time": time.time()-t0,
                              "result": {"success": exec_result["success"],
                                         "reward": actual_reward,
                                         "env_connected": self.environment is not None}})
            logger.debug(f"[Core] 阶段7执行完成 time={time.time()-t0:.4f}s "
                         f"reward={actual_reward:.3f}")

            # ===== 阶段8：误差评估 =====
            t0 = time.time()
            error_context = {"actual_reward": actual_reward}
            error = self.modules["error_evaluator"].evaluate(
                perception, prediction, action,
                actual_state=perception,  # 简化：用当前状态作为实际状态
                context=error_context
            )
            attribution = self.modules["error_evaluator"].attribute_error(error)
            cycle_log.append({"stage": "error_evaluator", "time": time.time()-t0,
                              "result": {"magnitude": error.magnitude,
                                         "surprise": error.surprise,
                                         "novel": error.is_novel}})
            logger.debug(f"[Core] 阶段8误差评估完成 time={time.time()-t0:.4f}s")

            # ===== 阶段9：在线学习 =====
            t0 = time.time()
            # 用误差更新预测模型参数（简化）
            if prediction is not None:
                updated = self.modules["optimizer"].update_online(
                    prediction.predicted_state,
                    perception.internal,
                    self.config.get("optimizer", "online_learning_rate")
                )
            cycle_log.append({"stage": "optimizer", "time": time.time()-t0,
                              "result": {"online_updates":
                                         self.modules["optimizer"].online_update_count}})
            logger.debug(f"[Core] 阶段9在线学习完成 time={time.time()-t0:.4f}s")

            # ===== 阶段10：记忆存储 =====
            t0 = time.time()
            record = ExperienceRecord(
                record_id=gen_id("exp"),
                timestamp=time.time(),
                state=perception,
                drives=drives,
                goal=goals[0] if goals else None,
                prediction=prediction,
                action=action,
                error=error,
                attribution=attribution,
                reward=actual_reward
            )
            self.modules["memory"].store_experience(record)
            # 定期巩固
            if (self.cycle_count % self.config.get("memory", "consolidation_interval") == 0
                and self.cycle_count > 0):
                self.modules["memory"].consolidate()
                # 在巩固阶段（已有开销）执行GC，避免GC在普通周期内触发尖峰
                gc.collect(0)
            cycle_log.append({"stage": "memory", "time": time.time()-t0,
                              "result": {"episodic_count":
                                         self.modules["memory"].episodic_count()}})
            logger.debug(f"[Core] 阶段10记忆存储完成 time={time.time()-t0:.4f}s")

            # ===== 阶段11：元认知（条件触发） =====
            t0 = time.time()
            if (self.config.get("core", "enable_metacognition") and
                self.cycle_count % self.config.get("core", "metacognition_interval") == 0):
                self._metacognition(error, attribution)
                cycle_log.append({"stage": "metacognition", "time": time.time()-t0,
                                  "result": {"triggered": True}})
            else:
                cycle_log.append({"stage": "metacognition", "time": time.time()-t0,
                                  "result": {"triggered": False}})
            logger.debug(f"[Core] 阶段11元认知完成 time={time.time()-t0:.4f}s")

            # ===== 阶段12：时序控制 =====
            t0 = time.time()
            self._timing_control()
            cycle_log.append({"stage": "timing_control", "time": time.time()-t0,
                              "result": {}})
            logger.debug(f"[Core] 阶段12时序控制完成 time={time.time()-t0:.4f}s")

        except Exception as e:
            logger.error(f"[Core] 周期异常: {e}", exc_info=True)
            cycle_log.append({"stage": "error", "time": 0,
                              "result": {"error": str(e)}})
            self.mode = "safe_fallback"

        cycle_time = time.time() - cycle_start
        # 修复Bug19: metrics列表添加上限，防止长期运行内存泄漏
        self.metrics["cycle_times"].append(cycle_time)
        if len(self.metrics["cycle_times"]) > 10000:
            self.metrics["cycle_times"] = self.metrics["cycle_times"][-5000:]
        self.metrics["total_reward"] += actual_reward
        if error is not None:
            self.metrics["prediction_errors"].append(error.magnitude)
            if len(self.metrics["prediction_errors"]) > 10000:
                self.metrics["prediction_errors"] = self.metrics["prediction_errors"][-5000:]
            self.metrics["surprises"].append(error.surprise)
            if len(self.metrics["surprises"]) > 10000:
                self.metrics["surprises"] = self.metrics["surprises"][-5000:]
        if action is not None:
            self.metrics["action_types"].append(action.type.value)
            if len(self.metrics["action_types"]) > 10000:
                self.metrics["action_types"] = self.metrics["action_types"][-5000:]
        if goals:
            self.metrics["goal_types"].append(goals[0].type.value)
            if len(self.metrics["goal_types"]) > 10000:
                self.metrics["goal_types"] = self.metrics["goal_types"][-5000:]

        self.last_cycle_log = cycle_log
        logger.info(f"[Core] ===== 周期 {self.cycle_count} 完成 "
                    f"time={cycle_time:.4f}s reward={actual_reward} =====")
        return {"cycle": self.cycle_count, "time": cycle_time,
                "log": cycle_log, "mode": self.mode}

    def _metacognition(self, error: ErrorSignal, attribution: ErrorAttribution):
        """元认知：自我反思"""
        logger.debug(f"[Core] 元认知触发 error_mag={error.magnitude:.3f}")
        # 如果误差主要由模型源引起，触发离线学习
        if attribution.model > 0.5:
            logger.info("[Core] 元认知：模型误差主导，触发离线复盘")
            replayed = self.modules["memory"].replay()
            if replayed:
                logger.debug(f"[Core] 元认知：回放{len(replayed)}条记忆用于离线学习")

    def _timing_control(self):
        """时序控制"""
        # 简化：检查是否需要降频
        if self.metrics["cycle_times"]:
            avg_time = float(np.mean(self.metrics["cycle_times"][-100:]))
            timeout = self.config.get("core", "cycle_timeout_ms") / 1000.0
            if avg_time > timeout:
                logger.warning(f"[Core] 平均周期时间{avg_time:.3f}s超过阈值{timeout:.3f}s")
        # 周期性GC：每100代执行gen0回收，每500代执行gen1回收
        # 防止GC在周期内触发尖峰
        if self.cycle_count % 500 == 0:
            gc.collect(1)
        elif self.cycle_count % 100 == 0:
            gc.collect(0)

    def run_cycle_with_logging(self) -> List[Dict]:
        """执行周期并返回详细日志（用于测试）"""
        self.run_cycle()
        return self.last_cycle_log

    def run_cycle_get_action(self) -> Action:
        """执行周期并返回动作（用于测试）"""
        self.run_cycle()
        return self.modules["decision_maker"].get_last_action()

    def run_with_environment(self, environment: Any, num_steps: int,
                             reset: bool = True) -> Dict[str, Any]:
        """
        阶段24: 附带环境运行多个周期
        - 自动附加环境、注入初始观测、运行num_steps步
        - 返回统计信息（总reward、平均reward、动作分布等）
        """
        self.attach_environment(environment)
        if reset:
            obs = environment.reset()
            self.inject_input(obs)

        rewards = []
        actions = []
        for _ in range(num_steps):
            if self._env_done:
                break
            result = self.run_cycle()
            rewards.append(result.get("time", 0))  # placeholder
            action = self.modules["decision_maker"].get_last_action()
            actions.append(action.type.value if action else "unknown")

        # 从executor获取真实reward统计
        exec_stats = self.modules["executor"].get_stats()
        return {
            "steps_completed": self.cycle_count,
            "total_real_reward": exec_stats.get("total_real_reward", 0.0),
            "env_done": self._env_done,
            "action_distribution": dict(zip(
                *np.unique(actions, return_counts=True)
            )) if actions else {},
            "env_state": environment.get_state() if hasattr(environment, "get_state") else {},
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return {
            "cycle_count": self.cycle_count,
            "total_reward": self.metrics["total_reward"],
            "avg_prediction_error": float(np.mean(self.metrics["prediction_errors"]))
                if self.metrics["prediction_errors"] else 0.0,
            "avg_surprise": float(np.mean(self.metrics["surprises"]))
                if self.metrics["surprises"] else 0.0,
            "avg_cycle_time": float(np.mean(self.metrics["cycle_times"]))
                if self.metrics["cycle_times"] else 0.0,
            "mode": self.mode,
            "action_distribution": dict(zip(
                *np.unique(self.metrics["action_types"], return_counts=True)
            )) if self.metrics["action_types"] else {},
        }

    def get_handled_messages(self) -> List:
        return []

    def reset(self):
        """重置系统"""
        self.cycle_count = 0
        self.mode = "normal"
        self.metrics = {
            "total_reward": 0.0,
            "prediction_errors": [],
            "surprises": [],
            "goal_types": [],
            "action_types": [],
            "cycle_times": [],
        }
        for mod in self.modules.values():
            if hasattr(mod, "reset"):
                mod.reset()
        self.modules["safety_ethics"].reset_circuit_breaker()
        logger.info("[Core] 重置完成")

    def health_check(self) -> bool:
        """健康检查"""
        for name, mod in self.modules.items():
            if hasattr(mod, "health_check") and not mod.health_check():
                logger.warning(f"[Core] 模块{name}健康检查失败")
                return False
        return True
