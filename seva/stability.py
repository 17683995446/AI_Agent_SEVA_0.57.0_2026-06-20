"""
系统稳定性与容错增强（阶段53）
对应第一性原理：稳定性 = 在故障发生时保持服务可用，通过隔离、降级、恢复使系统自愈

提供能力:
    - CircuitBreaker: 熔断器（CLOSED/OPEN/HALF_OPEN三态，失败计数、超时恢复）
    - RetryPolicy: 重试策略（指数退避、最大重试次数、可重试异常过滤）
    - HealthCheck: 健康检查（定期检查、健康状态、报告）
    - FaultRecovery: 故障恢复（降级、回退、自动恢复策略）
    - StabilityManager: 稳定性管理器（单例，整合所有组件）
    - 便捷函数: circuit / retry / health_check / recover / get_stability_report

使用示例:
    from seva import CircuitBreaker, retry_with_policy, HealthCheck

    # 熔断器保护
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    result = cb.call(risky_function, *args)

    # 带策略的重试
    policy = RetryPolicy(max_retries=3, backoff="exponential")
    result = retry_with_policy(unstable_function, policy, *args)

    # 健康检查
    hc = HealthCheck("db", check_func)
    status = hc.check()
"""
from __future__ import annotations

import os
import sys
import time
import json
import threading
import functools
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
STABILITY_FILE = SEVA_HOME / "stability.json"


# ============================================================================
# 异常定义
# ============================================================================

class CircuitBreakerOpenError(Exception):
    """熔断器处于开启状态时抛出"""

    def __init__(self, name: str, recovery_time: float):
        self.name = name
        self.recovery_time = recovery_time
        super().__init__(
            f"熔断器 '{name}' 处于开启状态，将在 {recovery_time:.1f}s 后尝试恢复"
        )


class HealthCheckError(Exception):
    """健康检查失败"""

    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message
        super().__init__(f"健康检查 '{name}' 失败: {message}")


# ============================================================================
# CircuitState - 熔断器状态枚举
# ============================================================================

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 正常，允许请求通过
    OPEN = "open"            # 熔断，拒绝所有请求
    HALF_OPEN = "half_open"  # 半开，允许少量请求试探


# ============================================================================
# HealthStatus - 健康状态枚举
# ============================================================================

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
# BackoffStrategy - 退避策略枚举
# ============================================================================

class BackoffStrategy(Enum):
    """退避策略"""
    FIXED = "fixed"            # 固定间隔
    LINEAR = "linear"          # 线性增长
    EXPONENTIAL = "exponential"  # 指数退避
    RANDOM = "random"          # 随机间隔


# ============================================================================
# CircuitBreaker - 熔断器
# ============================================================================

class CircuitBreaker:
    """
    熔断器：在连续失败达到阈值时切断请求，保护系统免受级联故障

    状态转换:
        CLOSED --(失败达阈值)--> OPEN --(超时后)--> HALF_OPEN
        HALF_OPEN --(成功)--> CLOSED
        HALF_OPEN --(失败)--> OPEN
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        expected_exception: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    ):
        """
        Args:
            name: 熔断器名称
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断后恢复尝试的超时时间（秒）
            success_threshold: HALF_OPEN状态下连续成功次数后恢复CLOSED
            expected_exception: 视为失败的异常类型
        """
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)
        self.success_threshold = max(1, success_threshold)
        self.expected_exception = expected_exception

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._lock = threading.RLock()

        # 统计
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejected = 0
        self._state_transitions: List[Dict[str, Any]] = []

    @property
    def state(self) -> CircuitState:
        """当前状态（自动检查是否应从OPEN转HALF_OPEN）"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._opened_at is not None:
                    elapsed = time.time() - self._opened_at
                    if elapsed >= self.recovery_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def is_closed(self) -> bool:
        """是否关闭（正常）状态"""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """是否开启（熔断）状态"""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """是否半开状态"""
        return self.state == CircuitState.HALF_OPEN

    @property
    def failure_count(self) -> int:
        """当前连续失败次数"""
        with self._lock:
            return self._failure_count

    @property
    def success_count(self) -> int:
        """HALF_OPEN状态下连续成功次数"""
        with self._lock:
            return self._success_count

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数

        Raises:
            CircuitBreakerOpenError: 熔断器开启时
        """
        with self._lock:
            current_state = self.state
            self._total_calls += 1

            if current_state == CircuitState.OPEN:
                self._total_rejected += 1
                remaining = self.recovery_timeout
                if self._opened_at is not None:
                    remaining = max(0, self.recovery_timeout - (time.time() - self._opened_at))
                raise CircuitBreakerOpenError(self.name, remaining)

        # 执行函数（锁外执行避免长时间持锁）
        try:
            result = func(*args, **kwargs)
        except self.expected_exception as e:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        """调用成功时处理"""
        with self._lock:
            self._total_successes += 1
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self) -> None:
        """调用失败时处理"""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下失败，重新熔断
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换"""
        if self._state == new_state:
            return
        old_state = self._state
        self._state = new_state
        self._opened_at = time.time() if new_state == CircuitState.OPEN else None

        # 重置计数器
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0

        self._state_transitions.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": time.time(),
        })

    def reset(self) -> None:
        """重置熔断器到初始状态"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._opened_at = None

    def force_open(self) -> None:
        """强制开启熔断器"""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """强制关闭熔断器"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "total_rejected": self._total_rejected,
                "failure_rate": (
                    self._total_failures / self._total_calls
                    if self._total_calls > 0 else 0.0
                ),
                "success_rate": (
                    self._total_successes / self._total_calls
                    if self._total_calls > 0 else 0.0
                ),
                "last_failure_time": self._last_failure_time,
                "opened_at": self._opened_at,
                "transitions": len(self._state_transitions),
            }

    def __repr__(self) -> str:
        return (f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
                f"failures={self._failure_count}/{self.failure_threshold})")


# ============================================================================
# RetryPolicy - 重试策略
# ============================================================================

@dataclass
class RetryPolicy:
    """
    重试策略：定义如何重试失败的操作

    Args:
        max_retries: 最大重试次数（不含首次）
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        backoff: 退避策略
        jitter: 是否添加随机抖动（避免惊群）
        retryable_exceptions: 可重试的异常类型
    """
    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    retryable_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception

    def __post_init__(self):
        if isinstance(self.backoff, str):
            self.backoff = BackoffStrategy(self.backoff)
        self.max_retries = max(0, self.max_retries)
        self.base_delay = max(0.0, self.base_delay)
        self.max_delay = max(self.base_delay, self.max_delay)

    def get_delay(self, attempt: int) -> float:
        """
        计算第attempt次重试的延迟（attempt从1开始）

        Returns:
            延迟秒数
        """
        if attempt <= 0:
            return 0.0

        if self.backoff == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.backoff == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.backoff == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.backoff == BackoffStrategy.RANDOM:
            delay = random.uniform(self.base_delay, self.base_delay * (2 ** attempt))
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            # 添加 ±25% 的抖动
            jitter_amount = delay * 0.25
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries + 1:
            return False
        return isinstance(exception, self.retryable_exceptions)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff": self.backoff.value,
            "jitter": self.jitter,
            "retryable_exceptions": (
                self.retryable_exceptions.__name__
                if isinstance(self.retryable_exceptions, type)
                else tuple(e.__name__ for e in self.retryable_exceptions)
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetryPolicy":
        """从字典创建"""
        backoff = data.get("backoff", "exponential")
        if isinstance(backoff, str):
            backoff = BackoffStrategy(backoff)
        return cls(
            max_retries=data.get("max_retries", 3),
            base_delay=data.get("base_delay", 0.1),
            max_delay=data.get("max_delay", 60.0),
            backoff=backoff,
            jitter=data.get("jitter", True),
        )


def retry_with_policy(
    func: Callable,
    policy: RetryPolicy,
    *args,
    sleep_func: Callable[[float], None] = time.sleep,
    **kwargs,
) -> Any:
    """
    按策略重试执行函数

    Args:
        func: 要执行的函数
        policy: 重试策略
        sleep_func: 睡眠函数（可注入用于测试）
        *args, **kwargs: 函数参数

    Returns:
        函数执行结果

    Raises:
        最后一次失败的异常
    """
    last_exception: Optional[Exception] = None
    total_attempts = policy.max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not policy.should_retry(e, attempt):
                raise
            if attempt < total_attempts:
                delay = policy.get_delay(attempt)
                sleep_func(delay)

    # 理论上不会到达
    if last_exception:
        raise last_exception
    raise RuntimeError("重试逻辑异常")


# ============================================================================
# HealthCheck - 健康检查
# ============================================================================

class HealthCheck:
    """
    健康检查：定期检查组件健康状态

    使用示例:
        def check_db():
            # 返回True/False或抛异常
            return db.ping()

        hc = HealthCheck("database", check_db)
        status = hc.check()  # HealthStatus.HEALTHY
    """

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Union[bool, None]],
        timeout: float = 5.0,
        critical: bool = True,
    ):
        """
        Args:
            name: 检查名称
            check_func: 检查函数（返回True/False/None，或抛异常）
            timeout: 超时时间（秒）
            critical: 是否关键组件（影响整体健康判定）
        """
        self.name = name
        self.check_func = check_func
        self.timeout = max(0.1, timeout)
        self.critical = critical

        self._last_status: HealthStatus = HealthStatus.UNKNOWN
        self._last_check_time: Optional[float] = None
        self._last_error: Optional[str] = None
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    @property
    def last_status(self) -> HealthStatus:
        """最后一次检查状态"""
        with self._lock:
            return self._last_status

    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self.last_status == HealthStatus.HEALTHY

    def check(self) -> HealthStatus:
        """执行健康检查"""
        start_time = time.time()
        status: HealthStatus
        error_msg: Optional[str] = None

        try:
            result = self.check_func()
            if result is True or result is None:
                status = HealthStatus.HEALTHY
            elif result is False:
                status = HealthStatus.UNHEALTHY
                error_msg = "检查函数返回False"
            else:
                status = HealthStatus.HEALTHY
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            error_msg = str(e)
        finally:
            elapsed = time.time() - start_time

        with self._lock:
            self._last_status = status
            self._last_check_time = time.time()
            self._last_error = error_msg

            if status == HealthStatus.HEALTHY:
                self._consecutive_successes += 1
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                self._consecutive_successes = 0

            # 记录历史（最多保留100条）
            self._history.append({
                "status": status.value,
                "timestamp": self._last_check_time,
                "elapsed": elapsed,
                "error": error_msg,
            })
            if len(self._history) > 100:
                self._history = self._history[-100:]

        return status

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        with self._lock:
            return {
                "name": self.name,
                "status": self._last_status.value,
                "critical": self.critical,
                "last_check_time": self._last_check_time,
                "last_error": self._last_error,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "history_count": len(self._history),
            }

    def reset(self) -> None:
        """重置状态"""
        with self._lock:
            self._last_status = HealthStatus.UNKNOWN
            self._last_check_time = None
            self._last_error = None
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._history.clear()

    def __repr__(self) -> str:
        return (f"HealthCheck(name={self.name!r}, status={self._last_status.value}, "
                f"critical={self.critical})")


# ============================================================================
# FaultRecovery - 故障恢复
# ============================================================================

class FaultRecovery:
    """
    故障恢复：提供降级、回退、自动恢复策略

    使用示例:
        recovery = FaultRecovery()

        # 注册降级策略
        recovery.register_fallback("primary_api", lambda: call_backup_api())

        # 注册恢复动作
        recovery.register_recovery_action("db_connection", reconnect_db)

        # 执行带恢复的操作
        result = recovery.execute_with_recovery(
            "db_query",
            primary_func=query_db,
            fallback_func=query_cache,
        )
    """

    def __init__(self):
        self._fallbacks: Dict[str, Callable] = {}
        self._recovery_actions: Dict[str, Callable] = {}
        self._degraded_mode: Dict[str, bool] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._lock = threading.RLock()

    def register_fallback(self, name: str, fallback_func: Callable) -> None:
        """注册降级/回退函数"""
        with self._lock:
            self._fallbacks[name] = fallback_func

    def register_recovery_action(self, name: str, action: Callable) -> None:
        """注册恢复动作"""
        with self._lock:
            self._recovery_actions[name] = action

    def execute_with_recovery(
        self,
        name: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        执行带恢复的操作：先尝试主函数，失败则用回退函数

        Args:
            name: 操作名称
            primary_func: 主函数
            fallback_func: 回退函数（None则使用注册的回退）
            *args, **kwargs: 函数参数

        Returns:
            主函数或回退函数的结果
        """
        try:
            result = primary_func(*args, **kwargs)
            # 成功则清除降级标记
            with self._lock:
                self._degraded_mode[name] = False
            return result
        except Exception as primary_error:
            # 尝试回退
            fb = fallback_func or self._fallbacks.get(name)
            if fb is not None:
                with self._lock:
                    self._degraded_mode[name] = True
                try:
                    return fb(*args, **kwargs)
                except Exception:
                    pass

            # 尝试恢复动作后重试主函数
            action = self._recovery_actions.get(name)
            if action is not None:
                with self._lock:
                    self._recovery_attempts[name] = self._recovery_attempts.get(name, 0) + 1
                try:
                    action()
                    return primary_func(*args, **kwargs)
                except Exception:
                    pass

            # 所有恢复手段失败，抛出原始异常
            raise primary_error

    def is_degraded(self, name: str) -> bool:
        """检查某操作是否处于降级模式"""
        with self._lock:
            return self._degraded_mode.get(name, False)

    def get_degraded_operations(self) -> List[str]:
        """获取所有处于降级模式的操作"""
        with self._lock:
            return [name for name, degraded in self._degraded_mode.items() if degraded]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "registered_fallbacks": list(self._fallbacks.keys()),
                "registered_recovery_actions": list(self._recovery_actions.keys()),
                "degraded_operations": self.get_degraded_operations(),
                "recovery_attempts": dict(self._recovery_attempts),
            }

    def reset(self) -> None:
        """重置"""
        with self._lock:
            self._degraded_mode.clear()
            self._recovery_attempts.clear()

    def __repr__(self) -> str:
        return (f"FaultRecovery(fallbacks={len(self._fallbacks)}, "
                f"degraded={len(self.get_degraded_operations())})")


# ============================================================================
# StabilityManager - 稳定性管理器（单例）
# ============================================================================

class StabilityManager:
    """
    稳定性管理器：整合熔断器、重试、健康检查、故障恢复

    单例模式，全局唯一实例
    """

    _instance: Optional["StabilityManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StabilityManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
        self._recovery = FaultRecovery()
        self._default_retry_policy = RetryPolicy()
        self._data_lock = threading.RLock()

    # ---- 熔断器管理 ----

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        with self._data_lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return self._circuit_breakers[name]

    def list_circuit_breakers(self) -> List[str]:
        """列出所有熔断器名称"""
        with self._data_lock:
            return list(self._circuit_breakers.keys())

    def reset_all_circuit_breakers(self) -> None:
        """重置所有熔断器"""
        with self._data_lock:
            for cb in self._circuit_breakers.values():
                cb.reset()

    # ---- 健康检查管理 ----

    def register_health_check(self, check: HealthCheck) -> None:
        """注册健康检查"""
        with self._data_lock:
            self._health_checks[check.name] = check

    def run_health_checks(self) -> Dict[str, HealthStatus]:
        """运行所有健康检查"""
        with self._data_lock:
            checks = list(self._health_checks.values())
        return {check.name: check.check() for check in checks}

    def get_overall_health(self) -> HealthStatus:
        """获取整体健康状态"""
        with self._data_lock:
            checks = list(self._health_checks.values())

        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [check.last_status for check in checks]
        critical_checks = [check for check in checks if check.critical]

        # 关键检查有不健康的，整体不健康
        for check in critical_checks:
            if check.last_status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY

        # 有降级的，整体降级
        if HealthStatus.UNHEALTHY in statuses or HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # 全部健康
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    # ---- 故障恢复 ----

    @property
    def recovery(self) -> FaultRecovery:
        """故障恢复器"""
        return self._recovery

    # ---- 重试策略 ----

    @property
    def default_retry_policy(self) -> RetryPolicy:
        """默认重试策略"""
        return self._default_retry_policy

    @default_retry_policy.setter
    def default_retry_policy(self, policy: RetryPolicy) -> None:
        self._default_retry_policy = policy

    # ---- 综合操作 ----

    def protected_call(
        self,
        name: str,
        func: Callable,
        *args,
        retry_policy: Optional[RetryPolicy] = None,
        use_circuit_breaker: bool = True,
        fallback_func: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """
        受保护调用：熔断器 + 重试 + 回退

        Args:
            name: 操作名称
            func: 要调用的函数
            retry_policy: 重试策略（None使用默认）
            use_circuit_breaker: 是否使用熔断器
            fallback_func: 回退函数
        """
        policy = retry_policy or self._default_retry_policy

        def execute():
            if use_circuit_breaker:
                cb = self.get_circuit_breaker(name)
                return cb.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        try:
            return retry_with_policy(execute, policy, sleep_func=time.sleep)
        except Exception:
            if fallback_func is not None:
                return fallback_func(*args, **kwargs)
            raise

    def get_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        with self._data_lock:
            cb_stats = {name: cb.get_stats() for name, cb in self._circuit_breakers.items()}
            hc_stats = {name: hc.get_status() for name, hc in self._health_checks.items()}

        return {
            "overall_health": self.get_overall_health().value,
            "circuit_breakers": cb_stats,
            "health_checks": hc_stats,
            "recovery": self._recovery.get_stats(),
            "default_retry_policy": self._default_retry_policy.to_dict(),
        }

    def format_report(self) -> str:
        """格式化报告为字符串"""
        report = self.get_report()
        lines = []
        lines.append("=" * 60)
        lines.append("SEVA 系统稳定性报告")
        lines.append("=" * 60)
        lines.append(f"整体健康状态: {report['overall_health']}")
        lines.append("")

        # 熔断器
        lines.append("--- 熔断器 ---")
        if report["circuit_breakers"]:
            for name, stats in report["circuit_breakers"].items():
                lines.append(
                    f"  [{stats['state'].upper()}] {name}: "
                    f"调用{stats['total_calls']}次, "
                    f"失败{stats['total_failures']}次, "
                    f"拒绝{stats['total_rejected']}次, "
                    f"失败率{stats['failure_rate']*100:.1f}%"
                )
        else:
            lines.append("  (无注册熔断器)")
        lines.append("")

        # 健康检查
        lines.append("--- 健康检查 ---")
        if report["health_checks"]:
            for name, stats in report["health_checks"].items():
                critical_tag = " [关键]" if stats["critical"] else ""
                lines.append(
                    f"  [{stats['status'].upper()}] {name}{critical_tag}: "
                    f"连续失败{stats['consecutive_failures']}次"
                )
                if stats["last_error"]:
                    lines.append(f"       错误: {stats['last_error']}")
        else:
            lines.append("  (无注册健康检查)")
        lines.append("")

        # 故障恢复
        lines.append("--- 故障恢复 ---")
        recovery = report["recovery"]
        lines.append(f"  注册回退: {len(recovery['registered_fallbacks'])}个")
        lines.append(f"  注册恢复动作: {len(recovery['registered_recovery_actions'])}个")
        degraded = recovery["degraded_operations"]
        if degraded:
            lines.append(f"  降级中的操作: {', '.join(degraded)}")
        else:
            lines.append("  降级中的操作: 无")
        lines.append("")

        # 重试策略
        lines.append("--- 默认重试策略 ---")
        rp = report["default_retry_policy"]
        lines.append(f"  最大重试: {rp['max_retries']}次")
        lines.append(f"  退避策略: {rp['backoff']}")
        lines.append(f"  基础延迟: {rp['base_delay']}s")
        lines.append(f"  最大延迟: {rp['max_delay']}s")
        lines.append(f"  随机抖动: {'是' if rp['jitter'] else '否'}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """重置所有状态"""
        with self._data_lock:
            for cb in self._circuit_breakers.values():
                cb.reset()
            for hc in self._health_checks.values():
                hc.reset()
            self._recovery.reset()

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            cls._instance = None


# ============================================================================
# 装饰器
# ============================================================================

def circuit_protected(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable:
    """
    装饰器：用熔断器保护函数

    使用示例:
        @circuit_protected("api_call", failure_threshold=3)
        def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = StabilityManager().get_circuit_breaker(
                name, failure_threshold, recovery_timeout
            )
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retryable(
    max_retries: int = 3,
    backoff: Union[BackoffStrategy, str] = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 0.1,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
) -> Callable:
    """
    装饰器：自动重试

    使用示例:
        @retryable(max_retries=5, backoff="exponential")
        def unstable_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(backoff, str):
                bs = BackoffStrategy(backoff)
            else:
                bs = backoff
            policy = RetryPolicy(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff=bs,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
            )
            return retry_with_policy(func, policy, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# 便捷函数
# ============================================================================

def get_stability_manager() -> StabilityManager:
    """获取稳定性管理器单例"""
    return StabilityManager()


def get_stability_report() -> str:
    """获取稳定性报告"""
    return StabilityManager().format_report()


def get_stability_status() -> Dict[str, Any]:
    """获取稳定性状态"""
    return StabilityManager().get_report()


def reset_stability() -> None:
    """重置所有稳定性状态"""
    StabilityManager().reset()


def circuit(name: str, **kwargs) -> CircuitBreaker:
    """获取或创建熔断器"""
    return StabilityManager().get_circuit_breaker(name, **kwargs)


def health_check(name: str, check_func: Callable, **kwargs) -> HealthCheck:
    """创建并注册健康检查"""
    hc = HealthCheck(name, check_func, **kwargs)
    StabilityManager().register_health_check(hc)
    return hc


def recover(name: str, primary: Callable, fallback: Optional[Callable] = None,
            *args, **kwargs) -> Any:
    """带恢复的执行"""
    return StabilityManager().recovery.execute_with_recovery(
        name, primary, fallback, *args, **kwargs
    )


def protected_call(name: str, func: Callable, *args, **kwargs) -> Any:
    """受保护调用"""
    return StabilityManager().protected_call(name, func, *args, **kwargs)
