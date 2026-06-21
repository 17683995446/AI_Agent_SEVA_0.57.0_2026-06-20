"""
阶段53测试：系统稳定性与容错增强
"""
import unittest
import time
import threading
from unittest.mock import MagicMock, patch

from seva.stability import (
    CircuitState, HealthStatus, BackoffStrategy,
    CircuitBreaker, CircuitBreakerOpenError, HealthCheckError,
    RetryPolicy, retry_with_policy,
    HealthCheck, FaultRecovery,
    StabilityManager,
    circuit_protected, retryable,
    get_stability_manager, get_stability_report, get_stability_status,
    reset_stability, circuit, health_check, recover, protected_call,
)


class TestCircuitState(unittest.TestCase):
    """测试熔断器状态枚举"""

    def test01_states(self):
        self.assertEqual(CircuitState.CLOSED.value, "closed")
        self.assertEqual(CircuitState.OPEN.value, "open")
        self.assertEqual(CircuitState.HALF_OPEN.value, "half_open")

    def test02_health_status(self):
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")
        self.assertEqual(HealthStatus.UNKNOWN.value, "unknown")

    def test03_backoff_strategy(self):
        self.assertEqual(BackoffStrategy.FIXED.value, "fixed")
        self.assertEqual(BackoffStrategy.LINEAR.value, "linear")
        self.assertEqual(BackoffStrategy.EXPONENTIAL.value, "exponential")
        self.assertEqual(BackoffStrategy.RANDOM.value, "random")


class TestCircuitBreaker(unittest.TestCase):
    """测试熔断器"""

    def test01_initial_state(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)
        self.assertTrue(cb.is_closed)
        self.assertFalse(cb.is_open)
        self.assertFalse(cb.is_half_open)
        self.assertEqual(cb.failure_count, 0)

    def test02_success_call(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        result = cb.call(lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertTrue(cb.is_closed)

    def test03_failure_triggers_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=10.0)

        def fail():
            raise ValueError("失败")

        for i in range(3):
            with self.assertRaises(ValueError):
                cb.call(fail)

        self.assertTrue(cb.is_open)
        self.assertEqual(cb.failure_count, 3)

    def test04_open_rejects_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=10.0)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        self.assertTrue(cb.is_open)
        with self.assertRaises(CircuitBreakerOpenError):
            cb.call(lambda: "ok")

    def test05_recovery_to_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        self.assertTrue(cb.is_open)
        time.sleep(0.15)
        # 访问state属性触发状态转换
        self.assertTrue(cb.is_half_open)

    def test06_half_open_success_to_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=2,
                            recovery_timeout=0.1, success_threshold=2)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        time.sleep(0.15)
        self.assertTrue(cb.is_half_open)

        # 两次成功后恢复
        cb.call(lambda: "ok1")
        self.assertTrue(cb.is_half_open)
        cb.call(lambda: "ok2")
        self.assertTrue(cb.is_closed)

    def test07_half_open_failure_to_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        time.sleep(0.15)
        self.assertTrue(cb.is_half_open)

        # 半开状态下失败，重新熔断
        with self.assertRaises(ValueError):
            cb.call(fail)
        self.assertTrue(cb.is_open)

    def test08_reset(self):
        cb = CircuitBreaker(name="test", failure_threshold=2)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        self.assertTrue(cb.is_open)
        cb.reset()
        self.assertTrue(cb.is_closed)
        self.assertEqual(cb.failure_count, 0)

    def test09_force_open_close(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.force_open()
        self.assertTrue(cb.is_open)
        cb.force_close()
        self.assertTrue(cb.is_closed)

    def test10_stats(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.call(lambda: "ok")
        cb.call(lambda: "ok")
        stats = cb.get_stats()
        self.assertEqual(stats["name"], "test")
        self.assertEqual(stats["state"], "closed")
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["total_successes"], 2)
        self.assertEqual(stats["total_failures"], 0)
        self.assertEqual(stats["failure_rate"], 0.0)
        self.assertEqual(stats["success_rate"], 1.0)

    def test11_repr(self):
        cb = CircuitBreaker(name="mycb", failure_threshold=5)
        r = repr(cb)
        self.assertIn("CircuitBreaker", r)
        self.assertIn("mycb", r)
        self.assertIn("closed", r)

    def test12_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)

        def fail():
            raise ValueError("失败")

        with self.assertRaises(ValueError):
            cb.call(fail)
        with self.assertRaises(ValueError):
            cb.call(fail)
        self.assertEqual(cb.failure_count, 2)

        # 成功调用重置失败计数
        cb.call(lambda: "ok")
        self.assertEqual(cb.failure_count, 0)

    def test13_expected_exception_filter(self):
        cb = CircuitBreaker(name="test", failure_threshold=3,
                            expected_exception=ValueError)

        # TypeError不计入失败
        with self.assertRaises(TypeError):
            cb.call(lambda: (_ for _ in ()).throw(TypeError("type error")))
        self.assertEqual(cb.failure_count, 0)
        self.assertTrue(cb.is_closed)


class TestRetryPolicy(unittest.TestCase):
    """测试重试策略"""

    def test01_default_policy(self):
        policy = RetryPolicy()
        self.assertEqual(policy.max_retries, 3)
        self.assertEqual(policy.backoff, BackoffStrategy.EXPONENTIAL)
        self.assertTrue(policy.jitter)

    def test02_fixed_delay(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.5,
                             backoff=BackoffStrategy.FIXED, jitter=False)
        self.assertAlmostEqual(policy.get_delay(1), 0.5)
        self.assertAlmostEqual(policy.get_delay(2), 0.5)
        self.assertAlmostEqual(policy.get_delay(3), 0.5)

    def test03_linear_delay(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.1,
                             backoff=BackoffStrategy.LINEAR, jitter=False)
        self.assertAlmostEqual(policy.get_delay(1), 0.1)
        self.assertAlmostEqual(policy.get_delay(2), 0.2)
        self.assertAlmostEqual(policy.get_delay(3), 0.3)

    def test04_exponential_delay(self):
        policy = RetryPolicy(max_retries=4, base_delay=0.1,
                             backoff=BackoffStrategy.EXPONENTIAL, jitter=False)
        self.assertAlmostEqual(policy.get_delay(1), 0.1)
        self.assertAlmostEqual(policy.get_delay(2), 0.2)
        self.assertAlmostEqual(policy.get_delay(3), 0.4)
        self.assertAlmostEqual(policy.get_delay(4), 0.8)

    def test05_max_delay_cap(self):
        policy = RetryPolicy(max_retries=10, base_delay=1.0,
                             max_delay=5.0,
                             backoff=BackoffStrategy.EXPONENTIAL, jitter=False)
        self.assertLessEqual(policy.get_delay(10), 5.0)

    def test06_jitter(self):
        policy = RetryPolicy(base_delay=1.0, backoff=BackoffStrategy.FIXED, jitter=True)
        delay = policy.get_delay(1)
        # 抖动范围 0.75 ~ 1.25
        self.assertGreaterEqual(delay, 0.75)
        self.assertLessEqual(delay, 1.25)

    def test07_should_retry(self):
        policy = RetryPolicy(max_retries=3, retryable_exceptions=ValueError)
        self.assertTrue(policy.should_retry(ValueError("test"), 1))
        self.assertFalse(policy.should_retry(TypeError("test"), 1))
        self.assertFalse(policy.should_retry(ValueError("test"), 4))

    def test08_to_dict_from_dict(self):
        policy = RetryPolicy(max_retries=5, base_delay=0.2, max_delay=10.0,
                             backoff=BackoffStrategy.LINEAR, jitter=False)
        d = policy.to_dict()
        self.assertEqual(d["max_retries"], 5)
        self.assertEqual(d["backoff"], "linear")

        policy2 = RetryPolicy.from_dict(d)
        self.assertEqual(policy2.max_retries, 5)
        self.assertEqual(policy2.backoff, BackoffStrategy.LINEAR)

    def test09_backoff_from_string(self):
        policy = RetryPolicy(backoff="fixed")
        self.assertEqual(policy.backoff, BackoffStrategy.FIXED)

    def test10_retry_with_policy_success(self):
        calls = [0]

        def func():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("重试")
            return "成功"

        sleeps = []
        policy = RetryPolicy(max_retries=5, base_delay=0.001, jitter=False)
        result = retry_with_policy(func, policy, sleep_func=sleeps.append)
        self.assertEqual(result, "成功")
        self.assertEqual(calls[0], 3)
        self.assertEqual(len(sleeps), 2)

    def test11_retry_with_policy_all_fail(self):
        calls = [0]

        def func():
            calls[0] += 1
            raise ValueError(f"失败{calls[0]}")

        policy = RetryPolicy(max_retries=2, base_delay=0.001, jitter=False)
        with self.assertRaises(ValueError):
            retry_with_policy(func, policy, sleep_func=lambda x: None)
        self.assertEqual(calls[0], 3)  # 1 + 2 retries

    def test12_retry_non_retryable_exception(self):
        calls = [0]

        def func():
            calls[0] += 1
            raise TypeError("不可重试")

        policy = RetryPolicy(max_retries=5, retryable_exceptions=ValueError)
        with self.assertRaises(TypeError):
            retry_with_policy(func, policy, sleep_func=lambda x: None)
        self.assertEqual(calls[0], 1)  # 不重试


class TestHealthCheck(unittest.TestCase):
    """测试健康检查"""

    def test01_healthy(self):
        hc = HealthCheck("test", lambda: True)
        status = hc.check()
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertTrue(hc.is_healthy)

    def test02_unhealthy_false(self):
        hc = HealthCheck("test", lambda: False)
        status = hc.check()
        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertFalse(hc.is_healthy)

    def test03_unhealthy_exception(self):
        def fail():
            raise RuntimeError("连接失败")

        hc = HealthCheck("test", fail)
        status = hc.check()
        self.assertEqual(status, HealthStatus.UNHEALTHY)

    def test04_none_is_healthy(self):
        hc = HealthCheck("test", lambda: None)
        status = hc.check()
        self.assertEqual(status, HealthStatus.HEALTHY)

    def test05_consecutive_counts(self):
        hc = HealthCheck("test", lambda: False)
        hc.check()
        hc.check()
        self.assertEqual(hc.get_status()["consecutive_failures"], 2)
        self.assertEqual(hc.get_status()["consecutive_successes"], 0)

    def test06_status_info(self):
        hc = HealthCheck("db", lambda: True, critical=True)
        hc.check()
        info = hc.get_status()
        self.assertEqual(info["name"], "db")
        self.assertEqual(info["status"], "healthy")
        self.assertTrue(info["critical"])
        self.assertEqual(info["consecutive_successes"], 1)

    def test07_reset(self):
        hc = HealthCheck("test", lambda: False)
        hc.check()
        hc.reset()
        self.assertEqual(hc.last_status, HealthStatus.UNKNOWN)

    def test08_repr(self):
        hc = HealthCheck("mycheck", lambda: True)
        r = repr(hc)
        self.assertIn("HealthCheck", r)
        self.assertIn("mycheck", r)

    def test09_history(self):
        hc = HealthCheck("test", lambda: True)
        for _ in range(5):
            hc.check()
        self.assertEqual(hc.get_status()["history_count"], 5)


class TestFaultRecovery(unittest.TestCase):
    """测试故障恢复"""

    def test01_register_fallback(self):
        fr = FaultRecovery()
        fr.register_fallback("op1", lambda: "fallback")
        self.assertIn("op1", fr.get_stats()["registered_fallbacks"])

    def test02_register_recovery_action(self):
        fr = FaultRecovery()
        fr.register_recovery_action("op1", lambda: None)
        self.assertIn("op1", fr.get_stats()["registered_recovery_actions"])

    def test03_execute_with_recovery_primary_success(self):
        fr = FaultRecovery()
        result = fr.execute_with_recovery("op", lambda: "primary")
        self.assertEqual(result, "primary")
        self.assertFalse(fr.is_degraded("op"))

    def test04_execute_with_recovery_fallback(self):
        fr = FaultRecovery()

        def primary():
            raise ValueError("主函数失败")

        def fallback():
            return "fallback"

        result = fr.execute_with_recovery("op", primary, fallback)
        self.assertEqual(result, "fallback")
        self.assertTrue(fr.is_degraded("op"))

    def test05_execute_with_recovery_registered_fallback(self):
        fr = FaultRecovery()
        fr.register_fallback("op", lambda: "registered_fallback")

        def primary():
            raise ValueError("失败")

        result = fr.execute_with_recovery("op", primary)
        self.assertEqual(result, "registered_fallback")

    def test06_execute_with_recovery_action(self):
        fr = FaultRecovery()
        reconnected = [False]

        def primary():
            if not reconnected[0]:
                raise ConnectionError("断开")
            return "重连后成功"

        def recovery_action():
            reconnected[0] = True

        fr.register_recovery_action("op", recovery_action)
        result = fr.execute_with_recovery("op", primary)
        self.assertEqual(result, "重连后成功")

    def test07_all_fail_raises(self):
        fr = FaultRecovery()

        def primary():
            raise ValueError("主函数失败")

        with self.assertRaises(ValueError):
            fr.execute_with_recovery("op", primary)

    def test08_degraded_operations(self):
        fr = FaultRecovery()
        fr.register_fallback("op1", lambda: "fb1")
        fr.register_fallback("op2", lambda: "fb2")

        def fail():
            raise ValueError("失败")

        fr.execute_with_recovery("op1", fail)
        fr.execute_with_recovery("op2", fail)

        degraded = fr.get_degraded_operations()
        self.assertIn("op1", degraded)
        self.assertIn("op2", degraded)

    def test09_reset(self):
        fr = FaultRecovery()
        fr.register_fallback("op", lambda: "fb")

        def fail():
            raise ValueError("失败")

        fr.execute_with_recovery("op", fail)
        self.assertTrue(fr.is_degraded("op"))
        fr.reset()
        self.assertFalse(fr.is_degraded("op"))

    def test10_repr(self):
        fr = FaultRecovery()
        r = repr(fr)
        self.assertIn("FaultRecovery", r)


class TestStabilityManager(unittest.TestCase):
    """测试稳定性管理器"""

    def setUp(self):
        StabilityManager.reset_instance()

    def tearDown(self):
        StabilityManager.reset_instance()

    def test01_singleton(self):
        mgr1 = StabilityManager()
        mgr2 = StabilityManager()
        self.assertIs(mgr1, mgr2)

    def test02_get_circuit_breaker(self):
        mgr = StabilityManager()
        cb = mgr.get_circuit_breaker("test", failure_threshold=3)
        self.assertIsInstance(cb, CircuitBreaker)
        self.assertEqual(cb.name, "test")

        # 同名返回同一实例
        cb2 = mgr.get_circuit_breaker("test")
        self.assertIs(cb, cb2)

    def test03_list_circuit_breakers(self):
        mgr = StabilityManager()
        mgr.get_circuit_breaker("cb1")
        mgr.get_circuit_breaker("cb2")
        names = mgr.list_circuit_breakers()
        self.assertIn("cb1", names)
        self.assertIn("cb2", names)

    def test04_register_health_check(self):
        mgr = StabilityManager()
        hc = HealthCheck("test", lambda: True)
        mgr.register_health_check(hc)
        results = mgr.run_health_checks()
        self.assertIn("test", results)
        self.assertEqual(results["test"], HealthStatus.HEALTHY)

    def test05_overall_health_empty(self):
        mgr = StabilityManager()
        self.assertEqual(mgr.get_overall_health(), HealthStatus.UNKNOWN)

    def test06_overall_health_all_healthy(self):
        mgr = StabilityManager()
        mgr.register_health_check(HealthCheck("h1", lambda: True))
        mgr.register_health_check(HealthCheck("h2", lambda: True))
        mgr.run_health_checks()
        self.assertEqual(mgr.get_overall_health(), HealthStatus.HEALTHY)

    def test07_overall_health_critical_unhealthy(self):
        mgr = StabilityManager()
        mgr.register_health_check(HealthCheck("critical", lambda: False, critical=True))
        mgr.register_health_check(HealthCheck("non_critical", lambda: True, critical=False))
        mgr.run_health_checks()
        self.assertEqual(mgr.get_overall_health(), HealthStatus.UNHEALTHY)

    def test08_overall_health_degraded(self):
        mgr = StabilityManager()
        mgr.register_health_check(HealthCheck("h1", lambda: True, critical=True))
        mgr.register_health_check(HealthCheck("h2", lambda: False, critical=False))
        mgr.run_health_checks()
        self.assertEqual(mgr.get_overall_health(), HealthStatus.DEGRADED)

    def test09_recovery_property(self):
        mgr = StabilityManager()
        self.assertIsInstance(mgr.recovery, FaultRecovery)

    def test10_default_retry_policy(self):
        mgr = StabilityManager()
        policy = mgr.default_retry_policy
        self.assertIsInstance(policy, RetryPolicy)

        new_policy = RetryPolicy(max_retries=10)
        mgr.default_retry_policy = new_policy
        self.assertEqual(mgr.default_retry_policy.max_retries, 10)

    def test11_protected_call_success(self):
        mgr = StabilityManager()
        result = mgr.protected_call("test", lambda: "ok")
        self.assertEqual(result, "ok")

    def test12_protected_call_with_fallback(self):
        mgr = StabilityManager()

        def fail():
            raise ValueError("失败")

        result = mgr.protected_call("test", fail,
                                     fallback_func=lambda: "fallback",
                                     retry_policy=RetryPolicy(max_retries=0, jitter=False))
        self.assertEqual(result, "fallback")

    def test13_get_report(self):
        mgr = StabilityManager()
        mgr.get_circuit_breaker("cb1")
        mgr.register_health_check(HealthCheck("hc1", lambda: True))
        report = mgr.get_report()
        self.assertIn("overall_health", report)
        self.assertIn("circuit_breakers", report)
        self.assertIn("health_checks", report)
        self.assertIn("recovery", report)
        self.assertIn("default_retry_policy", report)

    def test14_format_report(self):
        mgr = StabilityManager()
        report = mgr.format_report()
        self.assertIsInstance(report, str)
        self.assertIn("SEVA 系统稳定性报告", report)
        self.assertIn("整体健康状态", report)

    def test15_reset(self):
        mgr = StabilityManager()
        cb = mgr.get_circuit_breaker("test", failure_threshold=2)

        def fail():
            raise ValueError("失败")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(fail)

        self.assertTrue(cb.is_open)
        mgr.reset()
        self.assertTrue(cb.is_closed)


class TestDecorators(unittest.TestCase):
    """测试装饰器"""

    def setUp(self):
        StabilityManager.reset_instance()

    def tearDown(self):
        StabilityManager.reset_instance()

    def test01_circuit_protected(self):
        @circuit_protected("decorated", failure_threshold=3)
        def func():
            return "ok"

        result = func()
        self.assertEqual(result, "ok")

    def test02_circuit_protected_failure(self):
        @circuit_protected("decorated_fail", failure_threshold=2, recovery_timeout=10.0)
        def func():
            raise ValueError("失败")

        with self.assertRaises(ValueError):
            func()
        with self.assertRaises(ValueError):
            func()

        # 第三次应该被熔断器拒绝
        with self.assertRaises(CircuitBreakerOpenError):
            func()

    def test03_retryable_success(self):
        calls = [0]

        @retryable(max_retries=3, base_delay=0.001, jitter=False)
        def func():
            calls[0] += 1
            if calls[0] < 2:
                raise ValueError("重试")
            return "成功"

        result = func()
        self.assertEqual(result, "成功")
        self.assertEqual(calls[0], 2)

    def test04_retryable_all_fail(self):
        @retryable(max_retries=2, base_delay=0.001, jitter=False)
        def func():
            raise ValueError("总是失败")

        with self.assertRaises(ValueError):
            func()

    def test05_retryable_with_backoff_string(self):
        @retryable(max_retries=1, backoff="fixed", base_delay=0.001, jitter=False)
        def func():
            return "ok"

        self.assertEqual(func(), "ok")


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        StabilityManager.reset_instance()

    def tearDown(self):
        StabilityManager.reset_instance()

    def test01_get_stability_manager(self):
        mgr = get_stability_manager()
        self.assertIsInstance(mgr, StabilityManager)

    def test02_get_stability_report(self):
        report = get_stability_report()
        self.assertIsInstance(report, str)
        self.assertIn("SEVA 系统稳定性报告", report)

    def test03_get_stability_status(self):
        status = get_stability_status()
        self.assertIn("overall_health", status)

    def test04_reset_stability(self):
        reset_stability()  # 不应抛异常

    def test05_circuit_function(self):
        cb = circuit("test_func", failure_threshold=5)
        self.assertIsInstance(cb, CircuitBreaker)
        self.assertEqual(cb.name, "test_func")

    def test06_health_check_function(self):
        hc = health_check("test_hc", lambda: True)
        self.assertIsInstance(hc, HealthCheck)
        self.assertEqual(hc.name, "test_hc")

    def test07_recover_function(self):
        result = recover("test", lambda: "primary")
        self.assertEqual(result, "primary")

    def test08_recover_with_fallback(self):
        def fail():
            raise ValueError("失败")

        result = recover("test", fail, fallback=lambda: "fallback")
        self.assertEqual(result, "fallback")

    def test09_protected_call_function(self):
        result = protected_call("test", lambda: "ok")
        self.assertEqual(result, "ok")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        StabilityManager.reset_instance()

    def tearDown(self):
        StabilityManager.reset_instance()

    def test01_full_workflow(self):
        """完整工作流：熔断器+重试+健康检查+恢复"""
        mgr = StabilityManager()

        # 注册健康检查
        mgr.register_health_check(HealthCheck("service", lambda: True))

        # 受保护调用
        result = mgr.protected_call("api", lambda: "result")
        self.assertEqual(result, "result")

        # 检查健康
        mgr.run_health_checks()
        self.assertEqual(mgr.get_overall_health(), HealthStatus.HEALTHY)

    def test02_circuit_breaker_with_retry(self):
        """熔断器+重试组合"""
        mgr = StabilityManager()
        calls = [0]

        def flaky():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("不稳定")
            return "成功"

        policy = RetryPolicy(max_retries=5, base_delay=0.001, jitter=False)
        result = mgr.protected_call("flaky_op", flaky, retry_policy=policy)
        self.assertEqual(result, "成功")

    def test03_degraded_mode_tracking(self):
        """降级模式跟踪"""
        mgr = StabilityManager()
        mgr.recovery.register_fallback("op", lambda: "fallback")

        def fail():
            raise ValueError("失败")

        # 第一次失败，使用回退
        result = mgr.recovery.execute_with_recovery("op", fail)
        self.assertEqual(result, "fallback")
        self.assertIn("op", mgr.recovery.get_degraded_operations())

        # 恢复后清除降级标记
        result = mgr.recovery.execute_with_recovery("op", lambda: "primary")
        self.assertEqual(result, "primary")
        self.assertNotIn("op", mgr.recovery.get_degraded_operations())

    def test04_thread_safety(self):
        """线程安全测试"""
        mgr = StabilityManager()
        cb = mgr.get_circuit_breaker("thread_test", failure_threshold=100)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    result = cb.call(lambda: "ok")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)

    def test05_report_contains_all_sections(self):
        """报告包含所有部分"""
        mgr = StabilityManager()
        mgr.get_circuit_breaker("cb1")
        mgr.register_health_check(HealthCheck("hc1", lambda: True))
        mgr.recovery.register_fallback("fb1", lambda: "fb")

        report = mgr.format_report()
        self.assertIn("熔断器", report)
        self.assertIn("健康检查", report)
        self.assertIn("故障恢复", report)
        self.assertIn("重试策略", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
