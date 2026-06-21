"""
阶段48测试：友好错误处理与恢复
验证 FriendlyError / ErrorRecovery / ErrorSuggester / SafeExecutor
及便捷函数和CLI集成
"""
import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import seva
from seva.error_handling import (FriendlyError, ErrorRecovery,
                                 ErrorSuggester, SafeExecutor,
                                 handle_error, suggest_fix,
                                 safe_run, retry, safe_execute,
                                 format_error, log_error,
                                 get_error_log, clear_error_log,
                                 ERROR_LOG_FILE)


class TestFriendlyError(unittest.TestCase):
    """测试FriendlyError"""

    def test01_init_basic(self):
        """基本初始化"""
        err = FriendlyError("测试错误")
        self.assertEqual(err.message, "测试错误")
        self.assertEqual(err.cause, "")
        self.assertEqual(err.suggestion, "")

    def test02_init_full(self):
        """完整初始化"""
        original = ValueError("原始错误")
        err = FriendlyError(
            message="配置错误",
            cause="配置文件格式错误",
            suggestion="运行 seva config init",
            recovery_action="create_config",
            details={"file": "config.yaml"},
            original_error=original,
        )
        self.assertEqual(err.message, "配置错误")
        self.assertEqual(err.cause, "配置文件格式错误")
        self.assertEqual(err.suggestion, "运行 seva config init")
        self.assertEqual(err.recovery_action, "create_config")
        self.assertEqual(err.details["file"], "config.yaml")
        self.assertIs(err.original_error, original)

    def test03_format(self):
        """format"""
        err = FriendlyError("错误", cause="原因", suggestion="建议")
        text = err.format()
        self.assertIn("错误", text)
        self.assertIn("原因", text)
        self.assertIn("建议", text)

    def test04_to_dict(self):
        """to_dict"""
        err = FriendlyError("msg", cause="c", suggestion="s",
                            recovery_action="a")
        d = err.to_dict()
        self.assertEqual(d["message"], "msg")
        self.assertEqual(d["cause"], "c")
        self.assertEqual(d["suggestion"], "s")
        self.assertEqual(d["recovery_action"], "a")

    def test05_str(self):
        """__str__"""
        err = FriendlyError("测试")
        self.assertIn("测试", str(err))

    def test06_is_exception(self):
        """是Exception子类"""
        err = FriendlyError("test")
        self.assertIsInstance(err, Exception)

    def test07_raise_and_catch(self):
        """抛出和捕获"""
        with self.assertRaises(FriendlyError) as ctx:
            raise FriendlyError("测试错误", cause="原因")
        self.assertEqual(ctx.exception.message, "测试错误")

    def test08_format_with_traceback(self):
        """带堆栈的格式化"""
        try:
            raise ValueError("inner")
        except ValueError as e:
            err = FriendlyError("outer", original_error=e)
            text = err.format(include_traceback=False)
            self.assertIn("outer", text)
            self.assertIn("ValueError", text)


class TestErrorRecovery(unittest.TestCase):
    """测试ErrorRecovery"""

    def test09_retry_success(self):
        """重试成功"""
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        recovery = ErrorRecovery(strategy=ErrorRecovery.RETRY,
                                 max_retries=3, retry_delay=0.01)
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertEqual(result, "success")
        self.assertIsNone(error)

    def test10_retry_failure(self):
        """重试失败"""
        def func():
            raise ValueError("always fail")

        recovery = ErrorRecovery(strategy=ErrorRecovery.RETRY,
                                 max_retries=2, retry_delay=0.01)
        success, result, error = recovery.execute(func)
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIsInstance(error, ValueError)

    def test11_fallback(self):
        """回退策略"""
        def func():
            raise ValueError("fail")

        recovery = ErrorRecovery(strategy=ErrorRecovery.FALLBACK,
                                 fallback_value="default")
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertEqual(result, "default")
        self.assertIsInstance(error, ValueError)

    def test12_fallback_success(self):
        """回退策略成功"""
        def func():
            return "real_value"

        recovery = ErrorRecovery(strategy=ErrorRecovery.FALLBACK,
                                 fallback_value="default")
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertEqual(result, "real_value")
        self.assertIsNone(error)

    def test13_skip(self):
        """跳过策略"""
        def func():
            raise ValueError("fail")

        recovery = ErrorRecovery(strategy=ErrorRecovery.SKIP)
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertIsNone(result)
        self.assertIsInstance(error, ValueError)

    def test14_abort(self):
        """中止策略"""
        def func():
            raise ValueError("fail")

        recovery = ErrorRecovery(strategy=ErrorRecovery.ABORT)
        success, result, error = recovery.execute(func)
        self.assertFalse(success)
        self.assertIsInstance(error, ValueError)

    def test15_abort_success(self):
        """中止策略成功"""
        def func():
            return "ok"

        recovery = ErrorRecovery(strategy=ErrorRecovery.ABORT)
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertEqual(result, "ok")

    def test16_custom_recovery(self):
        """自定义恢复"""
        def func():
            raise ValueError("fail")

        def recovery_fn(e):
            return "recovered"

        recovery = ErrorRecovery(strategy=ErrorRecovery.CUSTOM,
                                 custom_recovery=recovery_fn)
        success, result, error = recovery.execute(func)
        self.assertTrue(success)
        self.assertEqual(result, "recovered")
        self.assertIsInstance(error, ValueError)

    def test17_with_args(self):
        """带参数执行"""
        def func(a, b, c=0):
            return a + b + c

        recovery = ErrorRecovery()
        success, result, error = recovery.execute(func, 1, 2, c=3)
        self.assertTrue(success)
        self.assertEqual(result, 6)


class TestErrorSuggester(unittest.TestCase):
    """测试ErrorSuggester"""

    def test18_file_not_found(self):
        """FileNotFoundError建议"""
        suggester = ErrorSuggester()
        err = FileNotFoundError("test.yaml")
        suggestion = suggester.suggest(err)
        self.assertIn("文件", suggestion["cause"])
        self.assertIn("suggestion", suggestion)

    def test19_value_error(self):
        """ValueError建议"""
        suggester = ErrorSuggester()
        err = ValueError("invalid")
        suggestion = suggester.suggest(err)
        self.assertIn("值", suggestion["cause"])

    def test20_key_error(self):
        """KeyError建议"""
        suggester = ErrorSuggester()
        err = KeyError("missing")
        suggestion = suggester.suggest(err)
        self.assertIn("键", suggestion["cause"])

    def test21_import_error(self):
        """ImportError建议"""
        suggester = ErrorSuggester()
        err = ImportError("nonexistent")
        suggestion = suggester.suggest(err)
        self.assertIn("pip install", suggestion["suggestion"])

    def test22_keyword_match(self):
        """关键词匹配"""
        suggester = ErrorSuggester()
        err = RuntimeError("config file is corrupted")
        suggestion = suggester.suggest(err)
        self.assertIn("配置", suggestion["cause"])

    def test23_unknown_error(self):
        """未知错误"""
        suggester = ErrorSuggester()
        err = RuntimeError("some weird error xyz123")
        suggestion = suggester.suggest(err)
        self.assertIn("RuntimeError", suggestion["cause"])

    def test24_to_friendly_error(self):
        """转为友好错误"""
        suggester = ErrorSuggester()
        err = FileNotFoundError("test.yaml")
        friendly = suggester.to_friendly_error(err)
        self.assertIsInstance(friendly, FriendlyError)
        self.assertEqual(friendly.original_error, err)
        self.assertNotEqual(friendly.cause, "")

    def test25_add_custom_suggestion(self):
        """添加自定义建议"""
        suggester = ErrorSuggester()
        suggester.add_suggestion(
            "CustomError",
            cause="自定义原因",
            suggestion="自定义建议",
            recovery_action="custom_action",
        )
        err = type("CustomError", (Exception,), {})("test")
        suggestion = suggester.suggest(err)
        self.assertEqual(suggestion["cause"], "自定义原因")


class TestSafeExecutor(unittest.TestCase):
    """测试SafeExecutor"""

    def test26_execute_success(self):
        """执行成功"""
        executor = SafeExecutor()
        success, result, error = executor.execute(lambda: 42)
        self.assertTrue(success)
        self.assertEqual(result, 42)
        self.assertIsNone(error)

    def test27_execute_failure(self):
        """执行失败"""
        executor = SafeExecutor(strategy=ErrorRecovery.ABORT)
        success, result, error = executor.execute(
            lambda: (_ for _ in ()).throw(ValueError("fail")))
        self.assertFalse(success)
        self.assertIsInstance(error, FriendlyError)

    def test28_fallback_strategy(self):
        """回退策略"""
        executor = SafeExecutor(strategy=ErrorRecovery.FALLBACK,
                                fallback_value="default")
        success, result, error = executor.execute(
            lambda: 1/0)
        self.assertTrue(success)
        self.assertEqual(result, "default")

    def test29_error_log(self):
        """错误日志"""
        executor = SafeExecutor(log_errors=True)
        executor.execute(lambda: 1/0)
        log = executor.get_error_log()
        self.assertEqual(len(log), 1)

    def test30_clear_log(self):
        """清空日志"""
        executor = SafeExecutor()
        executor.execute(lambda: 1/0)
        executor.clear_error_log()
        self.assertEqual(len(executor.get_error_log()), 0)

    def test31_save_log(self):
        """保存日志"""
        executor = SafeExecutor()
        executor.execute(lambda: 1/0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = executor.save_error_log(os.path.join(tmpdir, "log.json"))
            self.assertTrue(os.path.exists(path))


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def test32_handle_error(self):
        """handle_error"""
        err = ValueError("test")
        friendly = handle_error(err)
        self.assertIsInstance(friendly, FriendlyError)
        self.assertNotEqual(friendly.cause, "")

    def test33_suggest_fix(self):
        """suggest_fix"""
        err = FileNotFoundError("test.yaml")
        suggestion = suggest_fix(err)
        self.assertIn("cause", suggestion)
        self.assertIn("suggestion", suggestion)

    def test34_safe_run_success(self):
        """safe_run成功"""
        result = safe_run(lambda: 42)
        self.assertEqual(result, 42)

    def test35_safe_run_failure(self):
        """safe_run失败"""
        result = safe_run(lambda: 1/0, default="fallback")
        self.assertEqual(result, "fallback")

    def test36_safe_run_no_default(self):
        """safe_run无默认值"""
        result = safe_run(lambda: 1/0)
        self.assertIsNone(result)

    def test37_retry_success(self):
        """retry成功"""
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = retry(func, max_retries=3, delay=0.01)
        self.assertEqual(result, "success")

    def test38_retry_failure(self):
        """retry失败"""
        with self.assertRaises(ValueError):
            retry(lambda: (_ for _ in ()).throw(ValueError("fail")),
                  max_retries=2, delay=0.01)

    def test39_safe_execute(self):
        """safe_execute"""
        success, result, error = safe_execute(lambda: 42)
        self.assertTrue(success)
        self.assertEqual(result, 42)

    def test40_safe_execute_failure(self):
        """safe_execute失败"""
        success, result, error = safe_execute(lambda: 1/0)
        self.assertFalse(success)
        self.assertIsInstance(error, FriendlyError)

    def test41_format_error(self):
        """format_error"""
        err = ValueError("test")
        text = format_error(err)
        self.assertIn("test", text)
        self.assertIn("原因", text)

    def test42_format_friendly_error(self):
        """format_error对FriendlyError"""
        err = FriendlyError("msg", cause="c", suggestion="s")
        text = format_error(err)
        self.assertIn("msg", text)
        self.assertIn("c", text)

    def test43_log_and_get_error(self):
        """记录和获取错误"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "error.json")
            err = ValueError("test error")
            log_error(err, path)
            errors = get_error_log(path)
            self.assertEqual(len(errors), 1)
            self.assertIn("test error", errors[0]["message"])

    def test44_clear_error_log(self):
        """清空错误日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "error.json")
            log_error(ValueError("test"), path)
            self.assertTrue(clear_error_log(path))
            self.assertFalse(os.path.exists(path))

    def test45_clear_nonexistent_log(self):
        """清空不存在的日志"""
        self.assertFalse(clear_error_log("/nonexistent/path.json"))

    def test46_get_empty_log(self):
        """获取空日志"""
        errors = get_error_log("/nonexistent/path.json")
        self.assertEqual(len(errors), 0)


class TestPackageExports(unittest.TestCase):
    """测试包导出"""

    def test47_friendly_error_exported(self):
        self.assertTrue(hasattr(seva, "FriendlyError"))

    def test48_error_recovery_exported(self):
        self.assertTrue(hasattr(seva, "ErrorRecovery"))

    def test49_safe_run_exported(self):
        self.assertTrue(hasattr(seva, "safe_run"))

    def test50_retry_exported(self):
        self.assertTrue(hasattr(seva, "retry"))

    def test51_handle_error_exported(self):
        self.assertTrue(hasattr(seva, "handle_error"))

    def test52_version_updated(self):
        self.assertGreaterEqual(seva.__version__, "0.48.0")

    def test53_all_exports(self):
        for name in ["FriendlyError", "ErrorRecovery",
                     "ErrorSuggester", "SafeExecutor",
                     "handle_error", "suggest_fix",
                     "safe_run", "retry", "safe_execute",
                     "format_error", "log_error",
                     "get_error_log", "clear_error_log"]:
            self.assertIn(name, seva.__all__,
                          f"{name} 未在__all__中")


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def _run_cli(self, *args):
        from seva.cli import main
        old_argv = sys.argv
        sys.argv = ["seva"] + list(args)
        try:
            return main()
        finally:
            sys.argv = old_argv

    def test54_cli_error_default(self):
        """seva error 默认"""
        code = self._run_cli("error")
        self.assertEqual(code, 0)

    def test55_cli_error_test(self):
        """seva error --test"""
        code = self._run_cli("error", "--test")
        self.assertEqual(code, 0)

    def test56_cli_error_suggest(self):
        """seva error --suggest"""
        code = self._run_cli("error", "--suggest",
                             "ValueError('test')")
        self.assertEqual(code, 0)

    def test57_cli_error_log_empty(self):
        """seva error --log 空日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.json")
            code = self._run_cli("error", "--log", path)
            self.assertEqual(code, 0)

    def test58_cli_error_log_with_data(self):
        """seva error --log 有数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_error(ValueError("test error"), path)
            code = self._run_cli("error", "--log", path)
            self.assertEqual(code, 0)

    def test59_cli_error_clear_log(self):
        """seva error --clear-log"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_error(ValueError("test"), path)
            code = self._run_cli("error", "--clear-log", path)
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(path))


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def test60_full_error_recovery_flow(self):
        """完整错误恢复流程"""
        # 1. 定义可能失败的函数
        call_count = [0]

        def risky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("network error")
            return "success"

        # 2. 使用retry重试
        result = retry(risky_function, max_retries=5, delay=0.01)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test61_safe_run_with_fallback(self):
        """safe_run带回退"""
        def failing_func():
            raise FileNotFoundError("config.yaml")

        result = safe_run(failing_func, default={"default": True})
        self.assertEqual(result, {"default": True})

    def test62_error_suggestion_chain(self):
        """错误建议链"""
        errors = [
            FileNotFoundError("test.yaml"),
            PermissionError("/root/file"),
            ValueError("invalid"),
            KeyError("missing"),
            ImportError("nonexistent"),
        ]

        for err in errors:
            suggestion = suggest_fix(err)
            self.assertIn("cause", suggestion)
            self.assertIn("suggestion", suggestion)
            self.assertIn("recovery_action", suggestion)
            # 建议不应为空
            self.assertTrue(suggestion["suggestion"])

    def test63_friendly_error_logging(self):
        """友好错误日志记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "errors.json")

            # 记录多个错误
            log_error(ValueError("error1"), path)
            log_error(FileNotFoundError("file2"), path)
            log_error(KeyError("key3"), path)

            # 获取日志
            errors = get_error_log(path)
            self.assertEqual(len(errors), 3)

            # 验证每个错误都有建议
            for err in errors:
                self.assertIn("message", err)
                self.assertIn("cause", err)
                self.assertIn("suggestion", err)

    def test64_safe_executor_workflow(self):
        """SafeExecutor工作流"""
        executor = SafeExecutor(strategy=ErrorRecovery.FALLBACK,
                                fallback_value="default",
                                log_errors=True)

        # 成功执行
        success, result, error = executor.execute(lambda: "real")
        self.assertTrue(success)
        self.assertEqual(result, "real")
        self.assertEqual(len(executor.get_error_log()), 0)

        # 失败执行
        success, result, error = executor.execute(lambda: 1/0)
        self.assertTrue(success)
        self.assertEqual(result, "default")
        self.assertEqual(len(executor.get_error_log()), 1)

        # 保存日志
        with tempfile.TemporaryDirectory() as tmpdir:
            path = executor.save_error_log(
                os.path.join(tmpdir, "log.json"))
            self.assertTrue(os.path.exists(path))

    def test65_all_error_types_have_suggestions(self):
        """所有预定义错误类型都有建议"""
        suggester = ErrorSuggester()
        for error_type, suggestion in suggester.SUGGESTIONS.items():
            self.assertIn("cause", suggestion)
            self.assertIn("suggestion", suggestion)
            self.assertIn("recovery_action", suggestion)
            self.assertTrue(suggestion["cause"])
            self.assertTrue(suggestion["suggestion"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
