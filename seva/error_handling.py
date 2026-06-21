"""
友好错误处理与恢复（阶段48）
对应第一性原理：容错 = 错误时给出可操作建议，帮助用户恢复

提供能力:
    - FriendlyError: 友好错误（含原因/建议/恢复操作）
    - ErrorRecovery: 错误恢复策略（重试/回退/默认值）
    - ErrorSuggester: 错误建议器（根据错误类型给出建议）
    - SafeExecutor: 安全执行器（自动捕获+恢复）
    - 便捷函数: handle_error / suggest_fix / safe_run / retry

使用示例:
    from seva import FriendlyError, safe_run, retry

    # 安全执行
    result = safe_run(risky_function, default="fallback")

    # 重试
    result = retry(unstable_function, max_retries=3)

    # 友好错误
    raise FriendlyError("配置文件不存在",
                        cause="文件路径错误",
                        suggestion="检查路径或运行 seva config init",
                        recovery_action="create_config")
"""
from __future__ import annotations

import os
import sys
import time
import json
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
ERROR_LOG_FILE = SEVA_HOME / "error_log.json"


# ============================================================================
# FriendlyError - 友好错误
# ============================================================================

class FriendlyError(Exception):
    """
    友好错误，包含原因、建议和恢复操作

    属性:
        message: 错误消息
        cause: 错误原因
        suggestion: 修复建议
        recovery_action: 恢复操作名称
        details: 额外详情
        original_error: 原始错误（如果有）
    """

    def __init__(self, message: str,
                 cause: str = "",
                 suggestion: str = "",
                 recovery_action: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.suggestion = suggestion
        self.recovery_action = recovery_action
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = time.time()

    def format(self, include_traceback: bool = False) -> str:
        """格式化为友好的错误文本"""
        lines = [
            "=" * 60,
            f"错误: {self.message}",
            "=" * 60,
        ]
        if self.cause:
            lines.append(f"\n原因: {self.cause}")
        if self.suggestion:
            lines.append(f"\n建议: {self.suggestion}")
        if self.recovery_action:
            lines.append(f"\n恢复操作: {self.recovery_action}")
        if self.details:
            lines.append("\n详情:")
            for k, v in self.details.items():
                lines.append(f"  {k}: {v}")
        if self.original_error:
            lines.append(f"\n原始错误: {type(self.original_error).__name__}: "
                        f"{self.original_error}")
        if include_traceback and self.original_error:
            lines.append("\n堆栈跟踪:")
            lines.append(traceback.format_exc())
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "message": self.message,
            "cause": self.cause,
            "suggestion": self.suggestion,
            "recovery_action": self.recovery_action,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
            "original_error_type": type(self.original_error).__name__ if self.original_error else None,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return self.format()


# ============================================================================
# ErrorRecovery - 错误恢复策略
# ============================================================================

class ErrorRecovery:
    """
    错误恢复策略

    策略类型:
        - retry: 重试
        - fallback: 回退到默认值
        - skip: 跳过
        - abort: 中止
        - custom: 自定义恢复函数
    """

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CUSTOM = "custom"

    def __init__(self, strategy: str = "abort",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 fallback_value: Any = None,
                 custom_recovery: Optional[Callable[[Exception], Any]] = None) -> None:
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_value = fallback_value
        self.custom_recovery = custom_recovery

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[Exception]]:
        """
        执行函数并应用恢复策略

        Returns:
            (success, result, error)
        """
        if self.strategy == self.RETRY:
            return self._execute_retry(func, *args, **kwargs)
        elif self.strategy == self.FALLBACK:
            return self._execute_fallback(func, *args, **kwargs)
        elif self.strategy == self.SKIP:
            return self._execute_skip(func, *args, **kwargs)
        elif self.strategy == self.CUSTOM:
            return self._execute_custom(func, *args, **kwargs)
        else:  # ABORT
            return self._execute_abort(func, *args, **kwargs)

    def _execute_retry(self, func, *args, **kwargs):
        """重试策略"""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return (True, result, None)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return (False, None, last_error)

    def _execute_fallback(self, func, *args, **kwargs):
        """回退策略"""
        try:
            result = func(*args, **kwargs)
            return (True, result, None)
        except Exception as e:
            return (True, self.fallback_value, e)

    def _execute_skip(self, func, *args, **kwargs):
        """跳过策略"""
        try:
            result = func(*args, **kwargs)
            return (True, result, None)
        except Exception as e:
            return (True, None, e)

    def _execute_abort(self, func, *args, **kwargs):
        """中止策略"""
        try:
            result = func(*args, **kwargs)
            return (True, result, None)
        except Exception as e:
            return (False, None, e)

    def _execute_custom(self, func, *args, **kwargs):
        """自定义策略"""
        try:
            result = func(*args, **kwargs)
            return (True, result, None)
        except Exception as e:
            if self.custom_recovery:
                try:
                    recovered = self.custom_recovery(e)
                    return (True, recovered, e)
                except Exception as recovery_error:
                    return (False, None, recovery_error)
            return (False, None, e)


# ============================================================================
# ErrorSuggester - 错误建议器
# ============================================================================

class ErrorSuggester:
    """
    错误建议器，根据错误类型给出修复建议

    用法:
        suggester = ErrorSuggester()
        suggestion = suggester.suggest(error)
    """

    # 错误类型到建议的映射
    SUGGESTIONS: Dict[str, Dict[str, str]] = {
        "FileNotFoundError": {
            "cause": "文件不存在",
            "suggestion": "检查文件路径是否正确，或使用 seva config init 创建配置文件",
            "recovery_action": "create_file",
        },
        "PermissionError": {
            "cause": "权限不足",
            "suggestion": "检查文件/目录权限，或使用 sudo 运行",
            "recovery_action": "fix_permissions",
        },
        "FileExistsError": {
            "cause": "文件已存在",
            "suggestion": "使用 --force 覆盖，或选择其他文件名",
            "recovery_action": "overwrite_or_rename",
        },
        "ValueError": {
            "cause": "值无效",
            "suggestion": "检查输入值的类型和范围",
            "recovery_action": "fix_input",
        },
        "TypeError": {
            "cause": "类型错误",
            "suggestion": "检查参数类型是否正确",
            "recovery_action": "fix_type",
        },
        "KeyError": {
            "cause": "键不存在",
            "suggestion": "检查键名拼写，或使用 get() 方法避免错误",
            "recovery_action": "use_get_method",
        },
        "IndexError": {
            "cause": "索引越界",
            "suggestion": "检查索引是否在有效范围内",
            "recovery_action": "check_bounds",
        },
        "AttributeError": {
            "cause": "属性不存在",
            "suggestion": "检查对象类型和属性名拼写",
            "recovery_action": "check_attribute",
        },
        "ImportError": {
            "cause": "模块导入失败",
            "suggestion": "运行 pip install <module> 安装缺失的依赖",
            "recovery_action": "install_dependency",
        },
        "ModuleNotFoundError": {
            "cause": "模块未找到",
            "suggestion": "运行 pip install <module> 安装缺失的依赖",
            "recovery_action": "install_dependency",
        },
        "ConnectionError": {
            "cause": "连接失败",
            "suggestion": "检查网络连接和服务地址是否正确",
            "recovery_action": "check_connection",
        },
        "TimeoutError": {
            "cause": "操作超时",
            "suggestion": "增加超时时间或检查系统负载",
            "recovery_action": "increase_timeout",
        },
        "MemoryError": {
            "cause": "内存不足",
            "suggestion": "减少数据量或关闭其他程序释放内存",
            "recovery_action": "reduce_memory_usage",
        },
        "RecursionError": {
            "cause": "递归过深",
            "suggestion": "检查是否有无限递归，或改用迭代实现",
            "recovery_action": "fix_recursion",
        },
        "ZeroDivisionError": {
            "cause": "除以零",
            "suggestion": "在除法前检查除数是否为零",
            "recovery_action": "check_divisor",
        },
        "JSONDecodeError": {
            "cause": "JSON解析失败",
            "suggestion": "检查JSON格式是否正确，使用 jsonlint 验证",
            "recovery_action": "fix_json",
        },
        "YAMLError": {
            "cause": "YAML解析失败",
            "suggestion": "检查YAML缩进和格式是否正确",
            "recovery_action": "fix_yaml",
        },
    }

    # 关键词到建议的映射（用于消息匹配）
    KEYWORD_SUGGESTIONS: Dict[str, Dict[str, str]] = {
        "config": {
            "cause": "配置问题",
            "suggestion": "运行 seva config init 重新生成配置，或 seva config validate 检查配置",
            "recovery_action": "fix_config",
        },
        "permission": {
            "cause": "权限问题",
            "suggestion": "检查文件权限或使用 sudo",
            "recovery_action": "fix_permissions",
        },
        "network": {
            "cause": "网络问题",
            "suggestion": "检查网络连接和代理设置",
            "recovery_action": "check_network",
        },
        "memory": {
            "cause": "内存问题",
            "suggestion": "减少数据量或增加系统内存",
            "recovery_action": "reduce_memory",
        },
        "timeout": {
            "cause": "超时",
            "suggestion": "增加超时时间设置",
            "recovery_action": "increase_timeout",
        },
    }

    def suggest(self, error: Exception) -> Dict[str, str]:
        """
        根据错误给出建议

        Returns:
            {"cause": ..., "suggestion": ..., "recovery_action": ...}
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # 1. 按错误类型匹配
        if error_type in self.SUGGESTIONS:
            return dict(self.SUGGESTIONS[error_type])

        # 2. 按关键词匹配
        for keyword, suggestion in self.KEYWORD_SUGGESTIONS.items():
            if keyword in error_msg:
                return dict(suggestion)

        # 3. 默认建议
        return {
            "cause": f"发生{error_type}错误",
            "suggestion": "查看错误详情，或运行 seva diagnose 进行系统诊断",
            "recovery_action": "diagnose",
        }

    def to_friendly_error(self, error: Exception,
                          message: Optional[str] = None) -> FriendlyError:
        """将普通错误转为友好错误"""
        suggestion = self.suggest(error)
        return FriendlyError(
            message=message or str(error),
            cause=suggestion["cause"],
            suggestion=suggestion["suggestion"],
            recovery_action=suggestion["recovery_action"],
            original_error=error,
        )

    def add_suggestion(self, error_type: str,
                       cause: str, suggestion: str,
                       recovery_action: str) -> None:
        """添加自定义建议"""
        self.SUGGESTIONS[error_type] = {
            "cause": cause,
            "suggestion": suggestion,
            "recovery_action": recovery_action,
        }


# ============================================================================
# SafeExecutor - 安全执行器
# ============================================================================

class SafeExecutor:
    """
    安全执行器，自动捕获错误并应用恢复策略

    用法:
        executor = SafeExecutor(strategy="fallback", fallback_value=0)
        result = executor.execute(risky_function, arg1, arg2)
    """

    def __init__(self, strategy: str = ErrorRecovery.ABORT,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 fallback_value: Any = None,
                 custom_recovery: Optional[Callable] = None,
                 log_errors: bool = True) -> None:
        self.recovery = ErrorRecovery(
            strategy=strategy,
            max_retries=max_retries,
            retry_delay=retry_delay,
            fallback_value=fallback_value,
            custom_recovery=custom_recovery,
        )
        self.log_errors = log_errors
        self._error_log: List[Dict[str, Any]] = []

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[FriendlyError]]:
        """
        安全执行函数

        Returns:
            (success, result, friendly_error)
        """
        success, result, error = self.recovery.execute(func, *args, **kwargs)

        if error is not None:
            suggester = ErrorSuggester()
            friendly = suggester.to_friendly_error(error)
            if self.log_errors:
                self._log_error(friendly)
            if not success:
                return (False, result, friendly)
            return (True, result, friendly)
        return (True, result, None)

    def _log_error(self, error: FriendlyError) -> None:
        """记录错误"""
        self._error_log.append(error.to_dict())

    def get_error_log(self) -> List[Dict[str, Any]]:
        """获取错误日志"""
        return list(self._error_log)

    def clear_error_log(self) -> None:
        """清空错误日志"""
        self._error_log.clear()

    def save_error_log(self, path: Optional[str] = None) -> str:
        """保存错误日志到文件"""
        path = path or str(ERROR_LOG_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._error_log, f, indent=2, ensure_ascii=False,
                      default=str)
        return path


# ============================================================================
# 便捷函数
# ============================================================================

def handle_error(error: Exception,
                 message: Optional[str] = None) -> FriendlyError:
    """将错误转为友好错误"""
    return ErrorSuggester().to_friendly_error(error, message)


def suggest_fix(error: Exception) -> Dict[str, str]:
    """获取错误修复建议"""
    return ErrorSuggester().suggest(error)


def safe_run(func: Callable, *args, default: Any = None,
             **kwargs) -> Any:
    """
    安全执行函数，出错返回默认值

    用法:
        result = safe_run(risky_func, default="fallback")
    """
    recovery = ErrorRecovery(strategy=ErrorRecovery.FALLBACK,
                             fallback_value=default)
    success, result, _ = recovery.execute(func, *args, **kwargs)
    return result


def retry(func: Callable, *args, max_retries: int = 3,
          delay: float = 1.0, **kwargs) -> Any:
    """
    重试执行函数

    用法:
        result = retry(unstable_func, max_retries=5)
    """
    recovery = ErrorRecovery(strategy=ErrorRecovery.RETRY,
                             max_retries=max_retries,
                             retry_delay=delay)
    success, result, error = recovery.execute(func, *args, **kwargs)
    if not success and error:
        raise error
    return result


def safe_execute(func: Callable, *args,
                 strategy: str = ErrorRecovery.ABORT,
                 **kwargs) -> Tuple[bool, Any, Optional[FriendlyError]]:
    """
    安全执行（带恢复策略）

    Returns:
        (success, result, friendly_error)
    """
    executor = SafeExecutor(strategy=strategy)
    return executor.execute(func, *args, **kwargs)


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """格式化错误为友好文本"""
    if isinstance(error, FriendlyError):
        return error.format(include_traceback)
    friendly = handle_error(error)
    return friendly.format(include_traceback)


def log_error(error: Exception, path: Optional[str] = None) -> str:
    """记录错误到文件"""
    path = path or str(ERROR_LOG_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 读取已有日志
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # 添加新错误
    if isinstance(error, FriendlyError):
        existing.append(error.to_dict())
    else:
        friendly = handle_error(error)
        existing.append(friendly.to_dict())

    # 保存
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
    return path


def get_error_log(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取错误日志"""
    path = path or str(ERROR_LOG_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def clear_error_log(path: Optional[str] = None) -> bool:
    """清空错误日志"""
    path = path or str(ERROR_LOG_FILE)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


__all__ = [
    "FriendlyError",
    "ErrorRecovery",
    "ErrorSuggester",
    "SafeExecutor",
    "handle_error",
    "suggest_fix",
    "safe_run",
    "retry",
    "safe_execute",
    "format_error",
    "log_error",
    "get_error_log",
    "clear_error_log",
]
