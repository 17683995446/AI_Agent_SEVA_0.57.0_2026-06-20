"""
友好错误信息与诊断模块（阶段31）
对应第一性原理：易用 = 出错能看懂，知道怎么改

提供：
- ErrorCode: 错误码体系（分类+编号）
- SevaError: 友好异常基类（含错误码、修复建议）
- Diagnoser: 诊断模式（输出系统健康报告）
- 常见问题自动检测

使用示例:
    from seva.diagnostics import ErrorCode, SevaError, Diagnoser

    # 抛出友好错误
    raise SevaError(ErrorCode.CONFIG_INVALID, "max_cycles必须>0",
                    suggestion="请将core.max_cycles设置为正整数")

    # 诊断
    report = Diagnoser(agent).diagnose()
"""
import sys
import os
import time
import json
import logging
import traceback
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("seva.diagnostics")


class ErrorCategory(Enum):
    """错误分类"""
    CONFIG = "CONFIG"           # 配置错误
    RUNTIME = "RUNTIME"         # 运行时错误
    ENVIRONMENT = "ENVIRONMENT" # 环境错误
    MEMORY = "MEMORY"           # 内存错误
    SAFETY = "SAFETY"           # 安全错误
    API = "API"                 # API使用错误
    INTERNAL = "INTERNAL"       # 内部错误


class ErrorCode:
    """
    错误码体系
    格式: SEVA_<CATEGORY>_<NUMBER>
    """
    # 配置错误 (1xx)
    CONFIG_INVALID = ("SEVA_CONFIG_001", ErrorCategory.CONFIG,
                      "配置项无效",
                      "请检查配置项的值是否符合要求，参考seva config show输出")
    CONFIG_MISSING = ("SEVA_CONFIG_002", ErrorCategory.CONFIG,
                      "配置项缺失",
                      "请使用seva config init生成完整配置文件")
    CONFIG_TYPE_ERROR = ("SEVA_CONFIG_003", ErrorCategory.CONFIG,
                         "配置项类型错误",
                         "请检查配置项类型：int/float/bool/str/list/dict")

    # 运行时错误 (2xx)
    RUNTIME_CYCLE_FAILED = ("SEVA_RUNTIME_001", ErrorCategory.RUNTIME,
                            "周期执行失败",
                            "查看日志中的[Core]周期异常信息，检查输入数据是否合法")
    RUNTIME_MODE_FALLBACK = ("SEVA_RUNTIME_002", ErrorCategory.RUNTIME,
                             "系统进入safe_fallback模式",
                             "系统检测到异常已自动恢复，如频繁出现请检查模块健康状态")
    RUNTIME_TIMEOUT = ("SEVA_RUNTIME_003", ErrorCategory.RUNTIME,
                       "周期执行超时",
                       "降低max_cycles或增大cycle_timeout_ms，或使用easy预设")

    # 环境错误 (3xx)
    ENV_NOT_ATTACHED = ("SEVA_ENV_001", ErrorCategory.ENVIRONMENT,
                        "环境未附加",
                        "请使用agent.attach_environment(env)或SEVAFactory.create()")
    ENV_DONE = ("SEVA_ENV_002", ErrorCategory.ENVIRONMENT,
                "环境已结束",
                "请重置环境env.reset()并重新附加agent.attach_environment(env)")
    ENV_INVALID_TYPE = ("SEVA_ENV_003", ErrorCategory.ENVIRONMENT,
                        "无效环境类型",
                        "可用环境类型: blank/mixed/drift/adversarial")

    # 内存错误 (4xx)
    MEMORY_OVERFLOW = ("SEVA_MEM_001", ErrorCategory.MEMORY,
                       "内存溢出",
                       "降低memory.episodic_capacity或增加consolidation_interval")
    MEMORY_CONSOLIDATION_FAILED = ("SEVA_MEM_002", ErrorCategory.MEMORY,
                                   "记忆巩固失败",
                                   "查看日志中的[Memory]异常信息")

    # 安全错误 (5xx)
    SAFETY_CIRCUIT_BREAKER = ("SEVA_SAFETY_001", ErrorCategory.SAFETY,
                              "熔断器触发",
                              "系统检测到过多危险动作，请检查决策模块或降低探索率")
    SAFETY_VETO = ("SEVA_SAFETY_002", ErrorCategory.SAFETY,
                   "动作被安全模块否决",
                   "这是正常的安全机制，系统已执行fallback动作")

    # API使用错误 (6xx)
    API_INVALID_PRESET = ("SEVA_API_001", ErrorCategory.API,
                          "无效预设模板",
                          "可用预设: easy/standard/research/production")
    API_INVALID_PARAM = ("SEVA_API_002", ErrorCategory.API,
                         "API参数无效",
                         "请查看API文档，检查参数类型和取值范围")
    API_NO_ENVIRONMENT = ("SEVA_API_003", ErrorCategory.API,
                          "Agent未附加环境",
                          "请使用SEVAFactory.create()或agent.attach_environment()")

    # 内部错误 (9xx)
    INTERNAL_UNKNOWN = ("SEVA_INTERNAL_001", ErrorCategory.INTERNAL,
                        "未知内部错误",
                        "请查看完整日志并报告问题")

    @classmethod
    def get(cls, code: str) -> tuple:
        """获取错误码信息"""
        for name in dir(cls):
            if name.startswith("_"):
                continue
            val = getattr(cls, name)
            if isinstance(val, tuple) and val[0] == code:
                return val
        return cls.INTERNAL_UNKNOWN


class SevaError(Exception):
    """
    友好异常类
    含错误码、分类、修复建议
    """
    def __init__(self, error_code: tuple, message: str,
                 suggestion: str = None, details: Dict = None):
        self.code = error_code[0]
        self.category = error_code[1]
        self.default_message = error_code[2]
        self.default_suggestion = error_code[3]
        self.message = message or self.default_message
        self.suggestion = suggestion or self.default_suggestion
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "code": self.code,
            "category": self.category.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }

    def format(self) -> str:
        """格式化友好错误信息"""
        lines = [
            f"[{self.code}] {self.message}",
            f"  分类: {self.category.value}",
        ]
        if self.suggestion:
            lines.append(f"  修复建议: {self.suggestion}")
        if self.details:
            lines.append(f"  详情: {self.details}")
        return "\n".join(lines)

    def __str__(self):
        return self.format()


class Diagnoser:
    """
    诊断器：输出系统健康报告
    自动检测常见问题
    """
    def __init__(self, agent=None, config=None):
        self.agent = agent
        self.config = config or (agent.config if agent else None)
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.info: List[Dict] = []

    def diagnose(self) -> Dict[str, Any]:
        """
        执行完整诊断
        返回诊断报告
        """
        self.issues.clear()
        self.warnings.clear()
        self.info.clear()

        # 1. 配置诊断
        self._diagnose_config()

        # 2. 环境诊断
        if self.agent:
            self._diagnose_environment()

            # 3. 模块健康诊断
            self._diagnose_modules()

            # 4. 运行状态诊断
            self._diagnose_runtime()

            # 5. 性能诊断
            self._diagnose_performance()

        report = {
            "timestamp": time.time(),
            "healthy": len(self.issues) == 0,
            "issues_count": len(self.issues),
            "warnings_count": len(self.warnings),
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
        }
        return report

    def _diagnose_config(self):
        """配置诊断"""
        if not self.config:
            self.issues.append({
                "code": "SEVA_CONFIG_002",
                "message": "配置对象缺失",
                "suggestion": "请提供Config对象",
            })
            return

        errors = self.config.validate()
        if errors:
            for e in errors:
                self.issues.append({
                    "code": "SEVA_CONFIG_001",
                    "message": e,
                    "suggestion": "请修正配置项",
                })
        else:
            self.info.append({"message": "配置校验通过"})

        # 检查关键配置项
        max_cycles = self.config.get("core", "max_cycles")
        if max_cycles <= 0:
            self.issues.append({
                "code": "SEVA_CONFIG_001",
                "message": f"max_cycles={max_cycles}必须>0",
                "suggestion": "请将core.max_cycles设置为正整数",
            })

        # 检查内存配置
        epi_cap = self.config.get("memory", "episodic_capacity")
        if epi_cap > 100000:
            self.warnings.append({
                "code": "SEVA_MEM_001",
                "message": f"episodic_capacity={epi_cap}较大，可能导致内存压力",
                "suggestion": "建议降低到50000以下，或增加consolidation_interval",
            })

    def _diagnose_environment(self):
        """环境诊断"""
        if self.agent.environment is None:
            self.warnings.append({
                "code": "SEVA_ENV_001",
                "message": "Agent未附加环境",
                "suggestion": "使用agent.attach_environment(env)附加环境以启用真实闭环",
            })
        else:
            self.info.append({"message": "环境已附加"})

        if getattr(self.agent, "_env_done", False):
            self.warnings.append({
                "code": "SEVA_ENV_002",
                "message": "环境已结束",
                "suggestion": "请重置环境env.reset()并重新附加",
            })

    def _diagnose_modules(self):
        """模块健康诊断"""
        if not self.agent.health_check():
            self.issues.append({
                "code": "SEVA_RUNTIME_001",
                "message": "模块健康检查失败",
                "suggestion": "查看日志确定具体模块问题",
            })
        else:
            self.info.append({"message": "所有模块健康检查通过"})

        # 检查各模块状态
        for name, mod in self.agent.modules.items():
            if not hasattr(mod, "health_check"):
                self.warnings.append({
                    "code": "SEVA_INTERNAL_001",
                    "message": f"模块{name}无health_check方法",
                    "suggestion": "模块应实现health_check()方法",
                })

    def _diagnose_runtime(self):
        """运行状态诊断"""
        mode = self.agent.mode
        if mode == "safe_fallback":
            self.warnings.append({
                "code": "SEVA_RUNTIME_002",
                "message": "系统处于safe_fallback模式",
                "suggestion": "系统将自动尝试恢复，如频繁出现请检查模块状态",
            })
        elif mode == "circuit_breaker":
            self.issues.append({
                "code": "SEVA_SAFETY_001",
                "message": "熔断器已触发",
                "suggestion": "请检查决策模块或降低探索率",
            })

        # 检查周期数
        if self.agent.cycle_count == 0:
            self.info.append({"message": "Agent尚未运行"})
        else:
            self.info.append({
                "message": f"已运行{self.agent.cycle_count}周期"
            })

    def _diagnose_performance(self):
        """性能诊断"""
        cycle_times = self.agent.metrics.get("cycle_times", [])
        if len(cycle_times) >= 10:
            import numpy as np
            avg_time = float(np.mean(cycle_times[-100:])) * 1000
            timeout = self.config.get("core", "cycle_timeout_ms")

            if avg_time > timeout:
                self.warnings.append({
                    "code": "SEVA_RUNTIME_003",
                    "message": f"平均周期时间{avg_time:.2f}ms超过阈值{timeout}ms",
                    "suggestion": "降低max_cycles或使用easy预设，或增大cycle_timeout_ms",
                })
            elif avg_time > timeout * 0.8:
                self.warnings.append({
                    "code": "SEVA_RUNTIME_003",
                    "message": f"平均周期时间{avg_time:.2f}ms接近阈值{timeout}ms",
                    "suggestion": "监控性能，考虑降低负载",
                })
            else:
                self.info.append({
                    "message": f"性能正常，平均周期时间{avg_time:.2f}ms"
                })

    def format_report(self, report: Dict = None) -> str:
        """格式化诊断报告为可读字符串"""
        if report is None:
            report = self.diagnose()

        lines = [
            "="*60,
            "SEVA系统诊断报告",
            "="*60,
        ]

        status = "健康" if report["healthy"] else "有问题"
        lines.append(f"总体状态: {status}")
        lines.append(f"问题数: {report['issues_count']}")
        lines.append(f"警告数: {report['warnings_count']}")

        if report["issues"]:
            lines.append("\n" + "-"*40)
            lines.append("问题 (需修复):")
            lines.append("-"*40)
            for i, issue in enumerate(report["issues"], 1):
                lines.append(f"  {i}. [{issue.get('code', 'UNKNOWN')}] "
                            f"{issue['message']}")
                if issue.get("suggestion"):
                    lines.append(f"     修复: {issue['suggestion']}")

        if report["warnings"]:
            lines.append("\n" + "-"*40)
            lines.append("警告 (建议关注):")
            lines.append("-"*40)
            for i, warn in enumerate(report["warnings"], 1):
                lines.append(f"  {i}. [{warn.get('code', 'WARN')}] "
                            f"{warn['message']}")
                if warn.get("suggestion"):
                    lines.append(f"     建议: {warn['suggestion']}")

        if report["info"]:
            lines.append("\n" + "-"*40)
            lines.append("信息:")
            lines.append("-"*40)
            for info in report["info"]:
                lines.append(f"  • {info['message']}")

        lines.append("="*60)
        return "\n".join(lines)


def diagnose_agent(agent) -> str:
    """
    诊断Agent并返回格式化报告
    便捷函数

    示例:
        from seva.diagnostics import diagnose_agent
        print(diagnose_agent(agent))
    """
    diagnoser = Diagnoser(agent)
    return diagnoser.format_report()


def format_exception(exc: Exception) -> str:
    """
    格式化异常为友好信息
    如果是SevaError，使用其format方法
    否则提供通用友好格式
    """
    if isinstance(exc, SevaError):
        return exc.format()

    # 常见Python异常的友好提示
    if isinstance(exc, ValueError):
        return (f"[SEVA_API_002] 值错误: {exc}\n"
                f"  修复建议: 请检查输入参数的取值范围")
    elif isinstance(exc, TypeError):
        return (f"[SEVA_API_002] 类型错误: {exc}\n"
                f"  修复建议: 请检查输入参数的类型")
    elif isinstance(exc, KeyError):
        return (f"[SEVA_CONFIG_002] 键缺失: {exc}\n"
                f"  修复建议: 请检查配置或字典是否包含所需键")
    elif isinstance(exc, RuntimeError):
        return (f"[SEVA_RUNTIME_001] 运行时错误: {exc}\n"
                f"  修复建议: 查看日志获取详细信息")
    else:
        return (f"[SEVA_INTERNAL_001] {type(exc).__name__}: {exc}\n"
                f"  修复建议: 请查看完整日志并报告问题")


def check_common_issues(agent) -> List[str]:
    """
    检查常见问题，返回问题列表
    """
    issues = []
    if agent is None:
        issues.append("Agent对象为None")
        return issues

    if not agent.is_running:
        issues.append("Agent未初始化，请调用agent.initialize()")

    if agent.environment is None:
        issues.append("Agent未附加环境，请使用agent.attach_environment(env)")

    if agent.cycle_count == 0:
        issues.append("Agent尚未运行任何周期")

    if agent.mode == "safe_fallback":
        issues.append("Agent处于safe_fallback模式，可能有模块异常")

    return issues
