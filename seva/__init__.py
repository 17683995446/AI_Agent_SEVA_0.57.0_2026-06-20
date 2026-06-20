"""
自驱永续进化预测修正Agent系统 - 核心包
基于设计文档实现，每句关键代码后打印日志用于测试验证

阶段29: 统一API入口
快速启动:
    from seva import quickstart
    agent, stats = quickstart(steps=100)
"""
from .config import Config
from .core import SelfEvolvingAgentCore
from .api import (quickstart, run, get_status, create_config,
                  SEVAFactory, PresetConfigs)
from .diagnostics import (SevaError, ErrorCode, ErrorCategory,
                          Diagnoser, diagnose_agent, format_exception,
                          check_common_issues)

__all__ = [
    "Config",
    "SelfEvolvingAgentCore",
    "quickstart",
    "run",
    "get_status",
    "create_config",
    "SEVAFactory",
    "PresetConfigs",
    "SevaError",
    "ErrorCode",
    "ErrorCategory",
    "Diagnoser",
    "diagnose_agent",
    "format_exception",
    "check_common_issues",
]

__version__ = "0.31.0"
