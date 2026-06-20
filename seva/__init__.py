"""
自驱永续进化预测修正Agent系统 - 核心包
基于设计文档实现，每句关键代码后打印日志用于测试验证

阶段29: 统一API入口
阶段34: 预设场景与开箱即用
快速启动:
    from seva import quickstart
    agent, stats = quickstart(steps=100)

    from seva import run_scenario, list_scenarios
    print(list_scenarios())  # 查看可用场景
    result = run_scenario("quick_validation")  # 一键运行场景
"""
from .config import Config
from .core import SelfEvolvingAgentCore
from .api import (quickstart, run, get_status, create_config,
                  SEVAFactory, PresetConfigs)
from .diagnostics import (SevaError, ErrorCode, ErrorCategory,
                          Diagnoser, diagnose_agent, format_exception,
                          check_common_issues)
from .monitoring import (StructuredLogger, MetricsCollector, Dashboard,
                         monitor, setup_logging, export_metrics)
from .scenarios import (Scenario, PresetScenarios, ScenarioRunner,
                        run_scenario, list_scenarios,
                        list_scenarios_detailed, compare_scenarios,
                        quick_demo, run_all_scenarios,
                        get_scenario_report)

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
    "StructuredLogger",
    "MetricsCollector",
    "Dashboard",
    "monitor",
    "setup_logging",
    "export_metrics",
    "Scenario",
    "PresetScenarios",
    "ScenarioRunner",
    "run_scenario",
    "list_scenarios",
    "list_scenarios_detailed",
    "compare_scenarios",
    "quick_demo",
    "run_all_scenarios",
    "get_scenario_report",
]

__version__ = "0.34.0"
