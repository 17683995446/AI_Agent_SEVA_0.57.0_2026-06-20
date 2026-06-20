"""
自驱永续进化预测修正Agent系统 - 核心包
基于设计文档实现，每句关键代码后打印日志用于测试验证

阶段29: 统一API入口
阶段34: 预设场景与开箱即用
阶段35: 配置文件支持
阶段36: 交互式REPL Shell
阶段37: Web Dashboard可视化
阶段38: 结果可视化与报告导出
阶段39: 会话持久化与恢复
阶段40: 打包发布与安装脚本
快速启动:
    from seva import quickstart
    agent, stats = quickstart(steps=100)

    from seva import run_scenario, list_scenarios
    print(list_scenarios())  # 查看可用场景
    result = run_scenario("quick_validation")  # 一键运行场景

    from seva import save_config, load_config, config_from_file
    save_config(agent.config, "my_config.yaml")  # 保存配置
    agent = config_from_file("my_config.yaml")   # 从文件创建Agent

    from seva import run_shell
    run_shell()  # 启动交互式Shell

    from seva import start_dashboard
    start_dashboard(agent, port=8080)  # 启动Web Dashboard

    from seva import visualize, export_report, generate_charts
    visualize(agent, "html", "report.html")  # 导出HTML报告
    generate_charts(agent, "./charts/")       # 生成SVG图表

    from seva import save_session, load_session
    save_session(agent, "my_session.json")  # 保存会话
    agent2 = load_session("my_session.json")  # 恢复会话
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
from .config_io import (save_config, load_config, config_from_file,
                        merge_configs, validate_config_file,
                        ConfigLoader, ConfigWizard)
from .shell import SevaShell, run_shell, execute_command
from .web_dashboard import (WebDashboard, start_dashboard,
                            generate_html_report)
from .visualization import (ResultVisualizer, SVGChartGenerator,
                            visualize, export_report,
                            generate_charts, compare_runs)
from .persistence import (SessionManager, CheckpointHook,
                          save_session, load_session,
                          list_sessions, delete_session,
                          get_session_info, checkpoint)

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
    "save_config",
    "load_config",
    "config_from_file",
    "merge_configs",
    "validate_config_file",
    "ConfigLoader",
    "ConfigWizard",
    "SevaShell",
    "run_shell",
    "execute_command",
    "WebDashboard",
    "start_dashboard",
    "generate_html_report",
    "ResultVisualizer",
    "SVGChartGenerator",
    "visualize",
    "export_report",
    "generate_charts",
    "compare_runs",
    "SessionManager",
    "CheckpointHook",
    "save_session",
    "load_session",
    "list_sessions",
    "delete_session",
    "get_session_info",
    "checkpoint",
]

__version__ = "0.40.0"
