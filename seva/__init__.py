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
阶段41: 首次运行向导与引导系统
阶段42: 智能命令补全与提示
阶段43: 交互式教程系统
阶段44: 性能优化与启动加速
阶段45: 用户偏好与个性化设置
阶段46: 帮助系统与文档集成
阶段47: 国际化与多语言支持
阶段48: 友好错误处理与恢复
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

    from seva import run_wizard, check_first_run
    if check_first_run():
        run_wizard()  # 首次运行向导

    from seva import (PerformanceMonitor, LazyLoader, StartupProfiler,
                      LRUCache, profile_operation, get_performance_report,
                      optimize_agent, benchmark, startup_benchmark)
    with PerformanceMonitor().measure("my_op"):
        do_something()  # 性能监控
    print(get_performance_report())  # 查看性能报告

    from seva import (Preferences, Profile, ProfileManager,
                      get_preference, set_preference, list_preferences,
                      create_profile, switch_profile)
    set_preference("display.theme", "dark")  # 设置偏好
    print(get_preference("display.theme"))    # 获取偏好
    create_profile("work", from_current=True) # 创建配置文件
    switch_profile("work")                    # 切换配置文件

    from seva import show_help, search_help, list_help_topics, help
    print(show_help("getting_started"))  # 显示帮助主题
    print(search_help("配置"))           # 搜索帮助
    print(list_help_topics())            # 列出主题

    from seva import set_language, t, list_languages
    set_language("en_US")           # 切换到英文
    print(t("welcome"))             # 输出英文欢迎语
    set_language("zh_CN")           # 切换回中文
    print(list_languages())         # 列出支持的语言

    from seva import safe_run, retry, handle_error, format_error
    result = safe_run(risky_func, default="fallback")  # 安全执行
    result = retry(unstable_func, max_retries=3)       # 重试执行
    print(format_error(some_error))                    # 友好错误格式
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
from .wizard import (FirstRunWizard, WizardState, GuidedTips,
                     run_wizard, check_first_run, show_tips,
                     reset_wizard)
from .completion import (CommandCompleter, CommandHistory,
                         ShellCompletionGenerator,
                         suggest_command, complete_command,
                         did_you_mean, get_command_help,
                         levenshtein_distance, similarity_score)
from .tutorial import (Tutorial, TutorialStep, TutorialProgress,
                       run_tutorial, list_tutorials,
                       get_tutorial_info, reset_tutorial_progress,
                       get_tutorial_progress)
from .performance import (PerformanceMonitor, LazyLoader, LazyAttribute,
                          StartupProfiler, LRUCache, MemoryTracker,
                          profile_operation, get_performance_report,
                          get_performance_stats, save_performance,
                          reset_performance, optimize_agent,
                          benchmark, benchmark_report,
                          startup_benchmark)
from .preferences import (PreferenceItem, PreferenceSchema, Preferences,
                          Profile, ProfileManager,
                          get_preference, set_preference,
                          reset_preference, reset_all_preferences,
                          list_preferences, list_preference_schema,
                          save_preferences, load_preferences,
                          export_preferences, import_preferences,
                          get_preferences_report,
                          create_profile, delete_profile,
                          switch_profile, list_profiles,
                          get_current_profile, get_profile)
from .help_system import (HelpTopic, HelpSystem, CommandHelp,
                          CommandHelpRegistry, FAQEntry, FAQRegistry,
                          GlossaryTerm, Glossary, CodeExample,
                          ExampleRegistry,
                          show_help, search_help, list_help_topics,
                          list_help_categories, get_help_report,
                          get_command_help, list_commands,
                          search_faq, list_faqs,
                          get_glossary_term, search_glossary,
                          list_glossary,
                          get_example, list_examples, search_examples,
                          help)
from .i18n import (TranslationCatalog, I18n,
                   DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES,
                   t, gettext, set_language, get_language,
                   set_fallback, get_fallback, list_languages,
                   get_i18n_stats, get_i18n_report,
                   register_catalog, get_catalog,
                   list_translation_keys)
from .error_handling import (FriendlyError, ErrorRecovery,
                             ErrorSuggester, SafeExecutor,
                             handle_error, suggest_fix,
                             safe_run, retry, safe_execute,
                             format_error, log_error,
                             get_error_log, clear_error_log)

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
    "FirstRunWizard",
    "WizardState",
    "GuidedTips",
    "run_wizard",
    "check_first_run",
    "show_tips",
    "reset_wizard",
    "CommandCompleter",
    "CommandHistory",
    "ShellCompletionGenerator",
    "suggest_command",
    "complete_command",
    "did_you_mean",
    "get_command_help",
    "levenshtein_distance",
    "similarity_score",
    "Tutorial",
    "TutorialStep",
    "TutorialProgress",
    "run_tutorial",
    "list_tutorials",
    "get_tutorial_info",
    "reset_tutorial_progress",
    "get_tutorial_progress",
    "PerformanceMonitor",
    "LazyLoader",
    "LazyAttribute",
    "StartupProfiler",
    "LRUCache",
    "MemoryTracker",
    "profile_operation",
    "get_performance_report",
    "get_performance_stats",
    "save_performance",
    "reset_performance",
    "optimize_agent",
    "benchmark",
    "benchmark_report",
    "startup_benchmark",
    "PreferenceItem",
    "PreferenceSchema",
    "Preferences",
    "Profile",
    "ProfileManager",
    "get_preference",
    "set_preference",
    "reset_preference",
    "reset_all_preferences",
    "list_preferences",
    "list_preference_schema",
    "save_preferences",
    "load_preferences",
    "export_preferences",
    "import_preferences",
    "get_preferences_report",
    "create_profile",
    "delete_profile",
    "switch_profile",
    "list_profiles",
    "get_current_profile",
    "get_profile",
    "HelpTopic",
    "HelpSystem",
    "CommandHelp",
    "CommandHelpRegistry",
    "FAQEntry",
    "FAQRegistry",
    "GlossaryTerm",
    "Glossary",
    "CodeExample",
    "ExampleRegistry",
    "show_help",
    "search_help",
    "list_help_topics",
    "list_help_categories",
    "get_help_report",
    "get_command_help",
    "list_commands",
    "search_faq",
    "list_faqs",
    "get_glossary_term",
    "search_glossary",
    "list_glossary",
    "get_example",
    "list_examples",
    "search_examples",
    "help",
    "TranslationCatalog",
    "I18n",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "t",
    "gettext",
    "set_language",
    "get_language",
    "set_fallback",
    "get_fallback",
    "list_languages",
    "get_i18n_stats",
    "get_i18n_report",
    "register_catalog",
    "get_catalog",
    "list_translation_keys",
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

__version__ = "0.48.0"
