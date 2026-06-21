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
阶段49: 进度反馈与状态指示
阶段50: 智能建议与推荐系统
阶段51: 可访问性与无障碍支持
阶段52: 用户行为分析与引导优化
阶段53: 系统稳定性与容错增强
阶段54: 插件系统与扩展架构
阶段55: 多Agent协作与编排
阶段56: 实时流处理与事件驱动
阶段57: 性能基准与自动调优
阶段58: LLM客户端与API对接
阶段59: Prompt工程与模板管理
阶段60: 16阶段循环LLM化
阶段61: 记忆与上下文管理
阶段62: 工具调用与Function Calling
阶段63: 智能决策与自主规划
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

    from seva import ProgressBar, Spinner, track_progress, TaskProgress
    with ProgressBar("处理中", total=100) as bar:      # 进度条
        for i in range(100):
            do_something()
            bar.update(1)
    for item in track_progress(items, "处理"):         # 跟踪迭代
        process(item)

    from seva import suggest, recommend, get_suggestions
    suggestions = suggest(agent)                       # 生成智能建议
    for s in suggestions:
        print(s.format())
    print(get_suggestions(agent))                      # 建议报告

    from seva import (AccessibilityConfig, format_accessible,
                      enable_high_contrast, enable_screen_reader)
    config = AccessibilityConfig(high_contrast=True)  # 高对比度
    text = format_accessible("错误: 文件未找到", config)  # 无障碍格式
    enable_screen_reader(True)                         # 启用屏幕阅读器

    from seva import track_event, get_analytics_report, start_session
    start_session()                                    # 开始会话
    track_event("command", name="run", properties={"steps": 100})
    print(get_analytics_report())                      # 行为分析报告

    from seva import (chat, render_prompt, register_prompt, get_prompt,
                      list_prompts, PromptTemplate, PromptSection, FewShotExample)
    response = chat([{"role": "user", "content": "你好"}])    # LLM对话
    template = PromptTemplate(name="my_task",
        sections=[PromptSection.task("处理{{input}}")])
    register_prompt(template)                                # 注册模板
    rendered = render_prompt("my_task", inputs={"input": "数据"})  # 渲染Prompt

    from seva import LLMLoop, get_llm_loop, LLMRole
    loop = get_llm_loop()                                       # LLM循环管理
    result = loop.call_stage("goal_generator", inputs={"input": "data"})
    print(result.parsed_output)                                 # 解析后的JSON

    from seva import get_memory_manager, MemoryType
    memory = get_memory_manager()                               # 记忆管理
    memory.add("重要信息:...", MemoryType.SEMANTIC, importance=0.9)
    context = memory.build_context("查询")                      # 构造LLM上下文

    from seva import (get_tool_registry, get_tool_executor,
                      ToolCallRequest, ToolCallLoop)
    registry = get_tool_registry()                              # 工具注册表
    executor = get_tool_executor()                              # 工具执行器
    req = ToolCallRequest(tool_name="calculator",
                         arguments={"expression": "2+3*4"})
    result = executor.execute(req)                              # 执行工具调用
    loop = ToolCallLoop(registry, executor)                     # LLM驱动工具循环
    out = loop.run("计算 sqrt(16) 的值")                        # 自然语言调用

    from seva import get_autonomous_agent, Goal
    agent = get_autonomous_agent()                              # 自主Agent
    result = agent.run(Goal(description="分析数据并生成报告"))  # 自主完成任务
    print(result.final_output)                                  # 最终输出
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
from .progress import (ProgressBar, Spinner, StatusIndicator, TaskProgress,
                       SPINNER_CHARS, SPINNER_CHARS_ASCII,
                       STATUS_ICONS, STATUS_ICONS_ASCII,
                       progress_bar, spinner, status,
                       track_progress, track_progress_async,
                       format_status, get_progress_report)
from .suggestions import (Suggestion, SuggestionType, SuggestionPriority,
                          UserSkillLevel, ContextAnalyzer, UserProfiler,
                          Recommender, SuggestionEngine,
                          suggest, recommend, recommend_commands,
                          recommend_learning, get_suggestions,
                          get_user_profile, get_user_profile_report,
                          record_command, record_feature, record_error,
                          record_session, reset_user_profile,
                          get_skill_level, analyze_context)
from .accessibility import (AccessibilityConfig, AccessibilityLevel,
                            ColorBlindnessType, ColorBlindnessSupport,
                            ScreenReaderSupport, HighContrastMode,
                            KeyboardNavigation, AccessibleFormatter,
                            AccessibilityManager,
                            format_accessible, get_accessibility_config,
                            set_accessibility_config,
                            update_accessibility_config,
                            reset_accessibility_config,
                            get_accessibility_report,
                            enable_high_contrast, enable_large_text,
                            enable_screen_reader, enable_reduce_motion,
                            set_color_blindness, get_color_blind_palette,
                            get_keybindings, format_keybindings_help)
from .analytics import (BehaviorEvent, EventType, SessionState,
                        BehaviorTracker, BehaviorAnalyzer, UsagePattern,
                        GuidanceOptimizer, AnalyticsReport,
                        track_event, track_command, track_feature,
                        track_error, start_session, end_session,
                        get_analytics_report, get_analytics_summary,
                        optimize_guidance, get_behavior_patterns,
                        get_behavior_anomalies, clear_analytics,
                        get_event_count, get_session_count)
from .stability import (CircuitState, HealthStatus, BackoffStrategy,
                        CircuitBreaker, CircuitBreakerOpenError, HealthCheckError,
                        RetryPolicy, retry_with_policy, HealthCheck, FaultRecovery,
                        StabilityManager, circuit_protected, retryable,
                        get_stability_manager, get_stability_report,
                        get_stability_status, reset_stability, circuit,
                        health_check, recover, protected_call)
from .plugins import (Plugin, PluginInfo, PluginState, PluginContext,
                      HookType, HookEntry, HookRegistry, PluginLoader,
                      PluginManager, register_hook, plugin_hook,
                      get_plugin_manager, register_plugin, unregister_plugin,
                      enable_plugin, disable_plugin, get_plugin, list_plugins,
                      list_enabled_plugins, invoke_hooks, emit_event, on_event,
                      get_plugin_report, get_plugin_status, reset_plugins,
                      discover_plugins)
from .orchestration import (TaskNode, TaskStatus, ExecutionMode, WorkflowStatus,
                            Workflow, WorkflowResult, Orchestrator,
                            get_orchestrator, create_workflow, run_workflow,
                            register_workflow, run_registered_workflow,
                            list_workflows, get_workflow_result,
                            get_orchestration_report, get_orchestration_status,
                            reset_orchestration)
from .streaming import (Event, EventBus, Stream, WindowType, StreamProcessor,
                        ReactiveStream, get_event_bus, set_event_bus,
                        publish, subscribe, unsubscribe, stream,
                        create_stream_processor, create_reactive_stream,
                        get_streaming_stats, reset_streaming)
from .benchmark import (Benchmark, BenchmarkSuite, BenchmarkResult,
                        TuningStrategy, TuningResult, ParamTuner,
                        ABTest, ABTestResult,
                        get_benchmark_suite, benchmark, add_benchmark,
                        run_benchmarks, tune, ab_test,
                        get_benchmark_report, get_benchmark_status,
                        reset_benchmarks)
from .llm import (Role, Message, TokenUsage, ChatRequest, ChatResponse,
                  StreamChunk, LLMError, RateLimitError, QuotaExhaustedError,
                  AuthenticationError, ContentFilterError, ModelOverloadedError,
                  NetworkError, ContextLengthError, RetryConfig, RateLimiter,
                  ResponseCache, LLMClient, LLMSession,
                  get_llm_client, set_llm_client, chat, stream_chat,
                  get_llm_stats, reset_llm_client)
from .prompts import (SectionType, PromptVariable, FewShotExample, PromptSection,
                      PromptFormat, PromptTemplate, RenderedPrompt,
                      PromptFormatter, PromptChain, PromptRegistry,
                      get_prompt_registry, register_prompt, get_prompt,
                      list_prompts, unregister_prompt, render_prompt,
                      delete_prompt, reset_prompts)
from .llm_loop import (LLMStageStatus, LLMRole, LLMStageResult,
                       parse_json_output, LLMLoop,
                       get_llm_loop, set_llm_loop, reset_llm_loop)
from .memory_llm import (MemoryType, MemoryEntry, TokenBudget,
                        ContextCompressor, MemoryManager,
                        get_memory_manager, set_memory_manager,
                        reset_memory_manager)
from .tools import (ParameterType, ToolParameter, ToolDefinition,
                    ToolCallRequest, ToolCallStatus, ToolCallResult,
                    ToolRegistry, ToolExecutor, ToolCallLoop,
                    get_tool_registry, get_tool_executor, reset_tools)
from .planner import (Goal, Task, TaskStatus, Plan, PlanStatus, PlanStrategy,
                      TaskResult, PlanResult, ReflectionResult, Issue,
                      RiskLevel, ActionType, Action,
                      Planner, PlanExecutor, Reflector, DecisionMaker,
                      AutonomousAgent,
                      get_planner, get_plan_executor, get_reflector,
                      get_decision_maker, get_autonomous_agent,
                      reset_planner, reset_reflector,
                      reset_decision_maker, reset_autonomous_agent)

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
    "ProgressBar",
    "Spinner",
    "StatusIndicator",
    "TaskProgress",
    "SPINNER_CHARS",
    "SPINNER_CHARS_ASCII",
    "STATUS_ICONS",
    "STATUS_ICONS_ASCII",
    "progress_bar",
    "spinner",
    "status",
    "track_progress",
    "track_progress_async",
    "format_status",
    "get_progress_report",
    "Suggestion",
    "SuggestionType",
    "SuggestionPriority",
    "UserSkillLevel",
    "ContextAnalyzer",
    "UserProfiler",
    "Recommender",
    "SuggestionEngine",
    "suggest",
    "recommend",
    "recommend_commands",
    "recommend_learning",
    "get_suggestions",
    "get_user_profile",
    "get_user_profile_report",
    "record_command",
    "record_feature",
    "record_error",
    "record_session",
    "reset_user_profile",
    "get_skill_level",
    "analyze_context",
    "AccessibilityConfig",
    "AccessibilityLevel",
    "ColorBlindnessType",
    "ColorBlindnessSupport",
    "ScreenReaderSupport",
    "HighContrastMode",
    "KeyboardNavigation",
    "AccessibleFormatter",
    "AccessibilityManager",
    "format_accessible",
    "get_accessibility_config",
    "set_accessibility_config",
    "update_accessibility_config",
    "reset_accessibility_config",
    "get_accessibility_report",
    "enable_high_contrast",
    "enable_large_text",
    "enable_screen_reader",
    "enable_reduce_motion",
    "set_color_blindness",
    "get_color_blind_palette",
    "get_keybindings",
    "format_keybindings_help",
    "BehaviorEvent",
    "EventType",
    "SessionState",
    "BehaviorTracker",
    "BehaviorAnalyzer",
    "UsagePattern",
    "GuidanceOptimizer",
    "AnalyticsReport",
    "track_event",
    "track_command",
    "track_feature",
    "track_error",
    "start_session",
    "end_session",
    "get_analytics_report",
    "get_analytics_summary",
    "optimize_guidance",
    "get_behavior_patterns",
    "get_behavior_anomalies",
    "clear_analytics",
    "get_event_count",
    "get_session_count",
    "CircuitState",
    "HealthStatus",
    "BackoffStrategy",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "HealthCheckError",
    "RetryPolicy",
    "retry_with_policy",
    "HealthCheck",
    "FaultRecovery",
    "StabilityManager",
    "circuit_protected",
    "retryable",
    "get_stability_manager",
    "get_stability_report",
    "get_stability_status",
    "reset_stability",
    "circuit",
    "health_check",
    "recover",
    "protected_call",
    "Plugin",
    "PluginInfo",
    "PluginState",
    "PluginContext",
    "HookType",
    "HookEntry",
    "HookRegistry",
    "PluginLoader",
    "PluginManager",
    "register_hook",
    "plugin_hook",
    "get_plugin_manager",
    "register_plugin",
    "unregister_plugin",
    "enable_plugin",
    "disable_plugin",
    "get_plugin",
    "list_plugins",
    "list_enabled_plugins",
    "invoke_hooks",
    "emit_event",
    "on_event",
    "get_plugin_report",
    "get_plugin_status",
    "reset_plugins",
    "discover_plugins",
    "TaskNode",
    "TaskStatus",
    "ExecutionMode",
    "WorkflowStatus",
    "Workflow",
    "WorkflowResult",
    "Orchestrator",
    "get_orchestrator",
    "create_workflow",
    "run_workflow",
    "register_workflow",
    "run_registered_workflow",
    "list_workflows",
    "get_workflow_result",
    "get_orchestration_report",
    "get_orchestration_status",
    "reset_orchestration",
    "Event",
    "EventBus",
    "Stream",
    "WindowType",
    "StreamProcessor",
    "ReactiveStream",
    "get_event_bus",
    "set_event_bus",
    "publish",
    "subscribe",
    "unsubscribe",
    "stream",
    "create_stream_processor",
    "create_reactive_stream",
    "get_streaming_stats",
    "reset_streaming",
    "Benchmark",
    "BenchmarkSuite",
    "BenchmarkResult",
    "TuningStrategy",
    "TuningResult",
    "ParamTuner",
    "ABTest",
    "ABTestResult",
    "get_benchmark_suite",
    "benchmark",
    "add_benchmark",
    "run_benchmarks",
    "tune",
    "ab_test",
    "get_benchmark_report",
    "get_benchmark_status",
    "reset_benchmarks",
    "Role",
    "Message",
    "TokenUsage",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "LLMError",
    "RateLimitError",
    "QuotaExhaustedError",
    "AuthenticationError",
    "ContentFilterError",
    "ModelOverloadedError",
    "NetworkError",
    "ContextLengthError",
    "RetryConfig",
    "RateLimiter",
    "ResponseCache",
    "LLMClient",
    "LLMSession",
    "get_llm_client",
    "set_llm_client",
    "chat",
    "stream_chat",
    "get_llm_stats",
    "reset_llm_client",
    "SectionType",
    "PromptVariable",
    "FewShotExample",
    "PromptSection",
    "PromptFormat",
    "PromptTemplate",
    "RenderedPrompt",
    "PromptFormatter",
    "PromptChain",
    "PromptRegistry",
    "get_prompt_registry",
    "register_prompt",
    "get_prompt",
    "list_prompts",
    "unregister_prompt",
    "render_prompt",
    "delete_prompt",
    "reset_prompts",
    "LLMStageStatus",
    "LLMRole",
    "LLMStageResult",
    "parse_json_output",
    "LLMLoop",
    "get_llm_loop",
    "set_llm_loop",
    "reset_llm_loop",
    "MemoryType",
    "MemoryEntry",
    "TokenBudget",
    "ContextCompressor",
    "MemoryManager",
    "get_memory_manager",
    "set_memory_manager",
    "reset_memory_manager",
    "ParameterType",
    "ToolParameter",
    "ToolDefinition",
    "ToolCallRequest",
    "ToolCallStatus",
    "ToolCallResult",
    "ToolRegistry",
    "ToolExecutor",
    "ToolCallLoop",
    "get_tool_registry",
    "get_tool_executor",
    "reset_tools",
    "Goal",
    "Task",
    "TaskStatus",
    "Plan",
    "PlanStatus",
    "PlanStrategy",
    "TaskResult",
    "PlanResult",
    "ReflectionResult",
    "Issue",
    "RiskLevel",
    "ActionType",
    "Action",
    "Planner",
    "PlanExecutor",
    "Reflector",
    "DecisionMaker",
    "AutonomousAgent",
    "get_planner",
    "get_plan_executor",
    "get_reflector",
    "get_decision_maker",
    "get_autonomous_agent",
    "reset_planner",
    "reset_reflector",
    "reset_decision_maker",
    "reset_autonomous_agent",
]

__version__ = "0.63.0"
