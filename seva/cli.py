"""
CLI命令行工具（阶段30/34）
对应第一性原理：易用 = 命令行操作，无需写代码

提供命令:
    seva run          启动Agent运行
    seva config init  生成配置文件
    seva config show  显示当前配置
    seva config validate  校验配置
    seva status       查看系统状态
    seva version      版本信息
    seva presets      列出预设模板
    seva demo         运行演示
    seva scenarios    列出/运行预设场景（阶段34）
    seva scenario run 运行指定场景
    seva scenario compare  对比多个场景
    seva diagnose     诊断系统
    seva errors       列出所有错误码
    seva monitor      实时监控
    seva logs         日志管理

使用示例:
    seva run --steps 100 --preset easy
    seva scenarios list
    seva scenario run quick_validation
    seva scenario compare quick_validation standard_run
"""
import sys
import os
import json
import argparse
import logging
import time
import numpy as np
from typing import List, Optional

from . import __version__
from .config import Config
from .api import (quickstart, run, get_status, create_config,
                  SEVAFactory, PresetConfigs)
from .diagnostics import Diagnoser, diagnose_agent, ErrorCode, SevaError
from .monitoring import (StructuredLogger, MetricsCollector, Dashboard,
                         setup_logging, export_metrics)
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


def cmd_run(args):
    """seva run: 启动Agent运行"""
    # 阶段35: 支持--config从文件加载配置
    if hasattr(args, "config") and args.config:
        print(f"从配置文件加载: {args.config}")
        agent = config_from_file(args.config, env_type=args.env_type,
                                 seed=args.seed)
        print(f"Agent已从配置文件创建")
    else:
        print(f"启动SEVA Agent (preset={args.preset}, env={args.env_type}, "
              f"steps={args.steps})")
        agent = SEVAFactory.create(
            preset=args.preset,
            env_type=args.env_type,
            seed=args.seed
        )

    print(f"Agent已创建，开始运行 {args.steps} 步...")
    verbose = args.verbose
    stats = run(agent, steps=args.steps, verbose=verbose)

    # 输出统计
    print("\n" + "="*50)
    print("运行结果")
    print("="*50)
    print(f"  完成步数:       {stats['steps_completed']}")
    print(f"  总奖励:         {stats['total_real_reward']:.4f}")
    print(f"  运行模式:       {stats['mode']}")
    print(f"  平均周期时间:   {stats['avg_cycle_time']*1000:.2f}ms")
    print(f"  平均预测误差:   {stats['avg_prediction_error']:.4f}")
    print(f"  环境结束:       {stats['env_done']}")

    task_stats = stats.get("task_stats", {})
    skill_stats = stats.get("skill_stats", {})
    evo_stats = stats.get("evolution_stats", {})
    print(f"  任务完成数:     {task_stats.get('total_completed', 0)}")
    print(f"  技能总数:       {skill_stats.get('total_skills', 0)}")
    print(f"  进化评估次数:   {evo_stats.get('total_evaluations', 0)}")
    print(f"  当前难度:       {evo_stats.get('current_difficulty', 'EASY')}")

    if args.output:
        # 保存结果到文件
        result = {
            "preset": args.preset,
            "env_type": args.env_type,
            "seed": args.seed,
            "steps": args.steps,
            "stats": {k: v for k, v in stats.items()
                     if isinstance(v, (int, float, str, bool))},
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {args.output}")

    return 0


def cmd_config(args):
    """seva config: 配置管理"""
    if args.config_action == "init":
        return _config_init(args)
    elif args.config_action == "show":
        return _config_show(args)
    elif args.config_action == "validate":
        return _config_validate(args)
    elif args.config_action == "save":
        return _config_save(args)
    elif args.config_action == "load":
        return _config_load(args)
    elif args.config_action == "wizard":
        return _config_wizard(args)
    else:
        print(f"未知config子命令: {args.config_action}")
        return 1


def _config_init(args):
    """生成配置文件"""
    config = create_config(args.preset)
    config_dict = config.to_dict()

    output = args.output or "seva_config.json"
    # 阶段35: 根据扩展名选择格式
    save_config(config, output)

    print(f"配置文件已生成: {output}")
    print(f"  预设: {args.preset}")
    print(f"  Section数: {len(config_dict)}")
    return 0


def _config_save(args):
    """阶段35: 保存配置到YAML/JSON文件"""
    wizard = ConfigWizard()
    config = wizard.generate(
        preset=args.preset,
        env_type=getattr(args, "env_type", "mixed"),
        steps=getattr(args, "steps", 100),
    )
    output = args.output or "seva_config.yaml"
    save_config(config, output)
    print(f"配置已保存到: {output}")
    print(f"  预设: {args.preset}")
    print(f"  格式: {output.split('.')[-1]}")
    return 0


def _config_load(args):
    """阶段35: 从文件加载并显示配置"""
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return 1

    try:
        config = load_config(file_path)
        config_dict = config.to_dict()

        print(f"\n已加载配置: {file_path}")
        print("="*50)

        if args.section:
            if args.section in config_dict:
                print(f"\n[{args.section}]")
                for k, v in config_dict[args.section].items():
                    print(f"  {k} = {v}")
            else:
                print(f"未知section: {args.section}")
                print(f"可用: {list(config_dict.keys())}")
                return 1
        else:
            for section, items in config_dict.items():
                if section == "_meta":
                    continue
                print(f"\n[{section}]")
                for k, v in items.items():
                    print(f"  {k} = {v}")
        return 0
    except Exception as e:
        print(f"加载配置失败: {e}")
        return 1


def _config_wizard(args):
    """阶段35: 配置向导"""
    print("\nSEVA配置向导")
    print("="*50)

    wizard = ConfigWizard(verbose=True)
    config = wizard.generate(
        preset=args.preset,
        env_type=getattr(args, "env_type", "mixed"),
        steps=getattr(args, "steps", 100),
        seed=getattr(args, "seed", 42),
    )

    output = args.output
    if output:
        wizard.save(config, output)
        print(f"\n配置已保存到: {output}")
    else:
        print("\n生成的配置:")
        for section, items in config.to_dict().items():
            print(f"\n[{section}]")
            for k, v in items.items():
                print(f"  {k} = {v}")
    return 0


def _config_show(args):
    """显示当前配置"""
    config = create_config(args.preset)
    config_dict = config.to_dict()

    if args.section:
        # 只显示指定section
        if args.section in config_dict:
            print(f"\n[{args.section}]")
            for k, v in config_dict[args.section].items():
                print(f"  {k} = {v}")
        else:
            print(f"未知section: {args.section}")
            print(f"可用: {list(config_dict.keys())}")
            return 1
    else:
        # 显示所有配置
        print(f"\nSEVA配置 (preset={args.preset})")
        print("="*50)
        for section, items in config_dict.items():
            print(f"\n[{section}]")
            for k, v in items.items():
                print(f"  {k} = {v}")
    return 0


def _config_validate(args):
    """校验配置"""
    config = create_config(args.preset)
    errors = config.validate()
    if errors:
        print("配置校验失败:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    else:
        print(f"配置校验通过 (preset={args.preset})")
        return 0


def cmd_status(args):
    """seva status: 查看系统状态"""
    print(f"\nSEVA系统状态")
    print("="*50)
    print(f"  版本:           {__version__}")
    print(f"  Python:         {sys.version.split()[0]}")

    # 列出可用预设
    presets = PresetConfigs.list_presets()
    print(f"  可用预设:       {', '.join(presets)}")

    # 列出可用环境类型
    env_types = ["blank", "mixed", "drift", "adversarial"]
    print(f"  可用环境类型:   {', '.join(env_types)}")

    # 显示默认配置规模
    print(f"\n预设规模对比:")
    print(f"  {'预设':<12} {'input_dim':<12} {'grid_size':<12} {'max_cycles':<12}")
    print(f"  {'-'*48}")
    for preset in presets:
        cfg = PresetConfigs.get(preset)
        print(f"  {preset:<12} "
              f"{cfg['perception']['input_dim']:<12} "
              f"{cfg['environment']['grid_size']:<12} "
              f"{cfg['core']['max_cycles']:<12}")
    return 0


def cmd_version(args):
    """seva version: 版本信息"""
    print(f"SEVA {__version__}")
    print(f"自驱永续进化预测修正Agent系统")
    print(f"Python {sys.version.split()[0]}")
    return 0


def cmd_presets(args):
    """seva presets: 列出预设模板"""
    presets = PresetConfigs.list_presets()
    print(f"\n可用预设模板:")
    print("="*50)

    descriptions = {
        "easy": "易用模式：小规模、快速、低资源占用（推荐入门）",
        "standard": "标准模式：平衡性能与资源（日常使用推荐）",
        "research": "研究模式：大规模、高精度、长时运行",
        "production": "生产模式：稳定优先、资源受限、监控密集",
    }

    for preset in presets:
        cfg = PresetConfigs.get(preset)
        print(f"\n  {preset}:")
        print(f"    描述: {descriptions.get(preset, '')}")
        print(f"    input_dim: {cfg['perception']['input_dim']}")
        print(f"    grid_size: {cfg['environment']['grid_size']}")
        print(f"    max_cycles: {cfg['core']['max_cycles']}")
    return 0


def cmd_demo(args):
    """seva demo: 运行演示"""
    print(f"\nSEVA演示 (preset={args.preset}, steps={args.steps})")
    print("="*50)
    print("正在启动Agent...")

    agent, stats = quickstart(steps=args.steps, preset=args.preset,
                              env_type=args.env_type, verbose=True)

    print("\n" + "="*50)
    print("演示完成")
    print("="*50)
    status = get_status(agent)
    print(f"  运行步数:       {status['cycle_count']}")
    print(f"  总奖励:         {status['total_reward']}")
    print(f"  平均周期时间:   {status['avg_cycle_time_ms']}ms")
    print(f"  健康状态:       {'健康' if status['healthy'] else '异常'}")
    print(f"  任务完成:       {status['tasks_completed']}")
    print(f"  技能总数:       {status['skills_total']}")
    print(f"  进化评估:       {status['evolution_evaluations']}")
    print(f"  当前难度:       {status['current_difficulty']}")
    return 0


def cmd_scenarios(args):
    """seva scenarios: 列出所有预设场景"""
    print(f"\nSEVA预设场景")
    print("="*50)

    scenarios = list_scenarios_detailed()
    print(f"\n共 {len(scenarios)} 个预设场景:\n")

    for s in scenarios:
        print(f"  {s['name']}")
        print(f"    描述: {s['description']}")
        print(f"    预设: {s['preset']}, 环境: {s['env_type']}, "
              f"步数: {s['steps']}")
        print(f"    标签: {', '.join(s['tags'])}")
        print()

    print("使用方式:")
    print("  seva scenario run <场景名>          运行指定场景")
    print("  seva scenario compare <场景1> <场景2>...  对比多个场景")
    print("  seva scenario run-all               运行所有场景")
    return 0


def cmd_scenario(args):
    """seva scenario: 场景管理"""
    if args.scenario_action == "run":
        return _scenario_run(args)
    elif args.scenario_action == "compare":
        return _scenario_compare(args)
    elif args.scenario_action == "run-all":
        return _scenario_run_all(args)
    elif args.scenario_action == "list":
        return cmd_scenarios(args)
    else:
        print(f"未知scenario子命令: {args.scenario_action}")
        return 1


def _scenario_run(args):
    """运行指定场景"""
    name = args.name
    try:
        scenario = PresetScenarios.get(name)
    except ValueError as e:
        print(f"错误: {e}")
        return 1

    print(f"\n运行场景: {name}")
    print(f"  描述: {scenario.description}")
    print(f"  预设: {scenario.preset}, 环境: {scenario.env_type}, "
          f"步数: {scenario.steps}")
    print()

    result = run_scenario(name, verbose=False)

    print("\n" + "="*50)
    print("运行结果")
    print("="*50)
    print(f"  完成步数:       {result['steps_completed']}/{result['steps']}")
    print(f"  总奖励:         {result['total_reward']:.4f}")
    print(f"  平均周期:       {result['avg_cycle_time']*1000:.2f}ms")
    print(f"  预测误差:       {result['avg_prediction_error']:.4f}")
    print(f"  运行时间:       {result['elapsed_time']:.2f}s")
    print(f"  运行模式:       {result['mode']}")
    print(f"  任务完成:       {result['tasks_completed']}")
    print(f"  技能总数:       {result['skills_total']}")
    print(f"  进化评估:       {result['evolution_evaluations']}")
    print(f"  当前难度:       {result['current_difficulty']}")
    print(f"  通过:           {'是' if result['passed'] else '否'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到 {args.output}")

    return 0 if result["passed"] else 1


def _scenario_compare(args):
    """对比多个场景"""
    names = args.names
    print(f"\n对比场景: {', '.join(names)}")
    print("="*50)

    report = compare_scenarios(names, verbose=False)

    # 打印对比表
    print(f"\n{'场景':<25} {'奖励':<12} {'时间(s)':<10} {'通过':<6}")
    print("-"*55)
    for name, result in report["results"].items():
        print(f"{name:<25} "
              f"{result['total_reward']:<12.4f} "
              f"{result['elapsed_time']:<10.2f} "
              f"{'是' if result['passed'] else '否':<6}")

    # 摘要
    summary = report["summary"]
    print(f"\n摘要:")
    print(f"  总场景数:   {summary['total_scenarios']}")
    print(f"  通过/失败:  {summary['passed']}/{summary['failed']}")
    print(f"  最佳奖励:   {summary['best_reward']:.4f} "
          f"({summary['best_scenario']})")
    print(f"  最差奖励:   {summary['worst_reward']:.4f}")
    print(f"  平均奖励:   {summary['avg_reward']:.4f}")
    print(f"  总时间:     {summary['total_time']:.2f}s")
    print(f"  最快场景:   {summary['fastest_scenario']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n对比报告已保存到 {args.output}")

    return 0


def _scenario_run_all(args):
    """运行所有场景"""
    print(f"\n运行所有预设场景")
    print("="*50)

    results = run_all_scenarios(verbose=False)

    # 打印结果表
    print(f"\n{'场景':<25} {'奖励':<12} {'时间(s)':<10} {'通过':<6}")
    print("-"*55)
    for name, result in results.items():
        print(f"{name:<25} "
              f"{result['total_reward']:<12.4f} "
              f"{result['elapsed_time']:<10.2f} "
              f"{'是' if result['passed'] else '否':<6}")

    # 生成详细报告
    report = get_scenario_report(results)
    print("\n" + report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n所有结果已保存到 {args.output}")

    passed = sum(1 for r in results.values() if r["passed"])
    return 0 if passed == len(results) else 1


def cmd_shell(args):
    """seva shell: 启动交互式Shell"""
    print(f"\n启动SEVA交互式Shell (preset={args.preset}, env={args.env_type})")
    return run_shell(preset=args.preset, env_type=args.env_type,
                     seed=args.seed)


def cmd_exec(args):
    """seva exec: 执行单条Shell命令（非交互式）"""
    command = " ".join(args.command) if args.command else ""
    if not command:
        print("用法: seva exec <command>")
        print("示例: seva exec status")
        print("      seva exec 'run 10'")
        return 1
    output = execute_command(command, preset=args.preset,
                             env_type=args.env_type, seed=args.seed)
    print(output, end="")
    return 0


def cmd_web(args):
    """seva web: 启动Web Dashboard"""
    print(f"\n启动SEVA Web Dashboard")
    print(f"  预设: {args.preset}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  按Ctrl+C停止")
    print("="*50)

    agent = SEVAFactory.create(args.preset)
    # 先运行几步产生数据
    if args.steps > 0:
        print(f"运行 {args.steps} 步产生初始数据...")
        run(agent, steps=args.steps)

    start_dashboard(agent=agent, host=args.host, port=args.port,
                    blocking=True)
    return 0


def cmd_report(args):
    """seva report: 生成HTML报告（阶段38增强：支持多格式）"""
    print(f"\n生成SEVA报告")
    print("="*50)

    agent = SEVAFactory.create(args.preset)
    if args.steps > 0:
        print(f"运行 {args.steps} 步采集数据...")
        run(agent, steps=args.steps)

    fmt = getattr(args, "format", "html") or "html"
    output = args.output

    if fmt == "html" and not output:
        output = "seva_report.html"
    elif fmt == "json" and not output:
        output = "seva_report.json"
    elif fmt == "csv" and not output:
        output = "seva_report.csv"
    elif fmt == "markdown" and not output:
        output = "seva_report.md"
    elif fmt == "txt" and not output:
        output = "seva_report.txt"

    from .visualization import visualize
    content = visualize(agent, fmt, output)
    print(f"\n报告已生成: {output} (格式: {fmt})")
    if fmt in ("html", "markdown", "txt"):
        print(f"可直接查看文件内容")
    return 0


def cmd_export(args):
    """seva export: 导出多格式报告（阶段38）"""
    print(f"\n导出SEVA报告")
    print("="*50)

    agent = SEVAFactory.create(args.preset)
    if args.steps > 0:
        print(f"运行 {args.steps} 步采集数据...")
        run(agent, steps=args.steps)

    from .visualization import export_report
    output = args.output
    if not output:
        ext = {"html": "html", "json": "json", "csv": "csv",
               "markdown": "md", "txt": "txt"}.get(args.format, "txt")
        output = f"seva_export.{ext}"

    content = export_report(agent, args.format, output)
    print(f"\n报告已导出: {output} (格式: {args.format})")
    return 0


def cmd_charts(args):
    """seva charts: 生成SVG图表（阶段38）"""
    print(f"\n生成SVG图表")
    print("="*50)

    agent = SEVAFactory.create(args.preset)
    if args.steps > 0:
        print(f"运行 {args.steps} 步采集数据...")
        run(agent, steps=args.steps)

    from .visualization import generate_charts
    output_dir = args.output_dir or "./seva_charts"
    files = generate_charts(agent, output_dir)
    print(f"\n已生成 {len(files)} 个图表到目录: {output_dir}")
    for f in files:
        print(f"  - {f}")
    return 0


def cmd_compare(args):
    """seva compare: 比较多个运行（阶段38）"""
    print(f"\n比较多个SEVA运行")
    print("="*50)

    presets = args.presets
    if len(presets) < 2:
        print("错误: 至少需要2个预设进行比较")
        return 1

    agents = []
    for preset in presets:
        print(f"创建Agent (preset={preset})...")
        agent = SEVAFactory.create(preset)
        if args.steps > 0:
            run(agent, steps=args.steps)
        agents.append(agent)

    from .visualization import compare_runs
    output = args.output or "seva_comparison.html"
    fmt = args.format
    content = compare_runs(agents, labels=presets,
                           output_file=output, format=fmt)
    print(f"\n比较报告已生成: {output} (格式: {fmt})")
    return 0


def cmd_session(args):
    """seva session: 会话持久化管理（阶段39）"""
    from .persistence import (SessionManager, save_session, load_session,
                              list_sessions, delete_session,
                              get_session_info)

    action = args.action

    if action == "save":
        print(f"\n保存SEVA会话")
        print("=" * 50)
        agent = SEVAFactory.create(args.preset)
        if args.steps > 0:
            print(f"运行 {args.steps} 步采集数据...")
            run(agent, steps=args.steps)
        path = save_session(agent, args.output)
        print(f"\n会话已保存: {path}")
        print(f"周期数: {agent.cycle_count}")
        return 0

    elif action == "load":
        print(f"\n加载SEVA会话")
        print("=" * 50)
        if not args.input:
            print("错误: 需要 --input 指定会话文件")
            return 1
        agent = load_session(args.input)
        print(f"\n会话已加载: {args.input}")
        print(f"周期数: {agent.cycle_count}")
        print(f"运行模式: {agent.mode}")
        return 0

    elif action == "list":
        print(f"\n已保存的会话")
        print("=" * 50)
        directory = args.directory or "."
        sessions = list_sessions(directory)
        if not sessions:
            print(f"目录 {directory} 中无会话文件")
            return 0
        print(f"{'名称':<40} {'周期数':<10} {'大小':<10} {'修改时间'}")
        print("-" * 80)
        for s in sessions:
            ts = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(s["modified"]))
            size_kb = s["size"] / 1024
            print(f"{s['name']:<40} {s['cycle_count']:<10} "
                  f"{size_kb:.1f}KB    {ts}")
        return 0

    elif action == "info":
        print(f"\n会话信息")
        print("=" * 50)
        if not args.input:
            print("错误: 需要 --input 指定会话文件")
            return 1
        info = get_session_info(args.input)
        print(f"文件: {info['file_path']}")
        print(f"大小: {info['file_size'] / 1024:.1f} KB")
        meta = info.get("meta", {})
        print(f"版本: {meta.get('version', 'unknown')}")
        print(f"预设: {meta.get('preset', 'unknown')}")
        ts = meta.get("timestamp", 0)
        if ts:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(ts))
            print(f"保存时间: {ts_str}")
        runtime = info.get("runtime", {})
        print(f"周期数: {runtime.get('cycle_count', 0)}")
        print(f"运行模式: {runtime.get('mode', 'unknown')}")
        return 0

    elif action == "delete":
        print(f"\n删除会话")
        print("=" * 50)
        if not args.input:
            print("错误: 需要 --input 指定会话文件")
            return 1
        if delete_session(args.input):
            print(f"已删除: {args.input}")
            return 0
        else:
            print(f"文件不存在: {args.input}")
            return 1

    elif action == "checkpoint":
        print(f"\n创建检查点")
        print("=" * 50)
        agent = SEVAFactory.create(args.preset)
        if args.steps > 0:
            print(f"运行 {args.steps} 步...")
            run(agent, steps=args.steps)
        from .persistence import checkpoint as ckpt
        directory = args.directory or "./checkpoints"
        path = ckpt(agent, directory=directory, max_keep=args.max_keep)
        print(f"\n检查点已创建: {path}")
        return 0

    else:
        print(f"未知操作: {action}")
        print("可用操作: save, load, list, info, delete, checkpoint")
        return 1


def cmd_wizard(args):
    """seva wizard: 首次运行向导（阶段41）"""
    from .wizard import (FirstRunWizard, WizardState, GuidedTips,
                         run_wizard, reset_wizard)

    if args.reset:
        reset_wizard()
        print("向导状态已重置，下次运行将重新触发首次向导")
        return 0

    if args.status:
        print(f"\nSEVA向导状态")
        print("=" * 50)
        print(f"首次运行: {'是' if WizardState.is_first_run() else '否'}")
        print(f"已完成: {'是' if WizardState.is_completed() else '否'}")
        state = WizardState.load_state()
        if state:
            print(f"预设: {state.get('preset', '未知')}")
            print(f"环境: {state.get('env_type', '未知')}")
            print(f"步数: {state.get('steps', 0)}")
            ts = state.get("completed_at", 0)
            if ts:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(ts))
                print(f"完成时间: {ts_str}")
        return 0

    if args.tips:
        GuidedTips.show_tips()
        return 0

    # 运行向导
    interactive = not args.silent
    if args.silent:
        # 静默模式使用指定参数
        from .wizard import FirstRunWizard
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run_silent(
            preset=args.preset,
            env_type=args.env_type,
            steps=args.steps,
        )
    else:
        result = run_wizard(interactive=True)

    if result.get("cancelled"):
        return 0
    return 0


def cmd_completion(args):
    """seva completion: 生成Shell补全脚本（阶段42）"""
    from .completion import ShellCompletionGenerator, CommandCompleter

    if args.suggest:
        # 拼写纠错建议
        from .completion import did_you_mean
        print(did_you_mean(args.suggest))
        return 0

    if args.list:
        # 列出所有命令
        completer = CommandCompleter()
        commands = completer.get_all_commands()
        print(f"\nSEVA可用命令 ({len(commands)}个):")
        print("=" * 50)
        for cmd in commands:
            info = completer.get_command_info(cmd)
            desc = info["description"] if info else ""
            print(f"  {cmd:<15} {desc}")
        return 0

    if args.help_cmd:
        # 获取命令帮助
        from .completion import get_command_help
        help_text = get_command_help(args.help_cmd)
        if help_text:
            print(help_text)
        else:
            print(f"未知命令: {args.help_cmd}")
            from .completion import did_you_mean
            print(did_you_mean(args.help_cmd))
        return 0

    # 生成补全脚本
    shell = args.shell or "bash"
    gen = ShellCompletionGenerator()

    if shell == "bash":
        script = gen.generate_bash()
    elif shell == "zsh":
        script = gen.generate_zsh()
    else:
        print(f"不支持的shell: {shell}，支持: bash, zsh")
        return 1

    if args.output:
        with open(args.output, "w") as f:
            f.write(script)
        print(f"补全脚本已保存到: {args.output}")
        print(f"安装方法: source {args.output}")
    else:
        print(script)
    return 0


def cmd_tutorial(args):
    """seva tutorial: 交互式教程（阶段43）"""
    from .tutorial import (run_tutorial, list_tutorials,
                           get_tutorial_info, reset_tutorial_progress,
                           get_tutorial_progress, TutorialProgress)

    if args.list:
        tutorials = list_tutorials()
        print(f"\nSEVA可用教程 ({len(tutorials)}个):")
        print("=" * 60)
        for t in tutorials:
            status = "✓ 已完成" if t["completed"] else \
                     f"进行中({t['current_step']}/{t['steps']})"
            print(f"\n  [{t['id']}] {t['title']}")
            print(f"    难度: {t['difficulty']}")
            print(f"    描述: {t['description']}")
            print(f"    步骤: {t['steps']}步")
            print(f"    状态: {status}")
        print()
        return 0

    if args.reset:
        if args.tutorial_id:
            reset_tutorial_progress(args.tutorial_id)
            print(f"已重置教程进度: {args.tutorial_id}")
        else:
            reset_tutorial_progress()
            print("已重置所有教程进度")
        return 0

    if args.progress:
        if not args.tutorial_id:
            print("错误: 需要 --tutorial-id 指定教程")
            return 1
        progress = get_tutorial_progress(args.tutorial_id)
        print(f"\n教程进度: {args.tutorial_id}")
        print("=" * 50)
        print(f"已完成步骤: {len(progress.get('completed_steps', []))}")
        print(f"当前步骤: {progress.get('current_step', 0)}")
        print(f"已完成: {'是' if progress.get('completed') else '否'}")
        return 0

    if not args.tutorial_id:
        print("错误: 需要指定教程ID")
        print("使用 --list 查看可用教程")
        return 1

    info = get_tutorial_info(args.tutorial_id)
    if info is None:
        print(f"未找到教程: {args.tutorial_id}")
        print("使用 --list 查看可用教程")
        return 1

    interactive = not args.silent
    resume = not args.no_resume
    completed = run_tutorial(args.tutorial_id, interactive=interactive,
                             resume=resume)
    return 0 if completed else 0


def cmd_perf(args):
    """seva perf: 性能监控与优化（阶段44）"""
    from .performance import (PerformanceMonitor, get_performance_report,
                              get_performance_stats, save_performance,
                              reset_performance, optimize_agent,
                              startup_benchmark, benchmark_report)

    if args.reset:
        reset_performance()
        print("已重置性能监控数据")
        return 0

    if args.report:
        print(get_performance_report())
        return 0

    if args.save:
        path = save_performance(args.save)
        print(f"性能数据已保存到: {path}")
        return 0

    if args.startup:
        print("测量Agent启动时间...")
        result = startup_benchmark(
            preset=args.preset,
            env_type=args.env_type,
            seed=args.seed,
        )
        print(f"\n启动时间: {result['total_time']*1000:.2f}ms")
        print(f"预设: {result['preset']}, 环境: {result['env_type']}")
        print("\n各阶段耗时:")
        print("-" * 50)
        for step in result["steps"]:
            print(f"  {step['name']:<25} {step['duration']*1000:>8.2f}ms")
        return 0

    if args.optimize:
        print("优化Agent配置...")
        agent = SEVAFactory.create(
            preset=args.preset,
            env_type=args.env_type,
            seed=args.seed,
        )
        result = optimize_agent(agent)
        print(f"\n优化结果:")
        print(f"  应用优化: {result['applied_count']}项")
        for a in result["applied"]:
            print(f"    - {a}")
        print(f"  优化建议: {result['suggestion_count']}条")
        for s in result["suggestions"]:
            print(f"    - {s}")
        return 0

    if args.stats:
        stats = get_performance_stats()
        if not stats:
            print("性能监控: 暂无数据")
            print("提示: 使用 --startup 测量启动时间")
            return 0
        print(f"\n性能统计 ({len(stats)}个操作):")
        print("=" * 60)
        for name, s in stats.items():
            print(f"\n  {name}:")
            print(f"    调用次数: {s['count']}")
            print(f"    总耗时:   {s['total']*1000:.2f}ms")
            print(f"    平均耗时: {s['avg']*1000:.4f}ms")
            print(f"    最大耗时: {s['max']*1000:.4f}ms")
        return 0

    # 默认显示报告
    report = get_performance_report()
    print(report)
    print("\n可用选项:")
    print("  --startup    测量启动时间")
    print("  --optimize   优化Agent配置")
    print("  --stats      查看详细统计")
    print("  --save PATH  保存性能数据")
    print("  --reset      重置监控数据")
    return 0


def cmd_prefs(args):
    """seva prefs: 用户偏好与个性化设置（阶段45）"""
    from .preferences import (Preferences, PreferenceSchema,
                              get_preference, set_preference,
                              reset_preference, reset_all_preferences,
                              list_preferences, list_preference_schema,
                              save_preferences, load_preferences,
                              export_preferences, import_preferences,
                              get_preferences_report,
                              ProfileManager, Profile,
                              create_profile, delete_profile,
                              switch_profile, list_profiles,
                              get_current_profile, get_profile)

    # 配置文件管理
    if args.profile_action:
        pm = ProfileManager()
        if args.profile_action == "list":
            profiles = list_profiles()
            if not profiles:
                print("暂无配置文件")
                print("使用 'seva prefs --profile create <name>' 创建")
                return 0
            print(f"\nSEVA配置文件 ({len(profiles)}个):")
            print("=" * 60)
            current = get_current_profile()
            for p in profiles:
                marker = " *" if p["name"] == current else "  "
                print(f"{marker} {p['name']}")
                print(f"     偏好数: {p['preference_count']}")
                print(f"     创建: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p['created_at']))}")
                print(f"     最后使用: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p['last_used']))}")
            print()
            return 0

        if args.profile_action == "create":
            if not args.profile_name:
                print("错误: 需要指定配置文件名")
                return 1
            profile = create_profile(args.profile_name,
                                     from_current=args.from_current)
            print(f"已创建配置文件: {args.profile_name}")
            print(f"  偏好数: {len(profile.preferences)}")
            return 0

        if args.profile_action == "delete":
            if not args.profile_name:
                print("错误: 需要指定配置文件名")
                return 1
            if delete_profile(args.profile_name):
                print(f"已删除配置文件: {args.profile_name}")
                return 0
            else:
                print(f"删除失败（可能不存在或是当前配置）")
                return 1

        if args.profile_action == "switch":
            if not args.profile_name:
                print("错误: 需要指定配置文件名")
                return 1
            if switch_profile(args.profile_name):
                print(f"已切换到配置文件: {args.profile_name}")
                return 0
            else:
                print(f"切换失败（配置文件不存在）")
                return 1

        if args.profile_action == "show":
            if not args.profile_name:
                current = get_current_profile()
                if current:
                    args.profile_name = current
                else:
                    print("无当前配置文件")
                    return 0
            profile = get_profile(args.profile_name)
            if profile is None:
                print(f"配置文件不存在: {args.profile_name}")
                return 1
            print(f"\n配置文件: {profile.name}")
            print("=" * 50)
            print(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(profile.created_at))}")
            print(f"最后使用: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(profile.last_used))}")
            print(f"偏好数量: {len(profile.preferences)}")
            if profile.preferences:
                print("\n偏好:")
                for k, v in profile.preferences.items():
                    print(f"  {k} = {v}")
            return 0

    # 设置偏好
    if args.set:
        parts = args.set.split("=", 1)
        if len(parts) != 2:
            print("错误: 格式应为 key=value")
            return 1
        key, value = parts
        # 尝试转换值类型
        item = PreferenceSchema.get(key)
        if item is None:
            print(f"错误: 未知偏好键 '{key}'")
            print("使用 'seva prefs --list' 查看可用偏好")
            return 1
        # 类型转换
        try:
            if item.value_type == bool:
                value = value.lower() in ("true", "1", "yes", "on")
            elif item.value_type == int:
                value = int(value)
            elif item.value_type == float:
                value = float(value)
        except ValueError:
            print(f"错误: 值类型不匹配，期望 {item.value_type.__name__}")
            return 1
        if set_preference(key, value):
            print(f"已设置: {key} = {value}")
            return 0
        else:
            print(f"设置失败（值不合法）")
            if item.choices:
                print(f"可选值: {item.choices}")
            return 1

    # 获取偏好
    if args.get:
        value = get_preference(args.get)
        print(f"{args.get} = {value}")
        return 0

    # 重置偏好
    if args.reset:
        if args.reset == "all":
            reset_all_preferences()
            print("已重置所有偏好")
        else:
            if reset_preference(args.reset):
                print(f"已重置: {args.reset}")
            else:
                print(f"重置失败（键不存在）")
                return 1
        return 0

    # 列出偏好
    if args.list:
        if args.category:
            prefs = list_preferences(args.category)
        else:
            prefs = list_preferences()
        print(f"\n用户偏好 ({len(prefs)}项):")
        print("=" * 60)
        current_category = None
        for key in sorted(prefs.keys()):
            item = PreferenceSchema.get(key)
            if item and item.category != current_category:
                current_category = item.category
                print(f"\n[{current_category}]")
            value = prefs[key]
            default_marker = "" if Preferences().has(key) else " (默认)"
            print(f"  {key} = {value}{default_marker}")
        print()
        return 0

    # 列出偏好定义
    if args.schema:
        if args.category:
            schema = list_preference_schema(args.category)
        else:
            schema = list_preference_schema()
        print(f"\n偏好定义 ({len(schema)}项):")
        print("=" * 60)
        for key, info in schema.items():
            print(f"\n  {key}:")
            print(f"    类型: {info['value_type']}")
            print(f"    默认: {info['default']}")
            print(f"    分类: {info['category']}")
            if info["description"]:
                print(f"    描述: {info['description']}")
            if info["choices"]:
                print(f"    可选: {info['choices']}")
            if info["min_value"] is not None:
                print(f"    最小: {info['min_value']}")
            if info["max_value"] is not None:
                print(f"    最大: {info['max_value']}")
        return 0

    # 导出
    if args.export:
        path = export_preferences(args.export)
        print(f"已导出偏好到: {path}")
        return 0

    # 导入
    if args.import_file:
        count = import_preferences(args.import_file)
        print(f"已导入 {count} 项偏好")
        return 0

    # 默认显示报告
    print(get_preferences_report())
    return 0


def cmd_help(args):
    """seva help: 帮助系统（阶段46）"""
    from .help_system import (show_help, search_help, list_help_topics,
                              list_help_categories, get_help_report,
                              get_command_help, list_commands,
                              search_faq, list_faqs,
                              get_glossary_term, search_glossary,
                              list_glossary,
                              get_example, list_examples, search_examples,
                              help as help_func)

    if args.search:
        results = search_help(args.search)
        if not results:
            print(f"未找到匹配 '{args.search}' 的主题")
            print("使用 --list 查看所有主题")
            return 0
        print(f"搜索 '{args.search}' 的结果 ({len(results)}个):")
        print("=" * 60)
        for r in results:
            print(f"  [{r['category']}] {r['topic_id']}: {r['title']}"
                  f" (相关度: {r['score']})")
        print()
        return 0

    if args.list:
        topics = list_help_topics()
        print(f"\n所有帮助主题 ({len(topics)}个):")
        print("=" * 60)
        current_cat = None
        for t in topics:
            if t["category"] != current_cat:
                current_cat = t["category"]
                print(f"\n[{current_cat}]")
            print(f"  {t['topic_id']}: {t['title']}")
        print()
        return 0

    if args.categories:
        cats = list_help_categories()
        print(f"\n帮助分类 ({len(cats)}个):")
        print("=" * 60)
        for cat in cats:
            from .help_system import HelpSystem
            count = len(HelpSystem().get_topics_by_category(cat))
            print(f"  {cat} ({count}个主题)")
        print()
        return 0

    if args.faq:
        faqs = list_faqs()
        if args.search:
            faqs = search_faq(args.search)
        print(f"\n常见问题 ({len(faqs)}个):")
        print("=" * 60)
        for i, f in enumerate(faqs, 1):
            print(f"\n{i}. Q: {f.question}")
            print(f"   A: {f.answer}")
        print()
        return 0

    if args.glossary:
        terms = list_glossary()
        if args.search:
            terms = search_glossary(args.search)
        print(f"\n术语表 ({len(terms)}个):")
        print("=" * 60)
        for t in sorted(terms, key=lambda x: x.term):
            print(f"  {t.term}: {t.definition}")
        print()
        return 0

    if args.examples:
        exs = list_examples()
        if args.search:
            exs = search_examples(args.search)
        print(f"\n代码示例 ({len(exs)}个):")
        print("=" * 60)
        current_cat = None
        for ex in exs:
            if ex.category != current_cat:
                current_cat = ex.category
                print(f"\n[{current_cat}]")
            print(f"  {ex.example_id}: {ex.title}")
        print()
        return 0

    if args.command:
        cmd_help_text = get_command_help(args.command)
        if cmd_help_text:
            print(cmd_help_text)
        else:
            print(f"未找到命令: {args.command}")
            print(f"可用命令: {', '.join(list_commands())}")
            return 1
        return 0

    if args.topic:
        print(show_help(args.topic))
        return 0

    if args.example_id:
        ex = get_example(args.example_id)
        if ex:
            print(ex.format())
        else:
            print(f"未找到示例: {args.example_id}")
            return 1
        return 0

    if args.term:
        definition = get_glossary_term(args.term)
        if definition:
            print(definition)
        else:
            print(f"未找到术语: {args.term}")
            return 1
        return 0

    # 默认显示报告
    print(get_help_report())
    print("\n可用选项:")
    print("  <topic>          显示主题内容")
    print("  --search QUERY   搜索主题")
    print("  --list           列出所有主题")
    print("  --categories     列出分类")
    print("  --faq            常见问题")
    print("  --glossary       术语表")
    print("  --examples       代码示例")
    print("  --command CMD    命令帮助")
    print("  --example-id ID  查看示例")
    print("  --term TERM      查看术语")
    return 0


def cmd_i18n(args):
    """seva i18n: 国际化与多语言支持（阶段47）"""
    from .i18n import (I18n, set_language, get_language, list_languages,
                       get_i18n_stats, get_i18n_report, t,
                       list_translation_keys, get_catalog,
                       SUPPORTED_LANGUAGES)

    if args.set:
        if set_language(args.set):
            print(f"已切换语言: {args.set}")
            print(f"  {t('welcome')}")
            return 0
        else:
            print(f"不支持的语言: {args.set}")
            print(f"支持的语言: {', '.join(list_languages())}")
            return 1

    if args.list:
        langs = list_languages()
        current = get_language()
        print(f"\n支持的语言 ({len(langs)}个):")
        print("=" * 50)
        for lang in langs:
            marker = " *" if lang == current else "  "
            catalog = get_catalog(lang)
            count = len(catalog) if catalog else 0
            print(f"{marker} {lang} ({count}条翻译)")
        print()
        return 0

    if args.keys:
        keys = list_translation_keys(args.keys if args.keys != "all" else None)
        print(f"\n翻译键 ({len(keys)}个):")
        print("=" * 50)
        for k in sorted(keys):
            print(f"  {k} = {t(k)}")
        return 0

    if args.translate:
        text = t(args.translate)
        print(f"{args.translate} = {text}")
        return 0

    if args.report:
        print(get_i18n_report())
        return 0

    if args.stats:
        stats = get_i18n_stats()
        print(f"\n国际化统计:")
        print("=" * 50)
        print(f"当前语言: {stats['current_language']}")
        print(f"回退语言: {stats['fallback_language']}")
        print(f"支持语言: {', '.join(stats['supported_languages'])}")
        print(f"\n翻译目录:")
        for lang, size in stats["catalog_sizes"].items():
            print(f"  {lang}: {size}条")
        return 0

    # 默认显示当前语言和报告
    print(f"当前语言: {get_language()}")
    print(f"回退语言: {I18n().get_fallback()}")
    print(f"\n示例翻译:")
    for key in ["welcome", "cmd.run", "msg.agent_created", "tip.first_run"]:
        print(f"  {key} = {t(key)}")
    print(f"\n使用 --list 查看所有支持的语言")
    print(f"使用 --set <lang> 切换语言")
    print(f"使用 --report 查看完整报告")
    return 0


def cmd_error(args):
    """seva error: 错误处理与恢复（阶段48）"""
    from .error_handling import (FriendlyError, ErrorSuggester,
                                 SafeExecutor, ErrorRecovery,
                                 handle_error, suggest_fix,
                                 safe_run, retry, format_error,
                                 log_error, get_error_log,
                                 clear_error_log)

    if args.suggest:
        # 模拟错误给出建议
        try:
            raise eval(args.suggest)
        except Exception as e:
            suggestion = suggest_fix(e)
            print(f"\n错误: {type(e).__name__}: {e}")
            print("=" * 50)
            print(f"原因: {suggestion['cause']}")
            print(f"建议: {suggestion['suggestion']}")
            print(f"恢复操作: {suggestion['recovery_action']}")
        return 0

    if args.log:
        # 查看错误日志
        errors = get_error_log(args.log)
        if not errors:
            print("错误日志为空")
            return 0
        print(f"\n错误日志 ({len(errors)}条):")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err.get('message', 'N/A')}")
            if err.get("cause"):
                print(f"   原因: {err['cause']}")
            if err.get("suggestion"):
                print(f"   建议: {err['suggestion']}")
        return 0

    if args.clear_log:
        if clear_error_log(args.clear_log):
            print(f"已清空错误日志: {args.clear_log}")
        else:
            print(f"日志文件不存在: {args.clear_log}")
        return 0

    if args.test:
        # 测试错误处理
        print("测试错误处理...")
        test_errors = [
            FileNotFoundError("test_config.yaml"),
            ValueError("invalid value"),
            KeyError("missing_key"),
            ImportError("nonexistent_module"),
        ]
        for err in test_errors:
            print(f"\n{'='*50}")
            print(format_error(err))
        return 0

    # 默认显示帮助
    print("SEVA错误处理工具")
    print("=" * 50)
    print("\n可用选项:")
    print("  --suggest 'Error(...)'  对模拟错误给出建议")
    print("  --log PATH              查看错误日志")
    print("  --clear-log PATH        清空错误日志")
    print("  --test                  测试错误处理")
    print("\nPython API:")
    print("  from seva import safe_run, retry, handle_error")
    print("  result = safe_run(func, default='fallback')")
    print("  result = retry(func, max_retries=3)")
    return 0


def cmd_progress(args):
    """seva progress: 进度反馈与状态指示（阶段49）"""
    from .progress import (ProgressBar, Spinner, StatusIndicator, TaskProgress,
                           progress_bar, spinner, status,
                           track_progress, track_progress_async,
                           format_status, get_progress_report,
                           STATUS_ICONS, STATUS_ICONS_ASCII)
    if args.demo:
        # 演示进度条
        print("演示进度条...")
        bar = ProgressBar("演示", total=20, use_unicode=not args.ascii)
        bar.start()
        for i in range(20):
            time.sleep(0.02)
            bar.update(1)
        bar.finish()
        print("完成")

        # 演示Spinner
        print("\n演示Spinner...")
        with Spinner("处理中...", use_unicode=not args.ascii) as sp:
            time.sleep(0.5)
        print("完成")

        # 演示TaskProgress
        print("\n演示多任务进度...")
        tracker = TaskProgress(use_unicode=not args.ascii)
        t1 = tracker.add_task("下载数据", total=10)
        t2 = tracker.add_task("处理数据", total=10)
        tracker.start_task(t1)
        for i in range(10):
            tracker.update_task(t1)
            time.sleep(0.01)
        tracker.complete_task(t1)
        tracker.start_task(t2)
        for i in range(10):
            tracker.update_task(t2)
            time.sleep(0.01)
        tracker.complete_task(t2)
        print(tracker.format())
        return 0

    if args.status_list:
        # 列出所有状态图标
        use_unicode = not args.ascii
        icons = STATUS_ICONS if use_unicode else STATUS_ICONS_ASCII
        print(f"\n状态图标 ({'Unicode' if use_unicode else 'ASCII'}模式):")
        print("=" * 40)
        for s, icon in icons.items():
            print(f"  {icon:>4}  {s}")
        return 0

    if args.track:
        # 跟踪迭代演示
        n = args.total or 20
        print(f"跟踪 {n} 个项目的进度...")
        items = list(range(n))
        for item in track_progress(items, "处理",
                                   use_unicode=not args.ascii):
            time.sleep(0.02)
        print("完成")
        return 0

    if args.tasks:
        # 多任务演示
        n = args.total or 5
        tracker = TaskProgress(use_unicode=not args.ascii)
        for i in range(3):
            tracker.add_task(f"任务{i+1}", total=n)
        for tid in list(tracker.get_all_tasks().keys()):
            tracker.start_task(tid)
            for _ in range(n):
                tracker.update_task(tid)
                time.sleep(0.01)
            tracker.complete_task(tid)
        print(tracker.format())
        return 0

    # 默认显示帮助
    print("SEVA进度反馈与状态指示")
    print("=" * 50)
    print("\n可用选项:")
    print("  --demo          演示所有进度组件")
    print("  --track         跟踪迭代进度")
    print("  --tasks         多任务进度跟踪")
    print("  --status-list   列出状态图标")
    print("  --total N       项目数量 (默认: 20)")
    print("  --ascii         使用ASCII模式（无Unicode）")
    print("\nPython API:")
    print("  from seva import ProgressBar, Spinner, track_progress")
    print("  with ProgressBar('处理', total=100) as bar:")
    print("      for i in range(100): bar.update(1)")
    return 0


def cmd_suggest(args):
    """seva suggest: 智能建议与推荐系统（阶段50）"""
    from .suggestions import (SuggestionEngine, Suggestion,
                              SuggestionType, SuggestionPriority,
                              UserSkillLevel, ContextAnalyzer,
                              UserProfiler, Recommender,
                              suggest, recommend, recommend_commands,
                              recommend_learning, get_suggestions,
                              get_user_profile, get_user_profile_report,
                              record_command, record_feature,
                              record_session, reset_user_profile,
                              get_skill_level, analyze_context)

    if args.profile:
        # 显示用户画像
        print(get_user_profile_report())
        return 0

    if args.reset_profile:
        reset_user_profile()
        print("已重置用户画像")
        return 0

    if args.recommend_config:
        # 推荐配置
        skill = get_skill_level()
        rec = recommend(scenario=args.recommend_config,
                        skill_level=skill)
        print(f"\n配置推荐 (场景: {args.recommend_config}):")
        print("=" * 50)
        print(f"  推荐预设: {rec['preset']}")
        print(f"  描述: {rec['description']}")
        print(f"  原因: {rec['reason']}")
        return 0

    if args.recommend_commands:
        # 推荐命令
        skill = get_skill_level()
        profile = get_user_profile()
        cmds = recommend_commands(
            skill_level=skill,
            used_commands=profile["commands_used"])
        print(f"\n命令推荐 (技能水平: {skill}):")
        print("=" * 50)
        for c in cmds:
            marker = "✓" if c["already_used"] else " "
            print(f"  {marker} seva {c['command']:<25} {c['description']}")
        return 0

    if args.recommend_learning:
        # 推荐学习资源
        skill = get_skill_level()
        profile = get_user_profile()
        resources = recommend_learning(
            skill_level=skill,
            used_features=profile["features_used"])
        print(f"\n学习资源推荐 (技能水平: {skill}):")
        print("=" * 50)
        for r in resources:
            marker = "✓" if r["already_used"] else " "
            print(f"  {marker} {r['resource_id']:<20} {r['description']}")
        return 0

    if args.analyze:
        # 分析Agent上下文
        agent = SEVAFactory.create(args.preset)
        if args.steps > 0:
            run(agent, steps=args.steps)
        context = analyze_context(agent)
        print(f"\nAgent上下文分析:")
        print("=" * 50)
        for k, v in context.items():
            if k == "issues":
                if v:
                    print(f"  问题:")
                    for issue in v:
                        print(f"    - {issue}")
            else:
                print(f"  {k}: {v}")
        return 0

    # 默认：生成建议
    if args.preset and args.steps > 0:
        agent = SEVAFactory.create(args.preset)
        run(agent, steps=args.steps)
        print(get_suggestions(agent))
    else:
        # 无Agent时基于用户画像生成建议
        engine = SuggestionEngine()
        suggestions = engine.generate()
        if not suggestions:
            print("暂无建议，系统运行良好")
        else:
            print(f"\nSEVA智能建议 ({len(suggestions)}条):")
            print("=" * 60)
            for s in suggestions:
                print()
                print(s.format())
    return 0


def cmd_accessibility(args):
    """seva accessibility: 可访问性与无障碍支持（阶段51）"""
    from .accessibility import (AccessibilityConfig, AccessibilityManager,
                                ColorBlindnessType, ColorBlindnessSupport,
                                ScreenReaderSupport, HighContrastMode,
                                KeyboardNavigation, AccessibleFormatter,
                                format_accessible, get_accessibility_config,
                                set_accessibility_config,
                                update_accessibility_config,
                                reset_accessibility_config,
                                get_accessibility_report,
                                enable_high_contrast, enable_large_text,
                                enable_screen_reader, enable_reduce_motion,
                                set_color_blindness,
                                get_color_blind_palette,
                                get_keybindings, format_keybindings_help)

    if args.report:
        print(get_accessibility_report())
        return 0

    if args.reset:
        reset_accessibility_config()
        print("已重置可访问性配置")
        return 0

    if args.high_contrast is not None:
        enabled = args.high_contrast == "on"
        enable_high_contrast(enabled)
        state = "启用" if enabled else "禁用"
        print(f"已{state}高对比度模式")
        return 0

    if args.large_text is not None:
        enabled = args.large_text == "on"
        scale = args.font_scale or 1.5
        enable_large_text(enabled, scale)
        state = "启用" if enabled else "禁用"
        print(f"已{state}大字体模式 (缩放: {scale}x)")
        return 0

    if args.screen_reader is not None:
        enabled = args.screen_reader == "on"
        enable_screen_reader(enabled)
        state = "启用" if enabled else "禁用"
        print(f"已{state}屏幕阅读器支持")
        return 0

    if args.reduce_motion is not None:
        enabled = args.reduce_motion == "on"
        enable_reduce_motion(enabled)
        state = "启用" if enabled else "禁用"
        print(f"已{state}减少动画模式")
        return 0

    if args.color_blindness:
        if args.color_blindness == "none":
            set_color_blindness(ColorBlindnessType.NONE)
            print("已清除色盲模式")
        else:
            set_color_blindness(args.color_blindness)
            print(f"已设置色盲类型: {args.color_blindness}")
            palette = get_color_blind_palette(args.color_blindness)
            if palette:
                print("颜色替代方案:")
                for semantic, color in palette.items():
                    print(f"  {semantic}: {color}")
        return 0

    if args.keybindings:
        print(format_keybindings_help())
        return 0

    if args.test:
        # 测试无障碍格式化
        print("测试无障碍格式化...")
        config = get_accessibility_config()
        formatter = AccessibleFormatter(config)
        print(f"\n当前配置: {config}")
        print(f"\n格式化示例:")
        print(f"  状态: {formatter.format_status('running', '任务1', '处理中')}")
        print(f"  进度: {formatter.format_progress(50, 100, '下载')}")
        print(f"  错误: {formatter.format_error('文件未找到', '路径错误', '检查路径')}")
        print(f"  成功: {formatter.format_success('操作完成')}")
        print(f"  警告: {formatter.format_warning('资源不足')}")
        return 0

    # 默认显示报告
    print(get_accessibility_report())
    print("\n可用选项:")
    print("  --report               查看报告")
    print("  --high-contrast on/off 启用/禁用高对比度")
    print("  --large-text on/off    启用/禁用大字体")
    print("  --screen-reader on/off 启用/禁用屏幕阅读器")
    print("  --reduce-motion on/off 启用/禁用减少动画")
    print("  --color-blindness TYPE 设置色盲类型")
    print("  --keybindings          查看键盘快捷键")
    print("  --test                 测试格式化")
    print("  --reset                重置配置")
    return 0


def cmd_analytics(args):
    """seva analytics: 用户行为分析与引导优化（阶段52）"""
    from .analytics import (BehaviorTracker, BehaviorAnalyzer,
                            UsagePattern, GuidanceOptimizer,
                            AnalyticsReport, EventType, SessionState,
                            track_event, track_command, track_feature,
                            track_error, start_session, end_session,
                            get_analytics_report, get_analytics_summary,
                            optimize_guidance, get_behavior_patterns,
                            get_behavior_anomalies, clear_analytics,
                            get_event_count, get_session_count)

    if args.clear:
        clear_analytics()
        print("已清空分析数据")
        return 0

    if args.track:
        # 记录测试事件
        print("记录测试事件...")
        sid = start_session()
        track_command("test_cmd", source="cli")
        track_feature("analytics_test")
        track_error("test_error")
        end_session()
        print(f"已记录测试事件，会话ID: {sid}")
        return 0

    if args.summary:
        summary = get_analytics_summary()
        print(f"\nSEVA行为分析摘要:")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    if args.patterns:
        patterns = get_behavior_patterns()
        if not patterns:
            print("暂无识别到的使用模式")
            return 0
        print(f"\n使用模式 ({len(patterns)}个):")
        print("=" * 50)
        for p in patterns:
            print(f"\n  [{p['type']}] {p['description']}")
            print(f"    模式: {p['pattern']}")
            print(f"    次数: {p['count']}")
        return 0

    if args.anomalies:
        anomalies = get_behavior_anomalies()
        if not anomalies:
            print("未检测到异常行为")
            return 0
        print(f"\n异常行为 ({len(anomalies)}个):")
        print("=" * 50)
        for a in anomalies:
            print(f"\n  [{a['severity']}] {a['type']}")
            print(f"    {a['description']}")
        return 0

    if args.optimize:
        opt = optimize_guidance()
        recommendations = opt.get("recommendations", [])
        if not recommendations:
            print("暂无优化建议")
            return 0
        print(f"\n引导优化建议 ({len(recommendations)}条):")
        print("=" * 60)
        for r in recommendations:
            print(f"\n  [{r['priority']}] {r['title']}")
            print(f"    {r['description']}")
        return 0

    if args.stats:
        print(f"\nSEVA分析统计:")
        print("=" * 50)
        print(f"  事件总数: {get_event_count()}")
        print(f"  会话总数: {get_session_count()}")
        return 0

    # 默认显示完整报告
    report = get_analytics_report()
    print(report)
    return 0


def cmd_stability(args):
    """seva stability: 系统稳定性与容错管理（阶段53）"""
    from .stability import (StabilityManager, CircuitBreaker, CircuitState,
                            RetryPolicy, BackoffStrategy, HealthCheck, HealthStatus,
                            FaultRecovery, get_stability_report, reset_stability,
                            circuit_protected, retryable)

    if args.reset:
        reset_stability()
        print("已重置所有稳定性状态")
        return 0

    if args.demo:
        # 演示熔断器
        print("演示熔断器...")
        StabilityManager.reset_instance()
        mgr = StabilityManager()
        cb = mgr.get_circuit_breaker("demo", failure_threshold=3, recovery_timeout=0.5)

        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] <= 5:
                raise ValueError(f"模拟失败 #{call_count[0]}")
            return "成功"

        for i in range(8):
            try:
                result = cb.call(flaky)
                print(f"  调用{i+1}: {result} (状态: {cb.state.value})")
            except Exception as e:
                print(f"  调用{i+1}: {type(e).__name__}: {e} (状态: {cb.state.value})")
            time.sleep(0.1)

        print(f"\n熔断器统计: {cb.get_stats()}")
        return 0

    if args.circuit:
        # 查看熔断器状态
        mgr = StabilityManager()
        names = mgr.list_circuit_breakers()
        if not names:
            print("无注册熔断器")
            return 0
        print("\n熔断器状态:")
        print("-" * 60)
        for name in names:
            cb = mgr.get_circuit_breaker(name)
            stats = cb.get_stats()
            print(f"  [{stats['state'].upper():9s}] {name}")
            print(f"    调用: {stats['total_calls']}, "
                  f"失败: {stats['total_failures']}, "
                  f"拒绝: {stats['total_rejected']}")
            print(f"    失败率: {stats['failure_rate']*100:.1f}%")
        return 0

    if args.health:
        # 运行健康检查
        mgr = StabilityManager()
        results = mgr.run_health_checks()
        if not results:
            print("无注册健康检查")
            return 0
        print("\n健康检查结果:")
        print("-" * 60)
        for name, status in results.items():
            print(f"  [{status.value.upper():9s}] {name}")
        print(f"\n整体健康状态: {mgr.get_overall_health().value}")
        return 0

    if args.retry_policy:
        # 显示/设置重试策略
        mgr = StabilityManager()
        if args.max_retries is not None or args.backoff is not None:
            current = mgr.default_retry_policy
            new_policy = RetryPolicy(
                max_retries=args.max_retries if args.max_retries is not None else current.max_retries,
                backoff=BackoffStrategy(args.backoff) if args.backoff else current.backoff,
                base_delay=current.base_delay,
                max_delay=current.max_delay,
            )
            mgr.default_retry_policy = new_policy
            print(f"已更新重试策略")
        rp = mgr.default_retry_policy
        print(f"\n当前重试策略:")
        print(f"  最大重试次数: {rp.max_retries}")
        print(f"  退避策略: {rp.backoff.value}")
        print(f"  基础延迟: {rp.base_delay}s")
        print(f"  最大延迟: {rp.max_delay}s")
        print(f"  随机抖动: {'是' if rp.jitter else '否'}")
        return 0

    # 默认显示完整报告
    print(get_stability_report())
    return 0


def cmd_plugin(args):
    """seva plugin: 插件系统管理（阶段54）"""
    from .plugins import (PluginManager, Plugin, PluginInfo, PluginState,
                          HookType, get_plugin_report, get_plugin_status,
                          reset_plugins, discover_plugins, list_plugins,
                          list_enabled_plugins, invoke_hooks, emit_event)

    if args.reset:
        reset_plugins()
        print("已重置所有插件状态")
        return 0

    if args.discover:
        print("发现并加载插件...")
        results = discover_plugins()
        if not results:
            print("未发现任何插件")
        else:
            for name, success in results.items():
                status = "成功" if success else "失败"
                print(f"  {name}: {status}")
        return 0

    if args.list:
        plugins = list_plugins()
        if not plugins:
            print("无注册插件")
            return 0
        print("\n已注册插件:")
        print("-" * 60)
        mgr = PluginManager()
        for name in plugins:
            info = mgr.get_plugin_info(name)
            state = info["state"]
            version = info["info"]["version"]
            print(f"  [{state.upper():8s}] {name} v{version}")
        return 0

    if args.enable:
        if PluginManager().enable(args.enable):
            print(f"插件 '{args.enable}' 已启用")
            return 0
        else:
            print(f"插件 '{args.enable}' 启用失败")
            return 1

    if args.disable:
        if PluginManager().disable(args.disable):
            print(f"插件 '{args.disable}' 已禁用")
            return 0
        else:
            print(f"插件 '{args.disable}' 禁用失败")
            return 1

    if args.hooks:
        status = get_plugin_status()
        hooks = status["hooks"]
        print("\n钩子统计:")
        print("-" * 60)
        print(f"  钩子总数: {hooks['total_hooks']}")
        print(f"  事件钩子总数: {hooks['total_event_hooks']}")
        if hooks["hook_types"]:
            print("  按类型:")
            for ht, count in hooks["hook_types"].items():
                print(f"    {ht}: {count}")
        if hooks["events"]:
            print("  按事件:")
            for event, count in hooks["events"].items():
                print(f"    {event}: {count}")
        return 0

    if args.demo:
        # 演示插件系统
        print("演示插件系统...")
        PluginManager.reset_instance()
        mgr = PluginManager()

        class DemoPlugin(Plugin):
            info = PluginInfo(name="demo", version="1.0.0",
                              description="演示插件")

            def __init__(self):
                super().__init__()
                self.events = []

            def on_load(self, context):
                context.log("插件加载")
                self.events.append("loaded")

            def on_enable(self):
                self.events.append("enabled")

            def on_disable(self):
                self.events.append("disabled")

            def on_unload(self):
                self.events.append("unloaded")

        plugin = DemoPlugin()
        print(f"注册插件: {mgr.register(plugin)}")
        print(f"启用插件: {mgr.enable('demo')}")
        print(f"插件状态: {plugin.state.value}")
        print(f"事件记录: {plugin.events}")
        print(f"禁用插件: {mgr.disable('demo')}")
        print(f"注销插件: {mgr.unregister('demo')}")
        return 0

    # 默认显示完整报告
    print(get_plugin_report())
    return 0


def cmd_orchestrate(args):
    """seva orchestrate: 多Agent协作与编排（阶段55）"""
    from .orchestration import (Orchestrator, Workflow, TaskNode, ExecutionMode,
                                WorkflowStatus, TaskStatus, create_workflow,
                                run_workflow, register_workflow,
                                run_registered_workflow, list_workflows,
                                get_workflow_result, get_orchestration_report,
                                get_orchestration_status, reset_orchestration)

    if args.reset:
        reset_orchestration()
        print("已重置所有编排状态")
        return 0

    if args.demo:
        print("演示工作流编排...")
        Orchestrator.reset_instance()

        # 创建数据处理工作流
        wf = create_workflow("demo_pipeline", mode=ExecutionMode.PARALLEL)

        def fetch_data(context=None):
            return [1, 2, 3, 4, 5]

        def process_data(context=None):
            return [x * 2 for x in context.get("fetch", [])]

        def save_data(context=None):
            data = context.get("process", [])
            return f"保存了{len(data)}条数据"

        wf.add_task(TaskNode("fetch", fetch_data))
        wf.add_task(TaskNode("process", process_data), depends_on=["fetch"])
        wf.add_task(TaskNode("save", save_data), depends_on=["process"])

        result = run_workflow(wf)
        print(f"\n{result.summary()}")
        for name, task in result.tasks.items():
            print(f"  {name}: {task.status.value} -> {task.result}")
        return 0

    if args.list:
        workflows = list_workflows()
        if not workflows:
            print("无注册工作流")
            return 0
        print("\n已注册工作流:")
        print("-" * 60)
        for name in workflows:
            wf = Orchestrator().get_workflow(name)
            info = wf.get_info()
            print(f"  [{info['status'].upper():9s}] {name} "
                  f"({info['task_count']}个任务, {info['mode']})")
        return 0

    if args.run:
        result = run_registered_workflow(args.run)
        if result is None:
            print(f"工作流 '{args.run}' 不存在")
            return 1
        print(f"\n{result.summary()}")
        for name, task in result.tasks.items():
            status = task.status.value
            print(f"  [{status:8s}] {name}")
            if task.error:
                print(f"            错误: {task.error}")
        return 0 if result.is_success else 1

    # 默认显示完整报告
    print(get_orchestration_report())
    return 0


def cmd_stream(args):
    """seva stream: 实时流处理与事件驱动（阶段56）"""
    from .streaming import (EventBus, Stream, StreamProcessor, WindowType,
                            ReactiveStream, Event, get_event_bus, publish,
                            subscribe, get_streaming_stats, reset_streaming,
                            stream as create_stream)

    if args.reset:
        reset_streaming()
        print("已重置流处理状态")
        return 0

    if args.demo:
        print("演示流处理...")
        reset_streaming()

        # 1. 事件总线演示
        print("\n1. 事件总线演示:")
        bus = get_event_bus()
        events = []
        unsubscribe = subscribe("user.*", lambda e: events.append(e.data))
        publish("user.login", {"user": "alice"})
        publish("user.logout", {"user": "bob"})
        publish("system.start", {"time": "now"})  # 不匹配
        print(f"  收到事件: {events}")
        unsubscribe()

        # 2. 数据流演示
        print("\n2. 数据流演示:")
        result = create_stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).map(
            lambda x: x * 2).filter(lambda x: x > 10).to_list()
        print(f"  map(x*2).filter(>10): {result}")

        total = create_stream([1, 2, 3, 4, 5]).reduce(lambda a, b: a + b, 0)
        print(f"  reduce(sum): {total}")

        # 3. 窗口处理演示
        print("\n3. 窗口处理演示:")
        processor = StreamProcessor(window_size=3, window_type=WindowType.TUMBLING)
        windows = processor.process([1, 2, 3, 4, 5, 6], lambda w: sum(w))
        print(f"  翻滚窗口(3): {windows}")

        processor2 = StreamProcessor(window_size=3, window_type=WindowType.SLIDING,
                                     slide_size=2)
        windows2 = processor2.process([1, 2, 3, 4, 5], lambda w: sum(w))
        print(f"  滑动窗口(3,slide=2): {windows2}")

        print(f"\n统计: {get_streaming_stats()}")
        return 0

    if args.stats:
        stats = get_streaming_stats()
        print("\n流处理统计:")
        print("-" * 60)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.publish:
        publish(args.publish, data=args.data)
        print(f"已发布事件: {args.publish}")
        return 0

    # 默认显示统计
    stats = get_streaming_stats()
    print("\n流处理统计:")
    print("-" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


def cmd_benchmark(args):
    """seva benchmark: 性能基准与自动调优（阶段57）"""
    from .benchmark import (Benchmark, BenchmarkSuite, BenchmarkResult,
                            TuningStrategy, ParamTuner, ABTest,
                            get_benchmark_suite, benchmark, add_benchmark,
                            run_benchmarks, tune, ab_test,
                            get_benchmark_report, get_benchmark_status,
                            reset_benchmarks)

    if args.reset:
        reset_benchmarks()
        print("已重置基准测试状态")
        return 0

    if args.demo:
        print("演示性能基准测试...")
        reset_benchmarks()

        # 1. 基准测试
        print("\n1. 基准测试:")
        result = benchmark("list_sort", lambda: sorted([3, 1, 4, 1, 5, 9, 2, 6] * 100),
                           iterations=5)
        print(f"  {result.summary()}")

        # 2. A/B测试
        print("\n2. A/B测试:")
        ab_result = ab_test(
            "sort_vs_sorted",
            func_a=lambda: sorted([3, 1, 4, 1, 5] * 1000),
            func_b=lambda: list(reversed(sorted([3, 1, 4, 1, 5] * 1000))),
            iterations=5,
        )
        print(f"  {ab_result.summary()}")

        # 3. 参数调优
        print("\n3. 参数调优:")
        def objective(x, y):
            return -(x**2 + y**2)  # 最大化（越接近0越好）

        tune_result = tune(objective, {"x": (0, 5), "y": (0, 5)},
                           strategy="grid", step=1)
        print(f"  {tune_result.summary()}")

        return 0

    if args.tune:
        # 演示调优
        print("参数调优演示...")
        def objective(x, y):
            return -(x**2 + y**2)

        result = tune(objective, {"x": (0, 10), "y": (0, 10)},
                      strategy=args.tune, step=2 if args.tune == "grid" else 1,
                      n_trials=20)
        print(f"\n{result.summary()}")
        print(f"\n所有结果:")
        for r in result.all_results[:5]:
            print(f"  params={r['params']}, score={r['score']:.4f}")
        if len(result.all_results) > 5:
            print(f"  ... (共{len(result.all_results)}个结果)")
        return 0

    if args.ab:
        # A/B测试
        print(f"A/B测试: {args.ab}")
        result = ab_test(
            args.ab,
            func_a=lambda: sorted(list(range(1000, 0, -1))),
            func_b=lambda: list(reversed(sorted(list(range(1000, 0, -1))))),
            iterations=10,
        )
        print(f"\n{result.summary()}")
        return 0

    if args.list:
        suite = get_benchmark_suite()
        tests = suite.list_tests()
        if not tests:
            print("无注册基准测试")
            return 0
        print("\n已注册基准测试:")
        for name in tests:
            print(f"  - {name}")
        return 0

    if args.run:
        # 运行全局套件
        print("运行所有基准测试...")
        results = run_benchmarks()
        if not results:
            print("无注册测试")
            return 0
        suite = get_benchmark_suite()
        print(suite.compare(results))
        return 0

    # 默认显示报告
    print(get_benchmark_report())
    return 0


def cmd_llm(args):
    """seva llm: LLM客户端与API管理（阶段58）"""
    import os
    from .llm import (LLMClient, Message, Role, ChatRequest, get_llm_client,
                      get_llm_stats, reset_llm_client)

    if args.reset:
        reset_llm_client()
        print("已重置LLM客户端")
        return 0

    if args.test:
        # 测试连接
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            print("未设置API Key (环境变量 SEVA_LLM_API_KEY)")
            return 1
        print(f"使用API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"基础URL: {os.environ.get('SEVA_LLM_BASE_URL', '默认')}")
        print(f"模型: {os.environ.get('SEVA_LLM_MODEL', '默认')}")
        print("测试LLM连接...")

        try:
            client = LLMClient(api_key=api_key)
            response = client.chat([
                Message.system("你是一个测试助手。"),
                Message.user("请用一句话介绍自己。"),
            ], max_tokens=100)
            print(f"\n响应成功!")
            print(f"  内容: {response.content}")
            print(f"  Token使用: {response.usage.total_tokens}")
            print(f"  结束原因: {response.finish_reason}")
            return 0
        except Exception as e:
            print(f"测试失败: {e}")
            return 1

    if args.info:
        client = get_llm_client()
        print(f"\nLLM客户端配置:")
        print("-" * 60)
        print(f"  模型: {client.model}")
        print(f"  基础URL: {client.base_url}")
        print(f"  超时: {client.timeout}s")
        print(f"  最大重试: {client.retry_config.max_retries}")
        print(f"  API Key: {client.api_key[:8] + '...' if client.api_key else '未设置'}")
        print(f"  启用缓存: {client.use_cache}")
        return 0

    # 默认显示统计
    stats = get_llm_stats()
    print("\nLLM客户端统计:")
    print("-" * 60)
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    return 0


def cmd_memory(args):
    """seva memory: LLM增强的记忆与上下文管理（阶段61）"""
    from .memory_llm import (get_memory_manager, MemoryType, reset_memory_manager)
    import json

    if args.memory_reset:
        reset_memory_manager()
        print("已重置全局记忆管理器")
        return 0

    manager = get_memory_manager()

    # 添加记忆
    if args.add:
        mt = MemoryType(args.type) if args.type else MemoryType.EPISODIC
        importance = float(args.importance) if args.importance else 0.5
        entry = manager.add(args.add, mt, importance=importance,
                           auto_summarize=False)
        print(f"已添加记忆: {entry.id}")
        print(f"  类型: {entry.memory_type.value}")
        print(f"  重要性: {entry.importance}")
        print(f"  关键词: {entry.keywords[:5]}")
        return 0

    # 搜索记忆
    if args.search:
        results = manager.retrieve(args.search, limit=args.limit or 5)
        print(f"\n搜索 '{args.search}' 结果 (共{len(results)}条):")
        print("-" * 60)
        for i, entry in enumerate(results, 1):
            print(f"[{i}] {entry.id} ({entry.memory_type.value})")
            print(f"    重要性: {entry.current_importance():.2f}")
            print(f"    内容: {entry.content[:80]}...")
        return 0

    # 构造上下文
    if args.context:
        context = manager.build_context(args.context, max_tokens=300)
        print(f"\n上下文 (查询: '{args.context}'):")
        print("-" * 60)
        print(context if context else "(无相关记忆)")
        return 0

    # 清空记忆
    if args.clear:
        if args.type:
            count = manager.clear(MemoryType(args.type))
            print(f"已清空 {args.type} 类型记忆 {count} 条")
        else:
            count = manager.clear()
            print(f"已清空所有记忆 {count} 条")
        return 0

    # 默认显示报告
    report = manager.get_report()
    print(f"\n=== 记忆管理器报告 ===")
    print(f"容量: {report['capacity']}")
    print(f"Embedding维度: {report['embedding_dim']}")
    print(f"记忆预算: {report['memory_budget']} tokens")
    print(f"LLM摘要: {'启用' if report['enable_llm_summary'] else '禁用'}")
    print(f"\n统计:")
    for k, v in report['stats'].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    return 0


def cmd_tools(args):
    """seva tools: 工具调用与Function Calling管理（阶段62）"""
    import json
    from .tools import (get_tool_registry, get_tool_executor, reset_tools,
                        ToolCallRequest, ToolCallLoop, ToolCallStatus)
    if args.tools_reset:
        reset_tools()
        print("已重置全局工具注册表和执行器")
        return 0

    registry = get_tool_registry()

    # 列出所有工具
    if args.list:
        tools = registry.list_all()
        category = args.category
        if category:
            tools = [t for t in tools if t.category == category]
        print(f"\n已注册工具 ({len(tools)}个):")
        print("-" * 60)
        for t in tools:
            status = "启用" if t.enabled else "禁用"
            danger = " [危险]" if t.dangerous else ""
            print(f"  - {t.name} ({t.category}) [{status}]{danger}")
            print(f"    描述: {t.description}")
            if t.parameters:
                params = ", ".join(
                    f"{p.name}{'?' if not p.required else ''}"
                    for p in t.parameters
                )
                print(f"    参数: {params}")
        return 0

    # 显示工具详情
    if args.show:
        tool = registry.get(args.show)
        if tool is None:
            print(f"工具不存在: {args.show}")
            return 1
        print(f"\n工具详情:")
        print("-" * 60)
        print(f"  名称: {tool.name}")
        print(f"  描述: {tool.description}")
        print(f"  类别: {tool.category}")
        print(f"  超时: {tool.timeout}s")
        print(f"  危险: {'是' if tool.dangerous else '否'}")
        print(f"  启用: {'是' if tool.enabled else '否'}")
        print(f"  参数 ({len(tool.parameters)}个):")
        for p in tool.parameters:
            req = "必填" if p.required else "可选"
            print(f"    - {p.name} ({p.param_type.value}, {req})")
            if p.description:
                print(f"      描述: {p.description}")
            if p.default is not None:
                print(f"      默认: {p.default}")
            if p.enum is not None:
                print(f"      枚举: {p.enum}")
        # OpenAI schema
        print(f"\n  OpenAI Schema:")
        print(json.dumps(tool.to_openai_schema(), ensure_ascii=False, indent=2))
        return 0

    # 执行工具
    if args.call:
        tool = registry.get(args.call)
        if tool is None:
            print(f"工具不存在: {args.call}")
            return 1
        # 解析参数
        arguments = {}
        if args.arguments:
            try:
                arguments = json.loads(args.arguments)
            except json.JSONDecodeError as e:
                print(f"参数JSON解析失败: {e}")
                return 1
        executor = get_tool_executor()
        req = ToolCallRequest(tool_name=args.call, arguments=arguments)
        result = executor.execute(req)
        print(f"\n工具调用结果:")
        print("-" * 60)
        print(f"  工具: {result.tool_name}")
        print(f"  状态: {result.status.value}")
        print(f"  延迟: {result.latency_ms:.1f}ms")
        if result.success:
            print(f"  结果:")
            print(json.dumps(result.result, ensure_ascii=False, indent=2))
        else:
            print(f"  错误: {result.error}")
        return 0 if result.success else 1

    # LLM驱动的工具调用循环
    if args.ask:
        loop = ToolCallLoop(registry, get_tool_executor(),
                           max_iterations=args.max_iterations or 5)
        result = loop.run(args.ask, use_cache=not args.no_cache)
        print(f"\n=== 工具调用循环结果 ===")
        print(f"轮次: {result['iterations']}")
        print(f"Tokens: {result['total_tokens']}")
        if result["tool_calls"]:
            print(f"\n工具调用:")
            for tc in result["tool_calls"]:
                print(f"  - {tc['tool_name']}: {tc['status']}")
        print(f"\n回答:")
        print("-" * 60)
        print(result["answer"])
        return 0

    # 显示统计
    if args.stats:
        executor = get_tool_executor()
        stats = executor.get_stats()
        print(f"\n=== 工具执行统计 ===")
        print("-" * 60)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        return 0

    # 默认显示概览
    tools = registry.list_all()
    categories = {}
    for t in tools:
        categories[t.category] = categories.get(t.category, 0) + 1
    print(f"\n=== 工具系统概览 ===")
    print(f"总工具数: {len(tools)}")
    print(f"启用工具: {len([t for t in tools if t.enabled])}")
    print(f"\n按类别:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    return 0


def cmd_planner(args):
    """seva planner: 智能决策与自主规划（阶段63）"""
    import json
    from .planner import (Goal, PlanStrategy, PlanStatus,
                          get_planner, get_plan_executor, get_reflector,
                          get_decision_maker, get_autonomous_agent,
                          reset_planner, reset_reflector,
                          reset_decision_maker, reset_autonomous_agent)

    if args.planner_reset:
        reset_planner()
        reset_reflector()
        reset_decision_maker()
        reset_autonomous_agent()
        print("已重置全局规划器/反思器/决策器/自主Agent")
        return 0

    # 运行自主Agent
    if args.run:
        agent = get_autonomous_agent()
        goal = Goal(
            description=args.run,
            success_criteria=args.criteria or "",
        )
        print(f"\n=== 启动自主Agent ===")
        print(f"目标: {goal.description}")
        if goal.success_criteria:
            print(f"成功标准: {goal.success_criteria}")
        print(f"最大重规划: {args.max_replans or 2}")
        print(f"最大迭代: {args.max_iterations or 10}")
        print("-" * 60)

        result = agent.run(
            goal,
            max_replans=args.max_replans or 2,
            max_iterations=args.max_iterations or 10,
        )

        print(f"\n=== 执行结果 ===")
        print(f"状态: {result.status.value}")
        print(f"迭代: {result.iterations}")
        print(f"Tokens: {result.total_tokens}")
        print(f"延迟: {result.total_latency_ms:.0f}ms")
        if result.final_output:
            print(f"\n最终输出:")
            print(result.final_output[:500])
        if result.task_results:
            print(f"\n任务结果 ({len(result.task_results)}个):")
            for tr in result.task_results:
                status_icon = "✓" if tr.success else "✗"
                print(f"  {status_icon} {tr.task_id}: {tr.status.value}")
        if result.reflections:
            print(f"\n反思 ({len(result.reflections)}次):")
            for ref in result.reflections:
                print(f"  - 成功: {ref.get('success')}")
                if ref.get('summary'):
                    print(f"    摘要: {ref['summary'][:100]}")
        return 0 if result.success else 1

    # 仅分解目标(不执行)
    if args.decompose:
        planner = get_planner()
        goal = Goal(description=args.decompose)
        tasks = planner.decompose_goal(goal)
        print(f"\n=== 目标分解结果 ===")
        print(f"目标: {args.decompose}")
        print(f"任务数: {len(tasks)}")
        print("-" * 60)
        for i, task in enumerate(tasks, 1):
            print(f"[{i}] {task.description}")
            if task.expected_output:
                print(f"    期望输出: {task.expected_output}")
            if task.tools_needed:
                print(f"    工具: {', '.join(task.tools_needed)}")
            print(f"    难度: {task.difficulty:.1f}")
            if task.dependencies:
                print(f"    依赖: {task.dependencies}")
        return 0

    # 显示报告
    if args.report:
        agent = get_autonomous_agent()
        report = agent.get_report()
        print(f"\n=== 自主Agent报告 ===")
        print("-" * 60)
        for component, stats in report.items():
            if isinstance(stats, dict):
                print(f"\n{component}:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{component}: {stats}")
        return 0

    # 显示历史
    if args.history:
        agent = get_autonomous_agent()
        history = agent.get_history()
        print(f"\n=== 执行历史 ({len(history)}条) ===")
        print("-" * 60)
        for i, h in enumerate(history, 1):
            print(f"[{i}] 计划: {h.get('plan_id', '?')} v{h.get('version', 1)}")
            print(f"    状态: {h.get('status', '?')}")
            if h.get('reflection', {}).get('summary'):
                print(f"    反思: {h['reflection']['summary'][:80]}")
        return 0

    # 默认显示帮助
    print("seva planner - 智能决策与自主规划")
    print()
    print("用法:")
    print("  seva planner --run '目标描述'              # 运行自主Agent")
    print("  seva planner --decompose '目标描述'         # 仅分解目标")
    print("  seva planner --report                       # 显示报告")
    print("  seva planner --history                      # 显示历史")
    print("  seva planner --planner-reset                # 重置全局状态")
    print()
    print("选项:")
    print("  --criteria '标准'       成功标准")
    print("  --max-replans N         最大重规划次数 (默认2)")
    print("  --max-iterations N      最大迭代次数 (默认10)")
    return 0


def cmd_llm_loop(args):
    """seva llm-loop: 16阶段循环LLM增强管理（阶段60）"""
    from .llm_loop import get_llm_loop, reset_llm_loop
    import json

    if args.llm_loop_reset:
        reset_llm_loop()
        print("已重置全局LLM Loop")
        return 0

    loop = get_llm_loop()

    # 设置启用阶段
    if args.enable_stages:
        loop._enabled_stages = set(args.enable_stages.split(","))
        print(f"已启用阶段: {args.enable_stages}")
        return 0

    if args.disable_stages:
        for stage in args.disable_stages.split(","):
            loop.disable_stage(stage)
        print(f"已禁用阶段: {args.disable_stages}")
        return 0

    if args.disable:
        loop.enable = False
        print("LLM Loop已禁用")
        return 0

    if args.enable:
        loop.enable = True
        print("LLM Loop已启用")
        return 0

    # 测试调用
    if args.test:
        result = loop.call_stage(
            args.test,
            inputs={"input": args.input or "测试输入"},
            use_cache=False,
        )
        print(f"\n=== LLM Loop测试 ({args.test}) ===")
        print(f"状态: {result.status.value}")
        print(f"Tokens: {result.total_tokens}")
        print(f"延迟: {result.latency_ms:.0f}ms")
        if result.llm_output:
            print(f"\nLLM输出:")
            print(result.llm_output[:500])
        if result.parsed_output:
            print(f"\n解析后:")
            print(json.dumps(result.parsed_output, ensure_ascii=False, indent=2))
        return 0

    # 默认显示报告
    report = loop.get_report()
    print(f"\n=== LLM Loop报告 ===")
    print(f"启用: {report['enabled']}")
    print(f"启用阶段: {report['enabled_stages']}")
    print(f"禁用阶段: {report['disabled_stages']}")
    print(f"\n统计:")
    for k, v in report['stats'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n已注册模板: {len(report['registered_templates'])} 个")
    for stage, name in list(report['registered_templates'].items())[:5]:
        print(f"  - {stage}: {name}")
    if len(report['registered_templates']) > 5:
        print(f"  ... 还有 {len(report['registered_templates']) - 5} 个")
    return 0


def cmd_prompt(args):
    """seva prompt: Prompt工程与模板管理（阶段59）"""
    import json
    from .prompts import (PromptTemplate, PromptSection, PromptVariable,
                          FewShotExample, list_prompts, get_prompt,
                          register_prompt, unregister_prompt, render_prompt,
                          delete_prompt, reset_prompts)

    # 重置
    if args.reset_prompts:
        reset_prompts()
        print("已重置所有Prompt模板")
        return 0

    # 注册模板
    if args.register:
        # 创建示例模板
        template = PromptTemplate(
            name=args.register,
            description="通过CLI注册的模板",
            sections=[
                PromptSection.role("你是一个助手"),
                PromptSection.task("处理以下输入: {{input}}"),
                PromptSection.output_format("JSON: {result: string}"),
            ],
            variables=[
                PromptVariable(name="input", description="用户输入", required=True),
            ],
            author="cli",
        )
        if register_prompt(template, overwrite=True):
            print(f"已注册模板: {args.register}")
            return 0
        print(f"注册失败: {args.register}")
        return 1

    # 注销模板
    if args.unregister:
        if unregister_prompt(args.unregister):
            print(f"已注销模板: {args.unregister}")
            return 0
        print(f"模板不存在: {args.unregister}")
        return 1

    # 删除模板
    if args.delete:
        if delete_prompt(args.delete):
            print(f"已删除模板: {args.delete}")
            return 0
        print(f"模板不存在: {args.delete}")
        return 1

    # 渲染模板
    if args.render:
        template = get_prompt(args.render)
        if template is None:
            print(f"模板不存在: {args.render}")
            return 1
        # 默认输入
        inputs = {}
        if args.input:
            try:
                inputs = json.loads(args.input)
            except json.JSONDecodeError:
                inputs = {"input": args.input}
        if not inputs and template.variables:
            for var in template.variables:
                if var.default is not None:
                    inputs[var.name] = var.default
                else:
                    inputs[var.name] = f"[示例:{var.name}]"
        rendered = render_prompt(args.render, inputs=inputs)
        if rendered is None:
            print(f"渲染失败: {args.render}")
            return 1
        print(f"\n渲染结果 (template={args.render}, version={rendered.template_version}, hash={rendered.hash}):")
        print("-" * 60)
        for msg in rendered.to_messages():
            print(f"[{msg['role']}]:")
            print(msg['content'])
            print()
        return 0

    # 显示模板详情
    if args.show:
        template = get_prompt(args.show)
        if template is None:
            print(f"模板不存在: {args.show}")
            return 1
        print(f"\nPrompt模板详情:")
        print("-" * 60)
        print(f"  名称: {template.name}")
        print(f"  版本: {template.version}")
        print(f"  描述: {template.description}")
        print(f"  作者: {template.author}")
        print(f"  标签: {', '.join(template.tags) if template.tags else '(无)'}")
        print(f"  输出格式: {template.output_format.value}")
        print(f"  模型: {template.model or '(默认)'}")
        print(f"  温度: {template.temperature}")
        print(f"  最大Tokens: {template.max_tokens}")
        print(f"  段数: {len(template.sections)}")
        for i, section in enumerate(template.sections, 1):
            print(f"    [{i}] {section.section_type.value}: {section.content[:50]}..."
                  f"{' (含' + str(len(section.section_examples)) + '示例)' if section.section_examples else ''}")
        if template.variables:
            print(f"  变量: {len(template.variables)}")
            for var in template.variables:
                req = "必填" if var.required else "可选"
                default = f", 默认={var.default!r}" if var.default is not None else ""
                print(f"    - {var.name} ({req}, 类型={var.var_type}{default}): {var.description}")
        return 0

    # 默认列出所有模板
    names = list_prompts()
    print(f"\nPrompt模板列表 (共 {len(names)} 个):")
    print("-" * 60)
    if not names:
        print("  (无模板)")
        print("\n使用 --register <name> 注册新模板")
        print("使用 --render <name> 渲染模板")
        print("使用 --show <name> 查看详情")
    else:
        for name in names:
            template = get_prompt(name)
            if template:
                print(f"  - {name} (v{template.version}, {len(template.sections)}段, "
                      f"{len(template.variables)}变量)")
    return 0


def cmd_diagnose(args):
    """seva diagnose: 诊断系统"""
    print(f"\nSEVA系统诊断 (preset={args.preset})")
    print("="*50)

    # 创建Agent（可选运行几步）
    agent = SEVAFactory.create(args.preset)
    if args.steps > 0:
        print(f"运行 {args.steps} 步采集数据...")
        run(agent, steps=args.steps)

    # 诊断
    report = Diagnoser(agent).diagnose()
    output = Diagnoser(agent).format_report(report)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n诊断报告已保存到 {args.output}")

    return 0 if report["healthy"] else 1


def cmd_errors(args):
    """seva errors: 列出所有错误码"""
    print(f"\nSEVA错误码体系")
    print("="*50)

    # 收集所有错误码
    error_codes = []
    for name in dir(ErrorCode):
        if name.startswith("_"):
            continue
        val = getattr(ErrorCode, name)
        if isinstance(val, tuple) and len(val) == 4:
            error_codes.append((name, val))

    # 按分类分组
    from .diagnostics import ErrorCategory
    for category in ErrorCategory:
        print(f"\n[{category.value}]")
        for name, val in error_codes:
            if val[1] == category:
                print(f"  {val[0]} ({name})")
                print(f"    描述: {val[2]}")
                print(f"    修复: {val[3]}")
    return 0


def cmd_monitor(args):
    """seva monitor: 实时监控Agent运行"""
    print(f"\nSEVA实时监控 (preset={args.preset}, steps={args.steps})")
    print("="*50)

    agent = SEVAFactory.create(args.preset)
    dashboard = Dashboard(agent)

    # 运行并定期显示仪表盘
    report_interval = max(args.steps // 5, 1)
    for i in range(args.steps):
        if agent._env_done:
            break
        if not hasattr(agent, "_pending_input") or agent._pending_input is None:
            agent.inject_input(np.random.randn(
                agent.config.get("perception", "input_dim")) * 0.3)
        agent.run_cycle()

        if (i + 1) % report_interval == 0 or i == args.steps - 1:
            print(f"\n--- 步骤 {i+1}/{args.steps} ---")
            dashboard.print(compact=True)

    # 最终详细仪表盘
    print("\n--- 最终状态 ---")
    dashboard.print(compact=False)

    # 导出指标
    if args.output:
        export_metrics(dashboard.collector, args.output)
        print(f"\n指标已导出到 {args.output}")

    return 0


def cmd_logs(args):
    """seva logs: 日志管理"""
    if args.logs_action == "show":
        return _logs_show(args)
    elif args.logs_action == "export":
        return _logs_export(args)
    else:
        print(f"未知logs子命令: {args.logs_action}")
        return 1


def _logs_show(args):
    """显示日志"""
    logger = setup_logging(level=args.level, format=args.format)
    # 运行Agent产生日志
    agent = SEVAFactory.create(args.preset)
    print(f"运行 {args.steps} 步产生日志...")
    run(agent, steps=args.steps)

    # 显示日志记录
    records = logger.get_records(limit=args.limit)
    print(f"\n最近 {len(records)} 条日志:")
    for r in records:
        print(f"  [{r['level']}] {r['message']}")
    return 0


def _logs_export(args):
    """导出日志"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "seva.log")
        logger = setup_logging(level="DEBUG", format=args.format,
                              output_file=log_file)
        agent = SEVAFactory.create(args.preset)
        run(agent, steps=args.steps)

        output = args.output or "seva_export.log"
        import shutil
        shutil.copy(log_file, output)
        print(f"日志已导出到 {output}")
    return 0


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="seva",
        description="SEVA - 自驱永续进化预测修正Agent系统"
    )
    parser.add_argument("--version", action="store_true",
                       help="显示版本信息")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run命令
    run_parser = subparsers.add_parser("run", help="启动Agent运行")
    run_parser.add_argument("--steps", type=int, default=100,
                           help="运行步数 (默认: 100)")
    run_parser.add_argument("--preset", type=str, default="easy",
                           choices=PresetConfigs.list_presets(),
                           help="预设模板 (默认: easy)")
    run_parser.add_argument("--env-type", type=str, default="mixed",
                           choices=["blank", "mixed", "drift", "adversarial"],
                           help="环境类型 (默认: mixed)")
    run_parser.add_argument("--seed", type=int, default=42,
                           help="随机种子 (默认: 42)")
    run_parser.add_argument("--verbose", action="store_true",
                           help="详细输出")
    run_parser.add_argument("--output", type=str, default=None,
                           help="结果保存文件路径")
    # 阶段35: 支持从配置文件加载
    run_parser.add_argument("--config", type=str, default=None,
                           help="从配置文件加载 (YAML/JSON)")
    run_parser.set_defaults(func=cmd_run)

    # config命令
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_sub = config_parser.add_subparsers(dest="config_action",
                                              help="配置子命令")

    init_parser = config_sub.add_parser("init", help="生成配置文件")
    init_parser.add_argument("--preset", type=str, default="standard",
                            choices=PresetConfigs.list_presets(),
                            help="预设模板 (默认: standard)")
    init_parser.add_argument("--output", type=str, default=None,
                            help="输出文件路径 (默认: seva_config.json)")

    show_parser = config_sub.add_parser("show", help="显示当前配置")
    show_parser.add_argument("--preset", type=str, default="standard",
                            choices=PresetConfigs.list_presets(),
                            help="预设模板 (默认: standard)")
    show_parser.add_argument("--section", type=str, default=None,
                            help="只显示指定section")

    val_parser = config_sub.add_parser("validate", help="校验配置")
    val_parser.add_argument("--preset", type=str, default="standard",
                           choices=PresetConfigs.list_presets(),
                           help="预设模板 (默认: standard)")

    # 阶段35: config save - 保存配置到YAML/JSON
    save_parser = config_sub.add_parser("save", help="保存配置到文件")
    save_parser.add_argument("--preset", type=str, default="standard",
                            choices=PresetConfigs.list_presets(),
                            help="预设模板 (默认: standard)")
    save_parser.add_argument("--output", type=str, default=None,
                            help="输出文件路径 (默认: seva_config.yaml)")
    save_parser.add_argument("--env-type", type=str, default="mixed",
                            choices=["blank", "mixed", "drift", "adversarial"],
                            help="环境类型 (默认: mixed)")
    save_parser.add_argument("--steps", type=int, default=100,
                            help="运行步数 (默认: 100)")

    # 阶段35: config load - 从文件加载配置
    load_parser = config_sub.add_parser("load", help="从文件加载配置")
    load_parser.add_argument("file", type=str, help="配置文件路径")
    load_parser.add_argument("--section", type=str, default=None,
                            help="只显示指定section")

    # 阶段35: config wizard - 配置向导
    wizard_parser = config_sub.add_parser("wizard", help="配置向导")
    wizard_parser.add_argument("--preset", type=str, default="standard",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (默认: standard)")
    wizard_parser.add_argument("--output", type=str, default=None,
                              help="保存到文件路径")
    wizard_parser.add_argument("--env-type", type=str, default="mixed",
                              choices=["blank", "mixed", "drift", "adversarial"],
                              help="环境类型 (默认: mixed)")
    wizard_parser.add_argument("--steps", type=int, default=100,
                              help="运行步数 (默认: 100)")
    wizard_parser.add_argument("--seed", type=int, default=42,
                              help="随机种子 (默认: 42)")

    config_parser.set_defaults(func=cmd_config)

    # status命令
    status_parser = subparsers.add_parser("status", help="查看系统状态")
    status_parser.set_defaults(func=cmd_status)

    # version命令
    version_parser = subparsers.add_parser("version", help="版本信息")
    version_parser.set_defaults(func=cmd_version)

    # presets命令
    presets_parser = subparsers.add_parser("presets", help="列出预设模板")
    presets_parser.set_defaults(func=cmd_presets)

    # demo命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    demo_parser.add_argument("--steps", type=int, default=50,
                            help="演示步数 (默认: 50)")
    demo_parser.add_argument("--preset", type=str, default="easy",
                            choices=PresetConfigs.list_presets(),
                            help="预设模板 (默认: easy)")
    demo_parser.add_argument("--env-type", type=str, default="mixed",
                            choices=["blank", "mixed", "drift", "adversarial"],
                            help="环境类型 (默认: mixed)")
    demo_parser.set_defaults(func=cmd_demo)

    # scenarios命令（阶段34）：列出所有预设场景
    scenarios_parser = subparsers.add_parser("scenarios",
                                              help="列出所有预设场景")
    scenarios_parser.set_defaults(func=cmd_scenarios)

    # scenario命令（阶段34）：场景管理
    scenario_parser = subparsers.add_parser("scenario", help="场景管理")
    scenario_sub = scenario_parser.add_subparsers(
        dest="scenario_action", help="场景子命令")

    # scenario run
    run_scn_parser = scenario_sub.add_parser("run", help="运行指定场景")
    run_scn_parser.add_argument("name", type=str,
                                help="场景名称 (如 quick_validation)")
    run_scn_parser.add_argument("--output", type=str, default=None,
                                help="结果保存路径")

    # scenario compare
    cmp_scn_parser = scenario_sub.add_parser("compare",
                                              help="对比多个场景")
    cmp_scn_parser.add_argument("names", nargs="+", help="场景名称列表")
    cmp_scn_parser.add_argument("--output", type=str, default=None,
                                help="对比报告保存路径")

    # scenario run-all
    all_scn_parser = scenario_sub.add_parser("run-all",
                                              help="运行所有场景")
    all_scn_parser.add_argument("--output", type=str, default=None,
                                help="所有结果保存路径")

    # scenario list
    list_scn_parser = scenario_sub.add_parser("list",
                                               help="列出所有场景")
    scenario_parser.set_defaults(func=cmd_scenario)

    # 阶段36: shell命令 - 启动交互式Shell
    shell_parser = subparsers.add_parser("shell", help="启动交互式Shell")
    shell_parser.add_argument("--preset", type=str, default="easy",
                             choices=PresetConfigs.list_presets(),
                             help="预设模板 (默认: easy)")
    shell_parser.add_argument("--env-type", type=str, default="mixed",
                             choices=["blank", "mixed", "drift", "adversarial"],
                             help="环境类型 (默认: mixed)")
    shell_parser.add_argument("--seed", type=int, default=42,
                             help="随机种子 (默认: 42)")
    shell_parser.set_defaults(func=cmd_shell)

    # 阶段36: exec命令 - 执行单条Shell命令
    exec_parser = subparsers.add_parser("exec", help="执行单条Shell命令")
    exec_parser.add_argument("command", nargs="*", help="要执行的命令")
    exec_parser.add_argument("--preset", type=str, default="easy",
                            choices=PresetConfigs.list_presets(),
                            help="预设模板 (默认: easy)")
    exec_parser.add_argument("--env-type", type=str, default="mixed",
                            choices=["blank", "mixed", "drift", "adversarial"],
                            help="环境类型 (默认: mixed)")
    exec_parser.add_argument("--seed", type=int, default=42,
                            help="随机种子 (默认: 42)")
    exec_parser.set_defaults(func=cmd_exec)

    # 阶段37: web命令 - 启动Web Dashboard
    web_parser = subparsers.add_parser("web", help="启动Web Dashboard")
    web_parser.add_argument("--host", type=str, default="localhost",
                           help="监听地址 (默认: localhost)")
    web_parser.add_argument("--port", type=int, default=8080,
                           help="监听端口 (默认: 8080)")
    web_parser.add_argument("--preset", type=str, default="easy",
                           choices=PresetConfigs.list_presets(),
                           help="预设模板 (默认: easy)")
    web_parser.add_argument("--steps", type=int, default=10,
                           help="启动前运行步数 (默认: 10)")
    web_parser.set_defaults(func=cmd_web)

    # 阶段37/38: report命令 - 生成多格式报告
    report_parser = subparsers.add_parser("report", help="生成报告")
    report_parser.add_argument("--output", type=str, default=None,
                              help="输出文件路径 (默认: seva_report.<ext>)")
    report_parser.add_argument("--preset", type=str, default="easy",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (默认: easy)")
    report_parser.add_argument("--steps", type=int, default=50,
                              help="报告前运行步数 (默认: 50)")
    report_parser.add_argument("--format", type=str, default="html",
                              choices=["html", "json", "csv", "markdown", "txt"],
                              help="报告格式 (默认: html)")
    report_parser.set_defaults(func=cmd_report)

    # 阶段38: export命令 - 导出多格式报告
    export_parser = subparsers.add_parser("export", help="导出多格式报告")
    export_parser.add_argument("--format", type=str, default="html",
                              choices=["html", "json", "csv", "markdown", "txt"],
                              help="导出格式 (默认: html)")
    export_parser.add_argument("--output", type=str, default=None,
                              help="输出文件路径 (默认: seva_export.<ext>)")
    export_parser.add_argument("--preset", type=str, default="easy",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (默认: easy)")
    export_parser.add_argument("--steps", type=int, default=50,
                              help="导出前运行步数 (默认: 50)")
    export_parser.set_defaults(func=cmd_export)

    # 阶段38: charts命令 - 生成SVG图表
    charts_parser = subparsers.add_parser("charts", help="生成SVG图表")
    charts_parser.add_argument("--output-dir", type=str, default=None,
                              dest="output_dir",
                              help="输出目录 (默认: ./seva_charts)")
    charts_parser.add_argument("--preset", type=str, default="easy",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (默认: easy)")
    charts_parser.add_argument("--steps", type=int, default=50,
                              help="生成图表前运行步数 (默认: 50)")
    charts_parser.set_defaults(func=cmd_charts)

    # 阶段38: compare命令 - 比较多个运行
    compare_parser = subparsers.add_parser("compare", help="比较多个运行")
    compare_parser.add_argument("presets", nargs="+",
                               help="要比较的预设列表 (至少2个)")
    compare_parser.add_argument("--steps", type=int, default=50,
                               help="每个运行步数 (默认: 50)")
    compare_parser.add_argument("--format", type=str, default="html",
                               choices=["html", "markdown", "txt", "json"],
                               help="比较报告格式 (默认: html)")
    compare_parser.add_argument("--output", type=str, default=None,
                               help="输出文件路径 (默认: seva_comparison.<ext>)")
    compare_parser.set_defaults(func=cmd_compare)

    # 阶段39: session命令 - 会话持久化管理
    session_parser = subparsers.add_parser("session", help="会话持久化管理")
    session_sub = session_parser.add_subparsers(dest="action")

    # session save
    sp_save = session_sub.add_parser("save", help="保存会话")
    sp_save.add_argument("--output", type=str, default=None,
                         help="输出文件路径")
    sp_save.add_argument("--preset", type=str, default="easy",
                         choices=PresetConfigs.list_presets(),
                         help="预设模板 (默认: easy)")
    sp_save.add_argument("--steps", type=int, default=50,
                         help="保存前运行步数 (默认: 50)")

    # session load
    sp_load = session_sub.add_parser("load", help="加载会话")
    sp_load.add_argument("--input", type=str, required=True,
                         help="会话文件路径")

    # session list
    sp_list = session_sub.add_parser("list", help="列出会话")
    sp_list.add_argument("--directory", type=str, default=None,
                         help="目录路径 (默认: 当前目录)")

    # session info
    sp_info = session_sub.add_parser("info", help="查看会话信息")
    sp_info.add_argument("--input", type=str, required=True,
                         help="会话文件路径")

    # session delete
    sp_del = session_sub.add_parser("delete", help="删除会话")
    sp_del.add_argument("--input", type=str, required=True,
                        help="会话文件路径")

    # session checkpoint
    sp_ckpt = session_sub.add_parser("checkpoint", help="创建检查点")
    sp_ckpt.add_argument("--preset", type=str, default="easy",
                         choices=PresetConfigs.list_presets(),
                         help="预设模板 (默认: easy)")
    sp_ckpt.add_argument("--steps", type=int, default=50,
                         help="检查点前运行步数 (默认: 50)")
    sp_ckpt.add_argument("--directory", type=str, default=None,
                         help="保存目录 (默认: ./checkpoints)")
    sp_ckpt.add_argument("--max-keep", type=int, default=0,
                         dest="max_keep",
                         help="保留最近N个检查点 (0=不限制)")

    session_parser.set_defaults(func=cmd_session)

    # 阶段41: wizard命令 - 首次运行向导
    wizard_parser = subparsers.add_parser("wizard", help="首次运行向导")
    wizard_parser.add_argument("--silent", action="store_true",
                              help="静默模式（非交互，使用默认或指定参数）")
    wizard_parser.add_argument("--preset", type=str, default="easy",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (静默模式使用，默认: easy)")
    wizard_parser.add_argument("--env-type", type=str, default="mixed",
                              dest="env_type",
                              choices=["blank", "mixed", "drift", "adversarial"],
                              help="环境类型 (静默模式使用，默认: mixed)")
    wizard_parser.add_argument("--steps", type=int, default=10,
                              help="运行步数 (静默模式使用，默认: 10)")
    wizard_parser.add_argument("--reset", action="store_true",
                              help="重置向导状态")
    wizard_parser.add_argument("--status", action="store_true",
                              help="查看向导状态")
    wizard_parser.add_argument("--tips", action="store_true",
                              help="显示使用提示")
    wizard_parser.set_defaults(func=cmd_wizard)

    # 阶段42: completion命令 - Shell补全与命令提示
    completion_parser = subparsers.add_parser("completion",
                                              help="生成Shell补全脚本")
    completion_parser.add_argument("--shell", type=str, default="bash",
                                  choices=["bash", "zsh"],
                                  help="Shell类型 (默认: bash)")
    completion_parser.add_argument("--output", type=str, default=None,
                                  help="输出文件路径")
    completion_parser.add_argument("--suggest", type=str, default=None,
                                  help="拼写纠错建议")
    completion_parser.add_argument("--list", action="store_true",
                                  help="列出所有命令")
    completion_parser.add_argument("--help-cmd", type=str, default=None,
                                  dest="help_cmd",
                                  help="获取命令帮助")

    completion_parser.set_defaults(func=cmd_completion)

    # 阶段43: tutorial命令 - 交互式教程
    tutorial_parser = subparsers.add_parser("tutorial", help="交互式教程")
    tutorial_parser.add_argument("tutorial_id", type=str, nargs="?",
                                 default=None,
                                 help="教程ID (beginner/intermediate/advanced)")
    tutorial_parser.add_argument("--list", action="store_true",
                                 help="列出所有教程")
    tutorial_parser.add_argument("--silent", action="store_true",
                                 help="静默模式（非交互）")
    tutorial_parser.add_argument("--no-resume", action="store_true",
                                 dest="no_resume",
                                 help="不恢复进度，从头开始")
    tutorial_parser.add_argument("--reset", action="store_true",
                                 help="重置教程进度")
    tutorial_parser.add_argument("--progress", action="store_true",
                                 help="查看教程进度")
    tutorial_parser.set_defaults(func=cmd_tutorial)

    # 阶段44: perf命令 - 性能监控与优化
    perf_parser = subparsers.add_parser("perf", help="性能监控与优化")
    perf_parser.add_argument("--startup", action="store_true",
                             help="测量Agent启动时间")
    perf_parser.add_argument("--optimize", action="store_true",
                             help="优化Agent配置")
    perf_parser.add_argument("--stats", action="store_true",
                             help="查看详细性能统计")
    perf_parser.add_argument("--report", action="store_true",
                             help="查看性能报告")
    perf_parser.add_argument("--save", type=str, default=None,
                             help="保存性能数据到文件")
    perf_parser.add_argument("--reset", action="store_true",
                             help="重置性能监控数据")
    perf_parser.add_argument("--preset", type=str, default="easy",
                             choices=PresetConfigs.list_presets(),
                             help="预设模板 (默认: easy)")
    perf_parser.add_argument("--env-type", type=str, default="mixed",
                             dest="env_type",
                             choices=["blank", "mixed", "drift", "adversarial"],
                             help="环境类型 (默认: mixed)")
    perf_parser.add_argument("--seed", type=int, default=42,
                             help="随机种子 (默认: 42)")
    perf_parser.set_defaults(func=cmd_perf)

    # 阶段45: prefs命令 - 用户偏好与个性化设置
    prefs_parser = subparsers.add_parser("prefs", help="用户偏好与个性化设置")
    prefs_parser.add_argument("--get", type=str, default=None,
                              help="获取偏好值")
    prefs_parser.add_argument("--set", type=str, default=None,
                              help="设置偏好值 (key=value)")
    prefs_parser.add_argument("--reset", type=str, default=None,
                              help="重置偏好 (key 或 all)")
    prefs_parser.add_argument("--list", action="store_true",
                              help="列出所有偏好")
    prefs_parser.add_argument("--schema", action="store_true",
                              help="列出偏好定义")
    prefs_parser.add_argument("--category", type=str, default=None,
                              help="按分类筛选")
    prefs_parser.add_argument("--export", type=str, default=None,
                              help="导出偏好到文件")
    prefs_parser.add_argument("--import-file", type=str, default=None,
                              dest="import_file",
                              help="从文件导入偏好")
    prefs_parser.add_argument("--profile", type=str, default=None,
                              dest="profile_action",
                              choices=["list", "create", "delete",
                                       "switch", "show"],
                              help="配置文件操作")
    prefs_parser.add_argument("profile_name", type=str, nargs="?",
                              default=None,
                              help="配置文件名")
    prefs_parser.add_argument("--from-current", action="store_true",
                              dest="from_current",
                              help="从当前偏好创建配置文件")
    prefs_parser.set_defaults(func=cmd_prefs)

    # 阶段46: help命令 - 帮助系统与文档集成
    help_parser = subparsers.add_parser("help", help="帮助系统与文档")
    help_parser.add_argument("topic", type=str, nargs="?",
                             default=None,
                             help="主题ID")
    help_parser.add_argument("--search", type=str, default=None,
                             help="搜索主题")
    help_parser.add_argument("--list", action="store_true",
                             help="列出所有主题")
    help_parser.add_argument("--categories", action="store_true",
                             help="列出分类")
    help_parser.add_argument("--faq", action="store_true",
                             help="常见问题")
    help_parser.add_argument("--glossary", action="store_true",
                             help="术语表")
    help_parser.add_argument("--examples", action="store_true",
                             help="代码示例")
    help_parser.add_argument("--command", type=str, default=None,
                             help="命令帮助")
    help_parser.add_argument("--example-id", type=str, default=None,
                             dest="example_id",
                             help="查看指定示例")
    help_parser.add_argument("--term", type=str, default=None,
                             help="查看指定术语")
    help_parser.set_defaults(func=cmd_help)

    # 阶段47: i18n命令 - 国际化与多语言支持
    i18n_parser = subparsers.add_parser("i18n", help="国际化与多语言支持")
    i18n_parser.add_argument("--set", type=str, default=None,
                             help="设置当前语言 (如 zh_CN/en_US/ja_JP)")
    i18n_parser.add_argument("--list", action="store_true",
                             help="列出所有支持的语言")
    i18n_parser.add_argument("--keys", type=str, nargs="?",
                             const="all", default=None,
                             help="列出翻译键 (可选指定语言)")
    i18n_parser.add_argument("--translate", type=str, default=None,
                             help="翻译指定键")
    i18n_parser.add_argument("--report", action="store_true",
                             help="查看i18n报告")
    i18n_parser.add_argument("--stats", action="store_true",
                             help="查看统计信息")
    i18n_parser.set_defaults(func=cmd_i18n)

    # 阶段48: error命令 - 错误处理与恢复
    error_parser = subparsers.add_parser("error", help="错误处理与恢复")
    error_parser.add_argument("--suggest", type=str, default=None,
                              help="对模拟错误给出建议 (如 'ValueError(\"test\")')")
    error_parser.add_argument("--log", type=str, default=None,
                              help="查看错误日志文件")
    error_parser.add_argument("--clear-log", type=str, default=None,
                              dest="clear_log",
                              help="清空错误日志文件")
    error_parser.add_argument("--test", action="store_true",
                              help="测试错误处理功能")
    error_parser.set_defaults(func=cmd_error)

    # 阶段49: progress命令 - 进度反馈与状态指示
    progress_parser = subparsers.add_parser("progress",
                                            help="进度反馈与状态指示")
    progress_parser.add_argument("--demo", action="store_true",
                                 help="演示所有进度组件")
    progress_parser.add_argument("--track", action="store_true",
                                 help="跟踪迭代进度演示")
    progress_parser.add_argument("--tasks", action="store_true",
                                 help="多任务进度跟踪演示")
    progress_parser.add_argument("--status-list", action="store_true",
                                 dest="status_list",
                                 help="列出所有状态图标")
    progress_parser.add_argument("--total", type=int, default=20,
                                 help="项目数量 (默认: 20)")
    progress_parser.add_argument("--ascii", action="store_true",
                                 help="使用ASCII模式（无Unicode）")
    progress_parser.set_defaults(func=cmd_progress)

    # 阶段50: suggest命令 - 智能建议与推荐系统
    suggest_parser = subparsers.add_parser("suggest",
                                           help="智能建议与推荐系统")
    suggest_parser.add_argument("--profile", action="store_true",
                                help="查看用户画像")
    suggest_parser.add_argument("--reset-profile", action="store_true",
                                dest="reset_profile",
                                help="重置用户画像")
    suggest_parser.add_argument("--recommend-config", type=str,
                                default=None,
                                dest="recommend_config",
                                help="推荐配置 (指定场景)")
    suggest_parser.add_argument("--recommend-commands", action="store_true",
                                dest="recommend_commands",
                                help="推荐命令")
    suggest_parser.add_argument("--recommend-learning", action="store_true",
                                dest="recommend_learning",
                                help="推荐学习资源")
    suggest_parser.add_argument("--analyze", action="store_true",
                                help="分析Agent上下文")
    suggest_parser.add_argument("--preset", type=str, default="easy",
                                choices=PresetConfigs.list_presets(),
                                help="预设模板 (默认: easy)")
    suggest_parser.add_argument("--steps", type=int, default=0,
                                help="运行步数 (默认: 0)")
    suggest_parser.set_defaults(func=cmd_suggest)

    # 阶段51: accessibility命令 - 可访问性与无障碍支持
    a11y_parser = subparsers.add_parser("accessibility",
                                        help="可访问性与无障碍支持")
    a11y_parser.add_argument("--report", action="store_true",
                             help="查看可访问性报告")
    a11y_parser.add_argument("--reset", action="store_true",
                             help="重置可访问性配置")
    a11y_parser.add_argument("--high-contrast", type=str, default=None,
                             dest="high_contrast",
                             choices=["on", "off"],
                             help="启用/禁用高对比度模式")
    a11y_parser.add_argument("--large-text", type=str, default=None,
                             dest="large_text",
                             choices=["on", "off"],
                             help="启用/禁用大字体模式")
    a11y_parser.add_argument("--font-scale", type=float, default=None,
                             dest="font_scale",
                             help="字体缩放 (默认: 1.5)")
    a11y_parser.add_argument("--screen-reader", type=str, default=None,
                             dest="screen_reader",
                             choices=["on", "off"],
                             help="启用/禁用屏幕阅读器支持")
    a11y_parser.add_argument("--reduce-motion", type=str, default=None,
                             dest="reduce_motion",
                             choices=["on", "off"],
                             help="启用/禁用减少动画")
    a11y_parser.add_argument("--color-blindness", type=str, default=None,
                             dest="color_blindness",
                             choices=["none", "protanopia",
                                      "deuteranopia", "tritanopia",
                                      "achromatopsia"],
                             help="设置色盲类型")
    a11y_parser.add_argument("--keybindings", action="store_true",
                             help="查看键盘快捷键")
    a11y_parser.add_argument("--test", action="store_true",
                             help="测试无障碍格式化")
    a11y_parser.set_defaults(func=cmd_accessibility)

    # 阶段52: analytics命令 - 用户行为分析与引导优化
    analytics_parser = subparsers.add_parser("analytics",
                                             help="用户行为分析与引导优化")
    analytics_parser.add_argument("--summary", action="store_true",
                                  help="查看分析摘要")
    analytics_parser.add_argument("--patterns", action="store_true",
                                  help="查看使用模式")
    analytics_parser.add_argument("--anomalies", action="store_true",
                                  help="查看异常行为")
    analytics_parser.add_argument("--optimize", action="store_true",
                                  help="引导优化建议")
    analytics_parser.add_argument("--stats", action="store_true",
                                  help="查看统计信息")
    analytics_parser.add_argument("--track", action="store_true",
                                  help="记录测试事件")
    analytics_parser.add_argument("--clear", action="store_true",
                                  help="清空分析数据")
    analytics_parser.set_defaults(func=cmd_analytics)

    # diagnose命令
    diagnose_parser = subparsers.add_parser("diagnose", help="诊断系统")
    diagnose_parser.add_argument("--steps", type=int, default=10,
                               help="诊断前运行步数 (默认: 10)")
    diagnose_parser.add_argument("--preset", type=str, default="easy",
                               choices=PresetConfigs.list_presets(),
                               help="预设模板 (默认: easy)")
    diagnose_parser.add_argument("--output", type=str, default=None,
                               help="诊断报告保存路径")
    diagnose_parser.set_defaults(func=cmd_diagnose)

    # errors命令
    errors_parser = subparsers.add_parser("errors", help="列出所有错误码")
    errors_parser.set_defaults(func=cmd_errors)

    # monitor命令
    monitor_parser = subparsers.add_parser("monitor", help="实时监控Agent运行")
    monitor_parser.add_argument("--steps", type=int, default=50,
                              help="监控步数 (默认: 50)")
    monitor_parser.add_argument("--preset", type=str, default="easy",
                              choices=PresetConfigs.list_presets(),
                              help="预设模板 (默认: easy)")
    monitor_parser.add_argument("--output", type=str, default=None,
                              help="指标导出路径")
    monitor_parser.set_defaults(func=cmd_monitor)

    # logs命令
    logs_parser = subparsers.add_parser("logs", help="日志管理")
    logs_sub = logs_parser.add_subparsers(dest="logs_action",
                                         help="日志子命令")

    show_parser = logs_sub.add_parser("show", help="显示日志")
    show_parser.add_argument("--steps", type=int, default=10,
                           help="运行步数 (默认: 10)")
    show_parser.add_argument("--preset", type=str, default="easy",
                           choices=PresetConfigs.list_presets(),
                           help="预设模板 (默认: easy)")
    show_parser.add_argument("--level", type=str, default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                           help="日志级别 (默认: INFO)")
    show_parser.add_argument("--format", type=str, default="text",
                           choices=["text", "json"],
                           help="日志格式 (默认: text)")
    show_parser.add_argument("--limit", type=int, default=20,
                           help="显示条数 (默认: 20)")

    export_parser = logs_sub.add_parser("export", help="导出日志")
    export_parser.add_argument("--steps", type=int, default=20,
                             help="运行步数 (默认: 20)")
    export_parser.add_argument("--preset", type=str, default="easy",
                             choices=PresetConfigs.list_presets(),
                             help="预设模板 (默认: easy)")
    export_parser.add_argument("--format", type=str, default="text",
                             choices=["text", "json"],
                             help="日志格式 (默认: text)")
    export_parser.add_argument("--output", type=str, default=None,
                             help="输出文件路径 (默认: seva_export.log)")

    logs_parser.set_defaults(func=cmd_logs)

    # stability命令（阶段53）
    stability_parser = subparsers.add_parser("stability",
                                              help="系统稳定性与容错管理")
    stability_parser.add_argument("--report", action="store_true",
                                   help="显示稳定性报告 (默认)")
    stability_parser.add_argument("--demo", action="store_true",
                                   help="演示熔断器工作流程")
    stability_parser.add_argument("--circuit", action="store_true",
                                   help="查看熔断器状态")
    stability_parser.add_argument("--health", action="store_true",
                                   help="运行健康检查")
    stability_parser.add_argument("--retry-policy", action="store_true",
                                   help="显示/设置重试策略")
    stability_parser.add_argument("--max-retries", type=int, default=None,
                                   help="设置最大重试次数")
    stability_parser.add_argument("--backoff", type=str, default=None,
                                   choices=["fixed", "linear", "exponential", "random"],
                                   help="退避策略")
    stability_parser.add_argument("--reset", action="store_true",
                                   help="重置所有稳定性状态")
    stability_parser.set_defaults(func=cmd_stability)

    # plugin命令（阶段54）
    plugin_parser = subparsers.add_parser("plugin", help="插件系统管理")
    plugin_parser.add_argument("--report", action="store_true",
                                help="显示插件报告 (默认)")
    plugin_parser.add_argument("--list", action="store_true",
                                help="列出所有插件")
    plugin_parser.add_argument("--enable", type=str, default=None,
                                help="启用指定插件")
    plugin_parser.add_argument("--disable", type=str, default=None,
                                help="禁用指定插件")
    plugin_parser.add_argument("--hooks", action="store_true",
                                help="查看钩子统计")
    plugin_parser.add_argument("--discover", action="store_true",
                                help="发现并加载插件目录中的插件")
    plugin_parser.add_argument("--demo", action="store_true",
                                help="演示插件系统")
    plugin_parser.add_argument("--reset", action="store_true",
                                help="重置所有插件状态")
    plugin_parser.set_defaults(func=cmd_plugin)

    # orchestrate命令（阶段55）
    orch_parser = subparsers.add_parser("orchestrate",
                                         help="多Agent协作与编排")
    orch_parser.add_argument("--report", action="store_true",
                              help="显示编排报告 (默认)")
    orch_parser.add_argument("--list", action="store_true",
                              help="列出所有工作流")
    orch_parser.add_argument("--run", type=str, default=None,
                              help="运行指定工作流")
    orch_parser.add_argument("--demo", action="store_true",
                              help="演示工作流编排")
    orch_parser.add_argument("--reset", action="store_true",
                              help="重置所有编排状态")
    orch_parser.set_defaults(func=cmd_orchestrate)

    # stream命令（阶段56）
    stream_parser = subparsers.add_parser("stream",
                                           help="实时流处理与事件驱动")
    stream_parser.add_argument("--stats", action="store_true",
                                help="显示流处理统计 (默认)")
    stream_parser.add_argument("--demo", action="store_true",
                                help="演示流处理功能")
    stream_parser.add_argument("--publish", type=str, default=None,
                                help="发布事件 (指定事件类型)")
    stream_parser.add_argument("--data", type=str, default=None,
                                help="事件数据")
    stream_parser.add_argument("--reset", action="store_true",
                                help="重置流处理状态")
    stream_parser.set_defaults(func=cmd_stream)

    # benchmark命令（阶段57）
    bench_parser = subparsers.add_parser("benchmark",
                                          help="性能基准与自动调优")
    bench_parser.add_argument("--report", action="store_true",
                               help="显示基准测试报告 (默认)")
    bench_parser.add_argument("--list", action="store_true",
                               help="列出所有基准测试")
    bench_parser.add_argument("--run", action="store_true",
                               help="运行所有基准测试")
    bench_parser.add_argument("--demo", action="store_true",
                               help="演示性能基准功能")
    bench_parser.add_argument("--tune", type=str, default=None,
                               choices=["grid", "random"],
                               help="参数调优 (指定策略)")
    bench_parser.add_argument("--ab", type=str, default=None,
                               help="A/B测试 (指定测试名称)")
    bench_parser.add_argument("--reset", action="store_true",
                               help="重置基准测试状态")
    bench_parser.set_defaults(func=cmd_benchmark)

    # llm命令（阶段58）
    llm_parser = subparsers.add_parser("llm", help="LLM客户端与API管理")
    llm_parser.add_argument("--stats", action="store_true",
                            help="显示LLM客户端统计 (默认)")
    llm_parser.add_argument("--info", action="store_true",
                            help="显示配置信息")
    llm_parser.add_argument("--reset", action="store_true",
                            help="重置LLM客户端")
    llm_parser.add_argument("--test", action="store_true",
                            help="测试LLM连接 (需要API Key)")
    llm_parser.set_defaults(func=cmd_llm)

    # prompt命令（阶段59）
    prompt_parser = subparsers.add_parser("prompt", help="Prompt工程与模板管理")
    prompt_parser.add_argument("--list", action="store_true",
                               help="列出所有模板 (默认)")
    prompt_parser.add_argument("--show", type=str, default=None,
                               help="显示模板详情")
    prompt_parser.add_argument("--register", type=str, default=None,
                               help="注册新模板 (传入名称)")
    prompt_parser.add_argument("--unregister", type=str, default=None,
                               help="注销模板")
    prompt_parser.add_argument("--delete", type=str, default=None,
                               help="删除模板 (同unregister)")
    prompt_parser.add_argument("--render", type=str, default=None,
                               help="渲染模板 (传入名称)")
    prompt_parser.add_argument("--input", type=str, default=None,
                               help="渲染输入 (JSON字符串或简单字符串)")
    prompt_parser.add_argument("--reset-prompts", action="store_true",
                               help="重置所有模板")
    prompt_parser.set_defaults(func=cmd_prompt)

    # llm-loop命令（阶段60）
    llm_loop_parser = subparsers.add_parser("llm-loop", help="16阶段循环LLM增强管理")
    llm_loop_parser.add_argument("--enable", action="store_true",
                                  help="启用LLM Loop")
    llm_loop_parser.add_argument("--disable", action="store_true",
                                  help="禁用LLM Loop")
    llm_loop_parser.add_argument("--enable-stages", type=str, default=None,
                                  help="启用指定阶段 (逗号分隔)")
    llm_loop_parser.add_argument("--disable-stages", type=str, default=None,
                                  help="禁用指定阶段 (逗号分隔)")
    llm_loop_parser.add_argument("--test", type=str, default=None,
                                  help="测试调用某个阶段 (传入阶段名)")
    llm_loop_parser.add_argument("--input", type=str, default=None,
                                  help="测试输入")
    llm_loop_parser.add_argument("--llm-loop-reset", action="store_true",
                                  help="重置全局LLM Loop")
    llm_loop_parser.set_defaults(func=cmd_llm_loop)

    # memory命令（阶段61）
    memory_parser = subparsers.add_parser("memory", help="LLM增强的记忆与上下文管理")
    memory_parser.add_argument("--add", type=str, default=None,
                                help="添加记忆 (传入内容)")
    memory_parser.add_argument("--type", type=str, default=None,
                                help="记忆类型 (episodic/semantic/procedural/working/sensory)")
    memory_parser.add_argument("--importance", type=str, default=None,
                                help="重要性 (0.0-1.0)")
    memory_parser.add_argument("--search", type=str, default=None,
                                help="搜索记忆")
    memory_parser.add_argument("--context", type=str, default=None,
                                help="构造LLM上下文")
    memory_parser.add_argument("--limit", type=int, default=None,
                                help="返回数量限制")
    memory_parser.add_argument("--clear", action="store_true",
                                help="清空记忆")
    memory_parser.add_argument("--memory-reset", action="store_true",
                                help="重置全局记忆管理器")
    memory_parser.set_defaults(func=cmd_memory)

    # tools 命令 (阶段62)
    tools_parser = subparsers.add_parser("tools", help="工具调用与Function Calling管理")
    tools_parser.add_argument("--list", action="store_true",
                               help="列出所有工具")
    tools_parser.add_argument("--category", type=str, default=None,
                               help="按类别筛选")
    tools_parser.add_argument("--show", type=str, default=None,
                               help="显示工具详情")
    tools_parser.add_argument("--call", type=str, default=None,
                               help="执行工具 (传入工具名)")
    tools_parser.add_argument("--arguments", type=str, default=None,
                               help="工具参数 (JSON格式)")
    tools_parser.add_argument("--ask", type=str, default=None,
                               help="LLM驱动的工具调用 (自然语言提问)")
    tools_parser.add_argument("--max-iterations", type=int, default=None,
                               help="最大迭代轮次")
    tools_parser.add_argument("--no-cache", action="store_true",
                               help="禁用LLM缓存")
    tools_parser.add_argument("--stats", action="store_true",
                               help="显示工具执行统计")
    tools_parser.add_argument("--tools-reset", action="store_true",
                               help="重置全局工具注册表和执行器")
    tools_parser.set_defaults(func=cmd_tools)

    # planner 命令 (阶段63)
    planner_parser = subparsers.add_parser("planner",
                                            help="智能决策与自主规划")
    planner_parser.add_argument("--run", type=str, default=None,
                                 help="运行自主Agent (传入目标)")
    planner_parser.add_argument("--decompose", type=str, default=None,
                                 help="仅分解目标 (不执行)")
    planner_parser.add_argument("--criteria", type=str, default=None,
                                 help="成功标准")
    planner_parser.add_argument("--max-replans", type=int, default=None,
                                 help="最大重规划次数 (默认2)")
    planner_parser.add_argument("--max-iterations", type=int, default=None,
                                 help="最大迭代次数 (默认10)")
    planner_parser.add_argument("--report", action="store_true",
                                 help="显示自主Agent报告")
    planner_parser.add_argument("--history", action="store_true",
                                 help="显示执行历史")
    planner_parser.add_argument("--planner-reset", action="store_true",
                                 help="重置全局规划器/反思器/决策器/Agent")
    planner_parser.set_defaults(func=cmd_planner)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI主入口"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\n已中断")
            return 130
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            if args.verbose if hasattr(args, "verbose") else False:
                import traceback
                traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
