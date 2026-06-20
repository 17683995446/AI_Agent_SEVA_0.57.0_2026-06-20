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
