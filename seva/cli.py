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
