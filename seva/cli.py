"""
CLI命令行工具（阶段30）
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

使用示例:
    seva run --steps 100 --preset easy
    seva config init --preset standard --output config.json
    seva config show --preset easy
    seva status
    seva version
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


def cmd_run(args):
    """seva run: 启动Agent运行"""
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
    else:
        print(f"未知config子命令: {args.config_action}")
        return 1


def _config_init(args):
    """生成配置文件"""
    config = create_config(args.preset)
    config_dict = config.to_dict()

    output = args.output or "seva_config.json"
    with open(output, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"配置文件已生成: {output}")
    print(f"  预设: {args.preset}")
    print(f"  Section数: {len(config_dict)}")
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
