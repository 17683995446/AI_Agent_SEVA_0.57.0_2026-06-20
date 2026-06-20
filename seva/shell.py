"""
交互式REPL Shell模块（阶段36）
对应第一性原理：易用 = 探索式使用，无需写脚本

提供：
- SevaShell: 交互式Shell（支持命令补全、历史记录）
- run_shell(): 启动交互式Shell
- ShellCommand: 命令定义基类

支持的命令:
    help              显示帮助
    status            查看Agent状态
    run <steps>       运行指定步数
    preset <name>     切换预设
    env <type>        切换环境
    scenarios         列出场景
    scenario <name>   运行场景
    config show       显示配置
    config save <f>   保存配置
    config load <f>   加载配置
    diagnose          诊断Agent
    monitor           显示监控
    metrics           显示指标
    reset             重置Agent
    exit/quit         退出

使用示例:
    from seva import run_shell
    run_shell()  # 启动交互式Shell
"""
import sys
import os
import cmd as cmd_module
from typing import Any, Dict, List, Optional

from .api import SEVAFactory, run, get_status, PresetConfigs
from .diagnostics import Diagnoser, diagnose_agent, check_common_issues
from .monitoring import Dashboard, MetricsCollector
from .scenarios import (PresetScenarios, run_scenario, list_scenarios,
                        list_scenarios_detailed)
from .config_io import save_config, load_config, config_from_file


class SevaShell(cmd_module.Cmd):
    """
    SEVA交互式Shell

    使用示例:
        shell = SevaShell()
        shell.cmdloop()
    """

    intro = """
============================================================
SEVA交互式Shell
自驱永续进化预测修正Agent系统
输入 'help' 查看可用命令，'exit' 退出
============================================================
"""

    prompt = "seva> "

    def __init__(self, preset: str = "easy", env_type: str = "mixed",
                 seed: int = 42, stdout=None):
        super().__init__(stdout=stdout or sys.stdout)
        self.preset = preset
        self.env_type = env_type
        self.seed = seed
        self.agent = None
        self.dashboard = None
        self._create_agent()

    def _create_agent(self):
        """创建Agent"""
        try:
            self.agent = SEVAFactory.create(
                preset=self.preset,
                env_type=self.env_type,
                seed=self.seed,
            )
            self.dashboard = Dashboard(self.agent)
        except Exception as e:
            print(f"创建Agent失败: {e}")

    # ============================================================
    # help命令
    # ============================================================
    def do_help(self, arg):
        """显示帮助: help [命令]"""
        if arg:
            # 显示特定命令帮助
            method = getattr(self, f"do_{arg}", None)
            if method and method.__doc__:
                print(f"\n{arg}:")
                print(f"  {method.__doc__}")
            else:
                print(f"未知命令: {arg}")
        else:
            print("\n可用命令:")
            print("-" * 50)
            commands = [
                ("help", "显示帮助"),
                ("status", "查看Agent状态"),
                ("run <steps>", "运行指定步数"),
                ("preset <name>", "切换预设"),
                ("env <type>", "切换环境类型"),
                ("scenarios", "列出预设场景"),
                ("scenario <name>", "运行指定场景"),
                ("config show", "显示当前配置"),
                ("config save <file>", "保存配置到文件"),
                ("config load <file>", "从文件加载配置"),
                ("diagnose", "诊断Agent"),
                ("monitor", "显示监控仪表盘"),
                ("metrics", "显示运行指标"),
                ("reset", "重置Agent"),
                ("version", "显示版本信息"),
                ("exit/quit", "退出Shell"),
            ]
            for cmd, desc in commands:
                print(f"  {cmd:<20} {desc}")
            print()

    # ============================================================
    # status命令
    # ============================================================
    def do_status(self, arg):
        """查看Agent状态: status"""
        if not self.agent:
            print("Agent未创建")
            return
        status = get_status(self.agent)
        print("\nAgent状态:")
        print("-" * 40)
        for key, value in status.items():
            print(f"  {key:<25} {value}")

    # ============================================================
    # run命令
    # ============================================================
    def do_run(self, arg):
        """运行指定步数: run <steps>"""
        if not self.agent:
            print("Agent未创建")
            return
        try:
            steps = int(arg) if arg else 10
        except ValueError:
            print(f"无效步数: {arg}")
            return

        print(f"运行 {steps} 步...")
        stats = run(self.agent, steps=steps, verbose=False)
        print(f"\n运行结果:")
        print(f"  完成步数:     {stats['steps_completed']}")
        print(f"  总奖励:       {stats['total_real_reward']:.4f}")
        print(f"  平均周期:     {stats['avg_cycle_time']*1000:.2f}ms")
        print(f"  预测误差:     {stats['avg_prediction_error']:.4f}")

    # ============================================================
    # preset命令
    # ============================================================
    def do_preset(self, arg):
        """切换预设: preset <name>"""
        if not arg:
            print(f"当前预设: {self.preset}")
            print(f"可用预设: {PresetConfigs.list_presets()}")
            return
        if arg not in PresetConfigs.list_presets():
            print(f"未知预设: {arg}")
            print(f"可用: {PresetConfigs.list_presets()}")
            return
        self.preset = arg
        self._create_agent()
        print(f"已切换到预设: {arg}")

    # ============================================================
    # env命令
    # ============================================================
    def do_env(self, arg):
        """切换环境类型: env <type>"""
        if not arg:
            print(f"当前环境: {self.env_type}")
            print(f"可用: blank/mixed/drift/adversarial")
            return
        if arg not in ["blank", "mixed", "drift", "adversarial"]:
            print(f"未知环境: {arg}")
            return
        self.env_type = arg
        self._create_agent()
        print(f"已切换到环境: {arg}")

    # ============================================================
    # scenarios命令
    # ============================================================
    def do_scenarios(self, arg):
        """列出预设场景: scenarios"""
        scenarios = list_scenarios_detailed()
        print(f"\n预设场景 ({len(scenarios)}个):")
        print("-" * 50)
        for s in scenarios:
            print(f"  {s['name']}")
            print(f"    {s['description']}")

    # ============================================================
    # scenario命令
    # ============================================================
    def do_scenario(self, arg):
        """运行指定场景: scenario <name>"""
        if not arg:
            print("用法: scenario <name>")
            print(f"可用场景: {list_scenarios()}")
            return
        if arg not in list_scenarios():
            print(f"未知场景: {arg}")
            return
        print(f"运行场景: {arg}")
        result = run_scenario(arg, verbose=False)
        print(f"\n结果:")
        print(f"  完成步数:     {result['steps_completed']}")
        print(f"  总奖励:       {result['total_reward']:.4f}")
        print(f"  运行时间:     {result['elapsed_time']:.2f}s")
        print(f"  通过:         {'是' if result['passed'] else '否'}")

    # ============================================================
    # config命令
    # ============================================================
    def do_config(self, arg):
        """配置管理: config show|save <file>|load <file>"""
        parts = arg.split() if arg else []
        if not parts:
            print("用法: config show|save <file>|load <file>")
            return

        action = parts[0]
        if action == "show":
            self._config_show(parts[1:] if len(parts) > 1 else None)
        elif action == "save":
            if len(parts) < 2:
                print("用法: config save <file>")
                return
            self._config_save(parts[1])
        elif action == "load":
            if len(parts) < 2:
                print("用法: config load <file>")
                return
            self._config_load(parts[1])
        else:
            print(f"未知config子命令: {action}")

    def _config_show(self, section):
        if not self.agent:
            print("Agent未创建")
            return
        config_dict = self.agent.config.to_dict()
        if section:
            s = section[0]
            if s in config_dict:
                print(f"\n[{s}]")
                for k, v in config_dict[s].items():
                    print(f"  {k} = {v}")
            else:
                print(f"未知section: {s}")
        else:
            print("\n当前配置:")
            for sec, items in config_dict.items():
                if sec == "_meta":
                    continue
                print(f"\n[{sec}]")
                for k, v in items.items():
                    print(f"  {k} = {v}")

    def _config_save(self, filepath):
        if not self.agent:
            print("Agent未创建")
            return
        try:
            save_config(self.agent.config, filepath)
            print(f"配置已保存到: {filepath}")
        except Exception as e:
            print(f"保存失败: {e}")

    def _config_load(self, filepath):
        try:
            self.agent = config_from_file(filepath,
                                          env_type=self.env_type,
                                          seed=self.seed)
            self.dashboard = Dashboard(self.agent)
            print(f"已从 {filepath} 加载配置")
        except Exception as e:
            print(f"加载失败: {e}")

    # ============================================================
    # diagnose命令
    # ============================================================
    def do_diagnose(self, arg):
        """诊断Agent: diagnose"""
        if not self.agent:
            print("Agent未创建")
            return
        report = diagnose_agent(self.agent)
        print(report)

    # ============================================================
    # monitor命令
    # ============================================================
    def do_monitor(self, arg):
        """显示监控仪表盘: monitor"""
        if not self.agent or not self.dashboard:
            print("Agent未创建")
            return
        self.dashboard.collect_from_agent()
        print(self.dashboard.render(compact=False))

    # ============================================================
    # metrics命令
    # ============================================================
    def do_metrics(self, arg):
        """显示运行指标: metrics"""
        if not self.agent:
            print("Agent未创建")
            return
        metrics = self.agent.get_metrics()
        print("\n运行指标:")
        print("-" * 40)
        print(f"  周期数:         {metrics['cycle_count']}")
        print(f"  总奖励:         {metrics['total_reward']:.4f}")
        print(f"  平均预测误差:   {metrics['avg_prediction_error']:.4f}")
        print(f"  平均周期时间:   {metrics['avg_cycle_time']*1000:.2f}ms")
        print(f"  运行模式:       {metrics['mode']}")
        task_stats = metrics.get("task_stats", {})
        skill_stats = metrics.get("skill_stats", {})
        evo_stats = metrics.get("evolution_stats", {})
        print(f"  任务完成:       {task_stats.get('total_completed', 0)}")
        print(f"  技能总数:       {skill_stats.get('total_skills', 0)}")
        print(f"  进化评估:       {evo_stats.get('total_evaluations', 0)}")

    # ============================================================
    # reset命令
    # ============================================================
    def do_reset(self, arg):
        """重置Agent: reset"""
        if not self.agent:
            print("Agent未创建")
            return
        self.agent.reset()
        self.dashboard = Dashboard(self.agent)
        print("Agent已重置")

    # ============================================================
    # version命令
    # ============================================================
    def do_version(self, arg):
        """显示版本信息: version"""
        from . import __version__
        print(f"SEVA {__version__}")
        print(f"自驱永续进化预测修正Agent系统")

    # ============================================================
    # exit/quit命令
    # ============================================================
    def do_exit(self, arg):
        """退出Shell: exit"""
        print("再见！")
        return True

    def do_quit(self, arg):
        """退出Shell: quit"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Ctrl-D退出"""
        print()
        return self.do_exit(arg)

    # ============================================================
    # 空行处理
    # ============================================================
    def emptyline(self):
        """空行不做任何事"""
        pass

    # ============================================================
    # 默认错误处理
    # ============================================================
    def default(self, line):
        """未知命令"""
        print(f"未知命令: {line}")
        print("输入 'help' 查看可用命令")


def run_shell(preset: str = "easy", env_type: str = "mixed",
              seed: int = 42):
    """
    启动SEVA交互式Shell

    参数:
        preset: 初始预设模板
        env_type: 初始环境类型
        seed: 随机种子

    示例:
        from seva import run_shell
        run_shell()
    """
    shell = SevaShell(preset=preset, env_type=env_type, seed=seed)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n再见！")
        return 0
    return 0


def execute_command(command: str, preset: str = "easy",
                    env_type: str = "mixed",
                    seed: int = 42) -> str:
    """
    执行单条Shell命令（非交互式，用于测试和脚本）

    参数:
        command: 命令字符串（如 "status", "run 10"）
        preset: 预设模板
        env_type: 环境类型
        seed: 随机种子

    返回:
        命令输出字符串

    示例:
        output = execute_command("status")
        output = execute_command("run 10")
    """
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    shell = SevaShell(preset=preset, env_type=env_type, seed=seed,
                      stdout=output)

    # 解析命令
    parts = command.strip().split()
    if not parts:
        return ""

    cmd = parts[0]
    arg = " ".join(parts[1:]) if len(parts) > 1 else ""

    method = getattr(shell, f"do_{cmd}", None)
    if method:
        with redirect_stdout(output):
            method(arg)
    else:
        with redirect_stdout(output):
            shell.default(command)

    return output.getvalue()
