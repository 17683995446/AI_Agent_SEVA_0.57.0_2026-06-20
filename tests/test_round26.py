"""
阶段36测试：交互式REPL Shell
=============================

验证交互式Shell模块的完整功能：
1. SevaShell类
2. run_shell函数
3. execute_command函数（非交互式）
4. 各命令功能
5. CLI shell/exec命令

对应第一性原理：易用 = 探索式使用，无需写脚本
"""
import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import SevaShell, run_shell, execute_command


# ============================================================
# 测试1-5：SevaShell类基础
# ============================================================
def test1_shell_creation():
    """测试1: SevaShell可创建"""
    shell = SevaShell()
    assert shell is not None
    assert shell.agent is not None
    assert shell.dashboard is not None


def test2_shell_with_preset():
    """测试2: 指定预设创建"""
    shell = SevaShell(preset="easy")
    assert shell.preset == "easy"


def test3_shell_with_env_type():
    """测试3: 指定环境创建"""
    shell = SevaShell(env_type="drift")
    assert shell.env_type == "drift"


def test4_shell_with_seed():
    """测试4: 指定种子创建"""
    shell = SevaShell(seed=123)
    assert shell.seed == 123


def test5_shell_prompt():
    """测试5: Shell提示符"""
    shell = SevaShell()
    assert shell.prompt == "seva> "


# ============================================================
# 测试6-10：execute_command基础
# ============================================================
def test6_execute_command_returns_string():
    """测试6: execute_command返回字符串"""
    output = execute_command("status")
    assert isinstance(output, str)


def test7_execute_command_empty():
    """测试7: 空命令返回空字符串"""
    output = execute_command("")
    assert output == ""


def test8_execute_command_unknown():
    """测试8: 未知命令"""
    output = execute_command("unknown_command")
    assert "未知命令" in output


def test9_execute_command_no_args():
    """测试9: 无参数命令"""
    output = execute_command("version")
    assert "SEVA" in output


def test10_execute_command_with_args():
    """测试10: 带参数命令"""
    output = execute_command("run 5")
    assert "运行" in output or "完成" in output


# ============================================================
# 测试11-20：各命令功能
# ============================================================
def test11_command_help():
    """测试11: help命令"""
    output = execute_command("help")
    assert "可用命令" in output
    assert "status" in output
    assert "run" in output


def test12_command_help_specific():
    """测试12: help <command>"""
    output = execute_command("help run")
    assert "run" in output


def test13_command_status():
    """测试13: status命令"""
    output = execute_command("status")
    assert "Agent状态" in output
    assert "cycle_count" in output
    assert "mode" in output


def test14_command_run():
    """测试14: run命令"""
    output = execute_command("run 10")
    assert "运行" in output
    assert "完成步数" in output


def test15_command_run_invalid():
    """测试15: run无效步数"""
    output = execute_command("run abc")
    assert "无效" in output


def test16_command_preset():
    """测试16: preset命令"""
    output = execute_command("preset")
    assert "当前预设" in output or "easy" in output


def test17_command_preset_switch():
    """测试17: preset切换"""
    output = execute_command("preset standard")
    assert "standard" in output


def test18_command_preset_invalid():
    """测试18: preset无效"""
    output = execute_command("preset invalid")
    assert "未知" in output


def test19_command_env():
    """测试19: env命令"""
    output = execute_command("env")
    assert "当前环境" in output or "mixed" in output


def test20_command_env_switch():
    """测试20: env切换"""
    output = execute_command("env drift")
    assert "drift" in output


# ============================================================
# 测试21-30：更多命令
# ============================================================
def test21_command_scenarios():
    """测试21: scenarios命令"""
    output = execute_command("scenarios")
    assert "预设场景" in output
    assert "quick_validation" in output


def test22_command_scenario():
    """测试22: scenario命令"""
    output = execute_command("scenario quick_validation")
    assert "运行场景" in output or "结果" in output


def test23_command_scenario_invalid():
    """测试23: scenario无效场景"""
    output = execute_command("scenario invalid")
    assert "未知" in output


def test24_command_config_show():
    """测试24: config show命令"""
    output = execute_command("config show")
    assert "配置" in output or "core" in output


def test25_command_config_show_section():
    """测试25: config show <section>"""
    output = execute_command("config show core")
    assert "max_cycles" in output


def test26_command_config_no_subcommand():
    """测试26: config无子命令"""
    output = execute_command("config")
    assert "用法" in output


def test27_command_diagnose():
    """测试27: diagnose命令"""
    output = execute_command("diagnose")
    # diagnose输出可能较长
    assert len(output) > 0


def test28_command_monitor():
    """测试28: monitor命令"""
    output = execute_command("monitor")
    assert len(output) > 0


def test29_command_metrics():
    """测试29: metrics命令"""
    output = execute_command("metrics")
    assert "运行指标" in output or "周期" in output


def test30_command_version():
    """测试30: version命令"""
    output = execute_command("version")
    assert "SEVA" in output
    assert "0.36" in output


# ============================================================
# 测试31-35：config save/load
# ============================================================
def test31_command_config_save():
    """测试31: config save命令"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shell_cfg.yaml")
        output = execute_command(f"config save {path}")
        assert "保存" in output or "已" in output
        assert os.path.exists(path)


def test32_command_config_load():
    """测试32: config load命令"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # 先保存
        path = os.path.join(tmpdir, "shell_cfg.yaml")
        execute_command(f"config save {path}")
        # 再加载
        output = execute_command(f"config load {path}")
        assert "加载" in output or "已" in output


def test33_command_config_load_not_found():
    """测试33: config load不存在文件"""
    output = execute_command("config load /nonexistent/cfg.yaml")
    assert "失败" in output or "错误" in output


def test34_command_config_save_no_file():
    """测试34: config save无文件参数"""
    output = execute_command("config save")
    assert "用法" in output


def test35_command_config_load_no_file():
    """测试35: config load无文件参数"""
    output = execute_command("config load")
    assert "用法" in output


# ============================================================
# 测试36-40：reset和组合命令
# ============================================================
def test36_command_reset():
    """测试36: reset命令"""
    # 先运行几步
    execute_command("run 5")
    # 重置
    output = execute_command("reset")
    assert "重置" in output


def test37_run_then_status():
    """测试37: 运行后查看状态"""
    # 注意：execute_command每次创建新Shell
    # 这里测试单个Shell内的连续操作
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    shell = SevaShell(stdout=output)
    with redirect_stdout(output):
        shell.do_run("10")
        shell.do_status("")
    result = output.getvalue()
    assert "运行" in result
    assert "Agent状态" in result


def test38_multiple_commands_in_shell():
    """测试38: Shell内执行多个命令"""
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    shell = SevaShell(stdout=output)
    with redirect_stdout(output):
        shell.do_version("")
        shell.do_help("")
        shell.do_status("")
    result = output.getvalue()
    assert "SEVA" in result
    assert "可用命令" in result
    assert "Agent状态" in result


def test39_shell_emptyline():
    """测试39: 空行不做任何事"""
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    shell = SevaShell(stdout=output)
    with redirect_stdout(output):
        shell.emptyline()
    # 空行不应产生输出
    assert output.getvalue() == ""


def test40_shell_default_handler():
    """测试40: 未知命令处理"""
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    shell = SevaShell(stdout=output)
    with redirect_stdout(output):
        shell.default("nonexistent")
    result = output.getvalue()
    assert "未知命令" in result


# ============================================================
# 测试41-50：CLI命令
# ============================================================
def run_cli(*args):
    """运行CLI命令"""
    cmd = [sys.executable, "-m", "seva"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


def test41_cli_exec_status():
    """测试41: seva exec status"""
    code, out, err = run_cli("exec", "status")
    assert code == 0
    assert "Agent状态" in out


def test42_cli_exec_run():
    """测试42: seva exec run"""
    code, out, err = run_cli("exec", "run", "5")
    assert code == 0
    assert "运行" in out


def test43_cli_exec_help():
    """测试43: seva exec help"""
    code, out, err = run_cli("exec", "help")
    assert code == 0
    assert "可用命令" in out


def test44_cli_exec_version():
    """测试44: seva exec version"""
    code, out, err = run_cli("exec", "version")
    assert code == 0
    assert "SEVA" in out


def test45_cli_exec_scenarios():
    """测试45: seva exec scenarios"""
    code, out, err = run_cli("exec", "scenarios")
    assert code == 0
    assert "预设场景" in out


def test46_cli_exec_no_command():
    """测试46: seva exec无命令"""
    code, out, err = run_cli("exec")
    # argparse在无必需参数时打印帮助，返回0或2
    assert "exec" in out or "usage" in out.lower()


def test47_cli_exec_metrics():
    """测试47: seva exec metrics"""
    code, out, err = run_cli("exec", "metrics")
    assert code == 0
    assert "指标" in out or "周期" in out


def test48_cli_exec_with_preset():
    """测试48: seva exec --preset"""
    code, out, err = run_cli("exec", "status", "--preset", "standard")
    assert code == 0
    assert "Agent状态" in out


def test49_cli_exec_scenario():
    """测试49: seva exec scenario"""
    code, out, err = run_cli("exec", "scenario", "quick_validation")
    assert code == 0
    assert "运行场景" in out or "结果" in out


def test50_cli_exec_quoted_command():
    """测试50: seva exec带引号命令"""
    code, out, err = run_cli("exec", "run", "10")
    assert code == 0
    assert "完成步数" in out


# ============================================================
# 测试51-55：模块导出与集成
# ============================================================
def test51_module_import():
    """测试51: 模块可导入"""
    from seva import shell
    assert hasattr(shell, "SevaShell")
    assert hasattr(shell, "run_shell")
    assert hasattr(shell, "execute_command")


def test52_all_exports_available():
    """测试52: 所有导出可用"""
    from seva import SevaShell, run_shell, execute_command
    assert SevaShell is not None
    assert callable(run_shell)
    assert callable(execute_command)


def test53_exports_in_all():
    """测试53: 导出在__all__中"""
    import seva
    assert "SevaShell" in seva.__all__
    assert "run_shell" in seva.__all__
    assert "execute_command" in seva.__all__


def test54_version_updated():
    """测试54: 版本号已更新"""
    import seva
    assert seva.__version__ == "0.36.0"


def test55_shell_inherits_cmd():
    """测试55: SevaShell继承cmd.Cmd"""
    import cmd as cmd_module
    assert issubclass(SevaShell, cmd_module.Cmd)


# ============================================================
# 测试56-60：Shell命令完整性
# ============================================================
def test56_all_commands_exist():
    """测试56: 所有命令方法存在"""
    shell = SevaShell()
    commands = ["help", "status", "run", "preset", "env",
                "scenarios", "scenario", "config", "diagnose",
                "monitor", "metrics", "reset", "version",
                "exit", "quit"]
    for cmd in commands:
        method = getattr(shell, f"do_{cmd}", None)
        assert method is not None, f"缺少命令: do_{cmd}"


def test57_exit_returns_true():
    """测试57: exit返回True（退出信号）"""
    shell = SevaShell()
    result = shell.do_exit("")
    assert result is True


def test58_quit_returns_true():
    """测试58: quit返回True"""
    shell = SevaShell()
    result = shell.do_quit("")
    assert result is True


def test59_eof_returns_true():
    """测试59: EOF返回True"""
    shell = SevaShell()
    result = shell.do_EOF("")
    assert result is True


def test60_shell_commands_have_docs():
    """测试60: 所有命令有文档字符串"""
    shell = SevaShell()
    commands = ["help", "status", "run", "preset", "env",
                "scenarios", "scenario", "config", "diagnose",
                "monitor", "metrics", "reset", "version"]
    for cmd in commands:
        method = getattr(shell, f"do_{cmd}", None)
        assert method is not None
        assert method.__doc__ is not None, f"do_{cmd}缺少文档字符串"


# ============================================================
# 性能测试
# ============================================================
def test61_execute_command_fast():
    """测试61: execute_command应快速"""
    import time
    start = time.time()
    execute_command("status")
    elapsed = time.time() - start
    assert elapsed < 5.0


def test62_shell_creation_fast():
    """测试62: Shell创建应快速"""
    import time
    start = time.time()
    SevaShell()
    elapsed = time.time() - start
    assert elapsed < 5.0


def test63_run_command_fast():
    """测试63: run命令应快速"""
    import time
    start = time.time()
    execute_command("run 5")
    elapsed = time.time() - start
    assert elapsed < 10.0


# ============================================================
# 完整性测试
# ============================================================
def test64_complete_shell_system():
    """测试64: Shell系统完整性"""
    # 1. 创建Shell
    shell = SevaShell()
    assert shell is not None

    # 2. 执行命令
    output = execute_command("version")
    assert "SEVA" in output

    # 3. CLI可用
    code, out, err = run_cli("exec", "status")
    assert code == 0


def test65_shell_user_friendly():
    """测试65: Shell用户友好性"""
    # help输出应包含中文说明
    output = execute_command("help")
    assert "显示帮助" in output or "查看" in output
    assert "运行" in output

    # 未知命令应有友好提示
    output = execute_command("unknown")
    assert "未知命令" in output
    assert "help" in output  # 提示用help


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
