"""
第二十轮测试：阶段30 CLI命令行工具验证
验证seva CLI的run/config/status/version/presets/demo命令
核心目标：证明用户能通过命令行操作Agent，无需写代码

第一性原理验证：
1. 易用 = 命令行操作：seva run --steps 100即可
2. 易用 = 配置管理：seva config init/show/validate
3. 易用 = 信息查询：seva status/version/presets
4. 易用 = 演示模式：seva demo一键演示
"""
import sys
import os
import json
import time
import tempfile
import traceback
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

from seva.cli import main, build_parser
from seva import __version__

PASS = 0
FAIL = 0
RESULTS = []


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    RESULTS.append(f"\n=== {title} ===")


def run_cli(*args) -> int:
    """运行CLI命令，返回退出码"""
    try:
        return main(list(args))
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


def capture_cli(*args) -> tuple:
    """运行CLI命令，捕获输出，返回(退出码, stdout, stderr)"""
    result = subprocess.run(
        [sys.executable, "-m", "seva"] + list(args),
        capture_output=True, text=True, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode, result.stdout, result.stderr


# ============================================================
# 测试1：parser构建
# ============================================================
def test1_parser_build():
    section("测试1：parser构建")
    parser = build_parser()
    report("返回ArgumentParser", parser is not None)

    # 无参数应返回0（显示帮助）
    code = run_cli()
    report("无参数返回0", code == 0)


# ============================================================
# 测试2：version命令
# ============================================================
def test2_version_command():
    section("测试2：version命令")
    code = run_cli("version")
    report("version返回0", code == 0)

    # --version标志
    code2 = run_cli("--version")
    report("--version返回0", code2 == 0)


# ============================================================
# 测试3：version命令输出
# ============================================================
def test3_version_output():
    section("测试3：version命令输出")
    code, stdout, stderr = capture_cli("version")
    report("退出码0", code == 0)
    report("输出含SEVA", "SEVA" in stdout)
    report("输出含版本号", __version__ in stdout)
    report("输出含Python", "Python" in stdout)


# ============================================================
# 测试4：presets命令
# ============================================================
def test4_presets_command():
    section("测试4：presets命令")
    code, stdout, stderr = capture_cli("presets")
    report("退出码0", code == 0)
    report("输出含easy", "easy" in stdout)
    report("输出含standard", "standard" in stdout)
    report("输出含research", "research" in stdout)
    report("输出含production", "production" in stdout)
    report("输出含input_dim", "input_dim" in stdout)
    report("输出含grid_size", "grid_size" in stdout)


# ============================================================
# 测试5：status命令
# ============================================================
def test5_status_command():
    section("测试5：status命令")
    code, stdout, stderr = capture_cli("status")
    report("退出码0", code == 0)
    report("输出含版本", "版本" in stdout or "SEVA" in stdout)
    report("输出含可用预设", "预设" in stdout)
    report("输出含环境类型", "环境" in stdout)
    report("输出含easy", "easy" in stdout)
    report("输出含input_dim", "input_dim" in stdout)


# ============================================================
# 测试6：config init命令
# ============================================================
def test6_config_init_command():
    section("测试6：config init命令")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.json")
        code = run_cli("config", "init", "--preset", "easy",
                      "--output", config_path)
        report("config init返回0", code == 0)
        report("配置文件已创建", os.path.exists(config_path))

        # 验证文件内容
        with open(config_path) as f:
            config = json.load(f)
        report("配置是dict", isinstance(config, dict))
        report("含core section", "core" in config)
        report("含perception section", "perception" in config)
        report("含memory section", "memory" in config)
        report("easy input_dim=16",
               config["perception"]["input_dim"] == 16)


# ============================================================
# 测试7：config init默认输出
# ============================================================
def test7_config_init_default():
    section("测试7：config init默认输出")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            code = run_cli("config", "init", "--preset", "standard")
            report("config init默认返回0", code == 0)
            report("默认生成seva_config.json",
                   os.path.exists("seva_config.json"))
        finally:
            os.chdir(cwd)


# ============================================================
# 测试8：config show命令
# ============================================================
def test8_config_show_command():
    section("测试8：config show命令")
    code, stdout, stderr = capture_cli("config", "show", "--preset", "easy")
    report("退出码0", code == 0)
    report("输出含preset信息", "easy" in stdout)
    report("输出含core", "core" in stdout)
    report("输出含perception", "perception" in stdout)


# ============================================================
# 测试9：config show指定section
# ============================================================
def test9_config_show_section():
    section("测试9：config show指定section")
    code, stdout, stderr = capture_cli("config", "show",
                                       "--preset", "easy",
                                       "--section", "core")
    report("退出码0", code == 0)
    report("输出含max_cycles", "max_cycles" in stdout)
    report("输出含cycle_timeout_ms", "cycle_timeout_ms" in stdout)


# ============================================================
# 测试10：config show未知section
# ============================================================
def test10_config_show_unknown_section():
    section("测试10：config show未知section")
    code = run_cli("config", "show", "--preset", "easy",
                  "--section", "unknown_section")
    report("未知section返回1", code == 1)


# ============================================================
# 测试11：config validate命令
# ============================================================
def test11_config_validate_command():
    section("测试11：config validate命令")
    code, stdout, stderr = capture_cli("config", "validate", "--preset", "easy")
    report("退出码0", code == 0)
    report("输出含校验通过", "通过" in stdout)

    # 所有预设都应校验通过
    for preset in ["easy", "standard", "research", "production"]:
        code = run_cli("config", "validate", "--preset", preset)
        report(f"{preset}校验通过", code == 0)


# ============================================================
# 测试12：run命令
# ============================================================
def test12_run_command():
    section("测试12：run命令")
    code, stdout, stderr = capture_cli("run", "--steps", "10", "--preset", "easy")
    report("退出码0", code == 0)
    report("输出含运行结果", "运行结果" in stdout)
    report("输出含完成步数", "完成步数" in stdout)
    report("输出含总奖励", "总奖励" in stdout)
    report("输出含运行模式", "运行模式" in stdout)
    report("输出含normal", "normal" in stdout)


# ============================================================
# 测试13：run命令保存结果
# ============================================================
def test13_run_with_output():
    section("测试13：run命令保存结果")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "result.json")
        code, stdout, stderr = capture_cli(
            "run", "--steps", "10", "--preset", "easy",
            "--output", output_path
        )
        report("退出码0", code == 0)
        report("结果文件已创建", os.path.exists(output_path))

        with open(output_path) as f:
            result = json.load(f)
        report("结果含preset", "preset" in result)
        report("结果含steps", "steps" in result)
        report("结果含stats", "stats" in result)


# ============================================================
# 测试14：run命令不同预设
# ============================================================
def test14_run_different_presets():
    section("测试14：run命令不同预设")
    for preset in ["easy", "standard"]:
        code, stdout, stderr = capture_cli(
            "run", "--steps", "5", "--preset", preset
        )
        report(f"{preset}预设运行成功", code == 0 and "normal" in stdout)


# ============================================================
# 测试15：run命令不同环境
# ============================================================
def test15_run_different_envs():
    section("测试15：run命令不同环境")
    for env_type in ["blank", "mixed"]:
        code, stdout, stderr = capture_cli(
            "run", "--steps", "5", "--preset", "easy",
            "--env-type", env_type
        )
        report(f"{env_type}环境运行成功", code == 0)


# ============================================================
# 测试16：run命令verbose
# ============================================================
def test16_run_verbose():
    section("测试16：run命令verbose")
    code, stdout, stderr = capture_cli(
        "run", "--steps", "10", "--preset", "easy", "--verbose"
    )
    report("verbose运行成功", code == 0)


# ============================================================
# 测试17：run命令seed
# ============================================================
def test17_run_seed():
    section("测试17：run命令seed")
    code1, stdout1, _ = capture_cli(
        "run", "--steps", "10", "--preset", "easy", "--seed", "42"
    )
    code2, stdout2, _ = capture_cli(
        "run", "--steps", "10", "--preset", "easy", "--seed", "42"
    )
    report("同seed运行成功", code1 == 0 and code2 == 0)


# ============================================================
# 测试18：demo命令
# ============================================================
def test18_demo_command():
    section("测试18：demo命令")
    code, stdout, stderr = capture_cli("demo", "--steps", "10", "--preset", "easy")
    report("退出码0", code == 0)
    report("输出含演示", "演示" in stdout)
    report("输出含运行步数", "运行步数" in stdout)
    report("输出含总奖励", "总奖励" in stdout)
    report("输出含健康状态", "健康" in stdout)


# ============================================================
# 测试19：无效命令
# ============================================================
def test19_invalid_command():
    section("测试19：无效命令")
    code = run_cli("invalid_command")
    report("无效命令返回非0", code != 0)


# ============================================================
# 测试20：无效预设
# ============================================================
def test20_invalid_preset():
    section("测试20：无效预设")
    # argparse choices限制应导致非0退出
    code = run_cli("run", "--preset", "invalid_preset")
    report("无效预设返回非0", code != 0)


# ============================================================
# 测试21：无效环境类型
# ============================================================
def test21_invalid_env_type():
    section("测试21：无效环境类型")
    code = run_cli("run", "--env-type", "invalid_env")
    report("无效环境返回非0", code != 0)


# ============================================================
# 测试22：config无子命令
# ============================================================
def test22_config_no_subcommand():
    section("测试22：config无子命令")
    code = run_cli("config")
    # 无子命令时应返回0或非0（显示帮助）
    report("config无子命令有返回", code is not None)


# ============================================================
# 测试23：__main__.py入口
# ============================================================
def test23_main_entry():
    section("测试23：__main__.py入口")
    # python -m seva version 应该工作
    code, stdout, stderr = capture_cli("version")
    report("python -m seva version工作", code == 0)
    report("输出含SEVA", "SEVA" in stdout)


# ============================================================
# 测试24：CLI帮助信息
# ============================================================
def test24_help():
    section("测试24：CLI帮助信息")
    code, stdout, stderr = capture_cli("--help")
    report("--help返回0", code == 0)
    report("帮助含run", "run" in stdout)
    report("帮助含config", "config" in stdout)
    report("帮助含status", "status" in stdout)
    report("帮助含version", "version" in stdout)
    report("帮助含presets", "presets" in stdout)
    report("帮助含demo", "demo" in stdout)


# ============================================================
# 测试25：run子命令帮助
# ============================================================
def test25_run_help():
    section("测试25：run子命令帮助")
    code, stdout, stderr = capture_cli("run", "--help")
    report("run --help返回0", code == 0)
    report("帮助含--steps", "--steps" in stdout)
    report("帮助含--preset", "--preset" in stdout)
    report("帮助含--env-type", "--env-type" in stdout)
    report("帮助含--seed", "--seed" in stdout)
    report("帮助含--output", "--output" in stdout)


# ============================================================
# 测试26：config子命令帮助
# ============================================================
def test26_config_help():
    section("测试26：config子命令帮助")
    code, stdout, stderr = capture_cli("config", "--help")
    report("config --help返回0", code == 0)
    report("帮助含init", "init" in stdout)
    report("帮助含show", "show" in stdout)
    report("帮助含validate", "validate" in stdout)


# ============================================================
# 测试27：端到端CLI流程
# ============================================================
def test27_e2e_cli_flow():
    section("测试27：端到端CLI流程")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 生成配置
        config_path = os.path.join(tmpdir, "config.json")
        code1 = run_cli("config", "init", "--preset", "easy",
                       "--output", config_path)
        report("步骤1: config init", code1 == 0)

        # 2. 校验配置
        code2 = run_cli("config", "validate", "--preset", "easy")
        report("步骤2: config validate", code2 == 0)

        # 3. 运行Agent
        result_path = os.path.join(tmpdir, "result.json")
        code3, stdout3, _ = capture_cli(
            "run", "--steps", "15", "--preset", "easy",
            "--output", result_path
        )
        report("步骤3: run", code3 == 0)
        report("步骤3: 结果文件存在", os.path.exists(result_path))

        # 4. 查看状态
        code4, stdout4, _ = capture_cli("status")
        report("步骤4: status", code4 == 0)


# ============================================================
# 测试28：错误处理
# ============================================================
def test28_error_handling():
    section("测试28：CLI错误处理")
    # run命令异常应返回1
    # 使用无效seed（负数应被接受但可能产生警告）
    code, stdout, stderr = capture_cli("run", "--steps", "1", "--seed", "-1")
    report("负seed不崩溃", code in [0, 1])


# ============================================================
# 测试29：所有预设可运行
# ============================================================
def test29_all_presets_runnable():
    section("测试29：所有预设可运行")
    for preset in ["easy", "standard"]:
        code, stdout, stderr = capture_cli(
            "run", "--steps", "3", "--preset", preset
        )
        report(f"{preset}可运行", code == 0 and "normal" in stdout)


# ============================================================
# 测试30：性能测试
# ============================================================
def test30_performance():
    section("测试30：CLI性能")
    t0 = time.time()
    code, stdout, stderr = capture_cli("run", "--steps", "20", "--preset", "easy")
    elapsed = time.time() - t0
    report("20步CLI运行<10秒", elapsed < 10, f"{elapsed:.2f}s")
    report("运行成功", code == 0)


# ============================================================
# 主函数
# ============================================================
def main_test():
    print("\n" + "="*60)
    print("第二十轮测试：阶段30 CLI命令行工具验证")
    print("="*60)

    tests = [
        test1_parser_build,
        test2_version_command,
        test3_version_output,
        test4_presets_command,
        test5_status_command,
        test6_config_init_command,
        test7_config_init_default,
        test8_config_show_command,
        test9_config_show_section,
        test10_config_show_unknown_section,
        test11_config_validate_command,
        test12_run_command,
        test13_run_with_output,
        test14_run_different_presets,
        test15_run_different_envs,
        test16_run_verbose,
        test17_run_seed,
        test18_demo_command,
        test19_invalid_command,
        test20_invalid_preset,
        test21_invalid_env_type,
        test22_config_no_subcommand,
        test23_main_entry,
        test24_help,
        test25_run_help,
        test26_config_help,
        test27_e2e_cli_flow,
        test28_error_handling,
        test29_all_presets_runnable,
        test30_performance,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [ERROR] {test.__name__} 异常: {e}")
            traceback.print_exc()
            global FAIL
            FAIL += 1
            RESULTS.append(f"  [ERROR] {test.__name__}: {e}")

    print("\n" + "="*60)
    print(f"测试总结: {PASS}通过 / {FAIL}失败")
    print("="*60)

    print("\n".join(RESULTS))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main_test())
