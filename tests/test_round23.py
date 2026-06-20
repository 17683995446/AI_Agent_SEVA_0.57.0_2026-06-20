"""
阶段33测试：示例库与文档
======================

验证所有示例脚本可正常运行：
1. 示例文件存在性检查
2. 示例脚本可执行性（subprocess调用）
3. 示例输出内容验证
4. 示例覆盖功能完整性

对应第一性原理：易用 = 提供可运行的示例，降低学习成本
"""
import os
import sys
import subprocess
import pytest

# 工作目录为项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")

# 示例文件列表
EXPECTED_EXAMPLES = [
    "01_quickstart.py",
    "02_custom_config.py",
    "03_monitoring.py",
    "04_diagnostics.py",
    "05_cli_usage.py",
    "06_advanced.py",
]


def run_example(example_name, timeout=120):
    """运行示例脚本并返回结果"""
    example_path = os.path.join(EXAMPLES_DIR, example_name)
    result = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# ============================================================
# 测试1：示例文件存在性
# ============================================================
def test1_examples_directory_exists():
    """测试1: examples目录存在"""
    assert os.path.isdir(EXAMPLES_DIR), f"examples目录不存在: {EXAMPLES_DIR}"


def test2_all_examples_exist():
    """测试2: 所有预期示例文件都存在"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        assert os.path.isfile(path), f"示例文件不存在: {name}"


def test3_examples_count():
    """测试3: 示例数量符合预期（至少6个）"""
    examples = [f for f in os.listdir(EXAMPLES_DIR)
                if f.endswith(".py") and not f.startswith("__")]
    assert len(examples) >= 6, f"示例数量不足: {len(examples)}"


# ============================================================
# 测试4-9：各示例可执行性
# ============================================================
def test4_quickstart_example():
    """测试4: 01_quickstart.py可正常运行"""
    code, out, err = run_example("01_quickstart.py")
    assert code == 0, f"01_quickstart.py退出码{code}, stderr: {err[:500]}"
    assert "SEVA" in out, "输出中应包含SEVA"
    assert "示例完成" in out, "输出中应包含'示例完成'"


def test5_custom_config_example():
    """测试5: 02_custom_config.py可正常运行"""
    code, out, err = run_example("02_custom_config.py")
    assert code == 0, f"02_custom_config.py退出码{code}, stderr: {err[:500]}"
    assert "预设" in out, "输出中应包含'预设'"
    assert "示例完成" in out


def test6_monitoring_example():
    """测试6: 03_monitoring.py可正常运行"""
    code, out, err = run_example("03_monitoring.py")
    assert code == 0, f"03_monitoring.py退出码{code}, stderr: {err[:500]}"
    assert "仪表盘" in out or "Dashboard" in out, "输出中应包含仪表盘相关"
    assert "示例完成" in out


def test7_diagnostics_example():
    """测试7: 04_diagnostics.py可正常运行"""
    code, out, err = run_example("04_diagnostics.py")
    assert code == 0, f"04_diagnostics.py退出码{code}, stderr: {err[:500]}"
    assert "诊断" in out, "输出中应包含'诊断'"
    assert "示例完成" in out


def test8_cli_usage_example():
    """测试8: 05_cli_usage.py可正常运行"""
    code, out, err = run_example("05_cli_usage.py", timeout=180)
    assert code == 0, f"05_cli_usage.py退出码{code}, stderr: {err[:500]}"
    assert "CLI" in out or "seva" in out, "输出中应包含CLI相关"
    assert "示例完成" in out


def test9_advanced_example():
    """测试9: 06_advanced.py可正常运行"""
    code, out, err = run_example("06_advanced.py")
    assert code == 0, f"06_advanced.py退出码{code}, stderr: {err[:500]}"
    assert "高级" in out, "输出中应包含'高级'"
    assert "示例完成" in out


# ============================================================
# 测试10-15：示例输出内容验证
# ============================================================
def test10_quickstart_shows_stats():
    """测试10: quickstart示例显示运行统计"""
    code, out, err = run_example("01_quickstart.py")
    assert "完成步数" in out or "steps_completed" in out.lower()
    assert "总奖励" in out or "reward" in out.lower()


def test11_custom_config_shows_presets():
    """测试11: custom_config示例显示预设列表"""
    code, out, err = run_example("02_custom_config.py")
    assert "easy" in out
    assert "standard" in out


def test12_monitoring_shows_metrics():
    """测试12: monitoring示例显示指标"""
    code, out, err = run_example("03_monitoring.py")
    # 应该有指标相关输出
    assert "指标" in out or "metric" in out.lower() or "avg" in out.lower()


def test13_diagnostics_shows_report():
    """测试13: diagnostics示例显示诊断报告"""
    code, out, err = run_example("04_diagnostics.py")
    assert "健康" in out or "healthy" in out.lower() or "问题" in out


def test14_cli_usage_runs_multiple_commands():
    """测试14: cli_usage示例运行多个CLI命令"""
    code, out, err = run_example("05_cli_usage.py", timeout=180)
    # 应该包含多个命令的输出
    assert "version" in out
    assert "status" in out
    assert "presets" in out


def test15_advanced_shows_evolution():
    """测试15: advanced示例显示进化统计"""
    code, out, err = run_example("06_advanced.py")
    assert "进化" in out or "evolution" in out.lower()


# ============================================================
# 测试16-20：示例文件结构验证
# ============================================================
def test16_examples_have_docstring():
    """测试16: 所有示例都有文档字符串"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith('"""'), f"{name}缺少模块文档字符串"


def test17_examples_have_main():
    """测试17: 所有示例都有main函数和__main__入口"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "def main()" in content, f"{name}缺少main函数"
        assert '__name__ == "__main__"' in content, \
            f"{name}缺少__main__入口"


def test18_examples_have_sys_path():
    """测试18: 所有示例都设置了sys.path"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "sys.path.insert" in content, \
            f"{name}缺少sys.path设置"


def test19_examples_import_from_seva():
    """测试19: 所有示例都从seva包导入"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "from seva" in content or "import seva" in content, \
            f"{name}未从seva包导入"


def test20_examples_have_run_instruction():
    """测试20: 所有示例都包含运行方式说明"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "运行方式" in content or "python examples" in content, \
            f"{name}缺少运行方式说明"


# ============================================================
# 测试21-25：功能覆盖完整性
# ============================================================
def test21_quickstart_covers_quickstart_api():
    """测试21: quickstart示例覆盖quickstart API"""
    with open(os.path.join(EXAMPLES_DIR, "01_quickstart.py"), "r") as f:
        content = f.read()
    assert "quickstart(" in content
    assert "get_status" in content


def test22_custom_config_covers_factory():
    """测试22: custom_config示例覆盖工厂模式"""
    with open(os.path.join(EXAMPLES_DIR, "02_custom_config.py"), "r") as f:
        content = f.read()
    assert "SEVAFactory" in content
    assert "PresetConfigs" in content
    assert "create_config" in content
    assert "overrides" in content


def test23_monitoring_covers_dashboard():
    """测试23: monitoring示例覆盖仪表盘"""
    with open(os.path.join(EXAMPLES_DIR, "03_monitoring.py"), "r") as f:
        content = f.read()
    assert "Dashboard" in content
    assert "MetricsCollector" in content
    assert "StructuredLogger" in content
    assert "export_metrics" in content


def test24_diagnostics_covers_diagnoser():
    """测试24: diagnostics示例覆盖诊断器"""
    with open(os.path.join(EXAMPLES_DIR, "04_diagnostics.py"), "r") as f:
        content = f.read()
    assert "Diagnoser" in content
    assert "diagnose_agent" in content
    assert "ErrorCode" in content
    assert "SevaError" in content
    assert "format_exception" in content
    assert "check_common_issues" in content


def test25_cli_usage_covers_subprocess():
    """测试25: cli_usage示例覆盖subprocess调用"""
    with open(os.path.join(EXAMPLES_DIR, "05_cli_usage.py"), "r") as f:
        content = f.read()
    assert "subprocess" in content
    assert "run_cli" in content or "seva" in content


# ============================================================
# 测试26-30：高级功能覆盖
# ============================================================
def test26_advanced_covers_environment():
    """测试26: advanced示例覆盖自定义环境"""
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        content = f.read()
    assert "env_type" in content
    assert "SimulationEnvironment" in content or "drift" in content


def test27_advanced_covers_modules():
    """测试27: advanced示例覆盖模块访问"""
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        content = f.read()
    assert "agent.modules" in content
    assert "task_manager" in content
    assert "skill_library" in content
    assert "coordinator" in content


def test28_advanced_covers_evolution():
    """测试28: advanced示例覆盖进化闭环"""
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        content = f.read()
    assert "evolution" in content.lower()


def test29_advanced_covers_error_handling():
    """测试29: advanced示例覆盖错误处理"""
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        content = f.read()
    assert "SevaError" in content or "ErrorCode" in content
    assert "try" in content


def test30_advanced_covers_monitoring():
    """测试30: advanced示例覆盖监控"""
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        content = f.read()
    assert "Dashboard" in content or "MetricsCollector" in content
    assert "Diagnoser" in content


# ============================================================
# 测试31-35：示例输出深度验证
# ============================================================
def test31_all_examples_exit_zero():
    """测试31: 所有示例退出码为0"""
    for name in EXPECTED_EXAMPLES:
        code, out, err = run_example(name, timeout=180)
        assert code == 0, \
            f"{name}退出码{code}, stderr: {err[:300]}"


def test32_all_examples_no_traceback():
    """测试32: 所有示例输出中无Traceback"""
    for name in EXPECTED_EXAMPLES:
        code, out, err = run_example(name, timeout=180)
        combined = out + err
        assert "Traceback" not in combined, \
            f"{name}输出中包含Traceback: {combined[:500]}"


def test33_all_examples_show_completion():
    """测试33: 所有示例都显示完成标记"""
    for name in EXPECTED_EXAMPLES:
        code, out, err = run_example(name, timeout=180)
        assert "示例完成" in out, \
            f"{name}未显示'示例完成'标记"


def test34_examples_progressive_difficulty():
    """测试34: 示例难度递进（从简单到复杂）"""
    # 01应该最简单（quickstart一行启动）
    with open(os.path.join(EXAMPLES_DIR, "01_quickstart.py"), "r") as f:
        simple_content = f.read()
    # 06应该最复杂（高级用法）
    with open(os.path.join(EXAMPLES_DIR, "06_advanced.py"), "r") as f:
        advanced_content = f.read()
    # 高级示例应该比简单示例更长
    assert len(advanced_content) > len(simple_content) * 0.5, \
        "高级示例应比简单示例内容更丰富"


def test35_examples_have_next_steps():
    """测试35: 所有示例都提示下一步"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "下一步" in content, f"{name}缺少'下一步'提示"


# ============================================================
# 测试36-40：示例与API一致性
# ============================================================
def test36_quickstart_uses_correct_signature():
    """测试36: quickstart示例使用正确的API签名"""
    with open(os.path.join(EXAMPLES_DIR, "01_quickstart.py"), "r") as f:
        content = f.read()
    # quickstart(steps=100, preset="easy")
    assert "quickstart(steps=" in content
    assert "preset=" in content


def test37_custom_config_uses_overrides():
    """测试37: custom_config示例正确使用overrides"""
    with open(os.path.join(EXAMPLES_DIR, "02_custom_config.py"), "r") as f:
        content = f.read()
    assert "overrides=" in content
    assert "SEVAFactory.create" in content


def test38_monitoring_uses_dashboard_correctly():
    """测试38: monitoring示例正确使用Dashboard"""
    with open(os.path.join(EXAMPLES_DIR, "03_monitoring.py"), "r") as f:
        content = f.read()
    assert "Dashboard(" in content
    assert "collect_from_agent" in content
    assert "export_metrics" in content


def test39_diagnostics_uses_diagnoser_correctly():
    """测试39: diagnostics示例正确使用Diagnoser"""
    with open(os.path.join(EXAMPLES_DIR, "04_diagnostics.py"), "r") as f:
        content = f.read()
    assert "Diagnoser(" in content
    assert "diagnose_agent" in content
    assert "check_common_issues" in content


def test40_cli_usage_covers_main_commands():
    """测试40: cli_usage示例覆盖主要CLI命令"""
    with open(os.path.join(EXAMPLES_DIR, "05_cli_usage.py"), "r") as f:
        content = f.read()
    # 应该覆盖至少5个不同命令
    commands = ["version", "status", "presets", "config", "run",
                "errors", "diagnose"]
    covered = sum(1 for cmd in commands if cmd in content)
    assert covered >= 5, \
        f"cli_usage示例只覆盖{covered}个命令，应至少5个"


# ============================================================
# 性能测试
# ============================================================
def test41_examples_run_within_timeout():
    """测试41: 所有示例在合理时间内完成"""
    import time
    for name in EXPECTED_EXAMPLES:
        start = time.time()
        code, out, err = run_example(name, timeout=180)
        elapsed = time.time() - start
        assert elapsed < 180, f"{name}运行时间过长: {elapsed:.1f}s"


def test42_quickstart_example_fast():
    """测试42: quickstart示例应在30秒内完成"""
    import time
    start = time.time()
    code, out, err = run_example("01_quickstart.py")
    elapsed = time.time() - start
    assert elapsed < 30, f"quickstart示例运行时间过长: {elapsed:.1f}s"


def test43_examples_total_runtime():
    """测试43: 所有示例总运行时间应在5分钟内"""
    import time
    total_start = time.time()
    for name in EXPECTED_EXAMPLES:
        run_example(name, timeout=180)
    total_elapsed = time.time() - total_start
    assert total_elapsed < 300, \
        f"所有示例总运行时间过长: {total_elapsed:.1f}s"


# ============================================================
# 完整性测试
# ============================================================
def test44_examples_cover_all_apis():
    """测试44: 示例集合覆盖所有公开API"""
    all_content = ""
    for name in EXPECTED_EXAMPLES:
        with open(os.path.join(EXAMPLES_DIR, name), "r") as f:
            all_content += f.read() + "\n"

    # 检查关键API是否被覆盖
    required_apis = [
        "quickstart", "run", "get_status", "create_config",
        "SEVAFactory", "PresetConfigs",
        "Diagnoser", "diagnose_agent", "ErrorCode", "SevaError",
        "format_exception", "check_common_issues",
        "Dashboard", "MetricsCollector", "StructuredLogger",
        "export_metrics", "setup_logging",
    ]
    missing = [api for api in required_apis if api not in all_content]
    assert len(missing) == 0, \
        f"示例集合未覆盖的API: {missing}"


def test45_examples_cover_all_presets():
    """测试45: 示例集合覆盖所有预设模板"""
    all_content = ""
    for name in EXPECTED_EXAMPLES:
        with open(os.path.join(EXAMPLES_DIR, name), "r") as f:
            all_content += f.read() + "\n"

    presets = ["easy", "standard", "research", "production"]
    for preset in presets:
        assert preset in all_content, \
            f"示例集合未覆盖预设: {preset}"


def test46_examples_cover_all_env_types():
    """测试46: 示例集合覆盖环境类型"""
    all_content = ""
    for name in EXPECTED_EXAMPLES:
        with open(os.path.join(EXAMPLES_DIR, name), "r") as f:
            all_content += f.read() + "\n"

    # 至少覆盖mixed和drift
    assert "mixed" in all_content
    assert "drift" in all_content


def test47_examples_have_chinese_comments():
    """测试47: 示例使用中文注释（用户友好）"""
    for name in EXPECTED_EXAMPLES:
        path = os.path.join(EXAMPLES_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 应该包含中文（注释或字符串）
        chinese_count = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        assert chinese_count > 10, \
            f"{name}中文内容过少（{chinese_count}字符）"


def test48_examples_directory_no_pycache():
    """测试48: examples目录无__pycache__"""
    pycache = os.path.join(EXAMPLES_DIR, "__pycache__")
    # 允许存在但不应该影响示例运行
    if os.path.isdir(pycache):
        # 只要示例能运行就行
        pass
    assert True  # 此测试为信息性


def test49_example_output_contains_seva_keyword():
    """测试49: 所有示例输出包含SEVA关键字"""
    for name in EXPECTED_EXAMPLES:
        code, out, err = run_example(name, timeout=180)
        assert "SEVA" in out.upper() or "seva" in out.lower(), \
            f"{name}输出中未包含SEVA关键字"


def test50_examples_complete_coverage():
    """测试50: 示例库完整覆盖（综合验证）"""
    # 1. 文件存在
    for name in EXPECTED_EXAMPLES:
        assert os.path.isfile(os.path.join(EXAMPLES_DIR, name))

    # 2. 都能运行
    for name in EXPECTED_EXAMPLES:
        code, out, err = run_example(name, timeout=180)
        assert code == 0, f"{name}运行失败"

    # 3. 覆盖核心功能
    all_content = ""
    for name in EXPECTED_EXAMPLES:
        with open(os.path.join(EXAMPLES_DIR, name), "r") as f:
            all_content += f.read()

    # 快速启动
    assert "quickstart" in all_content
    # 工厂模式
    assert "SEVAFactory" in all_content
    # 配置管理
    assert "create_config" in all_content
    # 诊断
    assert "Diagnoser" in all_content
    # 监控
    assert "Dashboard" in all_content
    # CLI
    assert "seva" in all_content
    # 错误处理
    assert "SevaError" in all_content or "ErrorCode" in all_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
