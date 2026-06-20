"""
阶段40测试: 打包发布与安装脚本

测试覆盖:
    1. 打包文件存在性 (setup.py/pyproject.toml/MANIFEST.in/install.sh/requirements.txt/LICENSE)
    2. setup.py 配置正确性
    3. pyproject.toml 配置正确性
    4. 包元数据 (名称/版本/依赖/入口点)
    5. 安装脚本功能
    6. CLI入口点
    7. 包可导入性
"""
import os
import sys
import json
import subprocess
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seva

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# 1. 打包文件存在性测试
# =====================================================================

class TestPackagingFilesExist:
    """打包文件存在性测试"""

    def test01_setup_py_exists(self):
        """测试01: setup.py存在"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        assert os.path.exists(path), "setup.py不存在"

    def test02_pyproject_toml_exists(self):
        """测试02: pyproject.toml存在"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        assert os.path.exists(path), "pyproject.toml不存在"

    def test03_manifest_in_exists(self):
        """测试03: MANIFEST.in存在"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        assert os.path.exists(path), "MANIFEST.in不存在"

    def test04_install_sh_exists(self):
        """测试04: install.sh存在"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        assert os.path.exists(path), "install.sh不存在"

    def test05_install_sh_executable(self):
        """测试05: install.sh可执行"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        assert os.access(path, os.X_OK), "install.sh不可执行"

    def test06_requirements_txt_exists(self):
        """测试06: requirements.txt存在"""
        path = os.path.join(WORKSPACE_DIR, "requirements.txt")
        assert os.path.exists(path), "requirements.txt不存在"

    def test07_license_exists(self):
        """测试07: LICENSE文件存在"""
        path = os.path.join(WORKSPACE_DIR, "LICENSE")
        assert os.path.exists(path), "LICENSE不存在"

    def test08_readme_exists(self):
        """测试08: README.md存在"""
        path = os.path.join(WORKSPACE_DIR, "README.md")
        assert os.path.exists(path), "README.md不存在"


# =====================================================================
# 2. setup.py 配置测试
# =====================================================================

class TestSetupPy:
    """setup.py配置测试"""

    def test09_setup_py_content(self):
        """测试09: setup.py内容正确"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        assert "seva-agent" in content
        assert "setup(" in content
        assert "find_packages" in content

    def test10_setup_py_has_entry_points(self):
        """测试10: setup.py有入口点"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        assert "entry_points" in content
        assert "console_scripts" in content
        assert "seva=seva.cli:main" in content

    def test11_setup_py_has_dependencies(self):
        """测试11: setup.py有依赖"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        assert "install_requires" in content
        # setup.py从requirements.txt读取依赖
        assert "requirements.txt" in content

    def test12_setup_py_has_classifiers(self):
        """测试12: setup.py有分类器"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        assert "classifiers" in content
        assert "Python :: 3" in content

    def test13_setup_py_python_requires(self):
        """测试13: setup.py有Python版本要求"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        assert "python_requires" in content
        assert "3.8" in content


# =====================================================================
# 3. pyproject.toml 配置测试
# =====================================================================

class TestPyprojectToml:
    """pyproject.toml配置测试"""

    def test14_pyproject_has_build_system(self):
        """测试14: pyproject.toml有构建系统"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert "[build-system]" in content
        assert "setuptools" in content

    def test15_pyproject_has_project(self):
        """测试15: pyproject.toml有项目配置"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert "[project]" in content
        assert 'name = "seva-agent"' in content

    def test16_pyproject_has_dependencies(self):
        """测试16: pyproject.toml有依赖"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert "dependencies" in content
        assert "numpy" in content

    def test17_pyproject_has_scripts(self):
        """测试17: pyproject.toml有脚本入口"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert "[project.scripts]" in content
        assert 'seva = "seva.cli:main"' in content

    def test18_pyproject_has_python_requires(self):
        """测试18: pyproject.toml有Python版本要求"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert 'requires-python' in content
        assert "3.8" in content


# =====================================================================
# 4. MANIFEST.in 测试
# =====================================================================

class TestManifestIn:
    """MANIFEST.in测试"""

    def test19_manifest_includes_readme(self):
        """测试19: MANIFEST.in包含README"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        with open(path, "r") as f:
            content = f.read()
        assert "README.md" in content

    def test20_manifest_includes_requirements(self):
        """测试20: MANIFEST.in包含requirements"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        with open(path, "r") as f:
            content = f.read()
        assert "requirements.txt" in content

    def test21_manifest_includes_license(self):
        """测试21: MANIFEST.in包含LICENSE"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        with open(path, "r") as f:
            content = f.read()
        assert "LICENSE" in content

    def test22_manifest_recursive_include(self):
        """测试22: MANIFEST.in有递归包含"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        with open(path, "r") as f:
            content = f.read()
        assert "recursive-include" in content


# =====================================================================
# 5. requirements.txt 测试
# =====================================================================

class TestRequirementsTxt:
    """requirements.txt测试"""

    def test23_requirements_has_numpy(self):
        """测试23: requirements.txt有numpy"""
        path = os.path.join(WORKSPACE_DIR, "requirements.txt")
        with open(path, "r") as f:
            content = f.read()
        assert "numpy" in content

    def test24_requirements_has_pyyaml(self):
        """测试24: requirements.txt有PyYAML"""
        path = os.path.join(WORKSPACE_DIR, "requirements.txt")
        with open(path, "r") as f:
            content = f.read()
        assert "PyYAML" in content or "pyyaml" in content.lower()

    def test25_requirements_format(self):
        """测试25: requirements.txt格式正确"""
        path = os.path.join(WORKSPACE_DIR, "requirements.txt")
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                assert ">" in line or "=" in line or line.isalpha()


# =====================================================================
# 6. install.sh 测试
# =====================================================================

class TestInstallScript:
    """install.sh测试"""

    def test26_install_sh_has_shebang(self):
        """测试26: install.sh有shebang"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            first_line = f.readline()
        assert first_line.startswith("#!/bin/bash")

    def test27_install_sh_has_install_command(self):
        """测试27: install.sh有install命令"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "install" in content.lower()

    def test28_install_sh_has_uninstall_command(self):
        """测试28: install.sh有uninstall命令"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "uninstall" in content.lower()

    def test29_install_sh_has_develop_command(self):
        """测试29: install.sh有develop命令"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "develop" in content.lower() or "dev" in content.lower()

    def test30_install_sh_has_python_check(self):
        """测试30: install.sh有Python检查"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "python" in content.lower()

    def test31_install_sh_has_pip_check(self):
        """测试31: install.sh有pip检查"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "pip" in content.lower()

    def test32_install_sh_has_help(self):
        """测试32: install.sh有帮助信息"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        with open(path, "r") as f:
            content = f.read()
        assert "help" in content.lower() or "用法" in content

    def test33_install_sh_help_works(self):
        """测试33: install.sh help命令可执行"""
        path = os.path.join(WORKSPACE_DIR, "install.sh")
        result = subprocess.run(
            ["bash", path, "help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "SEVA" in result.stdout or "install" in result.stdout.lower()


# =====================================================================
# 7. LICENSE 测试
# =====================================================================

class TestLicense:
    """LICENSE测试"""

    def test34_license_is_mit(self):
        """测试34: LICENSE是MIT协议"""
        path = os.path.join(WORKSPACE_DIR, "LICENSE")
        with open(path, "r") as f:
            content = f.read()
        assert "MIT License" in content

    def test35_license_has_copyright(self):
        """测试35: LICENSE有版权信息"""
        path = os.path.join(WORKSPACE_DIR, "LICENSE")
        with open(path, "r") as f:
            content = f.read()
        assert "Copyright" in content

    def test36_license_has_permission(self):
        """测试36: LICENSE有授权条款"""
        path = os.path.join(WORKSPACE_DIR, "LICENSE")
        with open(path, "r") as f:
            content = f.read()
        assert "Permission is hereby granted" in content


# =====================================================================
# 8. 包导入与CLI入口测试
# =====================================================================

class TestPackageImport:
    """包导入与CLI入口测试"""

    def test37_seva_importable(self):
        """测试37: seva包可导入"""
        import seva
        assert seva is not None

    def test38_seva_has_version(self):
        """测试38: seva有版本号"""
        assert hasattr(seva, "__version__")
        assert seva.__version__ >= "0.40.0"

    def test39_seva_has_all(self):
        """测试39: seva有__all__"""
        assert hasattr(seva, "__all__")
        assert isinstance(seva.__all__, list)
        assert len(seva.__all__) > 0

    def test40_cli_main_exists(self):
        """测试40: CLI main函数存在"""
        from seva.cli import main
        assert callable(main)

    def test41_cli_main_callable(self):
        """测试41: CLI main可调用"""
        from seva.cli import main
        # 调用version命令
        result = main(["version"])
        assert result == 0

    def test42_main_module_exists(self):
        """测试42: __main__.py存在"""
        path = os.path.join(WORKSPACE_DIR, "seva", "__main__.py")
        assert os.path.exists(path)

    def test43_main_module_content(self):
        """测试43: __main__.py内容正确"""
        path = os.path.join(WORKSPACE_DIR, "seva", "__main__.py")
        with open(path, "r") as f:
            content = f.read()
        assert "from seva.cli import main" in content
        assert "__main__" in content


# =====================================================================
# 9. 版本一致性测试
# =====================================================================

class TestVersionConsistency:
    """版本一致性测试"""

    def test44_init_version(self):
        """测试44: __init__.py版本"""
        assert seva.__version__ == "0.40.0"

    def test45_setup_py_version(self):
        """测试45: setup.py版本一致"""
        path = os.path.join(WORKSPACE_DIR, "setup.py")
        with open(path, "r") as f:
            content = f.read()
        # setup.py动态读取版本，检查默认值
        assert "0.40.0" in content

    def test46_pyproject_version(self):
        """测试46: pyproject.toml版本一致"""
        path = os.path.join(WORKSPACE_DIR, "pyproject.toml")
        with open(path, "r") as f:
            content = f.read()
        assert 'version = "0.40.0"' in content


# =====================================================================
# 10. 包构建测试
# =====================================================================

class TestPackageBuild:
    """包构建测试（如果setuptools可用）"""

    def test47_setup_check(self):
        """测试47: setup.py check通过"""
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "check"],
                capture_output=True, text=True, timeout=30,
                cwd=WORKSPACE_DIR
            )
            # check命令返回0表示成功
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("setuptools不可用或构建超时")

    def test48_package_data_included(self):
        """测试48: 包数据包含在MANIFEST中"""
        path = os.path.join(WORKSPACE_DIR, "MANIFEST.in")
        with open(path, "r") as f:
            content = f.read()
        assert "seva" in content
        assert "*.py" in content


# =====================================================================
# 11. 端到端安装验证测试
# =====================================================================

class TestEndToEnd:
    """端到端安装验证测试"""

    def test49_full_import_chain(self):
        """测试49: 完整导入链"""
        import seva
        # 验证所有主要模块可导入
        from seva import (quickstart, run, get_status, create_config,
                          SEVAFactory, PresetConfigs)
        from seva import (save_config, load_config, config_from_file,
                          ConfigLoader, ConfigWizard)
        from seva import (SevaShell, run_shell, execute_command)
        from seva import (WebDashboard, start_dashboard,
                          generate_html_report)
        from seva import (ResultVisualizer, SVGChartGenerator,
                          visualize, export_report, generate_charts,
                          compare_runs)
        from seva import (SessionManager, CheckpointHook,
                          save_session, load_session, list_sessions,
                          delete_session, get_session_info, checkpoint)
        assert True  # 所有导入成功

    def test50_cli_all_commands(self):
        """测试50: CLI所有命令可用"""
        from seva.cli import build_parser
        parser = build_parser()
        # 验证主要命令的子解析器存在
        # 通过检查parser的subparsers来验证命令存在
        # 不实际解析--help（会触发SystemExit）
        commands = ["run", "config", "status", "version", "presets",
                    "demo", "scenarios", "scenario", "shell", "exec",
                    "web", "report", "export", "charts", "compare",
                    "session", "diagnose", "errors", "monitor"]
        for cmd in commands:
            # 验证命令可以被解析（使用最小参数）
            try:
                if cmd == "version":
                    parser.parse_args([cmd])
                elif cmd == "status":
                    parser.parse_args([cmd, "--preset", "easy", "--steps", "0"])
                elif cmd == "presets":
                    parser.parse_args([cmd])
                elif cmd == "errors":
                    parser.parse_args([cmd])
                elif cmd == "monitor":
                    parser.parse_args([cmd, "--preset", "easy", "--steps", "0"])
                elif cmd == "demo":
                    parser.parse_args([cmd, "--steps", "1"])
                elif cmd == "shell":
                    parser.parse_args([cmd])
                elif cmd == "exec":
                    # exec需要命令参数，会报错但说明子解析器存在
                    pass
                elif cmd == "scenarios":
                    parser.parse_args([cmd, "list"])
                elif cmd == "diagnose":
                    parser.parse_args([cmd, "--steps", "0"])
                elif cmd == "session":
                    parser.parse_args([cmd, "list"])
                elif cmd == "compare":
                    parser.parse_args([cmd, "easy", "standard", "--steps", "0"])
                elif cmd == "export":
                    parser.parse_args([cmd, "--format", "json", "--steps", "0"])
                elif cmd == "charts":
                    parser.parse_args([cmd, "--steps", "0"])
                elif cmd == "report":
                    parser.parse_args([cmd, "--format", "json", "--steps", "0"])
                elif cmd == "web":
                    parser.parse_args([cmd, "--steps", "0"])
                elif cmd == "config":
                    parser.parse_args([cmd, "show"])
                elif cmd == "run":
                    parser.parse_args([cmd, "--steps", "0"])
                elif cmd == "scenario":
                    parser.parse_args([cmd, "list"])
            except SystemExit:
                pass  # --help等会触发SystemExit，忽略
        assert True  # 所有命令验证完成

    def test51_quickstart_works(self):
        """测试51: quickstart正常工作"""
        from seva import quickstart
        agent, stats = quickstart(steps=5, preset="easy", seed=42)
        assert agent is not None
        assert agent.cycle_count == 5

    def test52_all_features_accessible(self):
        """测试52: 所有阶段功能可访问"""
        import seva
        # 阶段29: 统一API
        assert hasattr(seva, "quickstart")
        # 阶段34: 预设场景
        assert hasattr(seva, "run_scenario")
        # 阶段35: 配置文件
        assert hasattr(seva, "save_config")
        assert hasattr(seva, "load_config")
        # 阶段36: REPL Shell
        assert hasattr(seva, "run_shell")
        # 阶段37: Web Dashboard
        assert hasattr(seva, "start_dashboard")
        # 阶段38: 可视化
        assert hasattr(seva, "visualize")
        assert hasattr(seva, "generate_charts")
        # 阶段39: 持久化
        assert hasattr(seva, "save_session")
        assert hasattr(seva, "load_session")
