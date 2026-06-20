"""
阶段41测试: 首次运行向导与引导系统

测试覆盖:
    1. WizardState - 向导状态管理
    2. FirstRunWizard - 首次运行向导
    3. GuidedTips - 引导提示系统
    4. 便捷函数 (run_wizard/check_first_run/show_tips/reset_wizard)
    5. CLI集成 (wizard命令)
    6. 状态持久化与重置
"""
import os
import sys
import json
import time
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seva
from seva import (FirstRunWizard, WizardState, GuidedTips,
                  run_wizard, check_first_run, show_tips, reset_wizard)
from seva.wizard import _get_version
from seva.core import SelfEvolvingAgentCore


# =====================================================================
# 测试夹具
# =====================================================================

@pytest.fixture
def temp_state_dir(monkeypatch):
    """临时状态目录"""
    d = tempfile.mkdtemp()
    monkeypatch.setattr(WizardState, "STATE_DIR", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def agent():
    """创建一个运行了若干步的Agent"""
    from seva import quickstart
    agent, _ = quickstart(steps=20, preset="easy", seed=42)
    return agent


# =====================================================================
# 1. WizardState 测试
# =====================================================================

class TestWizardState:
    """向导状态管理测试"""

    def test01_is_first_run_true(self, temp_state_dir):
        """测试01: 首次运行为True"""
        assert WizardState.is_first_run() is True

    def test02_is_first_run_false_after_mark(self, temp_state_dir):
        """测试02: 标记完成后不再首次运行"""
        WizardState.mark_completed(preset="easy", env_type="mixed",
                                   steps=10)
        assert WizardState.is_first_run() is False

    def test03_is_completed_false_initially(self, temp_state_dir):
        """测试03: 初始未完成"""
        assert WizardState.is_completed() is False

    def test04_is_completed_true_after_mark(self, temp_state_dir):
        """测试04: 标记完成后为True"""
        WizardState.mark_completed(preset="easy", env_type="mixed",
                                   steps=10)
        assert WizardState.is_completed() is True

    def test05_load_state_empty(self, temp_state_dir):
        """测试05: 空状态加载"""
        state = WizardState.load_state()
        assert state == {}

    def test06_save_and_load_state(self, temp_state_dir):
        """测试06: 保存和加载状态"""
        test_state = {"completed": True, "preset": "standard"}
        WizardState.save_state(test_state)
        loaded = WizardState.load_state()
        assert loaded["completed"] is True
        assert loaded["preset"] == "standard"

    def test07_mark_completed(self, temp_state_dir):
        """测试07: 标记完成"""
        WizardState.mark_completed(preset="easy", env_type="mixed",
                                   steps=50)
        state = WizardState.load_state()
        assert state["completed"] is True
        assert state["preset"] == "easy"
        assert state["env_type"] == "mixed"
        assert state["steps"] == 50
        assert "completed_at" in state
        assert "version" in state

    def test08_reset(self, temp_state_dir):
        """测试08: 重置状态"""
        WizardState.mark_completed(preset="easy", env_type="mixed",
                                   steps=10)
        assert WizardState.is_completed() is True
        WizardState.reset()
        assert WizardState.is_first_run() is True
        assert WizardState.is_completed() is False

    def test09_get_state_path(self, temp_state_dir):
        """测试09: 获取状态文件路径"""
        path = WizardState.get_state_path()
        assert path.endswith("wizard_state.json")
        assert temp_state_dir in path

    def test10_state_persistence(self, temp_state_dir):
        """测试10: 状态持久化"""
        WizardState.mark_completed(preset="standard", env_type="drift",
                                   steps=100)
        # 重新加载应该保持状态
        assert WizardState.is_completed() is True
        state = WizardState.load_state()
        assert state["preset"] == "standard"


# =====================================================================
# 2. FirstRunWizard 测试
# =====================================================================

class TestFirstRunWizard:
    """首次运行向导测试"""

    def test11_init_default(self):
        """测试11: 默认初始化"""
        wizard = FirstRunWizard()
        assert wizard.interactive is True
        assert wizard.choices == {}

    def test12_init_non_interactive(self):
        """测试12: 非交互模式初始化"""
        wizard = FirstRunWizard(interactive=False)
        assert wizard.interactive is False

    def test13_preset_options(self):
        """测试13: 预设选项存在"""
        assert len(FirstRunWizard.PRESET_OPTIONS) >= 4
        values = [v for v, _ in FirstRunWizard.PRESET_OPTIONS]
        assert "easy" in values
        assert "standard" in values

    def test14_env_options(self):
        """测试14: 环境选项存在"""
        assert len(FirstRunWizard.ENV_OPTIONS) >= 4
        values = [v for v, _ in FirstRunWizard.ENV_OPTIONS]
        assert "mixed" in values
        assert "blank" in values

    def test15_steps_options(self):
        """测试15: 步数选项存在"""
        assert len(FirstRunWizard.STEPS_OPTIONS) >= 3
        values = [v for v, _ in FirstRunWizard.STEPS_OPTIONS]
        assert 10 in values

    def test16_run_silent(self, temp_state_dir):
        """测试16: 静默运行"""
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run_silent(preset="easy", env_type="mixed",
                                   steps=5)
        assert result["preset"] == "easy"
        assert result["env_type"] == "mixed"
        assert result["steps"] == 5
        assert result["cancelled"] is False
        assert "agent" in result
        assert "stats" in result
        assert isinstance(result["agent"], SelfEvolvingAgentCore)

    def test17_run_silent_marks_completed(self, temp_state_dir):
        """测试17: 静默运行标记完成"""
        wizard = FirstRunWizard(interactive=False)
        wizard.run_silent(preset="easy", env_type="mixed", steps=5)
        assert WizardState.is_completed() is True

    def test18_run_silent_agent_runs(self, temp_state_dir):
        """测试18: 静默运行后Agent已运行"""
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run_silent(preset="easy", env_type="mixed",
                                   steps=10)
        assert result["agent"].cycle_count == 10

    def test19_run_non_interactive(self, temp_state_dir):
        """测试19: 非交互模式运行"""
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run()
        assert result["cancelled"] is False
        assert result["preset"] == "easy"
        assert result["env_type"] == "mixed"
        assert result["steps"] == 10

    def test20_run_creates_agent(self, temp_state_dir):
        """测试20: 运行创建Agent"""
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run()
        assert isinstance(result["agent"], SelfEvolvingAgentCore)
        assert result["agent"].cycle_count > 0


# =====================================================================
# 3. GuidedTips 测试
# =====================================================================

class TestGuidedTips:
    """引导提示系统测试"""

    def test21_tips_first_run(self):
        """测试21: 首次运行提示"""
        tips = GuidedTips.get_tips(context="first_run")
        assert len(tips) > 0
        assert any("向导" in t or "wizard" in t for t in tips)

    def test22_tips_low_cycles(self):
        """测试22: 低周期数提示"""
        tips = GuidedTips.get_tips(context="low_cycles")
        assert len(tips) > 0
        assert any("步" in t for t in tips)

    def test23_tips_high_error(self):
        """测试23: 高误差提示"""
        tips = GuidedTips.get_tips(context="high_error")
        assert len(tips) > 0
        assert any("诊断" in t or "diagnose" in t for t in tips)

    def test24_tips_no_skills(self):
        """测试24: 无技能提示"""
        tips = GuidedTips.get_tips(context="no_skills")
        assert len(tips) > 0
        assert any("技能" in t for t in tips)

    def test25_tips_healthy(self):
        """测试25: 健康状态提示"""
        tips = GuidedTips.get_tips(context="healthy")
        assert len(tips) > 0

    def test26_tips_with_agent(self, agent):
        """测试26: 带Agent的提示"""
        tips = GuidedTips.get_tips(agent=agent)
        assert len(tips) > 0
        assert len(tips) <= 3  # 最多3条

    def test27_tips_default(self):
        """测试27: 默认提示"""
        tips = GuidedTips.get_tips()
        assert len(tips) > 0

    def test28_tips_deduplication(self, agent):
        """测试28: 提示去重"""
        tips = GuidedTips.get_tips(agent=agent, context="first_run")
        assert len(tips) == len(set(tips))  # 无重复

    def test29_tips_max_three(self, agent):
        """测试29: 最多3条提示"""
        tips = GuidedTips.get_tips(agent=agent, context="first_run")
        assert len(tips) <= 3

    def test30_show_tips_no_error(self, agent, capsys):
        """测试30: show_tips不报错"""
        GuidedTips.show_tips(agent)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test31_tips_completed(self):
        """测试31: 完成向导提示"""
        tips = GuidedTips.get_tips(context="completed")
        assert len(tips) > 0


# =====================================================================
# 4. 便捷函数测试
# =====================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test32_check_first_run_true(self, temp_state_dir):
        """测试32: 首次运行检测为True"""
        assert check_first_run() is True

    def test33_check_first_run_false(self, temp_state_dir):
        """测试33: 标记后非首次运行"""
        WizardState.mark_completed()
        assert check_first_run() is False

    def test34_reset_wizard(self, temp_state_dir):
        """测试34: 重置向导"""
        WizardState.mark_completed()
        assert check_first_run() is False
        reset_wizard()
        assert check_first_run() is True

    def test35_show_tips_function(self, capsys):
        """测试35: show_tips函数"""
        show_tips()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test36_show_tips_with_agent(self, agent, capsys):
        """测试36: 带Agent的show_tips"""
        show_tips(agent)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test37_run_wizard_non_interactive(self, temp_state_dir):
        """测试37: 非交互式run_wizard"""
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run()
        assert result["cancelled"] is False
        assert WizardState.is_completed() is True


# =====================================================================
# 5. 包导出与版本测试
# =====================================================================

class TestPackageExports:
    """包导出测试"""

    def test38_first_run_wizard_exported(self):
        """测试38: FirstRunWizard已导出"""
        assert hasattr(seva, "FirstRunWizard")
        assert "FirstRunWizard" in seva.__all__

    def test39_wizard_state_exported(self):
        """测试39: WizardState已导出"""
        assert hasattr(seva, "WizardState")
        assert "WizardState" in seva.__all__

    def test40_guided_tips_exported(self):
        """测试40: GuidedTips已导出"""
        assert hasattr(seva, "GuidedTips")
        assert "GuidedTips" in seva.__all__

    def test41_run_wizard_exported(self):
        """测试41: run_wizard已导出"""
        assert hasattr(seva, "run_wizard")
        assert "run_wizard" in seva.__all__

    def test42_check_first_run_exported(self):
        """测试42: check_first_run已导出"""
        assert hasattr(seva, "check_first_run")
        assert "check_first_run" in seva.__all__

    def test43_show_tips_exported(self):
        """测试43: show_tips已导出"""
        assert hasattr(seva, "show_tips")
        assert "show_tips" in seva.__all__

    def test44_reset_wizard_exported(self):
        """测试44: reset_wizard已导出"""
        assert hasattr(seva, "reset_wizard")
        assert "reset_wizard" in seva.__all__

    def test45_version_updated(self):
        """测试45: 版本号已更新"""
        assert seva.__version__ >= "0.41.0"

    def test46_get_version_helper(self):
        """测试46: 版本号辅助函数"""
        v = _get_version()
        assert isinstance(v, str)
        assert v != "unknown"


# =====================================================================
# 6. CLI集成测试
# =====================================================================

class TestCLIIntegration:
    """CLI集成测试"""

    def test47_cli_wizard_command(self):
        """测试47: CLI wizard命令存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--silent", "--steps", "5"])
        assert args.silent is True
        assert args.steps == 5
        assert hasattr(args, "func")

    def test48_cli_wizard_reset(self):
        """测试48: CLI wizard --reset"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--reset"])
        assert args.reset is True

    def test49_cli_wizard_status(self):
        """测试49: CLI wizard --status"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--status"])
        assert args.status is True

    def test50_cli_wizard_tips(self):
        """测试50: CLI wizard --tips"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--tips"])
        assert args.tips is True

    def test51_cli_wizard_preset(self):
        """测试51: CLI wizard --preset"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--silent", "--preset",
                                  "standard"])
        assert args.preset == "standard"

    def test52_cli_wizard_env_type(self):
        """测试52: CLI wizard --env-type"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["wizard", "--silent", "--env-type",
                                  "drift"])
        assert args.env_type == "drift"


# =====================================================================
# 7. 端到端集成测试
# =====================================================================

class TestEndToEnd:
    """端到端集成测试"""

    def test53_full_wizard_flow(self, temp_state_dir):
        """测试53: 完整向导流程"""
        # 1. 检测首次运行
        assert check_first_run() is True

        # 2. 运行向导
        wizard = FirstRunWizard(interactive=False)
        result = wizard.run_silent(preset="easy", env_type="mixed",
                                   steps=10)

        # 3. 验证完成
        assert WizardState.is_completed() is True
        assert check_first_run() is False

        # 4. 验证Agent
        assert result["agent"].cycle_count == 10

    def test54_reset_and_rerun(self, temp_state_dir):
        """测试54: 重置后重新运行"""
        # 首次运行
        wizard = FirstRunWizard(interactive=False)
        wizard.run_silent(steps=5)
        assert WizardState.is_completed() is True

        # 重置
        reset_wizard()
        assert check_first_run() is True
        assert WizardState.is_completed() is False

        # 重新运行
        wizard2 = FirstRunWizard(interactive=False)
        wizard2.run_silent(steps=10)
        assert WizardState.is_completed() is True

    def test55_tips_after_wizard(self, temp_state_dir, capsys):
        """测试55: 向导完成后显示提示"""
        wizard = FirstRunWizard(interactive=False)
        wizard.run_silent(steps=20)

        # 获取Agent并显示提示
        agent = wizard.choices.get("agent") or wizard.run_silent(
            steps=20)["agent"]
        show_tips(agent)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test56_wizard_different_presets(self, temp_state_dir):
        """测试56: 不同预设的向导"""
        for preset in ["easy", "standard"]:
            WizardState.reset()
            wizard = FirstRunWizard(interactive=False)
            result = wizard.run_silent(preset=preset, steps=5)
            assert result["preset"] == preset
            assert WizardState.is_completed() is True

    def test57_wizard_state_survives_reload(self, temp_state_dir):
        """测试57: 状态在重新加载后保持"""
        WizardState.mark_completed(preset="standard", env_type="drift",
                                   steps=100)

        # 模拟重新加载
        state = WizardState.load_state()
        assert state["preset"] == "standard"
        assert state["env_type"] == "drift"
        assert state["steps"] == 100
        assert WizardState.is_completed() is True
