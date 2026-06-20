"""
阶段43测试: 交互式教程系统

测试覆盖:
    1. TutorialProgress - 教程进度管理
    2. TutorialStep - 教程步骤
    3. Tutorial - 教程类
    4. 内置教程 (beginner/intermediate/advanced)
    5. 便捷函数
    6. CLI集成
"""
import os
import sys
import json
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seva
from seva import (Tutorial, TutorialStep, TutorialProgress,
                  run_tutorial, list_tutorials,
                  get_tutorial_info, reset_tutorial_progress,
                  get_tutorial_progress)


# =====================================================================
# 测试夹具
# =====================================================================

@pytest.fixture
def temp_state_dir(monkeypatch):
    """临时状态目录"""
    d = tempfile.mkdtemp()
    monkeypatch.setattr(TutorialProgress, "STATE_DIR", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =====================================================================
# 1. TutorialProgress 测试
# =====================================================================

class TestTutorialProgress:
    """教程进度管理测试"""

    def test01_load_all_empty(self, temp_state_dir):
        """测试01: 空状态加载"""
        assert TutorialProgress.load_all() == {}

    def test02_get_progress_default(self, temp_state_dir):
        """测试02: 默认进度"""
        progress = TutorialProgress.get_progress("beginner")
        assert progress["completed_steps"] == []
        assert progress["current_step"] == 0
        assert progress["completed"] is False

    def test03_mark_step_completed(self, temp_state_dir):
        """测试03: 标记步骤完成"""
        TutorialProgress.mark_step_completed("beginner", 0)
        progress = TutorialProgress.get_progress("beginner")
        assert 0 in progress["completed_steps"]
        assert progress["current_step"] == 1
        assert progress["started_at"] is not None

    def test04_mark_multiple_steps(self, temp_state_dir):
        """测试04: 标记多个步骤"""
        TutorialProgress.mark_step_completed("beginner", 0)
        TutorialProgress.mark_step_completed("beginner", 1)
        progress = TutorialProgress.get_progress("beginner")
        assert len(progress["completed_steps"]) == 2
        assert progress["current_step"] == 2

    def test05_mark_completed(self, temp_state_dir):
        """测试05: 标记教程完成"""
        TutorialProgress.mark_completed("beginner")
        progress = TutorialProgress.get_progress("beginner")
        assert progress["completed"] is True
        assert progress["completed_at"] is not None

    def test06_is_completed(self, temp_state_dir):
        """测试06: 检查完成状态"""
        assert TutorialProgress.is_completed("beginner") is False
        TutorialProgress.mark_completed("beginner")
        assert TutorialProgress.is_completed("beginner") is True

    def test07_reset_specific(self, temp_state_dir):
        """测试07: 重置特定教程"""
        TutorialProgress.mark_completed("beginner")
        TutorialProgress.reset("beginner")
        assert TutorialProgress.is_completed("beginner") is False

    def test08_reset_all(self, temp_state_dir):
        """测试08: 重置所有"""
        TutorialProgress.mark_completed("beginner")
        TutorialProgress.mark_completed("intermediate")
        TutorialProgress.reset()
        assert TutorialProgress.load_all() == {}

    def test09_persistence(self, temp_state_dir):
        """测试09: 进度持久化"""
        TutorialProgress.mark_step_completed("beginner", 0)
        TutorialProgress.mark_step_completed("beginner", 1)
        # 重新加载
        progress = TutorialProgress.get_progress("beginner")
        assert len(progress["completed_steps"]) == 2

    def test10_no_duplicate_steps(self, temp_state_dir):
        """测试10: 步骤不重复"""
        TutorialProgress.mark_step_completed("beginner", 0)
        TutorialProgress.mark_step_completed("beginner", 0)
        progress = TutorialProgress.get_progress("beginner")
        assert len(progress["completed_steps"]) == 1


# =====================================================================
# 2. TutorialStep 测试
# =====================================================================

class TestTutorialStep:
    """教程步骤测试"""

    def test11_init_basic(self):
        """测试11: 基本初始化"""
        step = TutorialStep("标题", "描述")
        assert step.title == "标题"
        assert step.description == "描述"
        assert step.example == ""
        assert step.verify is None
        assert step.tip == ""

    def test12_init_full(self):
        """测试12: 完整初始化"""
        verify_fn = lambda: True
        step = TutorialStep("标题", "描述", "示例", verify_fn, "提示")
        assert step.title == "标题"
        assert step.description == "描述"
        assert step.example == "示例"
        assert step.verify is verify_fn
        assert step.tip == "提示"

    def test13_step_with_verify(self):
        """测试13: 带验证的步骤"""
        step = TutorialStep("测试", "描述", verify=lambda: True)
        assert step.verify() is True


# =====================================================================
# 3. Tutorial 测试
# =====================================================================

class TestTutorial:
    """教程类测试"""

    def test14_init(self):
        """测试14: 初始化"""
        steps = [TutorialStep("步骤1", "描述1")]
        tutorial = Tutorial("test", "测试教程", "测试描述", steps)
        assert tutorial.id == "test"
        assert tutorial.title == "测试教程"
        assert tutorial.description == "测试描述"
        assert len(tutorial.steps) == 1
        assert tutorial.difficulty == "beginner"

    def test15_run_silent(self, temp_state_dir):
        """测试15: 静默运行"""
        steps = [
            TutorialStep("步骤1", "描述1"),
            TutorialStep("步骤2", "描述2"),
        ]
        tutorial = Tutorial("test_silent", "测试", "描述", steps)
        result = tutorial.run(interactive=False)
        assert result is True
        assert TutorialProgress.is_completed("test_silent")

    def test16_run_marks_steps(self, temp_state_dir):
        """测试16: 运行标记步骤"""
        steps = [
            TutorialStep("步骤1", "描述1"),
            TutorialStep("步骤2", "描述2"),
        ]
        tutorial = Tutorial("test_steps", "测试", "描述", steps)
        tutorial.run(interactive=False)
        progress = TutorialProgress.get_progress("test_steps")
        assert len(progress["completed_steps"]) == 2

    def test17_run_with_verify(self, temp_state_dir):
        """测试17: 带验证的运行"""
        verified = [False]
        def verify():
            verified[0] = True
            return True

        steps = [TutorialStep("步骤1", "描述1", verify=verify)]
        tutorial = Tutorial("test_verify", "测试", "描述", steps)
        tutorial.run(interactive=False)
        assert verified[0] is True

    def test18_register_and_get(self):
        """测试18: 注册和获取"""
        steps = [TutorialStep("步骤1", "描述1")]
        tutorial = Tutorial("test_register", "测试", "描述", steps)
        Tutorial.register(tutorial)
        assert Tutorial.get_tutorial("test_register") is tutorial

    def test19_get_tutorial_not_found(self):
        """测试19: 获取不存在的教程"""
        assert Tutorial.get_tutorial("nonexistent") is None

    def test20_list_tutorials(self):
        """测试20: 列出教程"""
        tutorials = Tutorial.list_tutorials()
        assert len(tutorials) >= 3  # 至少3个内置教程

    def test21_run_from_step(self, temp_state_dir):
        """测试21: 从指定步骤开始"""
        steps = [
            TutorialStep("步骤1", "描述1"),
            TutorialStep("步骤2", "描述2"),
            TutorialStep("步骤3", "描述3"),
        ]
        tutorial = Tutorial("test_resume", "测试", "描述", steps)
        tutorial.run(interactive=False, start_step=1)
        progress = TutorialProgress.get_progress("test_resume")
        # 从步骤1开始，应该完成步骤1和2
        assert 1 in progress["completed_steps"]
        assert 2 in progress["completed_steps"]
        assert 0 not in progress["completed_steps"]


# =====================================================================
# 4. 内置教程测试
# =====================================================================

class TestBuiltinTutorials:
    """内置教程测试"""

    def test22_beginner_exists(self):
        """测试22: 入门教程存在"""
        tutorial = Tutorial.get_tutorial("beginner")
        assert tutorial is not None
        assert tutorial.title == "SEVA入门教程"
        assert len(tutorial.steps) >= 3

    def test23_intermediate_exists(self):
        """测试23: 进阶教程存在"""
        tutorial = Tutorial.get_tutorial("intermediate")
        assert tutorial is not None
        assert tutorial.title == "SEVA进阶教程"
        assert len(tutorial.steps) >= 3

    def test24_advanced_exists(self):
        """测试24: 高级教程存在"""
        tutorial = Tutorial.get_tutorial("advanced")
        assert tutorial is not None
        assert tutorial.title == "SEVA高级教程"
        assert len(tutorial.steps) >= 3

    def test25_beginner_run_silent(self, temp_state_dir):
        """测试25: 入门教程静默运行"""
        result = run_tutorial("beginner", interactive=False,
                              resume=False)
        assert result is True
        assert TutorialProgress.is_completed("beginner")

    def test26_intermediate_run_silent(self, temp_state_dir):
        """测试26: 进阶教程静默运行"""
        result = run_tutorial("intermediate", interactive=False,
                              resume=False)
        assert result is True

    def test27_advanced_run_silent(self, temp_state_dir):
        """测试27: 高级教程静默运行"""
        result = run_tutorial("advanced", interactive=False,
                              resume=False)
        assert result is True

    def test28_tutorials_have_descriptions(self):
        """测试28: 教程都有描述"""
        for tid in ["beginner", "intermediate", "advanced"]:
            tutorial = Tutorial.get_tutorial(tid)
            assert len(tutorial.description) > 0

    def test29_tutorials_have_steps(self):
        """测试29: 教程都有步骤"""
        for tid in ["beginner", "intermediate", "advanced"]:
            tutorial = Tutorial.get_tutorial(tid)
            assert len(tutorial.steps) > 0

    def test30_steps_have_titles(self):
        """测试30: 步骤都有标题"""
        for tid in ["beginner", "intermediate", "advanced"]:
            tutorial = Tutorial.get_tutorial(tid)
            for step in tutorial.steps:
                assert len(step.title) > 0
                assert len(step.description) > 0


# =====================================================================
# 5. 便捷函数测试
# =====================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test31_list_tutorials(self):
        """测试31: list_tutorials函数"""
        tutorials = list_tutorials()
        assert len(tutorials) >= 3
        ids = [t["id"] for t in tutorials]
        assert "beginner" in ids
        assert "intermediate" in ids
        assert "advanced" in ids

    def test32_get_tutorial_info(self):
        """测试32: get_tutorial_info函数"""
        info = get_tutorial_info("beginner")
        assert info is not None
        assert info["id"] == "beginner"
        assert "title" in info
        assert "steps" in info

    def test33_get_tutorial_info_not_found(self):
        """测试33: 不存在的教程信息"""
        assert get_tutorial_info("nonexistent") is None

    def test34_run_tutorial_not_found(self, temp_state_dir):
        """测试34: 运行不存在的教程"""
        result = run_tutorial("nonexistent", interactive=False)
        assert result is False

    def test35_reset_progress(self, temp_state_dir):
        """测试35: 重置进度"""
        TutorialProgress.mark_completed("beginner")
        reset_tutorial_progress("beginner")
        assert not TutorialProgress.is_completed("beginner")

    def test36_reset_all_progress(self, temp_state_dir):
        """测试36: 重置所有进度"""
        TutorialProgress.mark_completed("beginner")
        TutorialProgress.mark_completed("intermediate")
        reset_tutorial_progress()
        assert not TutorialProgress.is_completed("beginner")
        assert not TutorialProgress.is_completed("intermediate")

    def test37_get_tutorial_progress(self, temp_state_dir):
        """测试37: 获取教程进度"""
        TutorialProgress.mark_step_completed("beginner", 0)
        progress = get_tutorial_progress("beginner")
        assert len(progress["completed_steps"]) == 1

    def test38_list_tutorials_includes_progress(self, temp_state_dir):
        """测试38: 列表包含进度信息"""
        TutorialProgress.mark_completed("beginner")
        tutorials = list_tutorials()
        for t in tutorials:
            if t["id"] == "beginner":
                assert t["completed"] is True

    def test39_resume_tutorial(self, temp_state_dir):
        """测试39: 断点续学"""
        # 完成第一步
        run_tutorial("beginner", interactive=False, resume=False)
        # 再次运行应该标记为已完成
        assert TutorialProgress.is_completed("beginner")


# =====================================================================
# 6. 包导出与版本测试
# =====================================================================

class TestPackageExports:
    """包导出测试"""

    def test40_tutorial_exported(self):
        """测试40: Tutorial已导出"""
        assert hasattr(seva, "Tutorial")
        assert "Tutorial" in seva.__all__

    def test41_tutorial_step_exported(self):
        """测试41: TutorialStep已导出"""
        assert hasattr(seva, "TutorialStep")
        assert "TutorialStep" in seva.__all__

    def test42_tutorial_progress_exported(self):
        """测试42: TutorialProgress已导出"""
        assert hasattr(seva, "TutorialProgress")
        assert "TutorialProgress" in seva.__all__

    def test43_run_tutorial_exported(self):
        """测试43: run_tutorial已导出"""
        assert hasattr(seva, "run_tutorial")
        assert "run_tutorial" in seva.__all__

    def test44_list_tutorials_exported(self):
        """测试44: list_tutorials已导出"""
        assert hasattr(seva, "list_tutorials")
        assert "list_tutorials" in seva.__all__

    def test45_version_updated(self):
        """测试45: 版本号已更新"""
        assert seva.__version__ >= "0.43.0"


# =====================================================================
# 7. CLI集成测试
# =====================================================================

class TestCLIIntegration:
    """CLI集成测试"""

    def test46_cli_tutorial_list(self):
        """测试46: CLI tutorial --list"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial", "--list"])
        assert args.list is True
        assert hasattr(args, "func")

    def test47_cli_tutorial_run(self):
        """测试47: CLI tutorial运行"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial", "beginner", "--silent",
                                  "--no-resume"])
        assert args.tutorial_id == "beginner"
        assert args.silent is True
        assert args.no_resume is True

    def test48_cli_tutorial_reset(self):
        """测试48: CLI tutorial --reset"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial", "--reset"])
        assert args.reset is True

    def test49_cli_tutorial_progress(self):
        """测试49: CLI tutorial --progress"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial", "beginner", "--progress"])
        assert args.progress is True
        assert args.tutorial_id == "beginner"

    def test50_cli_tutorial_no_id(self):
        """测试50: CLI tutorial无ID"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial"])
        assert args.tutorial_id is None


# =====================================================================
# 8. 端到端集成测试
# =====================================================================

class TestEndToEnd:
    """端到端集成测试"""

    def test51_full_tutorial_flow(self, temp_state_dir):
        """测试51: 完整教程流程"""
        # 1. 列出教程
        tutorials = list_tutorials()
        assert len(tutorials) >= 3

        # 2. 运行入门教程
        result = run_tutorial("beginner", interactive=False,
                              resume=False)
        assert result is True

        # 3. 验证完成
        assert TutorialProgress.is_completed("beginner")

        # 4. 查看进度
        progress = get_tutorial_progress("beginner")
        assert progress["completed"] is True

    def test52_all_tutorials_completable(self, temp_state_dir):
        """测试52: 所有教程可完成"""
        for tid in ["beginner", "intermediate", "advanced"]:
            reset_tutorial_progress(tid)
            result = run_tutorial(tid, interactive=False, resume=False)
            assert result is True, f"教程 {tid} 未完成"
            assert TutorialProgress.is_completed(tid)

    def test53_progress_tracking(self, temp_state_dir):
        """测试53: 进度跟踪"""
        tutorial = Tutorial.get_tutorial("beginner")
        total_steps = len(tutorial.steps)

        # 逐步完成
        for i in range(total_steps):
            TutorialProgress.mark_step_completed("beginner", i)
            progress = get_tutorial_progress("beginner")
            assert len(progress["completed_steps"]) == i + 1

        # 标记完成
        TutorialProgress.mark_completed("beginner")
        assert TutorialProgress.is_completed("beginner")

    def test54_reset_and_rerun(self, temp_state_dir):
        """测试54: 重置后重新运行"""
        run_tutorial("beginner", interactive=False, resume=False)
        assert TutorialProgress.is_completed("beginner")

        reset_tutorial_progress("beginner")
        assert not TutorialProgress.is_completed("beginner")

        run_tutorial("beginner", interactive=False, resume=False)
        assert TutorialProgress.is_completed("beginner")

    def test55_tutorial_info_complete(self):
        """测试55: 教程信息完整"""
        for tid in ["beginner", "intermediate", "advanced"]:
            info = get_tutorial_info(tid)
            assert info is not None
            assert "id" in info
            assert "title" in info
            assert "description" in info
            assert "difficulty" in info
            assert "steps" in info
            assert "completed" in info
            assert "current_step" in info
