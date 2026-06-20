"""
阶段42测试: 智能命令补全与提示

测试覆盖:
    1. Levenshtein距离算法
    2. CommandCompleter - 命令补全器
    3. CommandHistory - 命令历史管理
    4. ShellCompletionGenerator - Shell补全脚本生成
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
from seva import (CommandCompleter, CommandHistory,
                  ShellCompletionGenerator,
                  suggest_command, complete_command,
                  did_you_mean, get_command_help,
                  levenshtein_distance, similarity_score)


# =====================================================================
# 1. Levenshtein距离算法测试
# =====================================================================

class TestLevenshtein:
    """Levenshtein距离算法测试"""

    def test01_identical_strings(self):
        """测试01: 相同字符串距离为0"""
        assert levenshtein_distance("hello", "hello") == 0

    def test02_empty_strings(self):
        """测试02: 空字符串"""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test03_one_edit(self):
        """测试03: 一次编辑"""
        assert levenshtein_distance("cat", "cats") == 1  # 插入
        assert levenshtein_distance("cats", "cat") == 1  # 删除
        assert levenshtein_distance("cat", "bat") == 1   # 替换

    def test04_multiple_edits(self):
        """测试04: 多次编辑"""
        assert levenshtein_distance("kitten", "sitting") == 3

    def test05_case_insensitive_similarity(self):
        """测试05: 相似度计算"""
        assert similarity_score("hello", "hello") == 1.0
        assert similarity_score("", "") == 1.0
        assert similarity_score("abc", "") == 0.0
        assert 0 < similarity_score("cat", "bat") < 1.0

    def test06_similarity_high(self):
        """测试06: 高相似度"""
        score = similarity_score("status", "statys")
        assert score > 0.8

    def test07_similarity_low(self):
        """测试07: 低相似度"""
        score = similarity_score("abc", "xyz")
        assert score < 0.5


# =====================================================================
# 2. CommandCompleter 测试
# =====================================================================

class TestCommandCompleter:
    """命令补全器测试"""

    def test08_init(self):
        """测试08: 初始化"""
        completer = CommandCompleter()
        assert len(completer.get_all_commands()) > 0

    def test09_complete_exact(self):
        """测试09: 精确前缀匹配"""
        completer = CommandCompleter()
        result = completer.complete("run")
        assert "run" in result

    def test10_complete_prefix(self):
        """测试10: 前缀匹配"""
        completer = CommandCompleter()
        result = completer.complete("s")
        assert "status" in result
        assert "session" in result
        assert "scenarios" in result

    def test11_complete_empty(self):
        """测试11: 空前缀返回所有"""
        completer = CommandCompleter()
        result = completer.complete("")
        assert len(result) > 0
        assert result == sorted(result)

    def test12_complete_no_match(self):
        """测试12: 无匹配"""
        completer = CommandCompleter()
        result = completer.complete("xyz123")
        assert result == []

    def test13_complete_case_insensitive(self):
        """测试13: 不区分大小写"""
        completer = CommandCompleter()
        result = completer.complete("RUN")
        assert "run" in result

    def test14_fuzzy_match(self):
        """测试14: 模糊匹配"""
        completer = CommandCompleter()
        result = completer.fuzzy_match("ess")
        assert "session" in result

    def test15_fuzzy_match_empty(self):
        """测试15: 空查询"""
        completer = CommandCompleter()
        assert completer.fuzzy_match("") == []

    def test16_suggest_exact(self):
        """测试16: 精确建议"""
        completer = CommandCompleter()
        result = completer.suggest("status")
        assert "status" in result

    def test17_suggest_typo(self):
        """测试17: 拼写错误建议"""
        completer = CommandCompleter()
        result = completer.suggest("statys")
        assert "status" in result

    def test18_suggest_close(self):
        """测试18: 相似命令建议"""
        completer = CommandCompleter()
        result = completer.suggest("ruun")
        assert "run" in result

    def test19_suggest_no_match(self):
        """测试19: 无相似命令"""
        completer = CommandCompleter()
        result = completer.suggest("xyz123")
        assert result == []

    def test20_suggest_max_limit(self):
        """测试20: 建议数量限制"""
        completer = CommandCompleter()
        result = completer.suggest("s")
        assert len(result) <= CommandCompleter.MAX_SUGGESTIONS

    def test21_get_command_info(self):
        """测试21: 获取命令信息"""
        completer = CommandCompleter()
        info = completer.get_command_info("run")
        assert info is not None
        assert "description" in info
        assert "args" in info

    def test22_get_command_info_not_found(self):
        """测试22: 不存在的命令"""
        completer = CommandCompleter()
        assert completer.get_command_info("nonexistent") is None

    def test23_complete_args(self):
        """测试23: 参数补全"""
        completer = CommandCompleter()
        result = completer.complete_args("run", "--st")
        assert "--steps" in result

    def test24_complete_args_empty(self):
        """测试24: 空前缀参数"""
        completer = CommandCompleter()
        result = completer.complete_args("run", "")
        assert len(result) > 0

    def test25_complete_args_unknown_cmd(self):
        """测试25: 未知命令的参数"""
        completer = CommandCompleter()
        assert completer.complete_args("nonexistent", "--x") == []

    def test26_complete_subcommands(self):
        """测试26: 子命令补全"""
        completer = CommandCompleter()
        result = completer.complete_subcommands("config", "s")
        assert "show" in result
        assert "save" in result

    def test27_complete_subcommands_empty(self):
        """测试27: 空前缀子命令"""
        completer = CommandCompleter()
        result = completer.complete_subcommands("session", "")
        assert "save" in result
        assert "load" in result

    def test28_get_all_commands(self):
        """测试28: 获取所有命令"""
        completer = CommandCompleter()
        commands = completer.get_all_commands()
        assert "run" in commands
        assert "status" in commands
        assert commands == sorted(commands)

    def test29_format_suggestions(self):
        """测试29: 格式化建议"""
        completer = CommandCompleter()
        text = completer.format_suggestions("stat", ["status"])
        assert "status" in text
        assert "您是不是想输入" in text

    def test30_format_suggestions_empty(self):
        """测试30: 空建议格式化"""
        completer = CommandCompleter()
        text = completer.format_suggestions("xyz", [])
        assert "未找到" in text

    def test31_commands_have_descriptions(self):
        """测试31: 所有命令都有描述"""
        completer = CommandCompleter()
        for cmd in completer.get_all_commands():
            info = completer.get_command_info(cmd)
            assert info is not None
            assert "description" in info
            assert len(info["description"]) > 0


# =====================================================================
# 3. CommandHistory 测试
# =====================================================================

class TestCommandHistory:
    """命令历史管理测试"""

    def test32_init_default(self, tmp_path):
        """测试32: 默认初始化"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        assert history.size() == 0

    def test33_add_command(self, tmp_path):
        """测试33: 添加命令"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.add("run --steps 100")
        assert history.size() == 1

    def test34_add_empty_command(self, tmp_path):
        """测试34: 添加空命令"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.add("")
        history.add("   ")
        assert history.size() == 0

    def test35_search(self, tmp_path):
        """测试35: 搜索命令"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.add("run --steps 100")
        history.add("status")
        history.add("run --steps 50")
        result = history.search("run")
        assert len(result) == 2
        assert "run --steps 50" in result

    def test36_search_no_match(self, tmp_path):
        """测试36: 搜索无匹配"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.add("run")
        assert history.search("xyz") == []

    def test37_get_recent(self, tmp_path):
        """测试37: 获取最近命令"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        for i in range(5):
            history.add(f"cmd{i}")
        recent = history.get_recent(3)
        assert len(recent) == 3
        assert "cmd4" in recent

    def test38_clear(self, tmp_path):
        """测试38: 清空历史"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.add("run")
        history.clear()
        assert history.size() == 0

    def test39_persistence(self, tmp_path):
        """测试39: 历史持久化"""
        history_file = str(tmp_path / "history.json")
        history1 = CommandHistory(history_file=history_file)
        history1.add("run --steps 100")

        # 重新加载
        history2 = CommandHistory(history_file=history_file)
        assert history2.size() == 1
        assert "run --steps 100" in history2.get_recent(1)

    def test40_max_history_limit(self, tmp_path):
        """测试40: 历史长度限制"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)
        history.MAX_HISTORY = 5
        for i in range(10):
            history.add(f"cmd{i}")
        assert history.size() <= 5


# =====================================================================
# 4. ShellCompletionGenerator 测试
# =====================================================================

class TestShellCompletionGenerator:
    """Shell补全脚本生成测试"""

    def test41_generate_bash(self):
        """测试41: 生成Bash补全脚本"""
        gen = ShellCompletionGenerator()
        script = gen.generate_bash()
        assert "_seva_completions" in script
        assert "complete" in script
        assert "seva" in script

    def test42_generate_zsh(self):
        """测试42: 生成Zsh补全脚本"""
        gen = ShellCompletionGenerator()
        script = gen.generate_zsh()
        assert "#compdef" in script
        assert "seva" in script

    def test43_save_bash(self, tmp_path):
        """测试43: 保存Bash脚本"""
        gen = ShellCompletionGenerator()
        path = str(tmp_path / "completion.sh")
        result = gen.save_bash(path)
        assert result == path
        assert os.path.exists(path)
        with open(path) as f:
            assert "_seva_completions" in f.read()

    def test44_save_zsh(self, tmp_path):
        """测试44: 保存Zsh脚本"""
        gen = ShellCompletionGenerator()
        path = str(tmp_path / "_seva")
        result = gen.save_zsh(path)
        assert result == path
        assert os.path.exists(path)

    def test45_bash_contains_commands(self):
        """测试45: Bash脚本包含所有命令"""
        gen = ShellCompletionGenerator()
        script = gen.generate_bash()
        completer = CommandCompleter()
        for cmd in completer.get_all_commands():
            assert cmd in script


# =====================================================================
# 5. 便捷函数测试
# =====================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test46_suggest_command(self):
        """测试46: suggest_command函数"""
        result = suggest_command("statys")
        assert "status" in result

    def test47_complete_command(self):
        """测试47: complete_command函数"""
        result = complete_command("ru")
        assert "run" in result

    def test48_did_you_mean(self):
        """测试48: did_you_mean函数"""
        text = did_you_mean("statys")
        assert "status" in text
        assert "您是不是想输入" in text

    def test49_did_you_mean_no_match(self):
        """测试49: 无匹配的did_you_mean"""
        text = did_you_mean("xyz123456")
        assert "未找到" in text

    def test50_get_command_help(self):
        """测试50: get_command_help函数"""
        help_text = get_command_help("run")
        assert help_text is not None
        assert "run" in help_text
        assert "描述" in help_text

    def test51_get_command_help_not_found(self):
        """测试51: 不存在的命令帮助"""
        assert get_command_help("nonexistent") is None


# =====================================================================
# 6. 包导出与版本测试
# =====================================================================

class TestPackageExports:
    """包导出测试"""

    def test52_command_completer_exported(self):
        """测试52: CommandCompleter已导出"""
        assert hasattr(seva, "CommandCompleter")
        assert "CommandCompleter" in seva.__all__

    def test53_command_history_exported(self):
        """测试53: CommandHistory已导出"""
        assert hasattr(seva, "CommandHistory")
        assert "CommandHistory" in seva.__all__

    def test54_shell_completion_generator_exported(self):
        """测试54: ShellCompletionGenerator已导出"""
        assert hasattr(seva, "ShellCompletionGenerator")
        assert "ShellCompletionGenerator" in seva.__all__

    def test55_suggest_command_exported(self):
        """测试55: suggest_command已导出"""
        assert hasattr(seva, "suggest_command")
        assert "suggest_command" in seva.__all__

    def test56_levenshtein_exported(self):
        """测试56: levenshtein_distance已导出"""
        assert hasattr(seva, "levenshtein_distance")
        assert "levenshtein_distance" in seva.__all__

    def test57_similarity_score_exported(self):
        """测试57: similarity_score已导出"""
        assert hasattr(seva, "similarity_score")
        assert "similarity_score" in seva.__all__

    def test58_version_updated(self):
        """测试58: 版本号已更新"""
        assert seva.__version__ >= "0.42.0"


# =====================================================================
# 7. CLI集成测试
# =====================================================================

class TestCLIIntegration:
    """CLI集成测试"""

    def test59_cli_completion_bash(self):
        """测试59: CLI completion bash"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["completion", "--shell", "bash"])
        assert args.shell == "bash"
        assert hasattr(args, "func")

    def test60_cli_completion_zsh(self):
        """测试60: CLI completion zsh"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["completion", "--shell", "zsh"])
        assert args.shell == "zsh"

    def test61_cli_completion_suggest(self):
        """测试61: CLI completion --suggest"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["completion", "--suggest", "statys"])
        assert args.suggest == "statys"

    def test62_cli_completion_list(self):
        """测试62: CLI completion --list"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["completion", "--list"])
        assert args.list is True

    def test63_cli_completion_help_cmd(self):
        """测试63: CLI completion --help-cmd"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["completion", "--help-cmd", "run"])
        assert args.help_cmd == "run"

    def test64_cli_completion_output(self, tmp_path):
        """测试64: CLI completion --output"""
        from seva.cli import build_parser
        parser = build_parser()
        output = str(tmp_path / "completion.sh")
        args = parser.parse_args(["completion", "--output", output])
        assert args.output == output


# =====================================================================
# 8. 端到端集成测试
# =====================================================================

class TestEndToEnd:
    """端到端集成测试"""

    def test65_full_completion_flow(self):
        """测试65: 完整补全流程"""
        completer = CommandCompleter()

        # 1. 前缀补全
        assert "run" in completer.complete("ru")

        # 2. 拼写纠错
        assert "run" in completer.suggest("ruun")

        # 3. 参数补全
        assert "--steps" in completer.complete_args("run", "--st")

        # 4. 子命令补全
        assert "show" in completer.complete_subcommands("config", "sh")

    def test66_history_workflow(self, tmp_path):
        """测试66: 历史工作流"""
        history_file = str(tmp_path / "history.json")
        history = CommandHistory(history_file=history_file)

        # 添加命令
        history.add("run --steps 100")
        history.add("status")
        history.add("run --steps 50")

        # 搜索
        assert len(history.search("run")) == 2

        # 获取最近
        recent = history.get_recent(2)
        assert len(recent) == 2

        # 清空
        history.clear()
        assert history.size() == 0

    def test67_shell_completion_generation(self, tmp_path):
        """测试67: Shell补全脚本生成"""
        gen = ShellCompletionGenerator()

        # 生成Bash
        bash_script = gen.generate_bash()
        assert len(bash_script) > 100

        # 生成Zsh
        zsh_script = gen.generate_zsh()
        assert len(zsh_script) > 100

        # 保存
        bash_path = str(tmp_path / "bash.sh")
        zsh_path = str(tmp_path / "zsh.sh")
        gen.save_bash(bash_path)
        gen.save_zsh(zsh_path)
        assert os.path.exists(bash_path)
        assert os.path.exists(zsh_path)

    def test68_typo_correction_scenarios(self):
        """测试68: 拼写纠错场景"""
        completer = CommandCompleter()

        test_cases = [
            ("statys", "status"),
            ("ruun", "run"),
            ("reprot", "report"),
            ("sesion", "session"),
            ("wizrd", "wizard"),
        ]

        for typo, expected in test_cases:
            suggestions = completer.suggest(typo)
            assert expected in suggestions, \
                f"建议 '{typo}' 应包含 '{expected}'"

    def test69_all_commands_have_info(self):
        """测试69: 所有命令都有完整信息"""
        completer = CommandCompleter()
        for cmd in completer.get_all_commands():
            info = completer.get_command_info(cmd)
            assert info is not None
            assert "description" in info
            # 必须有args或subcommands之一
            assert "args" in info or "subcommands" in info
