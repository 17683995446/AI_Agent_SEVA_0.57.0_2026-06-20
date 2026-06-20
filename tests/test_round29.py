"""
阶段39测试: 会话持久化与恢复

测试覆盖:
    1. SessionManager - 会话管理器
    2. CheckpointHook - 自动检查点钩子
    3. 便捷函数 (save_session/load_session/list_sessions等)
    4. CLI集成 (session命令)
    5. 状态恢复正确性
    6. 边界条件与错误处理
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
from seva import (quickstart, SessionManager, CheckpointHook,
                  save_session, load_session, list_sessions,
                  delete_session, get_session_info, checkpoint)
from seva.core import SelfEvolvingAgentCore
from seva.persistence import _get_version


# =====================================================================
# 测试夹具
# =====================================================================

@pytest.fixture
def agent():
    """创建一个运行了若干步的Agent"""
    agent, _ = quickstart(steps=20, preset="easy", seed=42)
    return agent


@pytest.fixture
def temp_dir():
    """临时目录"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def session_file(agent, temp_dir):
    """创建一个会话文件"""
    path = os.path.join(temp_dir, "test_session.json")
    save_session(agent, path)
    return path


# =====================================================================
# 1. SessionManager 基础测试
# =====================================================================

class TestSessionManagerBasic:
    """SessionManager基础功能测试"""

    def test01_init_default_dir(self):
        """测试01: 默认目录初始化"""
        mgr = SessionManager()
        assert mgr.default_dir == "."

    def test02_init_custom_dir(self):
        """测试02: 自定义目录初始化"""
        mgr = SessionManager("/tmp/sessions/")
        assert mgr.default_dir == "/tmp/sessions/"

    def test03_supported_formats(self):
        """测试03: 支持的格式"""
        assert "json" in SessionManager.SUPPORTED_FORMATS

    def test04_session_extension(self):
        """测试04: 会话文件扩展名"""
        assert SessionManager.SESSION_EXTENSION == ".json"


# =====================================================================
# 2. 会话保存测试
# =====================================================================

class TestSaveSession:
    """会话保存测试"""

    def test05_save_to_file(self, agent, temp_dir):
        """测试05: 保存到指定文件"""
        path = os.path.join(temp_dir, "session.json")
        result = save_session(agent, path)
        assert result == path
        assert os.path.exists(path)

    def test06_save_with_name(self, agent, temp_dir):
        """测试06: 用名称保存"""
        mgr = SessionManager(temp_dir)
        path = mgr.save_session(agent, name="my_session")
        assert os.path.exists(path)
        assert "my_session.json" in path

    def test07_save_default_name(self, agent, temp_dir):
        """测试07: 默认名称"""
        mgr = SessionManager(temp_dir)
        path = mgr.save_session(agent)
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test08_save_creates_directory(self, agent, temp_dir):
        """测试08: 自动创建目录"""
        path = os.path.join(temp_dir, "sub", "deep", "session.json")
        save_session(agent, path)
        assert os.path.exists(path)

    def test09_save_file_is_valid_json(self, agent, temp_dir):
        """测试09: 保存的文件是有效JSON"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test10_save_contains_meta(self, agent, temp_dir):
        """测试10: 保存内容包含元数据"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "meta" in data
        assert "timestamp" in data["meta"]
        assert "version" in data["meta"]

    def test11_save_contains_config(self, agent, temp_dir):
        """测试11: 保存内容包含配置"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "config" in data

    def test12_save_contains_runtime(self, agent, temp_dir):
        """测试12: 保存内容包含运行时状态"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "runtime" in data
        assert data["runtime"]["cycle_count"] == agent.cycle_count
        assert data["runtime"]["mode"] == agent.mode

    def test13_save_contains_metrics(self, agent, temp_dir):
        """测试13: 保存内容包含指标"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "metrics" in data
        assert "total_reward" in data["metrics"]

    def test14_save_contains_status(self, agent, temp_dir):
        """测试14: 保存内容包含状态摘要"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "status" in data

    def test15_save_contains_module_stats(self, agent, temp_dir):
        """测试15: 保存内容包含模块统计"""
        path = os.path.join(temp_dir, "session.json")
        save_session(agent, path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "module_stats" in data


# =====================================================================
# 3. 会话加载测试
# =====================================================================

class TestLoadSession:
    """会话加载测试"""

    def test16_load_basic(self, session_file):
        """测试16: 基本加载"""
        agent2 = load_session(session_file)
        assert agent2 is not None
        assert isinstance(agent2, SelfEvolvingAgentCore)

    def test17_load_restores_cycle_count(self, agent, session_file):
        """测试17: 恢复周期数"""
        agent2 = load_session(session_file)
        assert agent2.cycle_count == agent.cycle_count

    def test18_load_restores_mode(self, agent, session_file):
        """测试18: 恢复运行模式"""
        agent2 = load_session(session_file)
        assert agent2.mode == agent.mode

    def test19_load_restores_metrics(self, agent, session_file):
        """测试19: 恢复指标"""
        agent2 = load_session(session_file)
        assert agent2.metrics["total_reward"] == agent.metrics["total_reward"]
        assert (agent2.metrics["prediction_errors"] ==
                agent.metrics["prediction_errors"])

    def test20_load_restores_last_cycle_log(self, agent, session_file):
        """测试20: 恢复最后周期日志"""
        agent2 = load_session(session_file)
        assert agent2.last_cycle_log == agent.last_cycle_log

    def test21_load_file_not_found(self, temp_dir):
        """测试21: 加载不存在的文件"""
        path = os.path.join(temp_dir, "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            load_session(path)

    def test22_load_can_run_after(self, session_file):
        """测试22: 加载后可以继续运行"""
        agent2 = load_session(session_file)
        original_count = agent2.cycle_count
        agent2.run_cycle()
        assert agent2.cycle_count == original_count + 1

    def test23_load_preserves_config(self, agent, session_file):
        """测试23: 加载后配置一致"""
        agent2 = load_session(session_file)
        assert (agent2.config.get("core", "max_cycles") ==
                agent.config.get("core", "max_cycles"))


# =====================================================================
# 4. 会话管理测试
# =====================================================================

class TestSessionManagement:
    """会话管理测试"""

    def test24_list_sessions_empty(self, temp_dir):
        """测试24: 空目录列出会话"""
        sessions = list_sessions(temp_dir)
        assert sessions == []

    def test25_list_sessions_with_files(self, agent, temp_dir):
        """测试25: 列出会话文件"""
        save_session(agent, os.path.join(temp_dir, "s1.json"))
        save_session(agent, os.path.join(temp_dir, "s2.json"))
        sessions = list_sessions(temp_dir)
        assert len(sessions) == 2

    def test26_list_sessions_contains_info(self, agent, temp_dir):
        """测试26: 列出会话包含信息"""
        save_session(agent, os.path.join(temp_dir, "s1.json"))
        sessions = list_sessions(temp_dir)
        s = sessions[0]
        assert "file_path" in s
        assert "name" in s
        assert "size" in s
        assert "modified" in s
        assert "cycle_count" in s

    def test27_list_sessions_sorted_by_time(self, agent, temp_dir):
        """测试27: 按时间排序"""
        save_session(agent, os.path.join(temp_dir, "old.json"))
        time.sleep(0.1)
        save_session(agent, os.path.join(temp_dir, "new.json"))
        sessions = list_sessions(temp_dir)
        assert sessions[0]["name"] == "new.json"

    def test28_delete_session(self, agent, temp_dir):
        """测试28: 删除会话"""
        path = os.path.join(temp_dir, "to_delete.json")
        save_session(agent, path)
        assert os.path.exists(path)
        result = delete_session(path)
        assert result is True
        assert not os.path.exists(path)

    def test29_delete_nonexistent(self, temp_dir):
        """测试29: 删除不存在的文件"""
        path = os.path.join(temp_dir, "nonexistent.json")
        result = delete_session(path)
        assert result is False

    def test30_get_session_info(self, agent, session_file):
        """测试30: 获取会话信息"""
        info = get_session_info(session_file)
        assert "meta" in info
        assert "runtime" in info
        assert "file_size" in info
        assert "file_path" in info

    def test31_get_session_info_not_found(self, temp_dir):
        """测试31: 获取不存在会话信息"""
        path = os.path.join(temp_dir, "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            get_session_info(path)


# =====================================================================
# 5. 检查点测试
# =====================================================================

class TestCheckpoint:
    """检查点功能测试"""

    def test32_checkpoint_basic(self, agent, temp_dir):
        """测试32: 基本检查点"""
        mgr = SessionManager(temp_dir)
        path = mgr.checkpoint(agent)
        assert os.path.exists(path)
        assert "checkpoint" in path

    def test33_checkpoint_with_name(self, agent, temp_dir):
        """测试33: 带名称检查点"""
        mgr = SessionManager(temp_dir)
        path = mgr.checkpoint(agent, name="my_ckpt")
        assert "my_ckpt" in path

    def test34_checkpoint_creates_directory(self, agent, temp_dir):
        """测试34: 检查点创建目录"""
        mgr = SessionManager()
        new_dir = os.path.join(temp_dir, "checkpoints")
        path = mgr.checkpoint(agent, directory=new_dir)
        assert os.path.exists(new_dir)
        assert os.path.exists(path)

    def test35_checkpoint_max_keep(self, agent, temp_dir):
        """测试35: 检查点保留数量限制"""
        mgr = SessionManager(temp_dir)
        # 创建3个检查点，保留2个
        for i in range(3):
            time.sleep(0.01)  # 确保时间戳不同
            mgr.checkpoint(agent, name="ckpt", max_keep=2)
        files = [f for f in os.listdir(temp_dir) if f.startswith("ckpt_")]
        assert len(files) <= 2

    def test36_checkpoint_function(self, agent, temp_dir):
        """测试36: checkpoint便捷函数"""
        path = checkpoint(agent, directory=temp_dir)
        assert os.path.exists(path)


# =====================================================================
# 6. CheckpointHook 测试
# =====================================================================

class TestCheckpointHook:
    """自动检查点钩子测试"""

    def test37_hook_init(self, temp_dir):
        """测试37: 钩子初始化"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=10, max_keep=3)
        assert hook.interval == 10
        assert hook.max_keep == 3

    def test38_hook_no_checkpoint_before_interval(self, agent, temp_dir):
        """测试38: 未到间隔不检查点"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=100)
        result = hook(agent, 50)  # 50 < 100
        assert result is None

    def test39_hook_checkpoint_at_interval(self, agent, temp_dir):
        """测试39: 到间隔创建检查点"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=10)
        result = hook(agent, 10)
        assert result is not None
        assert os.path.exists(result)

    def test40_hook_multiple_checkpoints(self, agent, temp_dir):
        """测试40: 多次检查点"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=10, name="auto")
        results = []
        for cycle in [10, 20, 30]:
            r = hook(agent, cycle)
            if r:
                results.append(r)
        assert len(results) == 3

    def test41_hook_reset(self, agent, temp_dir):
        """测试41: 重置钩子"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=10)
        hook(agent, 10)  # 创建一个检查点
        hook.reset()
        assert hook._last_checkpoint == 0

    def test42_hook_get_checkpoints(self, agent, temp_dir):
        """测试42: 获取检查点列表"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=10)
        hook(agent, 10)
        hook(agent, 20)
        ckpts = hook.get_checkpoints()
        assert len(ckpts) == 2

    def test43_hook_default_cycle_count(self, agent, temp_dir):
        """测试43: 默认使用agent.cycle_count"""
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=1)
        # agent.cycle_count >= 1 (运行了20步)
        result = hook(agent)  # 不传cycle_count
        assert result is not None


# =====================================================================
# 7. 便捷函数测试
# =====================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test44_save_session_function(self, agent, temp_dir):
        """测试44: save_session便捷函数"""
        path = os.path.join(temp_dir, "conv.json")
        result = save_session(agent, path)
        assert result == path
        assert os.path.exists(path)

    def test45_load_session_function(self, agent, temp_dir):
        """测试45: load_session便捷函数"""
        path = os.path.join(temp_dir, "conv.json")
        save_session(agent, path)
        agent2 = load_session(path)
        assert agent2.cycle_count == agent.cycle_count

    def test46_list_sessions_function(self, agent, temp_dir):
        """测试46: list_sessions便捷函数"""
        save_session(agent, os.path.join(temp_dir, "a.json"))
        sessions = list_sessions(temp_dir)
        assert len(sessions) == 1

    def test47_delete_session_function(self, agent, temp_dir):
        """测试47: delete_session便捷函数"""
        path = os.path.join(temp_dir, "del.json")
        save_session(agent, path)
        assert delete_session(path) is True

    def test48_get_session_info_function(self, agent, temp_dir):
        """测试48: get_session_info便捷函数"""
        path = os.path.join(temp_dir, "info.json")
        save_session(agent, path)
        info = get_session_info(path)
        assert "file_size" in info

    def test49_checkpoint_function(self, agent, temp_dir):
        """测试49: checkpoint便捷函数"""
        path = checkpoint(agent, directory=temp_dir)
        assert os.path.exists(path)


# =====================================================================
# 8. 状态恢复正确性测试
# =====================================================================

class TestStateRestoration:
    """状态恢复正确性测试"""

    def test50_restore_different_cycle_counts(self, temp_dir):
        """测试50: 恢复不同周期数"""
        agent1, _ = quickstart(steps=10, preset="easy", seed=42)
        path1 = os.path.join(temp_dir, "s10.json")
        save_session(agent1, path1)

        agent2, _ = quickstart(steps=50, preset="easy", seed=42)
        path2 = os.path.join(temp_dir, "s50.json")
        save_session(agent2, path2)

        loaded1 = load_session(path1)
        loaded2 = load_session(path2)
        assert loaded1.cycle_count == 10
        assert loaded2.cycle_count == 50

    def test51_restore_metrics_history(self, temp_dir):
        """测试51: 恢复指标历史"""
        agent, _ = quickstart(steps=30, preset="easy", seed=42)
        path = os.path.join(temp_dir, "metrics.json")
        save_session(agent, path)

        loaded = load_session(path)
        assert (len(loaded.metrics["prediction_errors"]) ==
                len(agent.metrics["prediction_errors"]))
        assert (len(loaded.metrics["cycle_times"]) ==
                len(agent.metrics["cycle_times"]))

    def test52_restore_total_reward(self, temp_dir):
        """测试52: 恢复总奖励"""
        agent, _ = quickstart(steps=20, preset="easy", seed=42)
        original_reward = agent.metrics["total_reward"]
        path = os.path.join(temp_dir, "reward.json")
        save_session(agent, path)

        loaded = load_session(path)
        assert loaded.metrics["total_reward"] == original_reward

    def test53_restore_mode(self, temp_dir):
        """测试53: 恢复运行模式"""
        agent, _ = quickstart(steps=10, preset="easy", seed=42)
        path = os.path.join(temp_dir, "mode.json")
        save_session(agent, path)

        loaded = load_session(path)
        assert loaded.mode == agent.mode

    def test54_restore_and_continue(self, temp_dir):
        """测试54: 恢复后继续运行"""
        agent, _ = quickstart(steps=20, preset="easy", seed=42)
        path = os.path.join(temp_dir, "continue.json")
        save_session(agent, path)

        loaded = load_session(path)
        original_count = loaded.cycle_count
        # 继续运行5步
        for _ in range(5):
            loaded.run_cycle()
        assert loaded.cycle_count == original_count + 5

    def test55_restore_last_cycle_log(self, temp_dir):
        """测试55: 恢复最后周期日志"""
        agent, _ = quickstart(steps=10, preset="easy", seed=42)
        path = os.path.join(temp_dir, "log.json")
        save_session(agent, path)

        loaded = load_session(path)
        assert loaded.last_cycle_log == agent.last_cycle_log


# =====================================================================
# 9. 包导出与版本测试
# =====================================================================

class TestPackageExports:
    """包导出测试"""

    def test56_session_manager_exported(self):
        """测试56: SessionManager已导出"""
        assert hasattr(seva, "SessionManager")
        assert "SessionManager" in seva.__all__

    def test57_checkpoint_hook_exported(self):
        """测试57: CheckpointHook已导出"""
        assert hasattr(seva, "CheckpointHook")
        assert "CheckpointHook" in seva.__all__

    def test58_save_session_exported(self):
        """测试58: save_session已导出"""
        assert hasattr(seva, "save_session")
        assert "save_session" in seva.__all__

    def test59_load_session_exported(self):
        """测试59: load_session已导出"""
        assert hasattr(seva, "load_session")
        assert "load_session" in seva.__all__

    def test60_list_sessions_exported(self):
        """测试60: list_sessions已导出"""
        assert hasattr(seva, "list_sessions")
        assert "list_sessions" in seva.__all__

    def test61_delete_session_exported(self):
        """测试61: delete_session已导出"""
        assert hasattr(seva, "delete_session")
        assert "delete_session" in seva.__all__

    def test62_get_session_info_exported(self):
        """测试62: get_session_info已导出"""
        assert hasattr(seva, "get_session_info")
        assert "get_session_info" in seva.__all__

    def test63_checkpoint_exported(self):
        """测试63: checkpoint已导出"""
        assert hasattr(seva, "checkpoint")
        assert "checkpoint" in seva.__all__

    def test64_version_updated(self):
        """测试64: 版本号已更新"""
        assert seva.__version__ >= "0.39.0"

    def test65_get_version_helper(self):
        """测试65: 版本号辅助函数"""
        v = _get_version()
        assert isinstance(v, str)
        assert v != "unknown"


# =====================================================================
# 10. CLI集成测试
# =====================================================================

class TestCLIIntegration:
    """CLI集成测试"""

    def test66_cli_session_save(self):
        """测试66: CLI session save命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "save", "--steps", "0",
                                  "--output", "test.json"])
        assert args.action == "save"
        assert args.output == "test.json"
        assert hasattr(args, "func")

    def test67_cli_session_load(self):
        """测试67: CLI session load命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "load",
                                  "--input", "test.json"])
        assert args.action == "load"
        assert args.input == "test.json"

    def test68_cli_session_list(self):
        """测试68: CLI session list命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "list",
                                  "--directory", "/tmp"])
        assert args.action == "list"
        assert args.directory == "/tmp"

    def test69_cli_session_info(self):
        """测试69: CLI session info命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "info",
                                  "--input", "test.json"])
        assert args.action == "info"

    def test70_cli_session_delete(self):
        """测试70: CLI session delete命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "delete",
                                  "--input", "test.json"])
        assert args.action == "delete"

    def test71_cli_session_checkpoint(self):
        """测试71: CLI session checkpoint命令"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["session", "checkpoint",
                                  "--steps", "0", "--max-keep", "3"])
        assert args.action == "checkpoint"
        assert args.max_keep == 3


# =====================================================================
# 11. 端到端集成测试
# =====================================================================

class TestEndToEnd:
    """端到端集成测试"""

    def test72_full_save_load_cycle(self, temp_dir):
        """测试72: 完整保存-加载循环"""
        # 创建并运行
        agent, _ = quickstart(steps=30, preset="easy", seed=42)
        original_count = agent.cycle_count
        original_reward = agent.metrics["total_reward"]

        # 保存
        path = os.path.join(temp_dir, "e2e.json")
        save_session(agent, path)

        # 加载
        loaded = load_session(path)

        # 验证
        assert loaded.cycle_count == original_count
        assert loaded.metrics["total_reward"] == original_reward

    def test73_multiple_save_load(self, temp_dir):
        """测试73: 多次保存加载"""
        agent, _ = quickstart(steps=10, preset="easy", seed=42)

        # 第一次保存
        path1 = os.path.join(temp_dir, "v1.json")
        save_session(agent, path1)

        # 继续运行
        for _ in range(10):
            agent.run_cycle()

        # 第二次保存
        path2 = os.path.join(temp_dir, "v2.json")
        save_session(agent, path2)

        # 加载两个版本
        v1 = load_session(path1)
        v2 = load_session(path2)
        assert v1.cycle_count == 10
        assert v2.cycle_count == 20

    def test74_checkpoint_hook_integration(self, temp_dir):
        """测试74: 检查点钩子集成"""
        agent, _ = quickstart(steps=0, preset="easy", seed=42)
        mgr = SessionManager(temp_dir)
        hook = CheckpointHook(mgr, interval=5, max_keep=3, name="integ")

        # 模拟运行循环
        for _ in range(20):
            agent.run_cycle()
            hook(agent, agent.cycle_count)

        # 验证检查点已创建
        ckpts = hook.get_checkpoints()
        assert len(ckpts) > 0
        for c in ckpts:
            assert os.path.exists(c)

    def test75_session_manager_all_operations(self, agent, temp_dir):
        """测试75: SessionManager所有操作"""
        mgr = SessionManager(temp_dir)

        # 保存
        path = mgr.save_session(agent, name="all_ops")
        assert os.path.exists(path)

        # 列出
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "all_ops.json"

        # 获取信息
        info = mgr.get_session_info(path)
        assert info["runtime"]["cycle_count"] == agent.cycle_count

        # 加载
        loaded = mgr.load_session(path)
        assert loaded.cycle_count == agent.cycle_count

        # 删除
        assert mgr.delete_session(path) is True
        assert not os.path.exists(path)

    def test76_save_load_different_presets(self, temp_dir):
        """测试76: 不同预设的保存加载"""
        for preset in ["easy", "standard"]:
            agent, _ = quickstart(steps=10, preset=preset, seed=42)
            path = os.path.join(temp_dir, f"{preset}.json")
            save_session(agent, path)

            loaded = load_session(path)
            assert loaded.cycle_count == 10

    def test77_session_file_size_reasonable(self, agent, temp_dir):
        """测试77: 会话文件大小合理"""
        path = os.path.join(temp_dir, "size.json")
        save_session(agent, path)
        size = os.path.getsize(path)
        # 应该大于1KB（有实际数据）小于10MB（不会过大）
        assert size > 1024
        assert size < 10 * 1024 * 1024
