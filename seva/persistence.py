"""
阶段39: 会话持久化与恢复

提供Agent会话的保存、加载、检查点功能，支持长时间运行任务的断点续传。

核心能力:
    1. 会话保存: 将Agent当前状态（配置+运行时状态+指标历史）序列化为JSON
    2. 会话加载: 从JSON文件恢复Agent到保存时的状态
    3. 检查点: 自动定时保存，支持保留最近N个检查点
    4. 会话管理: 列出、删除、查看会话信息

设计原则（第一性原理）:
    - 保存可重建的状态: 配置 + 运行时计数器 + 指标时序数据
    - 不保存复杂内部状态: 神经网络权重、HNSW索引等（重建成本高，ROI低）
    - 加载时创建新Agent并恢复运行时状态，模块内部从初始状态开始
    - 这足以支持: 监控数据连续性、运行进度恢复、配置复用

快速启动:
    from seva import quickstart, save_session, load_session

    agent, _ = quickstart(steps=100)
    path = save_session(agent, "my_session.json")  # 保存

    agent2 = load_session("my_session.json")  # 加载
    print(agent2.cycle_count)  # 100（恢复进度）

    from seva import SessionManager, CheckpointHook
    mgr = SessionManager("./sessions/")
    hook = CheckpointHook(mgr, interval=50)
    for i in range(200):
        agent.run_cycle()
        hook(agent, agent.cycle_count)  # 每50步自动检查点
"""
import os
import json
import time
import shutil
import glob
from typing import Any, Dict, List, Optional, Callable

from .core import SelfEvolvingAgentCore
from .config import Config
from .api import SEVAFactory, get_status


# =====================================================================
# 会话持久化管理器
# =====================================================================

class SessionManager:
    """
    会话持久化管理器

    提供Agent会话的保存、加载、检查点、管理功能。

    支持的格式: json
    保存内容: 配置 + 运行时状态 + 指标历史 + 模块统计摘要
    """

    SUPPORTED_FORMATS = ["json"]
    SESSION_EXTENSION = ".json"

    def __init__(self, default_dir: str = "."):
        """
        初始化会话管理器

        Args:
            default_dir: 默认会话保存目录
        """
        self.default_dir = default_dir

    def save_session(self, agent: SelfEvolvingAgentCore,
                     file_path: Optional[str] = None,
                     name: Optional[str] = None) -> str:
        """
        保存Agent会话到文件

        Args:
            agent: SEVA Agent实例
            file_path: 文件路径（可选，优先于name）
            name: 会话名称（可选，保存到default_dir/name.json）

        Returns:
            实际保存的文件路径

        Example:
            >>> mgr = SessionManager("./sessions/")
            >>> path = mgr.save_session(agent, name="run_001")
        """
        if file_path is None:
            if name is None:
                name = f"session_{int(time.time())}"
            if not name.endswith(self.SESSION_EXTENSION):
                name += self.SESSION_EXTENSION
            file_path = os.path.join(self.default_dir, name)

        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        session_data = self._serialize_agent(agent)
        self._write_file(file_path, session_data)
        return file_path

    def load_session(self, file_path: str) -> SelfEvolvingAgentCore:
        """
        从文件加载会话

        Args:
            file_path: 会话文件路径

        Returns:
            恢复后的Agent实例

        Example:
            >>> mgr = SessionManager()
            >>> agent = mgr.load_session("session.json")
            >>> print(agent.cycle_count)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"会话文件不存在: {file_path}")

        data = self._read_file(file_path)
        return self._deserialize_agent(data)

    def list_sessions(self, directory: Optional[str] = None) -> List[Dict]:
        """
        列出已保存的会话

        Args:
            directory: 目录路径（可选，默认使用default_dir）

        Returns:
            会话信息列表，每个元素包含:
            - file_path: 文件路径
            - name: 会话名称
            - size: 文件大小（字节）
            - modified: 最后修改时间戳
            - cycle_count: 保存时的周期数（从文件读取）
            - preset: 保存时的预设
        """
        directory = directory or self.default_dir
        if not os.path.exists(directory):
            return []

        pattern = os.path.join(directory, f"*{self.SESSION_EXTENSION}")
        files = sorted(glob.glob(pattern), key=os.path.getmtime,
                       reverse=True)

        sessions = []
        for f in files:
            info = {
                "file_path": f,
                "name": os.path.basename(f),
                "size": os.path.getsize(f),
                "modified": os.path.getmtime(f),
                "cycle_count": 0,
                "preset": "unknown",
            }
            # 尝试读取摘要信息
            try:
                data = self._read_file(f)
                info["cycle_count"] = data.get("runtime", {}).get(
                    "cycle_count", 0)
                info["preset"] = data.get("meta", {}).get("preset",
                                                          "unknown")
            except Exception:
                pass
            sessions.append(info)

        return sessions

    def delete_session(self, file_path: str) -> bool:
        """
        删除会话文件

        Args:
            file_path: 会话文件路径

        Returns:
            是否删除成功
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def checkpoint(self, agent: SelfEvolvingAgentCore,
                   directory: Optional[str] = None,
                   name: Optional[str] = None,
                   max_keep: int = 0) -> str:
        """
        创建检查点（带时间戳）

        Args:
            agent: Agent实例
            directory: 保存目录（可选，默认default_dir）
            name: 基础名称（可选）
            max_keep: 保留最近N个检查点（0=不限制，删除旧的）

        Returns:
            检查点文件路径
        """
        directory = directory or self.default_dir
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = name or "checkpoint"
        filename = f"{base_name}_{timestamp}{self.SESSION_EXTENSION}"
        file_path = os.path.join(directory, filename)

        # 保存
        self.save_session(agent, file_path=file_path)

        # 清理旧检查点
        if max_keep > 0:
            self._cleanup_checkpoints(directory, base_name, max_keep)

        return file_path

    def get_session_info(self, file_path: str) -> Dict:
        """
        获取会话信息（不加载Agent）

        Args:
            file_path: 会话文件路径

        Returns:
            会话信息字典，包含:
            - meta: 元数据（版本、时间戳、预设）
            - runtime: 运行时状态摘要
            - status: 状态摘要
            - file_size: 文件大小
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"会话文件不存在: {file_path}")

        data = self._read_file(file_path)
        return {
            "meta": data.get("meta", {}),
            "runtime": data.get("runtime", {}),
            "status": data.get("status", {}),
            "file_size": os.path.getsize(file_path),
            "file_path": file_path,
        }

    # ================ 内部方法 ================

    def _serialize_agent(self, agent: SelfEvolvingAgentCore) -> Dict:
        """序列化Agent状态"""
        # 获取配置
        config_dict = agent.config.to_dict()

        # 获取运行时状态
        runtime = {
            "cycle_count": agent.cycle_count,
            "mode": agent.mode,
            "is_running": agent.is_running,
            "env_done": getattr(agent, "_env_done", False),
            "last_cycle_log": agent.last_cycle_log,
        }

        # 获取指标（深拷贝避免引用问题）
        import copy
        metrics = copy.deepcopy(agent.metrics)

        # 获取状态摘要
        try:
            status = get_status(agent)
        except Exception:
            status = {}

        # 获取模块统计摘要（仅用于信息展示，不用于恢复）
        module_stats = {}
        for name, mod in agent.modules.items():
            try:
                if hasattr(mod, "get_stats"):
                    module_stats[name] = mod.get_stats()
                elif hasattr(mod, "get_status"):
                    module_stats[name] = mod.get_status()
                elif hasattr(mod, "get_progress_report"):
                    module_stats[name] = mod.get_progress_report()
            except Exception:
                pass

        # 推断预设
        preset = "unknown"
        try:
            from .api import PresetConfigs
            for p in PresetConfigs.list_presets():
                if PresetConfigs.get(p) == config_dict:
                    preset = p
                    break
        except Exception:
            pass

        return {
            "meta": {
                "version": _get_version(),
                "seva_version": _get_version(),
                "timestamp": time.time(),
                "format": "json",
                "preset": preset,
                "description": "SEVA Agent Session",
            },
            "config": config_dict,
            "runtime": runtime,
            "metrics": metrics,
            "status": status,
            "module_stats": module_stats,
        }

    def _deserialize_agent(self, data: Dict) -> SelfEvolvingAgentCore:
        """从数据反序列化Agent"""
        config_dict = data.get("config", {})
        runtime = data.get("runtime", {})
        metrics = data.get("metrics", {})

        # 直接用保存的配置创建Agent（确保配置完全一致）
        config = Config(instance_overrides=config_dict)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        # 附加环境（如果原始配置中有环境类型）
        env_type = config_dict.get("environment", {}).get(
            "env_type", "mixed")
        seed = config_dict.get("environment", {}).get("seed", 42)
        try:
            from .environment import SimulationEnvironment
            env = SimulationEnvironment(config, env_type=env_type, seed=seed)
            agent.attach_environment(env)
            obs = env.reset()
            agent.inject_input(obs)
        except Exception:
            pass  # 环境创建失败不影响加载

        # 恢复运行时状态
        agent.cycle_count = runtime.get("cycle_count", 0)
        agent.mode = runtime.get("mode", "normal")
        agent.is_running = runtime.get("is_running", False)
        if hasattr(agent, "_env_done"):
            agent._env_done = runtime.get("env_done", False)
        agent.last_cycle_log = runtime.get("last_cycle_log", [])

        # 恢复指标
        if metrics:
            import copy
            agent.metrics = copy.deepcopy(metrics)

        return agent

    def _cleanup_checkpoints(self, directory: str, base_name: str,
                             max_keep: int) -> None:
        """清理旧检查点，保留最近max_keep个"""
        pattern = os.path.join(directory, f"{base_name}_*{self.SESSION_EXTENSION}")
        files = sorted(glob.glob(pattern), key=os.path.getmtime,
                       reverse=True)
        # 删除超出数量的旧文件
        for f in files[max_keep:]:
            try:
                os.remove(f)
            except Exception:
                pass

    @staticmethod
    def _write_file(file_path: str, data: Dict) -> None:
        """写入JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _read_file(file_path: str) -> Dict:
        """读取JSON文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


# =====================================================================
# 自动检查点钩子
# =====================================================================

class CheckpointHook:
    """
    自动检查点钩子

    在Agent运行循环中定期保存检查点。

    Example:
        >>> from seva import quickstart, SessionManager, CheckpointHook
        >>> agent, _ = quickstart(steps=0)
        >>> mgr = SessionManager("./checkpoints/")
        >>> hook = CheckpointHook(mgr, interval=50, max_keep=3)
        >>> for i in range(200):
        ...     agent.run_cycle()
        ...     hook(agent, agent.cycle_count)
    """

    def __init__(self, manager: SessionManager,
                 directory: Optional[str] = None,
                 interval: int = 100,
                 max_keep: int = 5,
                 name: str = "checkpoint"):
        """
        初始化检查点钩子

        Args:
            manager: SessionManager实例
            directory: 检查点保存目录（可选，默认使用manager.default_dir）
            interval: 检查点间隔（周期数）
            max_keep: 保留最近N个检查点（0=不限制）
            name: 检查点基础名称
        """
        self.manager = manager
        self.directory = directory or manager.default_dir
        self.interval = interval
        self.max_keep = max_keep
        self.name = name
        self._last_checkpoint = 0
        self._checkpoints: List[str] = []

    def __call__(self, agent: SelfEvolvingAgentCore,
                 cycle_count: Optional[int] = None) -> Optional[str]:
        """
        检查是否需要创建检查点

        Args:
            agent: Agent实例
            cycle_count: 当前周期数（可选，默认使用agent.cycle_count）

        Returns:
            如果创建了检查点则返回文件路径，否则返回None
        """
        if cycle_count is None:
            cycle_count = agent.cycle_count

        if cycle_count >= self._last_checkpoint + self.interval:
            path = self.manager.checkpoint(
                agent,
                directory=self.directory,
                name=self.name,
                max_keep=self.max_keep
            )
            self._last_checkpoint = cycle_count
            self._checkpoints.append(path)
            return path
        return None

    def reset(self) -> None:
        """重置钩子状态"""
        self._last_checkpoint = 0
        self._checkpoints = []

    def get_checkpoints(self) -> List[str]:
        """获取已创建的检查点列表"""
        return list(self._checkpoints)


# =====================================================================
# 便捷函数
# =====================================================================

def save_session(agent: SelfEvolvingAgentCore,
                 file_path: Optional[str] = None,
                 name: Optional[str] = None) -> str:
    """
    保存Agent会话到文件

    Args:
        agent: SEVA Agent实例
        file_path: 文件路径（可选）
        name: 会话名称（可选，保存到当前目录）

    Returns:
        保存的文件路径

    Example:
        >>> from seva import quickstart, save_session
        >>> agent, _ = quickstart(steps=50)
        >>> path = save_session(agent, "my_session.json")
    """
    mgr = SessionManager()
    return mgr.save_session(agent, file_path=file_path, name=name)


def load_session(file_path: str) -> SelfEvolvingAgentCore:
    """
    从文件加载会话

    Args:
        file_path: 会话文件路径

    Returns:
        恢复后的Agent实例

    Example:
        >>> from seva import load_session
        >>> agent = load_session("my_session.json")
        >>> print(agent.cycle_count)
    """
    mgr = SessionManager()
    return mgr.load_session(file_path)


def list_sessions(directory: str = ".") -> List[Dict]:
    """
    列出目录中的会话文件

    Args:
        directory: 目录路径

    Returns:
        会话信息列表

    Example:
        >>> from seva import list_sessions
        >>> for s in list_sessions("./sessions/"):
        ...     print(s["name"], s["cycle_count"])
    """
    mgr = SessionManager(directory)
    return mgr.list_sessions()


def delete_session(file_path: str) -> bool:
    """
    删除会话文件

    Args:
        file_path: 会话文件路径

    Returns:
        是否删除成功
    """
    mgr = SessionManager()
    return mgr.delete_session(file_path)


def get_session_info(file_path: str) -> Dict:
    """
    获取会话信息（不加载Agent）

    Args:
        file_path: 会话文件路径

    Returns:
        会话信息字典
    """
    mgr = SessionManager()
    return mgr.get_session_info(file_path)


def checkpoint(agent: SelfEvolvingAgentCore,
               directory: str = ".",
               name: Optional[str] = None,
               max_keep: int = 0) -> str:
    """
    创建检查点

    Args:
        agent: Agent实例
        directory: 保存目录
        name: 基础名称
        max_keep: 保留最近N个（0=不限制）

    Returns:
        检查点文件路径
    """
    mgr = SessionManager(directory)
    return mgr.checkpoint(agent, directory=directory, name=name,
                          max_keep=max_keep)


def _get_version() -> str:
    """获取SEVA版本号"""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
