"""
插件系统与扩展架构（阶段54）
对应第一性原理：可扩展性 = 通过标准化接口允许第三方扩展功能，而不修改核心代码

提供能力:
    - Plugin: 插件基类（生命周期：on_load/on_enable/on_disable/on_unload）
    - PluginInfo: 插件元数据（名称、版本、作者、描述、依赖）
    - PluginState: 插件状态（UNLOADED/LOADED/ENABLED/DISABLED/ERROR）
    - HookType: 钩子类型（pre_run/post_run/pre_step/post_step/on_error/on_event等）
    - PluginManager: 插件管理器（单例，加载、注册、启用、禁用、卸载）
    - HookRegistry: 钩子注册表
    - PluginLoader: 插件加载器（从模块、文件、目录加载）
    - PluginContext: 插件上下文（提供给插件的运行环境）
    - 装饰器: register_hook / plugin_hook
    - 便捷函数: get_plugin_manager / register_plugin / load_plugins 等

使用示例:
    from seva import Plugin, PluginInfo, register_hook, HookType

    class MyPlugin(Plugin):
        info = PluginInfo(name="my_plugin", version="1.0.0")

        def on_load(self, context):
            print("插件加载")

        @register_hook(HookType.POST_STEP)
        def after_step(self, context, **kwargs):
            print(f"步骤完成: {kwargs}")
"""
from __future__ import annotations

import os
import sys
import time
import json
import importlib
import importlib.util
import inspect
import threading
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
PLUGINS_DIR = SEVA_HOME / "plugins"


# ============================================================================
# PluginState - 插件状态枚举
# ============================================================================

class PluginState(Enum):
    """插件状态"""
    UNLOADED = "unloaded"    # 未加载
    LOADED = "loaded"        # 已加载但未启用
    ENABLED = "enabled"      # 已启用
    DISABLED = "disabled"    # 已禁用
    ERROR = "error"          # 出错


# ============================================================================
# HookType - 钩子类型枚举
# ============================================================================

class HookType(Enum):
    """钩子类型"""
    PRE_RUN = "pre_run"            # 运行前
    POST_RUN = "post_run"          # 运行后
    PRE_STEP = "pre_step"          # 每步前
    POST_STEP = "post_step"        # 每步后
    PRE_CYCLE = "pre_cycle"        # 每个循环前
    POST_CYCLE = "post_cycle"      # 每个循环后
    ON_ERROR = "on_error"          # 错误发生时
    ON_EVENT = "on_event"          # 事件发生时
    ON_CONFIG_LOAD = "on_config_load"  # 配置加载时
    ON_CONFIG_SAVE = "on_config_save"  # 配置保存时
    ON_STARTUP = "on_startup"      # 系统启动时
    ON_SHUTDOWN = "on_shutdown"    # 系统关闭时
    CUSTOM = "custom"              # 自定义钩子


# ============================================================================
# PluginInfo - 插件元数据
# ============================================================================

@dataclass
class PluginInfo:
    """插件元数据"""
    name: str
    version: str = "0.1.0"
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "tags": list(self.tags),
            "homepage": self.homepage,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginInfo":
        return cls(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
        )

    def __repr__(self) -> str:
        return f"PluginInfo(name={self.name!r}, version={self.version!r})"


# ============================================================================
# PluginContext - 插件上下文
# ============================================================================

class PluginContext:
    """
    插件上下文：提供给插件的运行环境

    插件通过上下文访问系统功能、注册钩子、与其他插件通信
    """

    def __init__(self, plugin_manager: "PluginManager", plugin_name: str):
        self._manager = plugin_manager
        self._plugin_name = plugin_name
        self._data: Dict[str, Any] = {}
        self._logger: List[str] = []

    @property
    def plugin_name(self) -> str:
        """插件名称"""
        return self._plugin_name

    @property
    def manager(self) -> "PluginManager":
        """插件管理器"""
        return self._manager

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        return self._data.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """设置配置"""
        self._data[key] = value

    def log(self, message: str) -> None:
        """记录日志"""
        entry = f"[{self._plugin_name}] {message}"
        self._logger.append(entry)

    def get_logger(self) -> List[str]:
        """获取日志"""
        return list(self._logger)

    def get_plugin(self, name: str) -> Optional["Plugin"]:
        """获取其他插件实例"""
        return self._manager.get_plugin(name)

    def register_hook(self, hook_type: HookType, callback: Callable,
                      priority: int = 0) -> None:
        """注册钩子"""
        self._manager.register_hook(hook_type, callback, self._plugin_name, priority)

    def unregister_hook(self, hook_type: HookType, callback: Callable) -> None:
        """注销钩子"""
        self._manager.unregister_hook(hook_type, callback)

    def emit_event(self, event_name: str, **data: Any) -> None:
        """发射事件"""
        self._manager.emit_event(event_name, source=self._plugin_name, **data)


# ============================================================================
# Plugin - 插件基类
# ============================================================================

class Plugin:
    """
    插件基类：子类继承并实现生命周期方法

    生命周期:
        on_load -> on_enable <-> on_disable -> on_unload
    """

    info: PluginInfo = PluginInfo(name="base")

    def __init__(self):
        self._context: Optional[PluginContext] = None
        self._state: PluginState = PluginState.UNLOADED
        self._error: Optional[str] = None

    @property
    def context(self) -> Optional[PluginContext]:
        """插件上下文"""
        return self._context

    @property
    def state(self) -> PluginState:
        """插件状态"""
        return self._state

    @property
    def error(self) -> Optional[str]:
        """错误信息"""
        return self._error

    @property
    def name(self) -> str:
        """插件名称"""
        return self.info.name

    @property
    def is_enabled(self) -> bool:
        """是否已启用"""
        return self._state == PluginState.ENABLED

    @property
    def is_loaded(self) -> bool:
        """是否已加载"""
        return self._state in (PluginState.LOADED, PluginState.ENABLED, PluginState.DISABLED)

    # ---- 生命周期方法（子类可覆盖）----

    def on_load(self, context: PluginContext) -> None:
        """加载时调用"""
        pass

    def on_enable(self) -> None:
        """启用时调用"""
        pass

    def on_disable(self) -> None:
        """禁用时调用"""
        pass

    def on_unload(self) -> None:
        """卸载时调用"""
        pass

    def on_error(self, error: Exception) -> None:
        """出错时调用"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            "info": self.info.to_dict(),
            "state": self._state.value,
            "error": self._error,
            "is_enabled": self.is_enabled,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, state={self._state.value})"


# ============================================================================
# HookEntry - 钩子条目
# ============================================================================

@dataclass
class HookEntry:
    """钩子条目"""
    hook_type: HookType
    callback: Callable
    plugin_name: str
    priority: int = 0
    is_enabled: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        if not self.is_enabled:
            return None
        return self.callback(*args, **kwargs)


# ============================================================================
# HookRegistry - 钩子注册表
# ============================================================================

class HookRegistry:
    """
    钩子注册表：管理所有钩子的注册和调用
    """

    def __init__(self):
        self._hooks: Dict[HookType, List[HookEntry]] = {}
        self._event_hooks: Dict[str, List[HookEntry]] = {}
        self._lock = threading.RLock()

    def register(self, hook_type: HookType, callback: Callable,
                 plugin_name: str = "", priority: int = 0) -> HookEntry:
        """注册钩子"""
        with self._lock:
            entry = HookEntry(
                hook_type=hook_type,
                callback=callback,
                plugin_name=plugin_name,
                priority=priority,
            )
            if hook_type not in self._hooks:
                self._hooks[hook_type] = []
            self._hooks[hook_type].append(entry)
            # 按优先级排序（数字小的先执行）
            self._hooks[hook_type].sort(key=lambda e: e.priority)
            return entry

    def register_event(self, event_name: str, callback: Callable,
                       plugin_name: str = "", priority: int = 0) -> HookEntry:
        """注册事件钩子"""
        with self._lock:
            entry = HookEntry(
                hook_type=HookType.ON_EVENT,
                callback=callback,
                plugin_name=plugin_name,
                priority=priority,
            )
            if event_name not in self._event_hooks:
                self._event_hooks[event_name] = []
            self._event_hooks[event_name].append(entry)
            self._event_hooks[event_name].sort(key=lambda e: e.priority)
            return entry

    def unregister(self, hook_type: HookType, callback: Callable) -> bool:
        """注销钩子"""
        with self._lock:
            if hook_type not in self._hooks:
                return False
            before = len(self._hooks[hook_type])
            self._hooks[hook_type] = [
                e for e in self._hooks[hook_type] if e.callback != callback
            ]
            return len(self._hooks[hook_type]) < before

    def unregister_plugin(self, plugin_name: str) -> int:
        """注销某插件的所有钩子"""
        count = 0
        with self._lock:
            for hook_type in list(self._hooks.keys()):
                before = len(self._hooks[hook_type])
                self._hooks[hook_type] = [
                    e for e in self._hooks[hook_type] if e.plugin_name != plugin_name
                ]
                count += before - len(self._hooks[hook_type])
            for event_name in list(self._event_hooks.keys()):
                before = len(self._event_hooks[event_name])
                self._event_hooks[event_name] = [
                    e for e in self._event_hooks[event_name] if e.plugin_name != plugin_name
                ]
                count += before - len(self._event_hooks[event_name])
        return count

    def get_hooks(self, hook_type: HookType) -> List[HookEntry]:
        """获取某类型的所有钩子"""
        with self._lock:
            return list(self._hooks.get(hook_type, []))

    def get_event_hooks(self, event_name: str) -> List[HookEntry]:
        """获取某事件的所有钩子"""
        with self._lock:
            return list(self._event_hooks.get(event_name, []))

    def invoke(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """
        调用钩子

        Returns:
            所有钩子的返回值列表
        """
        results = []
        with self._lock:
            entries = list(self._hooks.get(hook_type, []))

        for entry in entries:
            if not entry.is_enabled:
                continue
            try:
                result = entry.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                # 钩子异常不中断流程，记录错误
                results.append(None)
        return results

    def invoke_event(self, event_name: str, *args, **kwargs) -> List[Any]:
        """调用事件钩子"""
        results = []
        with self._lock:
            entries = list(self._event_hooks.get(event_name, []))

        for entry in entries:
            if not entry.is_enabled:
                continue
            try:
                result = entry.callback(*args, **kwargs)
                results.append(result)
            except Exception:
                results.append(None)
        return results

    def clear(self) -> None:
        """清空所有钩子"""
        with self._lock:
            self._hooks.clear()
            self._event_hooks.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "hook_types": {
                    ht.value: len(entries) for ht, entries in self._hooks.items()
                },
                "events": {
                    name: len(entries) for name, entries in self._event_hooks.items()
                },
                "total_hooks": sum(len(e) for e in self._hooks.values()),
                "total_event_hooks": sum(len(e) for e in self._event_hooks.values()),
            }


# ============================================================================
# PluginLoader - 插件加载器
# ============================================================================

class PluginLoader:
    """
    插件加载器：从模块、文件、目录加载插件
    """

    @staticmethod
    def load_from_class(plugin_class: Type[Plugin]) -> Plugin:
        """从类加载插件"""
        if not issubclass(plugin_class, Plugin):
            raise TypeError(f"{plugin_class} 不是 Plugin 子类")
        return plugin_class()

    @staticmethod
    def load_from_module(module_name: str) -> List[Plugin]:
        """
        从模块加载插件

        模块中需定义 Plugin 子类，或通过 get_plugin() / get_plugins() 函数返回
        """
        module = importlib.import_module(module_name)

        plugins: List[Plugin] = []

        # 优先调用 get_plugins() 或 get_plugin()
        if hasattr(module, "get_plugins"):
            result = module.get_plugins()
            if isinstance(result, list):
                plugins.extend(result)
            else:
                plugins.append(result)
        elif hasattr(module, "get_plugin"):
            plugins.append(module.get_plugin())
        else:
            # 扫描模块中的 Plugin 子类
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and issubclass(obj, Plugin)
                        and obj is not Plugin):
                    plugins.append(obj())

        return plugins

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> List[Plugin]:
        """从文件加载插件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"插件文件不存在: {file_path}")

        module_name = f"seva_plugin_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件模块: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return PluginLoader.load_from_module(module_name)

    @staticmethod
    def load_from_directory(dir_path: Union[str, Path]) -> List[Plugin]:
        """从目录加载所有插件"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []

        plugins: List[Plugin] = []
        for file_path in sorted(dir_path.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            try:
                plugins.extend(PluginLoader.load_from_file(file_path))
            except Exception:
                # 加载失败跳过
                pass
        return plugins

    @staticmethod
    def discover_plugins() -> List[Plugin]:
        """发现并加载默认插件目录中的插件"""
        if PLUGINS_DIR.exists():
            return PluginLoader.load_from_directory(PLUGINS_DIR)
        return []


# ============================================================================
# PluginManager - 插件管理器（单例）
# ============================================================================

class PluginManager:
    """
    插件管理器：管理插件的完整生命周期

    单例模式，全局唯一实例
    """

    _instance: Optional["PluginManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PluginManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._plugins: Dict[str, Plugin] = {}
        self._hooks = HookRegistry()
        self._data_lock = threading.RLock()
        self._event_listeners: Dict[str, List[Callable]] = {}

    # ---- 插件管理 ----

    def register(self, plugin: Plugin) -> bool:
        """
        注册插件

        Returns:
            是否注册成功
        """
        with self._data_lock:
            name = plugin.name
            if name in self._plugins:
                return False

            # 创建上下文
            context = PluginContext(self, name)
            plugin._context = context

            # 加载插件
            try:
                plugin._state = PluginState.LOADED
                plugin.on_load(context)
                self._plugins[name] = plugin
                return True
            except Exception as e:
                plugin._state = PluginState.ERROR
                plugin._error = str(e)
                plugin.on_error(e)
                return False

    def unregister(self, name: str) -> bool:
        """注销插件"""
        with self._data_lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]

            # 如果已启用，先禁用
            if plugin.is_enabled:
                self._disable_plugin(plugin)

            # 卸载
            try:
                plugin.on_unload()
            except Exception:
                pass

            # 清理钩子
            self._hooks.unregister_plugin(name)

            del self._plugins[name]
            return True

    def enable(self, name: str) -> bool:
        """启用插件"""
        with self._data_lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]
            if plugin.is_enabled:
                return True

            return self._enable_plugin(plugin)

    def disable(self, name: str) -> bool:
        """禁用插件"""
        with self._data_lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]
            if not plugin.is_enabled:
                return True

            return self._disable_plugin(plugin)

    def _enable_plugin(self, plugin: Plugin) -> bool:
        """启用插件（内部）"""
        try:
            # 检查依赖
            for dep in plugin.info.dependencies:
                dep_plugin = self._plugins.get(dep)
                if dep_plugin is None or not dep_plugin.is_enabled:
                    plugin._state = PluginState.ERROR
                    plugin._error = f"依赖插件未启用: {dep}"
                    return False

            plugin.on_enable()
            plugin._state = PluginState.ENABLED
            return True
        except Exception as e:
            plugin._state = PluginState.ERROR
            plugin._error = str(e)
            plugin.on_error(e)
            return False

    def _disable_plugin(self, plugin: Plugin) -> bool:
        """禁用插件（内部）"""
        try:
            plugin.on_disable()
            plugin._state = PluginState.DISABLED
            return True
        except Exception as e:
            plugin._state = PluginState.ERROR
            plugin._error = str(e)
            plugin.on_error(e)
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """获取插件实例"""
        with self._data_lock:
            return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """列出所有插件名称"""
        with self._data_lock:
            return list(self._plugins.keys())

    def list_enabled_plugins(self) -> List[str]:
        """列出已启用的插件"""
        with self._data_lock:
            return [name for name, p in self._plugins.items() if p.is_enabled]

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取插件信息"""
        with self._data_lock:
            plugin = self._plugins.get(name)
            if plugin is None:
                return None
            return plugin.get_info()

    # ---- 钩子管理 ----

    @property
    def hooks(self) -> HookRegistry:
        """钩子注册表"""
        return self._hooks

    def register_hook(self, hook_type: HookType, callback: Callable,
                      plugin_name: str = "", priority: int = 0) -> HookEntry:
        """注册钩子"""
        return self._hooks.register(hook_type, callback, plugin_name, priority)

    def unregister_hook(self, hook_type: HookType, callback: Callable) -> bool:
        """注销钩子"""
        return self._hooks.unregister(hook_type, callback)

    def invoke_hooks(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """调用钩子"""
        return self._hooks.invoke(hook_type, *args, **kwargs)

    # ---- 事件管理 ----

    def on(self, event_name: str, callback: Callable) -> None:
        """监听事件"""
        with self._data_lock:
            if event_name not in self._event_listeners:
                self._event_listeners[event_name] = []
            self._event_listeners[event_name].append(callback)

    def off(self, event_name: str, callback: Callable) -> bool:
        """取消监听事件"""
        with self._data_lock:
            if event_name not in self._event_listeners:
                return False
            before = len(self._event_listeners[event_name])
            self._event_listeners[event_name] = [
                c for c in self._event_listeners[event_name] if c != callback
            ]
            return len(self._event_listeners[event_name]) < before

    def emit_event(self, event_name: str, source: str = "", **data: Any) -> None:
        """发射事件"""
        # 调用事件监听器
        with self._data_lock:
            listeners = list(self._event_listeners.get(event_name, []))

        for listener in listeners:
            try:
                listener(event_name, source=source, **data)
            except Exception:
                pass

        # 调用事件钩子（避免event_name参数冲突，使用_positional传递）
        with self._data_lock:
            event_hooks = list(self._hooks._event_hooks.get(event_name, []))

        for entry in event_hooks:
            if not entry.is_enabled:
                continue
            try:
                entry.callback(event_name, source=source, **data)
            except Exception:
                pass

    # ---- 批量操作 ----

    def load_plugins(self, plugins: List[Plugin]) -> Dict[str, bool]:
        """批量加载插件"""
        results = {}
        for plugin in plugins:
            results[plugin.name] = self.register(plugin)
        return results

    def load_from_directory(self, dir_path: Union[str, Path]) -> Dict[str, bool]:
        """从目录加载插件"""
        plugins = PluginLoader.load_from_directory(dir_path)
        return self.load_plugins(plugins)

    def discover(self) -> Dict[str, bool]:
        """发现并加载默认插件目录中的插件"""
        plugins = PluginLoader.discover_plugins()
        return self.load_plugins(plugins)

    def enable_all(self) -> Dict[str, bool]:
        """启用所有已加载的插件"""
        results = {}
        with self._data_lock:
            names = list(self._plugins.keys())
        for name in names:
            results[name] = self.enable(name)
        return results

    def disable_all(self) -> Dict[str, bool]:
        """禁用所有插件"""
        results = {}
        with self._data_lock:
            names = list(self._plugins.keys())
        for name in names:
            results[name] = self.disable(name)
        return results

    # ---- 报告 ----

    def get_report(self) -> Dict[str, Any]:
        """获取插件报告"""
        with self._data_lock:
            plugins_info = {
                name: plugin.get_info() for name, plugin in self._plugins.items()
            }

        return {
            "total_plugins": len(self._plugins),
            "enabled_plugins": len(self.list_enabled_plugins()),
            "plugins": plugins_info,
            "hooks": self._hooks.get_stats(),
        }

    def format_report(self) -> str:
        """格式化报告为字符串"""
        report = self.get_report()
        lines = []
        lines.append("=" * 60)
        lines.append("SEVA 插件系统报告")
        lines.append("=" * 60)
        lines.append(f"插件总数: {report['total_plugins']}")
        lines.append(f"已启用: {report['enabled_plugins']}")
        lines.append("")

        lines.append("--- 已注册插件 ---")
        if report["plugins"]:
            for name, info in report["plugins"].items():
                state = info["state"]
                version = info["info"]["version"]
                desc = info["info"]["description"]
                tag = f" - {desc}" if desc else ""
                lines.append(f"  [{state.upper():8s}] {name} v{version}{tag}")
                if info["error"]:
                    lines.append(f"            错误: {info['error']}")
        else:
            lines.append("  (无注册插件)")
        lines.append("")

        lines.append("--- 钩子统计 ---")
        hooks = report["hooks"]
        lines.append(f"  钩子总数: {hooks['total_hooks']}")
        lines.append(f"  事件钩子总数: {hooks['total_event_hooks']}")
        if hooks["hook_types"]:
            lines.append("  按类型:")
            for ht, count in hooks["hook_types"].items():
                lines.append(f"    {ht}: {count}")
        if hooks["events"]:
            lines.append("  按事件:")
            for event, count in hooks["events"].items():
                lines.append(f"    {event}: {count}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """重置所有状态"""
        with self._data_lock:
            # 卸载所有插件
            for plugin in list(self._plugins.values()):
                if plugin.is_enabled:
                    try:
                        plugin.on_disable()
                    except Exception:
                        pass
                try:
                    plugin.on_unload()
                except Exception:
                    pass
            self._plugins.clear()
            self._hooks.clear()
            self._event_listeners.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.reset()
            cls._instance = None


# ============================================================================
# 装饰器
# ============================================================================

def register_hook(hook_type: HookType, priority: int = 0) -> Callable:
    """
    装饰器：标记方法为钩子，在插件启用时自动注册

    使用示例:
        class MyPlugin(Plugin):
            @register_hook(HookType.POST_STEP)
            def after_step(self, context, **kwargs):
                ...
    """
    def decorator(func: Callable) -> Callable:
        func._hook_type = hook_type
        func._hook_priority = priority
        return func
    return decorator


def plugin_hook(hook_type: HookType, priority: int = 0) -> Callable:
    """register_hook 的别名"""
    return register_hook(hook_type, priority)


# ============================================================================
# 便捷函数
# ============================================================================

def get_plugin_manager() -> PluginManager:
    """获取插件管理器单例"""
    return PluginManager()


def register_plugin(plugin: Plugin) -> bool:
    """注册插件"""
    return PluginManager().register(plugin)


def unregister_plugin(name: str) -> bool:
    """注销插件"""
    return PluginManager().unregister(name)


def enable_plugin(name: str) -> bool:
    """启用插件"""
    return PluginManager().enable(name)


def disable_plugin(name: str) -> bool:
    """禁用插件"""
    return PluginManager().disable(name)


def get_plugin(name: str) -> Optional[Plugin]:
    """获取插件"""
    return PluginManager().get_plugin(name)


def list_plugins() -> List[str]:
    """列出所有插件"""
    return PluginManager().list_plugins()


def list_enabled_plugins() -> List[str]:
    """列出已启用的插件"""
    return PluginManager().list_enabled_plugins()


def invoke_hooks(hook_type: HookType, *args, **kwargs) -> List[Any]:
    """调用钩子"""
    return PluginManager().invoke_hooks(hook_type, *args, **kwargs)


def emit_event(event_name: str, source: str = "", **data: Any) -> None:
    """发射事件"""
    PluginManager().emit_event(event_name, source=source, **data)


def on_event(event_name: str, callback: Callable) -> None:
    """监听事件"""
    PluginManager().on(event_name, callback)


def get_plugin_report() -> str:
    """获取插件报告"""
    return PluginManager().format_report()


def get_plugin_status() -> Dict[str, Any]:
    """获取插件状态"""
    return PluginManager().get_report()


def reset_plugins() -> None:
    """重置所有插件"""
    PluginManager().reset()


def discover_plugins() -> Dict[str, bool]:
    """发现并加载插件"""
    return PluginManager().discover()
