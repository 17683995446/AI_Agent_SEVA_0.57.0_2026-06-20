"""
阶段54测试：插件系统与扩展架构
"""
import unittest
import time
import threading
import tempfile
import os
from pathlib import Path

from seva.plugins import (
    Plugin, PluginInfo, PluginState, PluginContext,
    HookType, HookEntry, HookRegistry, PluginLoader,
    PluginManager, register_hook, plugin_hook,
    get_plugin_manager, register_plugin, unregister_plugin,
    enable_plugin, disable_plugin, get_plugin, list_plugins,
    list_enabled_plugins, invoke_hooks, emit_event, on_event,
    get_plugin_report, get_plugin_status, reset_plugins,
    discover_plugins,
)


# ============================================================================
# 测试用插件
# ============================================================================

class SimplePlugin(Plugin):
    info = PluginInfo(name="simple", version="1.0.0", description="简单插件")

    def __init__(self):
        super().__init__()
        self.events = []

    def on_load(self, context):
        self.events.append("loaded")

    def on_enable(self):
        self.events.append("enabled")

    def on_disable(self):
        self.events.append("disabled")

    def on_unload(self):
        self.events.append("unloaded")


class ErrorPlugin(Plugin):
    info = PluginInfo(name="error_plugin", version="1.0.0")

    def on_load(self, context):
        raise ValueError("加载失败")


class DependentPlugin(Plugin):
    info = PluginInfo(name="dependent", version="1.0.0", dependencies=["simple"])

    def on_enable(self):
        pass


# ============================================================================
# 测试枚举
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_plugin_state(self):
        self.assertEqual(PluginState.UNLOADED.value, "unloaded")
        self.assertEqual(PluginState.LOADED.value, "loaded")
        self.assertEqual(PluginState.ENABLED.value, "enabled")
        self.assertEqual(PluginState.DISABLED.value, "disabled")
        self.assertEqual(PluginState.ERROR.value, "error")

    def test02_hook_type(self):
        self.assertEqual(HookType.PRE_RUN.value, "pre_run")
        self.assertEqual(HookType.POST_RUN.value, "post_run")
        self.assertEqual(HookType.PRE_STEP.value, "pre_step")
        self.assertEqual(HookType.POST_STEP.value, "post_step")
        self.assertEqual(HookType.ON_ERROR.value, "on_error")
        self.assertEqual(HookType.ON_EVENT.value, "on_event")
        self.assertEqual(HookType.CUSTOM.value, "custom")


# ============================================================================
# 测试PluginInfo
# ============================================================================

class TestPluginInfo(unittest.TestCase):
    """测试插件元数据"""

    def test01_creation(self):
        info = PluginInfo(name="test", version="1.0.0", author="me",
                          description="测试", dependencies=["dep1"],
                          tags=["tag1", "tag2"])
        self.assertEqual(info.name, "test")
        self.assertEqual(info.version, "1.0.0")
        self.assertEqual(info.author, "me")
        self.assertEqual(info.description, "测试")
        self.assertEqual(info.dependencies, ["dep1"])
        self.assertEqual(info.tags, ["tag1", "tag2"])

    def test02_defaults(self):
        info = PluginInfo(name="test")
        self.assertEqual(info.version, "0.1.0")
        self.assertEqual(info.author, "")
        self.assertEqual(info.dependencies, [])
        self.assertEqual(info.tags, [])

    def test03_to_dict(self):
        info = PluginInfo(name="test", version="1.0.0", author="me")
        d = info.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["version"], "1.0.0")
        self.assertEqual(d["author"], "me")

    def test04_from_dict(self):
        data = {"name": "test", "version": "2.0.0", "author": "you",
                "dependencies": ["dep"]}
        info = PluginInfo.from_dict(data)
        self.assertEqual(info.name, "test")
        self.assertEqual(info.version, "2.0.0")
        self.assertEqual(info.author, "you")
        self.assertEqual(info.dependencies, ["dep"])

    def test05_repr(self):
        info = PluginInfo(name="test", version="1.0.0")
        r = repr(info)
        self.assertIn("PluginInfo", r)
        self.assertIn("test", r)
        self.assertIn("1.0.0", r)


# ============================================================================
# 测试Plugin基类
# ============================================================================

class TestPlugin(unittest.TestCase):
    """测试插件基类"""

    def test01_initial_state(self):
        plugin = SimplePlugin()
        self.assertEqual(plugin.state, PluginState.UNLOADED)
        self.assertFalse(plugin.is_enabled)
        self.assertFalse(plugin.is_loaded)
        self.assertEqual(plugin.name, "simple")

    def test02_get_info(self):
        plugin = SimplePlugin()
        info = plugin.get_info()
        self.assertEqual(info["info"]["name"], "simple")
        self.assertEqual(info["state"], "unloaded")
        self.assertFalse(info["is_enabled"])

    def test03_repr(self):
        plugin = SimplePlugin()
        r = repr(plugin)
        self.assertIn("SimplePlugin", r)
        self.assertIn("simple", r)
        self.assertIn("unloaded", r)

    def test04_lifecycle_methods_exist(self):
        plugin = SimplePlugin()
        self.assertTrue(hasattr(plugin, "on_load"))
        self.assertTrue(hasattr(plugin, "on_enable"))
        self.assertTrue(hasattr(plugin, "on_disable"))
        self.assertTrue(hasattr(plugin, "on_unload"))
        self.assertTrue(hasattr(plugin, "on_error"))


# ============================================================================
# 测试HookRegistry
# ============================================================================

class TestHookRegistry(unittest.TestCase):
    """测试钩子注册表"""

    def setUp(self):
        self.registry = HookRegistry()

    def test01_register_and_invoke(self):
        calls = []
        self.registry.register(HookType.POST_STEP, lambda: calls.append(1))
        self.registry.invoke(HookType.POST_STEP)
        self.assertEqual(calls, [1])

    def test02_multiple_hooks(self):
        calls = []
        self.registry.register(HookType.POST_STEP, lambda: calls.append("a"))
        self.registry.register(HookType.POST_STEP, lambda: calls.append("b"))
        self.registry.invoke(HookType.POST_STEP)
        self.assertEqual(calls, ["a", "b"])

    def test03_priority_order(self):
        calls = []
        self.registry.register(HookType.POST_STEP, lambda: calls.append("low"),
                               priority=10)
        self.registry.register(HookType.POST_STEP, lambda: calls.append("high"),
                               priority=1)
        self.registry.invoke(HookType.POST_STEP)
        self.assertEqual(calls, ["high", "low"])

    def test04_unregister(self):
        callback = lambda: None
        self.registry.register(HookType.POST_STEP, callback)
        self.assertTrue(self.registry.unregister(HookType.POST_STEP, callback))
        self.assertEqual(len(self.registry.get_hooks(HookType.POST_STEP)), 0)

    def test05_unregister_nonexistent(self):
        self.assertFalse(self.registry.unregister(HookType.POST_STEP, lambda: None))

    def test06_unregister_plugin(self):
        cb1 = lambda: None
        cb2 = lambda: None
        self.registry.register(HookType.POST_STEP, cb1, plugin_name="plugin1")
        self.registry.register(HookType.POST_STEP, cb2, plugin_name="plugin2")
        count = self.registry.unregister_plugin("plugin1")
        self.assertEqual(count, 1)
        self.assertEqual(len(self.registry.get_hooks(HookType.POST_STEP)), 1)

    def test07_invoke_with_args(self):
        results = []
        self.registry.register(HookType.PRE_STEP, lambda x, y: x + y)
        results = self.registry.invoke(HookType.PRE_STEP, 1, 2)
        self.assertEqual(results, [3])

    def test08_invoke_empty(self):
        results = self.registry.invoke(HookType.POST_RUN)
        self.assertEqual(results, [])

    def test09_hook_exception_handled(self):
        def bad_hook():
            raise ValueError("钩子错误")

        self.registry.register(HookType.POST_STEP, bad_hook)
        # 不应抛异常
        results = self.registry.invoke(HookType.POST_STEP)
        self.assertEqual(results, [None])

    def test10_event_hooks(self):
        calls = []
        self.registry.register_event("custom_event", lambda **kw: calls.append(kw))
        self.registry.invoke_event("custom_event", data="test")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["data"], "test")

    def test11_clear(self):
        self.registry.register(HookType.POST_STEP, lambda: None)
        self.registry.register_event("event", lambda: None)
        self.registry.clear()
        self.assertEqual(len(self.registry.get_hooks(HookType.POST_STEP)), 0)

    def test12_stats(self):
        self.registry.register(HookType.POST_STEP, lambda: None)
        self.registry.register(HookType.PRE_RUN, lambda: None)
        self.registry.register_event("event1", lambda: None)
        stats = self.registry.get_stats()
        self.assertEqual(stats["total_hooks"], 2)
        self.assertEqual(stats["total_event_hooks"], 1)

    def test13_disabled_hook_not_called(self):
        calls = []
        entry = self.registry.register(HookType.POST_STEP, lambda: calls.append(1))
        entry.is_enabled = False
        self.registry.invoke(HookType.POST_STEP)
        self.assertEqual(calls, [])


# ============================================================================
# 测试PluginManager
# ============================================================================

class TestPluginManager(unittest.TestCase):
    """测试插件管理器"""

    def setUp(self):
        PluginManager.reset_instance()

    def tearDown(self):
        PluginManager.reset_instance()

    def test01_singleton(self):
        mgr1 = PluginManager()
        mgr2 = PluginManager()
        self.assertIs(mgr1, mgr2)

    def test02_register_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        self.assertTrue(mgr.register(plugin))
        self.assertEqual(plugin.state, PluginState.LOADED)
        self.assertIn("loaded", plugin.events)
        self.assertIn("simple", mgr.list_plugins())

    def test03_register_duplicate(self):
        mgr = PluginManager()
        plugin1 = SimplePlugin()
        plugin2 = SimplePlugin()
        self.assertTrue(mgr.register(plugin1))
        self.assertFalse(mgr.register(plugin2))

    def test04_unregister_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertTrue(mgr.unregister("simple"))
        self.assertNotIn("simple", mgr.list_plugins())
        self.assertIn("unloaded", plugin.events)

    def test05_unregister_nonexistent(self):
        mgr = PluginManager()
        self.assertFalse(mgr.unregister("nonexistent"))

    def test06_enable_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertTrue(mgr.enable("simple"))
        self.assertEqual(plugin.state, PluginState.ENABLED)
        self.assertIn("enabled", plugin.events)
        self.assertIn("simple", mgr.list_enabled_plugins())

    def test07_disable_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")
        self.assertTrue(mgr.disable("simple"))
        self.assertEqual(plugin.state, PluginState.DISABLED)
        self.assertIn("disabled", plugin.events)

    def test08_enable_nonexistent(self):
        mgr = PluginManager()
        self.assertFalse(mgr.enable("nonexistent"))

    def test09_register_error_plugin(self):
        mgr = PluginManager()
        plugin = ErrorPlugin()
        self.assertFalse(mgr.register(plugin))
        self.assertEqual(plugin.state, PluginState.ERROR)
        self.assertIsNotNone(plugin.error)

    def test10_get_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertIs(mgr.get_plugin("simple"), plugin)

    def test11_get_plugin_nonexistent(self):
        mgr = PluginManager()
        self.assertIsNone(mgr.get_plugin("nonexistent"))

    def test12_get_plugin_info(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        info = mgr.get_plugin_info("simple")
        self.assertEqual(info["info"]["name"], "simple")
        self.assertEqual(info["state"], "loaded")

    def test13_list_plugins(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        self.assertEqual(mgr.list_plugins(), ["simple"])

    def test14_list_enabled_plugins(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")
        self.assertEqual(mgr.list_enabled_plugins(), ["simple"])

    def test15_register_hook(self):
        mgr = PluginManager()
        calls = []
        mgr.register_hook(HookType.POST_STEP, lambda: calls.append(1))
        mgr.invoke_hooks(HookType.POST_STEP)
        self.assertEqual(calls, [1])

    def test16_unregister_hook(self):
        mgr = PluginManager()
        callback = lambda: None
        mgr.register_hook(HookType.POST_STEP, callback)
        self.assertTrue(mgr.unregister_hook(HookType.POST_STEP, callback))

    def test17_event_system(self):
        mgr = PluginManager()
        calls = []
        mgr.on("test_event", lambda name, **data: calls.append(data))
        mgr.emit_event("test_event", source="test", value=42)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["value"], 42)

    def test18_off_event(self):
        mgr = PluginManager()
        callback = lambda name, **data: None
        mgr.on("test_event", callback)
        self.assertTrue(mgr.off("test_event", callback))

    def test19_load_plugins_batch(self):
        mgr = PluginManager()
        plugins = [SimplePlugin()]
        results = mgr.load_plugins(plugins)
        self.assertTrue(results["simple"])

    def test20_enable_all(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        results = mgr.enable_all()
        self.assertTrue(results["simple"])

    def test21_disable_all(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")
        results = mgr.disable_all()
        self.assertTrue(results["simple"])

    def test22_dependency_check(self):
        mgr = PluginManager()
        # 先注册但不启用simple
        mgr.register(SimplePlugin())
        # 注册依赖插件
        dep = DependentPlugin()
        mgr.register(dep)
        # 依赖未启用，应该启用失败
        self.assertFalse(mgr.enable("dependent"))
        self.assertEqual(dep.state, PluginState.ERROR)

    def test23_dependency_satisfied(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        dep = DependentPlugin()
        mgr.register(dep)
        self.assertTrue(mgr.enable("dependent"))

    def test24_unregister_cleans_hooks(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")

        callback = lambda: None
        plugin.context.register_hook(HookType.POST_STEP, callback)
        self.assertEqual(len(mgr.hooks.get_hooks(HookType.POST_STEP)), 1)

        mgr.unregister("simple")
        self.assertEqual(len(mgr.hooks.get_hooks(HookType.POST_STEP)), 0)

    def test25_get_report(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        report = mgr.get_report()
        self.assertEqual(report["total_plugins"], 1)
        self.assertIn("plugins", report)
        self.assertIn("hooks", report)

    def test26_format_report(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        report = mgr.format_report()
        self.assertIsInstance(report, str)
        self.assertIn("SEVA 插件系统报告", report)
        self.assertIn("simple", report)

    def test27_reset(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.reset()
        self.assertEqual(len(mgr.list_plugins()), 0)


# ============================================================================
# 测试PluginContext
# ============================================================================

class TestPluginContext(unittest.TestCase):
    """测试插件上下文"""

    def setUp(self):
        PluginManager.reset_instance()

    def tearDown(self):
        PluginManager.reset_instance()

    def test01_config(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        ctx = plugin.context
        ctx.set_config("key", "value")
        self.assertEqual(ctx.get_config("key"), "value")
        self.assertEqual(ctx.get_config("nonexistent", "default"), "default")

    def test02_log(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        ctx = plugin.context
        ctx.log("测试日志")
        logger = ctx.get_logger()
        self.assertEqual(len(logger), 1)
        self.assertIn("测试日志", logger[0])
        self.assertIn("simple", logger[0])

    def test03_plugin_name(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertEqual(plugin.context.plugin_name, "simple")

    def test04_manager_access(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertIs(plugin.context.manager, mgr)

    def test05_get_other_plugin(self):
        mgr = PluginManager()
        plugin1 = SimplePlugin()
        mgr.register(plugin1)
        ctx = plugin1.context
        self.assertIs(ctx.get_plugin("simple"), plugin1)
        self.assertIsNone(ctx.get_plugin("nonexistent"))

    def test06_register_hook_via_context(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        calls = []
        plugin.context.register_hook(HookType.POST_STEP, lambda: calls.append(1))
        mgr.invoke_hooks(HookType.POST_STEP)
        self.assertEqual(calls, [1])

    def test07_emit_event_via_context(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        calls = []
        mgr.on("test", lambda name, **data: calls.append(data))
        plugin.context.emit_event("test", value=123)
        self.assertEqual(len(calls), 1)


# ============================================================================
# 测试PluginLoader
# ============================================================================

class TestPluginLoader(unittest.TestCase):
    """测试插件加载器"""

    def test01_load_from_class(self):
        plugin = PluginLoader.load_from_class(SimplePlugin)
        self.assertIsInstance(plugin, Plugin)
        self.assertEqual(plugin.name, "simple")

    def test02_load_from_class_invalid(self):
        with self.assertRaises(TypeError):
            PluginLoader.load_from_class(dict)

    def test03_load_from_file(self):
        # 创建临时插件文件
        plugin_code = '''
from seva.plugins import Plugin, PluginInfo

class FilePlugin(Plugin):
    info = PluginInfo(name="file_plugin", version="1.0.0")

    def on_load(self, context):
        pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            f.write(plugin_code)
            f.flush()
            file_path = f.name

        try:
            plugins = PluginLoader.load_from_file(file_path)
            self.assertEqual(len(plugins), 1)
            self.assertEqual(plugins[0].name, "file_plugin")
        finally:
            os.unlink(file_path)

    def test04_load_from_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            PluginLoader.load_from_file("/nonexistent/path/plugin.py")

    def test05_load_from_directory(self):
        # 创建临时目录和插件文件
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_code = '''
from seva.plugins import Plugin, PluginInfo

class DirPlugin(Plugin):
    info = PluginInfo(name="dir_plugin", version="1.0.0")
'''
            with open(os.path.join(tmpdir, "plugin1.py"), "w") as f:
                f.write(plugin_code)

            plugins = PluginLoader.load_from_directory(tmpdir)
            self.assertEqual(len(plugins), 1)
            self.assertEqual(plugins[0].name, "dir_plugin")

    def test06_load_from_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins = PluginLoader.load_from_directory(tmpdir)
            self.assertEqual(len(plugins), 0)

    def test07_load_from_nonexistent_directory(self):
        plugins = PluginLoader.load_from_directory("/nonexistent/dir")
        self.assertEqual(len(plugins), 0)


# ============================================================================
# 测试装饰器
# ============================================================================

class TestDecorators(unittest.TestCase):
    """测试装饰器"""

    def setUp(self):
        PluginManager.reset_instance()

    def tearDown(self):
        PluginManager.reset_instance()

    def test01_register_hook_decorator(self):
        class MyPlugin(Plugin):
            info = PluginInfo(name="test_decorator")

            @register_hook(HookType.POST_STEP)
            def my_hook(self):
                return "hooked"

        plugin = MyPlugin()
        self.assertTrue(hasattr(plugin.my_hook, "_hook_type"))
        self.assertEqual(plugin.my_hook._hook_type, HookType.POST_STEP)

    def test02_plugin_hook_alias(self):
        class MyPlugin(Plugin):
            info = PluginInfo(name="test_alias")

            @plugin_hook(HookType.PRE_RUN)
            def my_hook(self):
                return "hooked"

        plugin = MyPlugin()
        self.assertEqual(plugin.my_hook._hook_type, HookType.PRE_RUN)

    def test03_hook_priority(self):
        class MyPlugin(Plugin):
            info = PluginInfo(name="test_priority")

            @register_hook(HookType.POST_STEP, priority=5)
            def my_hook(self):
                return "hooked"

        plugin = MyPlugin()
        self.assertEqual(plugin.my_hook._hook_priority, 5)


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        PluginManager.reset_instance()

    def tearDown(self):
        PluginManager.reset_instance()

    def test01_get_plugin_manager(self):
        mgr = get_plugin_manager()
        self.assertIsInstance(mgr, PluginManager)

    def test02_register_plugin(self):
        plugin = SimplePlugin()
        self.assertTrue(register_plugin(plugin))

    def test03_unregister_plugin(self):
        register_plugin(SimplePlugin())
        self.assertTrue(unregister_plugin("simple"))

    def test04_enable_plugin(self):
        register_plugin(SimplePlugin())
        self.assertTrue(enable_plugin("simple"))

    def test05_disable_plugin(self):
        register_plugin(SimplePlugin())
        enable_plugin("simple")
        self.assertTrue(disable_plugin("simple"))

    def test06_get_plugin(self):
        plugin = SimplePlugin()
        register_plugin(plugin)
        self.assertIs(get_plugin("simple"), plugin)

    def test07_list_plugins(self):
        register_plugin(SimplePlugin())
        self.assertIn("simple", list_plugins())

    def test08_list_enabled_plugins(self):
        register_plugin(SimplePlugin())
        enable_plugin("simple")
        self.assertIn("simple", list_enabled_plugins())

    def test09_invoke_hooks(self):
        calls = []
        PluginManager().register_hook(HookType.POST_STEP, lambda: calls.append(1))
        invoke_hooks(HookType.POST_STEP)
        self.assertEqual(calls, [1])

    def test10_emit_event(self):
        calls = []
        on_event("test", lambda name, **data: calls.append(data))
        emit_event("test", source="unit", value=1)
        self.assertEqual(len(calls), 1)

    def test11_get_plugin_report(self):
        register_plugin(SimplePlugin())
        report = get_plugin_report()
        self.assertIsInstance(report, str)
        self.assertIn("SEVA 插件系统报告", report)

    def test12_get_plugin_status(self):
        register_plugin(SimplePlugin())
        status = get_plugin_status()
        self.assertEqual(status["total_plugins"], 1)

    def test13_reset_plugins(self):
        register_plugin(SimplePlugin())
        reset_plugins()
        self.assertEqual(len(list_plugins()), 0)

    def test14_discover_plugins(self):
        # 默认插件目录不存在时应返回空
        results = discover_plugins()
        self.assertIsInstance(results, dict)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        PluginManager.reset_instance()

    def tearDown(self):
        PluginManager.reset_instance()

    def test01_full_lifecycle(self):
        """完整生命周期测试"""
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        self.assertIn("loaded", plugin.events)

        mgr.enable("simple")
        self.assertIn("enabled", plugin.events)
        self.assertTrue(plugin.is_enabled)

        mgr.disable("simple")
        self.assertIn("disabled", plugin.events)

        mgr.unregister("simple")
        self.assertIn("unloaded", plugin.events)

    def test02_hook_invocation_with_plugins(self):
        """插件钩子调用"""
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")

        calls = []
        plugin.context.register_hook(HookType.POST_STEP, lambda **kw: calls.append(kw))
        mgr.invoke_hooks(HookType.POST_STEP, step=1)
        self.assertEqual(len(calls), 1)

    def test03_multiple_plugins_interaction(self):
        """多插件交互"""
        mgr = PluginManager()

        class PluginA(Plugin):
            info = PluginInfo(name="a")
            def on_load(self, ctx):
                ctx.set_config("value", 42)

        class PluginB(Plugin):
            info = PluginInfo(name="b")
            def on_load(self, ctx):
                a = ctx.get_plugin("a")
                ctx.set_config("a_value", a.context.get_config("value"))

        mgr.register(PluginA())
        mgr.register(PluginB())

        b = mgr.get_plugin("b")
        self.assertEqual(b.context.get_config("a_value"), 42)

    def test04_thread_safety(self):
        """线程安全测试"""
        mgr = PluginManager()
        results = []
        errors = []

        def worker(i):
            try:
                class TPlugin(Plugin):
                    info = PluginInfo(name=f"thread_{i}")

                plugin = TPlugin()
                mgr.register(plugin)
                mgr.enable(f"thread_{i}")
                mgr.disable(f"thread_{i}")
                mgr.unregister(f"thread_{i}")
                results.append(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)

    def test05_report_contains_all_sections(self):
        """报告包含所有部分"""
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        mgr.register_hook(HookType.POST_STEP, lambda: None)

        report = mgr.format_report()
        self.assertIn("插件系统报告", report)
        self.assertIn("已注册插件", report)
        self.assertIn("钩子统计", report)
        self.assertIn("simple", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
