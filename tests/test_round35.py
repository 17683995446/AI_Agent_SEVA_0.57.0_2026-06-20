"""
阶段45测试：用户偏好与个性化设置
验证 PreferenceSchema / Preferences / Profile / ProfileManager
及便捷函数和CLI集成
"""
import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


import seva
from seva.preferences import (PreferenceItem, PreferenceSchema, Preferences,
                              Profile, ProfileManager,
                              get_preference, set_preference,
                              reset_preference, reset_all_preferences,
                              list_preferences, list_preference_schema,
                              save_preferences, load_preferences,
                              export_preferences, import_preferences,
                              get_preferences_report,
                              create_profile, delete_profile,
                              switch_profile, list_profiles,
                              get_current_profile, get_profile,
                              PREFS_FILE, PROFILES_DIR, CURRENT_PROFILE_FILE)


def _reset_all():
    """重置所有单例和状态"""
    Preferences.reset_instance()
    ProfileManager.reset_instance()


class TestPreferenceItem(unittest.TestCase):
    """测试PreferenceItem"""

    def test01_init_basic(self):
        """基本初始化"""
        item = PreferenceItem(
            key="test.key",
            default="value",
            category="test",
            description="测试项",
        )
        self.assertEqual(item.key, "test.key")
        self.assertEqual(item.default, "value")
        self.assertEqual(item.category, "test")
        self.assertEqual(item.description, "测试项")
        self.assertIsNone(item.choices)

    def test02_init_with_choices(self):
        """带可选值的初始化"""
        item = PreferenceItem(
            key="test.color",
            default="red",
            category="test",
            choices=["red", "green", "blue"],
        )
        self.assertEqual(item.choices, ["red", "green", "blue"])

    def test03_validate_correct_type(self):
        """验证正确类型"""
        item = PreferenceItem(
            key="test.num",
            default=10,
            category="test",
            value_type=int,
        )
        self.assertTrue(item.validate(20))
        self.assertTrue(item.validate(0))

    def test04_validate_wrong_type(self):
        """验证错误类型"""
        item = PreferenceItem(
            key="test.num",
            default=10,
            category="test",
            value_type=int,
        )
        self.assertFalse(item.validate("not a number"))
        self.assertFalse(item.validate(10.5))

    def test05_validate_choices(self):
        """验证可选值"""
        item = PreferenceItem(
            key="test.color",
            default="red",
            category="test",
            choices=["red", "green", "blue"],
        )
        self.assertTrue(item.validate("red"))
        self.assertFalse(item.validate("yellow"))

    def test06_validate_range(self):
        """验证范围"""
        item = PreferenceItem(
            key="test.num",
            default=50,
            category="test",
            value_type=int,
            min_value=0,
            max_value=100,
        )
        self.assertTrue(item.validate(50))
        self.assertTrue(item.validate(0))
        self.assertTrue(item.validate(100))
        self.assertFalse(item.validate(-1))
        self.assertFalse(item.validate(101))

    def test07_validate_bool(self):
        """验证布尔值"""
        item = PreferenceItem(
            key="test.flag",
            default=True,
            category="test",
            value_type=bool,
        )
        self.assertTrue(item.validate(True))
        self.assertTrue(item.validate(False))

    def test08_to_dict(self):
        """to_dict"""
        item = PreferenceItem(
            key="test.key",
            default="value",
            category="test",
            description="测试",
            choices=["a", "b"],
        )
        d = item.to_dict()
        self.assertEqual(d["key"], "test.key")
        self.assertEqual(d["default"], "value")
        self.assertEqual(d["category"], "test")
        self.assertEqual(d["description"], "测试")
        self.assertEqual(d["choices"], ["a", "b"])
        self.assertEqual(d["value_type"], "str")


class TestPreferenceSchema(unittest.TestCase):
    """测试PreferenceSchema"""

    def test09_register_and_get(self):
        """注册和获取"""
        item = PreferenceItem(
            key="custom.test",
            default="x",
            category="test",
        )
        PreferenceSchema.register(item)
        self.assertIs(PreferenceSchema.get("custom.test"), item)

    def test10_get_nonexistent(self):
        """获取不存在的项"""
        self.assertIsNone(PreferenceSchema.get("nonexistent.key"))

    def test11_get_all(self):
        """获取所有项"""
        all_items = PreferenceSchema.get_all()
        self.assertIsInstance(all_items, dict)
        self.assertGreater(len(all_items), 0)

    def test12_get_by_category(self):
        """按分类获取"""
        display_items = PreferenceSchema.get_by_category("display")
        self.assertIsInstance(display_items, dict)
        for item in display_items.values():
            self.assertEqual(item.category, "display")

    def test13_get_defaults(self):
        """获取默认值"""
        defaults = PreferenceSchema.get_defaults()
        self.assertIsInstance(defaults, dict)
        self.assertGreater(len(defaults), 0)

    def test14_get_keys(self):
        """获取所有键"""
        keys = PreferenceSchema.get_keys()
        self.assertIsInstance(keys, list)
        self.assertGreater(len(keys), 0)

    def test15_validate_key(self):
        """验证键"""
        self.assertTrue(PreferenceSchema.validate_key("display.theme"))
        self.assertFalse(PreferenceSchema.validate_key("nonexistent.key"))

    def test16_validate_value(self):
        """验证值"""
        self.assertTrue(PreferenceSchema.validate_value("display.theme", "dark"))
        self.assertFalse(PreferenceSchema.validate_value("display.theme", "invalid"))
        self.assertFalse(PreferenceSchema.validate_value("nonexistent", "x"))

    def test17_default_categories(self):
        """默认分类"""
        for cat in ["display", "behavior", "performance",
                    "notifications", "advanced"]:
            self.assertIn(cat, PreferenceSchema.CATEGORIES)

    def test18_has_all_categories_items(self):
        """每个分类都有项"""
        for cat in PreferenceSchema.CATEGORIES:
            items = PreferenceSchema.get_by_category(cat)
            self.assertGreater(len(items), 0,
                               f"分类 {cat} 没有偏好项")


class TestPreferences(unittest.TestCase):
    """测试Preferences"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test19_singleton(self):
        """单例模式"""
        p1 = Preferences()
        p2 = Preferences()
        self.assertIs(p1, p2)

    def test20_get_default(self):
        """获取默认值"""
        prefs = Preferences()
        prefs.reset_all()
        theme = prefs.get("display.theme")
        self.assertEqual(theme, "auto")  # 默认值

    def test21_set_and_get(self):
        """设置和获取"""
        prefs = Preferences()
        prefs.reset_all()
        self.assertTrue(prefs.set("display.theme", "dark"))
        self.assertEqual(prefs.get("display.theme"), "dark")

    def test22_set_invalid_key(self):
        """设置无效键"""
        prefs = Preferences()
        prefs.reset_all()
        self.assertFalse(prefs.set("nonexistent.key", "value"))

    def test23_set_invalid_value(self):
        """设置无效值"""
        prefs = Preferences()
        prefs.reset_all()
        self.assertFalse(prefs.set("display.theme", "invalid_theme"))

    def test24_reset(self):
        """重置单个偏好"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        prefs.reset("display.theme")
        self.assertEqual(prefs.get("display.theme"), "auto")  # 回到默认

    def test25_reset_all(self):
        """重置所有偏好"""
        prefs = Preferences()
        prefs.set("display.theme", "dark")
        prefs.set("display.verbose", True)
        prefs.reset_all()
        self.assertEqual(prefs.get("display.theme"), "auto")
        self.assertEqual(prefs.get("display.verbose"), False)

    def test26_reset_category(self):
        """重置分类"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        prefs.set("behavior.default_steps", 200)
        prefs.reset_category("display")
        self.assertEqual(prefs.get("display.theme"), "auto")
        self.assertEqual(prefs.get("behavior.default_steps"), 200)

    def test27_has(self):
        """has方法"""
        prefs = Preferences()
        prefs.reset_all()
        self.assertFalse(prefs.has("display.theme"))
        prefs.set("display.theme", "dark")
        self.assertTrue(prefs.has("display.theme"))

    def test28_to_dict(self):
        """to_dict包含默认值"""
        prefs = Preferences()
        prefs.reset_all()
        d = prefs.to_dict()
        self.assertIn("display.theme", d)
        self.assertEqual(d["display.theme"], "auto")

    def test29_to_dict_custom_only(self):
        """to_dict_custom_only仅自定义值"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        d = prefs.to_dict_custom_only()
        self.assertEqual(len(d), 1)
        self.assertIn("display.theme", d)

    def test30_from_dict(self):
        """from_dict"""
        prefs = Preferences()
        prefs.reset_all()
        count = prefs.from_dict({"display.theme": "dark",
                                 "behavior.default_steps": 50})
        self.assertEqual(count, 2)
        self.assertEqual(prefs.get("display.theme"), "dark")
        self.assertEqual(prefs.get("behavior.default_steps"), 50)

    def test31_from_dict_invalid(self):
        """from_dict跳过无效项"""
        prefs = Preferences()
        prefs.reset_all()
        count = prefs.from_dict({"display.theme": "dark",
                                 "invalid.key": "x"})
        self.assertEqual(count, 1)

    def test32_get_category(self):
        """获取分类"""
        prefs = Preferences()
        prefs.reset_all()
        display = prefs.get_category("display")
        self.assertIsInstance(display, dict)
        self.assertIn("display.theme", display)

    def test33_get_categories(self):
        """获取所有分类"""
        prefs = Preferences()
        cats = prefs.get_categories()
        self.assertIn("display", cats)
        self.assertIn("behavior", cats)

    def test34_save_and_load(self):
        """保存和加载"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prefs.json")
            saved = prefs.save(path)
            self.assertEqual(saved, path)
            self.assertTrue(os.path.exists(path))
            # 重置后加载
            prefs.reset_all()
            self.assertEqual(prefs.get("display.theme"), "auto")
            count = prefs.load(path)
            self.assertGreater(count, 0)
            self.assertEqual(prefs.get("display.theme"), "dark")

    def test35_export_import(self):
        """导出导入"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        prefs.set("behavior.default_steps", 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "export.json")
            prefs.export(export_path)
            # 重置
            prefs.reset_all()
            # 导入
            count = prefs.import_from(export_path)
            self.assertEqual(count, 2)
            self.assertEqual(prefs.get("display.theme"), "dark")
            self.assertEqual(prefs.get("behavior.default_steps"), 50)

    def test36_get_report(self):
        """获取报告"""
        prefs = Preferences()
        prefs.reset_all()
        report = prefs.get_report()
        self.assertIn("用户偏好设置", report)
        self.assertIn("display", report)


class TestProfile(unittest.TestCase):
    """测试Profile"""

    def setUp(self):
        _reset_all()
        # 使用临时目录
        self._tmpdir = tempfile.mkdtemp()
        self._profiles_dir_patch = patch("seva.preferences.PROFILES_DIR",
                                          Path(self._tmpdir) / "profiles")
        self._profiles_dir_patch.start()

    def tearDown(self):
        self._profiles_dir_patch.stop()
        _reset_all()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test37_init_basic(self):
        """基本初始化"""
        profile = Profile("test")
        self.assertEqual(profile.name, "test")
        self.assertEqual(profile.preferences, {})
        self.assertGreater(profile.created_at, 0)

    def test38_init_with_preferences(self):
        """带偏好的初始化"""
        profile = Profile("test", {"display.theme": "dark"})
        self.assertEqual(profile.preferences, {"display.theme": "dark"})

    def test39_get_set(self):
        """get/set"""
        profile = Profile("test")
        self.assertTrue(profile.set("display.theme", "dark"))
        self.assertEqual(profile.get("display.theme"), "dark")

    def test40_set_invalid(self):
        """设置无效值"""
        profile = Profile("test")
        self.assertFalse(profile.set("invalid.key", "x"))
        self.assertFalse(profile.set("display.theme", "invalid"))

    def test41_remove(self):
        """remove"""
        profile = Profile("test", {"display.theme": "dark"})
        self.assertTrue(profile.remove("display.theme"))
        self.assertFalse(profile.remove("display.theme"))

    def test42_clear(self):
        """clear"""
        profile = Profile("test", {"display.theme": "dark"})
        profile.clear()
        self.assertEqual(profile.preferences, {})

    def test43_to_dict(self):
        """to_dict"""
        profile = Profile("test", {"display.theme": "dark"})
        d = profile.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["preferences"], {"display.theme": "dark"})
        self.assertIn("created_at", d)
        self.assertIn("last_used", d)

    def test44_from_dict(self):
        """from_dict"""
        data = {
            "name": "test",
            "preferences": {"display.theme": "dark"},
            "created_at": 1000,
            "last_used": 2000,
        }
        profile = Profile.from_dict(data)
        self.assertEqual(profile.name, "test")
        self.assertEqual(profile.preferences, {"display.theme": "dark"})
        self.assertEqual(profile.created_at, 1000)
        self.assertEqual(profile.last_used, 2000)

    def test45_save_and_load(self):
        """保存和加载"""
        profile = Profile("test", {"display.theme": "dark"})
        path = profile.save()
        self.assertTrue(os.path.exists(path))
        # 加载
        loaded = Profile.load("test")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "test")
        self.assertEqual(loaded.get("display.theme"), "dark")

    def test46_load_nonexistent(self):
        """加载不存在的配置"""
        self.assertIsNone(Profile.load("nonexistent"))

    def test47_delete(self):
        """删除"""
        profile = Profile("test", {"display.theme": "dark"})
        profile.save()
        self.assertTrue(profile.delete())
        self.assertIsNone(Profile.load("test"))

    def test48_exists(self):
        """exists"""
        profile = Profile("test")
        self.assertFalse(profile.exists())
        profile.save()
        self.assertTrue(profile.exists())

    def test49_apply_to(self):
        """apply_to"""
        profile = Profile("test", {"display.theme": "dark",
                                    "behavior.default_steps": 50})
        prefs = Preferences()
        prefs.reset_all()
        count = profile.apply_to(prefs)
        self.assertEqual(count, 2)
        self.assertEqual(prefs.get("display.theme"), "dark")
        self.assertEqual(prefs.get("behavior.default_steps"), 50)

    def test50_capture_current(self):
        """capture_current"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        profile = Profile.capture_current("snapshot", prefs)
        self.assertEqual(profile.name, "snapshot")
        self.assertEqual(profile.get("display.theme"), "dark")

    def test51_update_last_used(self):
        """update_last_used"""
        profile = Profile("test")
        old_time = profile.last_used
        time.sleep(0.01)
        profile.update_last_used()
        self.assertGreater(profile.last_used, old_time)


class TestProfileManager(unittest.TestCase):
    """测试ProfileManager"""

    def setUp(self):
        _reset_all()
        self._tmpdir = tempfile.mkdtemp()
        self._profiles_dir_patch = patch("seva.preferences.PROFILES_DIR",
                                          Path(self._tmpdir) / "profiles")
        self._current_file_patch = patch("seva.preferences.CURRENT_PROFILE_FILE",
                                          Path(self._tmpdir) / "current")
        self._profiles_dir_patch.start()
        self._current_file_patch.start()

    def tearDown(self):
        self._profiles_dir_patch.stop()
        self._current_file_patch.stop()
        _reset_all()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test52_singleton(self):
        """单例"""
        pm1 = ProfileManager()
        pm2 = ProfileManager()
        self.assertIs(pm1, pm2)

    def test53_create(self):
        """创建配置文件"""
        pm = ProfileManager()
        profile = pm.create("test")
        self.assertEqual(profile.name, "test")
        self.assertTrue(profile.exists())

    def test54_create_from_current(self):
        """从当前偏好创建"""
        prefs = Preferences()
        prefs.reset_all()
        prefs.set("display.theme", "dark")
        pm = ProfileManager()
        profile = pm.create("snapshot", from_current=True)
        self.assertEqual(profile.get("display.theme"), "dark")

    def test55_delete(self):
        """删除配置文件"""
        pm = ProfileManager()
        pm.create("test")
        self.assertTrue(pm.delete("test"))
        self.assertIsNone(pm.get("test"))

    def test56_delete_nonexistent(self):
        """删除不存在的配置"""
        pm = ProfileManager()
        self.assertFalse(pm.delete("nonexistent"))

    def test57_get(self):
        """获取配置文件"""
        pm = ProfileManager()
        pm.create("test")
        profile = pm.get("test")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "test")

    def test58_list_profiles(self):
        """列出配置文件"""
        pm = ProfileManager()
        pm.create("test1")
        pm.create("test2")
        profiles = pm.list_profiles()
        self.assertEqual(len(profiles), 2)
        names = [p["name"] for p in profiles]
        self.assertIn("test1", names)
        self.assertIn("test2", names)

    def test59_list_profiles_empty(self):
        """空列表"""
        pm = ProfileManager()
        profiles = pm.list_profiles()
        self.assertEqual(len(profiles), 0)

    def test60_switch(self):
        """切换配置文件"""
        pm = ProfileManager()
        pm.create("test")
        self.assertTrue(pm.switch("test"))
        self.assertEqual(pm.get_current(), "test")

    def test61_switch_nonexistent(self):
        """切换不存在的配置"""
        pm = ProfileManager()
        self.assertFalse(pm.switch("nonexistent"))

    def test62_get_current(self):
        """获取当前配置"""
        pm = ProfileManager()
        self.assertIsNone(pm.get_current())
        pm.create("test")
        pm.switch("test")
        self.assertEqual(pm.get_current(), "test")

    def test63_rename(self):
        """重命名"""
        pm = ProfileManager()
        pm.create("old")
        self.assertTrue(pm.rename("old", "new"))
        self.assertIsNotNone(pm.get("new"))
        self.assertIsNone(pm.get("old"))

    def test64_rename_nonexistent(self):
        """重命名不存在的配置"""
        pm = ProfileManager()
        self.assertFalse(pm.rename("nonexistent", "new"))

    def test65_duplicate(self):
        """复制配置文件"""
        pm = ProfileManager()
        pm.create("src")
        self.assertTrue(pm.duplicate("src", "dst"))
        self.assertIsNotNone(pm.get("dst"))

    def test66_delete_current_fails(self):
        """不能删除当前配置"""
        pm = ProfileManager()
        pm.create("current")
        pm.switch("current")
        self.assertFalse(pm.delete("current"))


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test67_get_preference(self):
        """get_preference"""
        self.assertEqual(get_preference("display.theme"), "auto")

    def test68_set_preference(self):
        """set_preference"""
        self.assertTrue(set_preference("display.theme", "dark"))
        self.assertEqual(get_preference("display.theme"), "dark")

    def test69_reset_preference(self):
        """reset_preference"""
        set_preference("display.theme", "dark")
        self.assertTrue(reset_preference("display.theme"))
        self.assertEqual(get_preference("display.theme"), "auto")

    def test70_reset_all_preferences(self):
        """reset_all_preferences"""
        set_preference("display.theme", "dark")
        set_preference("behavior.default_steps", 50)
        reset_all_preferences()
        self.assertEqual(get_preference("display.theme"), "auto")
        self.assertEqual(get_preference("behavior.default_steps"), 100)

    def test71_list_preferences(self):
        """list_preferences"""
        prefs = list_preferences()
        self.assertIsInstance(prefs, dict)
        self.assertIn("display.theme", prefs)

    def test72_list_preferences_by_category(self):
        """按分类列出偏好"""
        prefs = list_preferences("display")
        for key in prefs:
            self.assertTrue(key.startswith("display."))

    def test73_list_preference_schema(self):
        """list_preference_schema"""
        schema = list_preference_schema()
        self.assertIn("display.theme", schema)
        self.assertIn("value_type", schema["display.theme"])

    def test74_save_and_load_preferences(self):
        """save/load_preferences"""
        set_preference("display.theme", "dark")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_preferences(os.path.join(tmpdir, "p.json"))
            self.assertTrue(os.path.exists(path))
            reset_all_preferences()
            count = load_preferences(path)
            self.assertGreater(count, 0)
            self.assertEqual(get_preference("display.theme"), "dark")

    def test75_export_import_preferences(self):
        """export/import_preferences"""
        set_preference("display.theme", "dark")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_preferences(os.path.join(tmpdir, "e.json"))
            reset_all_preferences()
            count = import_preferences(path)
            self.assertEqual(get_preference("display.theme"), "dark")

    def test76_get_preferences_report(self):
        """get_preferences_report"""
        report = get_preferences_report()
        self.assertIn("用户偏好设置", report)


class TestPackageExports(unittest.TestCase):
    """测试包导出"""

    def test77_preferences_exported(self):
        self.assertTrue(hasattr(seva, "Preferences"))

    def test78_profile_exported(self):
        self.assertTrue(hasattr(seva, "Profile"))

    def test79_profile_manager_exported(self):
        self.assertTrue(hasattr(seva, "ProfileManager"))

    def test80_preference_schema_exported(self):
        self.assertTrue(hasattr(seva, "PreferenceSchema"))

    def test81_preference_item_exported(self):
        self.assertTrue(hasattr(seva, "PreferenceItem"))

    def test82_version_updated(self):
        self.assertGreaterEqual(seva.__version__, "0.45.0")

    def test83_all_exports(self):
        for name in ["PreferenceItem", "PreferenceSchema", "Preferences",
                     "Profile", "ProfileManager",
                     "get_preference", "set_preference",
                     "reset_preference", "reset_all_preferences",
                     "list_preferences", "list_preference_schema",
                     "save_preferences", "load_preferences",
                     "export_preferences", "import_preferences",
                     "get_preferences_report",
                     "create_profile", "delete_profile",
                     "switch_profile", "list_profiles",
                     "get_current_profile", "get_profile"]:
            self.assertIn(name, seva.__all__,
                          f"{name} 未在__all__中")


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def _run_cli(self, *args):
        from seva.cli import main
        old_argv = sys.argv
        sys.argv = ["seva"] + list(args)
        try:
            return main()
        finally:
            sys.argv = old_argv

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test84_cli_prefs_default(self):
        """seva prefs 默认显示报告"""
        code = self._run_cli("prefs")
        self.assertEqual(code, 0)

    def test85_cli_prefs_list(self):
        """seva prefs --list"""
        code = self._run_cli("prefs", "--list")
        self.assertEqual(code, 0)

    def test86_cli_prefs_schema(self):
        """seva prefs --schema"""
        code = self._run_cli("prefs", "--schema")
        self.assertEqual(code, 0)

    def test87_cli_prefs_get(self):
        """seva prefs --get"""
        code = self._run_cli("prefs", "--get", "display.theme")
        self.assertEqual(code, 0)

    def test88_cli_prefs_set(self):
        """seva prefs --set"""
        code = self._run_cli("prefs", "--set", "display.theme=dark")
        self.assertEqual(code, 0)
        self.assertEqual(get_preference("display.theme"), "dark")

    def test89_cli_prefs_set_invalid(self):
        """seva prefs --set 无效值"""
        code = self._run_cli("prefs", "--set", "display.theme=invalid")
        self.assertEqual(code, 1)

    def test90_cli_prefs_set_invalid_key(self):
        """seva prefs --set 无效键"""
        code = self._run_cli("prefs", "--set", "invalid.key=x")
        self.assertEqual(code, 1)

    def test91_cli_prefs_reset(self):
        """seva prefs --reset"""
        set_preference("display.theme", "dark")
        code = self._run_cli("prefs", "--reset", "display.theme")
        self.assertEqual(code, 0)
        self.assertEqual(get_preference("display.theme"), "auto")

    def test92_cli_prefs_reset_all(self):
        """seva prefs --reset all"""
        set_preference("display.theme", "dark")
        code = self._run_cli("prefs", "--reset", "all")
        self.assertEqual(code, 0)
        self.assertEqual(get_preference("display.theme"), "auto")

    def test93_cli_prefs_export_import(self):
        """seva prefs --export / --import-file"""
        set_preference("display.theme", "dark")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prefs.json")
            code = self._run_cli("prefs", "--export", path)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path))
            reset_all_preferences()
            code = self._run_cli("prefs", "--import-file", path)
            self.assertEqual(code, 0)
            self.assertEqual(get_preference("display.theme"), "dark")

    def test94_cli_prefs_profile_list(self):
        """seva prefs --profile list"""
        code = self._run_cli("prefs", "--profile", "list")
        self.assertEqual(code, 0)


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def setUp(self):
        _reset_all()
        self._tmpdir = tempfile.mkdtemp()
        self._profiles_dir_patch = patch("seva.preferences.PROFILES_DIR",
                                          Path(self._tmpdir) / "profiles")
        self._current_file_patch = patch("seva.preferences.CURRENT_PROFILE_FILE",
                                          Path(self._tmpdir) / "current")
        self._profiles_dir_patch.start()
        self._current_file_patch.start()

    def tearDown(self):
        self._profiles_dir_patch.stop()
        self._current_file_patch.stop()
        _reset_all()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test95_full_preferences_flow(self):
        """完整偏好流程"""
        # 设置偏好
        set_preference("display.theme", "dark")
        set_preference("display.verbose", True)
        set_preference("behavior.default_steps", 200)

        # 验证
        self.assertEqual(get_preference("display.theme"), "dark")
        self.assertEqual(get_preference("display.verbose"), True)
        self.assertEqual(get_preference("behavior.default_steps"), 200)

        # 导出
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_preferences(os.path.join(tmpdir, "prefs.json"))
            # 重置
            reset_all_preferences()
            # 导入
            count = import_preferences(path)
            self.assertEqual(count, 3)
            # 验证
            self.assertEqual(get_preference("display.theme"), "dark")

    def test96_profile_workflow(self):
        """配置文件工作流"""
        # 设置当前偏好
        set_preference("display.theme", "dark")
        set_preference("behavior.default_steps", 50)

        # 创建配置文件
        profile = create_profile("work", from_current=True)
        self.assertEqual(profile.get("display.theme"), "dark")

        # 修改当前偏好
        set_preference("display.theme", "light")
        set_preference("behavior.default_steps", 100)

        # 切换回work配置
        self.assertTrue(switch_profile("work"))
        self.assertEqual(get_preference("display.theme"), "dark")
        self.assertEqual(get_preference("behavior.default_steps"), 50)

    def test97_multiple_profiles(self):
        """多配置文件"""
        # 创建多个配置
        create_profile("home")
        create_profile("work")
        create_profile("lab")

        profiles = list_profiles()
        self.assertEqual(len(profiles), 3)

        # 切换
        self.assertTrue(switch_profile("work"))
        self.assertEqual(get_current_profile(), "work")
        self.assertTrue(switch_profile("home"))
        self.assertEqual(get_current_profile(), "home")

    def test98_preferences_persistence(self):
        """偏好持久化"""
        set_preference("display.theme", "dark")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_preferences(os.path.join(tmpdir, "prefs.json"))
            # 重置单例
            _reset_all()
            # 重新加载
            count = load_preferences(path)
            self.assertGreater(count, 0)
            self.assertEqual(get_preference("display.theme"), "dark")

    def test99_category_management(self):
        """分类管理"""
        # 设置多个分类的偏好
        set_preference("display.theme", "dark")
        set_preference("behavior.default_steps", 50)
        set_preference("performance.cache_size", 256)

        # 验证所有分类
        for cat in ["display", "behavior", "performance"]:
            prefs = list_preferences(cat)
            self.assertGreater(len(prefs), 0)

        # 重置display分类
        from seva.preferences import Preferences
        Preferences().reset_category("display")
        self.assertEqual(get_preference("display.theme"), "auto")
        # 其他分类不受影响
        self.assertEqual(get_preference("behavior.default_steps"), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
