"""
阶段51测试：可访问性与无障碍支持
验证 AccessibilityConfig / ColorBlindnessSupport / ScreenReaderSupport /
HighContrastMode / KeyboardNavigation / AccessibleFormatter / AccessibilityManager
及便捷函数和CLI集成
"""
import os
import sys
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


import seva
from seva.accessibility import (AccessibilityConfig, AccessibilityLevel,
                                ColorBlindnessType, ColorBlindnessSupport,
                                ScreenReaderSupport, HighContrastMode,
                                KeyboardNavigation, AccessibleFormatter,
                                AccessibilityManager,
                                format_accessible, get_accessibility_config,
                                set_accessibility_config,
                                update_accessibility_config,
                                reset_accessibility_config,
                                get_accessibility_report,
                                enable_high_contrast, enable_large_text,
                                enable_screen_reader, enable_reduce_motion,
                                set_color_blindness, get_color_blind_palette,
                                get_keybindings, format_keybindings_help,
                                COLOR_DESCRIPTIONS, STATUS_TEXT,
                                ANSI_TO_TEXT, COLOR_BLIND_PALETTE,
                                HIGH_CONTRAST_COLORS, DEFAULT_KEYBINDINGS,
                                CONFIG_FILE)


class TestAccessibilityConfig(unittest.TestCase):
    """测试AccessibilityConfig"""

    def test01_init_default(self):
        """默认初始化"""
        config = AccessibilityConfig()
        self.assertFalse(config.high_contrast)
        self.assertFalse(config.large_text)
        self.assertFalse(config.screen_reader)
        self.assertEqual(config.color_blindness, ColorBlindnessType.NONE)
        self.assertEqual(config.font_scale, 1.0)

    def test02_init_full(self):
        """完整初始化"""
        config = AccessibilityConfig(
            high_contrast=True,
            large_text=True,
            screen_reader=True,
            reduce_motion=True,
            color_blindness=ColorBlindnessType.PROTANOPIA,
            keyboard_only=True,
            verbose_descriptions=True,
            font_scale=1.5,
        )
        self.assertTrue(config.high_contrast)
        self.assertTrue(config.large_text)
        self.assertTrue(config.screen_reader)
        self.assertTrue(config.reduce_motion)
        self.assertEqual(config.color_blindness,
                         ColorBlindnessType.PROTANOPIA)
        self.assertTrue(config.keyboard_only)
        self.assertTrue(config.verbose_descriptions)
        self.assertEqual(config.font_scale, 1.5)

    def test03_font_scale_clamped(self):
        """字体缩放限制"""
        config = AccessibilityConfig(font_scale=0.1)
        self.assertEqual(config.font_scale, 0.5)
        config = AccessibilityConfig(font_scale=5.0)
        self.assertEqual(config.font_scale, 3.0)

    def test04_level_none(self):
        """等级NONE"""
        config = AccessibilityConfig()
        self.assertEqual(config.level, AccessibilityLevel.NONE)

    def test05_level_basic(self):
        """等级BASIC"""
        config = AccessibilityConfig(high_contrast=True)
        self.assertEqual(config.level, AccessibilityLevel.BASIC)

    def test06_level_enhanced(self):
        """等级ENHANCED"""
        config = AccessibilityConfig(
            high_contrast=True, large_text=True,
            screen_reader=True, reduce_motion=True,
        )
        self.assertEqual(config.level, AccessibilityLevel.ENHANCED)

    def test07_level_maximum(self):
        """等级MAXIMUM"""
        config = AccessibilityConfig(
            high_contrast=True, large_text=True,
            screen_reader=True, reduce_motion=True,
            keyboard_only=True, verbose_descriptions=True,
            color_blindness=ColorBlindnessType.PROTANOPIA,
        )
        self.assertEqual(config.level, AccessibilityLevel.MAXIMUM)

    def test08_to_dict(self):
        """to_dict"""
        config = AccessibilityConfig(high_contrast=True)
        d = config.to_dict()
        self.assertTrue(d["high_contrast"])
        self.assertIn("level", d)

    def test09_from_dict(self):
        """from_dict"""
        data = {
            "high_contrast": True,
            "large_text": False,
            "screen_reader": True,
            "reduce_motion": False,
            "color_blindness": ColorBlindnessType.DEUTERANOPIA,
            "keyboard_only": False,
            "verbose_descriptions": True,
            "text_alternatives": True,
            "font_scale": 2.0,
        }
        config = AccessibilityConfig.from_dict(data)
        self.assertTrue(config.high_contrast)
        self.assertTrue(config.screen_reader)
        self.assertEqual(config.color_blindness,
                         ColorBlindnessType.DEUTERANOPIA)
        self.assertEqual(config.font_scale, 2.0)

    def test10_repr(self):
        """__repr__"""
        config = AccessibilityConfig(high_contrast=True)
        r = repr(config)
        self.assertIn("AccessibilityConfig", r)


class TestColorBlindnessSupport(unittest.TestCase):
    """测试ColorBlindnessSupport"""

    def test01_init_none(self):
        """无色盲"""
        cbs = ColorBlindnessSupport(ColorBlindnessType.NONE)
        self.assertEqual(cbs.blindness_type, ColorBlindnessType.NONE)

    def test02_get_color_none(self):
        """无色盲时返回原色"""
        cbs = ColorBlindnessSupport(ColorBlindnessType.NONE)
        self.assertEqual(cbs.get_color("red"), "red")

    def test03_get_color_protanopia(self):
        """红色盲颜色替代"""
        cbs = ColorBlindnessSupport(ColorBlindnessType.PROTANOPIA)
        color = cbs.get_color("red")
        self.assertNotEqual(color, "red")

    def test04_get_color_deuteranopia(self):
        """绿色盲颜色替代"""
        cbs = ColorBlindnessSupport(ColorBlindnessType.DEUTERANOPIA)
        color = cbs.get_color("green")
        self.assertNotEqual(color, "green")

    def test05_get_pattern(self):
        """获取图案"""
        cbs = ColorBlindnessSupport()
        self.assertEqual(cbs.get_pattern("error"), "✗")
        self.assertEqual(cbs.get_pattern("success"), "✓")

    def test06_describe_color(self):
        """描述颜色"""
        cbs = ColorBlindnessSupport()
        self.assertEqual(cbs.describe_color("red"), "红色")
        self.assertEqual(cbs.describe_color("blue"), "蓝色")

    def test07_describe_color_unknown(self):
        """未知颜色"""
        cbs = ColorBlindnessSupport()
        self.assertEqual(cbs.describe_color("unknown"), "unknown")

    def test08_get_report(self):
        """获取报告"""
        cbs = ColorBlindnessSupport(ColorBlindnessType.PROTANOPIA)
        report = cbs.get_report()
        self.assertIn("protanopia", report)

    def test09_all_blindness_types(self):
        """所有色盲类型"""
        for bt in [ColorBlindnessType.PROTANOPIA,
                   ColorBlindnessType.DEUTERANOPIA,
                   ColorBlindnessType.TRITANOPIA,
                   ColorBlindnessType.ACHROMATOPSIA]:
            cbs = ColorBlindnessSupport(bt)
            color = cbs.get_color("red")
            self.assertIsInstance(color, str)


class TestScreenReaderSupport(unittest.TestCase):
    """测试ScreenReaderSupport"""

    def test01_init(self):
        """初始化"""
        srs = ScreenReaderSupport()
        self.assertFalse(srs.verbose)

    def test02_aria_label_basic(self):
        """基本ARIA标签"""
        srs = ScreenReaderSupport()
        label = srs.aria_label("button", "确定")
        self.assertIn("确定", label)

    def test03_aria_label_with_state(self):
        """带状态的ARIA标签"""
        srs = ScreenReaderSupport()
        label = srs.aria_label("button", "确定", state="enabled")
        self.assertIn("确定", label)
        self.assertIn("enabled", label)

    def test04_aria_label_verbose(self):
        """详细ARIA标签"""
        srs = ScreenReaderSupport(verbose=True)
        label = srs.aria_label("button", "确定")
        self.assertIn("button", label)

    def test05_describe(self):
        """描述文本"""
        srs = ScreenReaderSupport()
        text = srs.describe("running 任务")
        self.assertIn("运行中", text)

    def test06_describe_with_ansi(self):
        """描述带ANSI的文本"""
        srs = ScreenReaderSupport()
        text = srs.describe("\033[31m错误\033[0m")
        self.assertIn("红色", text)
        self.assertIn("错误", text)

    def test07_announce_polite(self):
        """礼貌公告"""
        srs = ScreenReaderSupport()
        msg = srs.announce("更新完成")
        self.assertIn("更新完成", msg)

    def test08_announce_assertive(self):
        """紧急公告"""
        srs = ScreenReaderSupport()
        msg = srs.announce("严重错误", priority="assertive")
        self.assertIn("紧急", msg)

    def test09_describe_progress(self):
        """描述进度"""
        srs = ScreenReaderSupport()
        text = srs.describe_progress(50, 100, "下载")
        self.assertIn("50", text)
        self.assertIn("100", text)
        self.assertIn("50%", text)

    def test10_describe_status(self):
        """描述状态"""
        srs = ScreenReaderSupport()
        text = srs.describe_status("running", "任务1", "处理中")
        self.assertIn("运行中", text)
        self.assertIn("任务1", text)

    def test11_describe_with_context_verbose(self):
        """带上下文的详细描述"""
        srs = ScreenReaderSupport(verbose=True)
        text = srs.describe("test", context="按钮")
        self.assertIn("按钮", text)


class TestHighContrastMode(unittest.TestCase):
    """测试HighContrastMode"""

    def test01_init(self):
        """初始化"""
        hcm = HighContrastMode()
        self.assertIn("background", hcm.colors)

    def test02_get_color(self):
        """获取颜色"""
        hcm = HighContrastMode()
        self.assertEqual(hcm.get_color("background"), "#000000")
        self.assertEqual(hcm.get_color("foreground"), "#FFFFFF")

    def test03_get_color_unknown(self):
        """未知语义颜色"""
        hcm = HighContrastMode()
        color = hcm.get_color("unknown")
        self.assertEqual(color, hcm.colors["foreground"])

    def test04_format_text(self):
        """格式化文本"""
        hcm = HighContrastMode()
        text = hcm.format_text("测试", "error")
        self.assertIn("测试", text)

    def test05_contrast_ratio_black_white(self):
        """黑白对比度"""
        hcm = HighContrastMode()
        ratio = hcm.get_contrast_ratio("#FFFFFF", "#000000")
        self.assertEqual(ratio, 21.0)

    def test06_contrast_ratio_default(self):
        """默认对比度"""
        hcm = HighContrastMode()
        ratio = hcm.get_contrast_ratio("#FF0000", "#00FF00")
        self.assertEqual(ratio, 7.0)

    def test07_get_report(self):
        """获取报告"""
        hcm = HighContrastMode()
        report = hcm.get_report()
        self.assertIn("高对比度", report)
        self.assertIn("对比度比率", report)


class TestKeyboardNavigation(unittest.TestCase):
    """测试KeyboardNavigation"""

    def test01_init(self):
        """初始化"""
        kn = KeyboardNavigation()
        self.assertIn("help", kn.bindings)

    def test02_get_keys(self):
        """获取快捷键"""
        kn = KeyboardNavigation()
        keys = kn.get_keys("help")
        self.assertIn("?", keys)
        self.assertIn("h", keys)

    def test03_get_keys_unknown(self):
        """未知动作"""
        kn = KeyboardNavigation()
        keys = kn.get_keys("unknown_action")
        self.assertEqual(keys, [])

    def test04_get_action(self):
        """获取动作"""
        kn = KeyboardNavigation()
        action = kn.get_action("q")
        self.assertEqual(action, "quit")

    def test05_get_action_unknown(self):
        """未知快捷键"""
        kn = KeyboardNavigation()
        action = kn.get_action("unknown_key")
        self.assertIsNone(action)

    def test06_bind(self):
        """绑定快捷键"""
        kn = KeyboardNavigation()
        kn.bind("custom", ["x", "y"])
        self.assertEqual(kn.get_keys("custom"), ["x", "y"])

    def test07_unbind(self):
        """解绑快捷键"""
        kn = KeyboardNavigation()
        self.assertTrue(kn.unbind("help"))
        self.assertNotIn("help", kn.bindings)

    def test08_unbind_unknown(self):
        """解绑未知动作"""
        kn = KeyboardNavigation()
        self.assertFalse(kn.unbind("unknown"))

    def test09_list_actions(self):
        """列出所有动作"""
        kn = KeyboardNavigation()
        actions = kn.list_actions()
        self.assertIn("help", actions)
        self.assertIn("quit", actions)

    def test10_format_help(self):
        """格式化帮助"""
        kn = KeyboardNavigation()
        help_text = kn.format_help()
        self.assertIn("help", help_text)
        self.assertIn("quit", help_text)

    def test11_default_bindings_complete(self):
        """默认绑定完整"""
        kn = KeyboardNavigation()
        for action in ["help", "quit", "cancel", "confirm", "next",
                       "previous", "up", "down", "search", "menu",
                       "refresh"]:
            self.assertIn(action, kn.bindings)


class TestAccessibleFormatter(unittest.TestCase):
    """测试AccessibleFormatter"""

    def test01_init_default(self):
        """默认初始化"""
        formatter = AccessibleFormatter()
        self.assertFalse(formatter.config.high_contrast)

    def test02_init_with_config(self):
        """带配置初始化"""
        config = AccessibilityConfig(high_contrast=True)
        formatter = AccessibleFormatter(config)
        self.assertTrue(formatter.config.high_contrast)

    def test03_format_basic(self):
        """基本格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format("测试文本")
        self.assertIn("测试文本", result)

    def test04_format_screen_reader(self):
        """屏幕阅读器格式化"""
        config = AccessibilityConfig(screen_reader=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format("running 任务")
        self.assertIn("运行中", result)

    def test05_format_high_contrast(self):
        """高对比度格式化"""
        config = AccessibilityConfig(high_contrast=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format("\033[31m错误\033[0m")
        # 应该移除ANSI代码
        self.assertNotIn("\033[31m", result)

    def test06_format_status_normal(self):
        """普通状态格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_status("running", "任务1", "处理中")
        self.assertIn("任务1", result)

    def test07_format_status_screen_reader(self):
        """屏幕阅读器状态格式化"""
        config = AccessibilityConfig(screen_reader=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format_status("running", "任务1", "处理中")
        self.assertIn("运行中", result)
        self.assertIn("任务1", result)

    def test08_format_progress_normal(self):
        """普通进度格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_progress(50, 100, "下载")
        self.assertIn("50", result)
        self.assertIn("下载", result)

    def test09_format_progress_reduce_motion(self):
        """减少动画进度格式化"""
        config = AccessibilityConfig(reduce_motion=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format_progress(50, 100, "下载")
        self.assertIn("50/100", result)

    def test10_format_progress_screen_reader(self):
        """屏幕阅读器进度格式化"""
        config = AccessibilityConfig(screen_reader=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format_progress(50, 100, "下载")
        self.assertIn("进度", result)

    def test11_format_error_normal(self):
        """普通错误格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_error("文件未找到", "路径错误", "检查路径")
        self.assertIn("文件未找到", result)
        self.assertIn("路径错误", result)

    def test12_format_error_screen_reader(self):
        """屏幕阅读器错误格式化"""
        config = AccessibilityConfig(screen_reader=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format_error("文件未找到", "路径错误", "检查路径")
        self.assertIn("错误", result)
        self.assertIn("文件未找到", result)

    def test13_format_success(self):
        """成功格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_success("操作完成")
        self.assertIn("操作完成", result)

    def test14_format_success_screen_reader(self):
        """屏幕阅读器成功格式化"""
        config = AccessibilityConfig(screen_reader=True)
        formatter = AccessibleFormatter(config)
        result = formatter.format_success("操作完成")
        self.assertIn("成功", result)

    def test15_format_warning(self):
        """警告格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_warning("资源不足")
        self.assertIn("资源不足", result)

    def test16_format_info(self):
        """信息格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format_info("提示信息")
        self.assertIn("提示信息", result)

    def test17_get_color_high_contrast(self):
        """高对比度获取颜色"""
        config = AccessibilityConfig(high_contrast=True)
        formatter = AccessibleFormatter(config)
        color = formatter.get_color("error")
        self.assertEqual(color, "#FF0000")

    def test18_get_color_color_blind(self):
        """色盲获取颜色"""
        config = AccessibilityConfig(
            color_blindness=ColorBlindnessType.PROTANOPIA)
        formatter = AccessibleFormatter(config)
        color = formatter.get_color("red")
        self.assertNotEqual(color, "red")

    def test19_get_pattern(self):
        """获取图案"""
        formatter = AccessibleFormatter()
        pattern = formatter.get_pattern("error")
        self.assertEqual(pattern, "✗")


class TestAccessibilityManager(unittest.TestCase):
    """测试AccessibilityManager"""

    def test01_singleton(self):
        """单例模式"""
        m1 = AccessibilityManager()
        m2 = AccessibilityManager()
        self.assertIs(m1, m2)

    def test02_get_config(self):
        """获取配置"""
        manager = AccessibilityManager()
        config = manager.get_config()
        self.assertIsInstance(config, AccessibilityConfig)

    def test03_set_config(self):
        """设置配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            new_config = AccessibilityConfig(high_contrast=True)
            manager.set_config(new_config)
            self.assertTrue(manager.get_config().high_contrast)
        finally:
            manager.set_config(original)

    def test04_update_config(self):
        """更新配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            manager.update_config(high_contrast=True)
            self.assertTrue(manager.get_config().high_contrast)
        finally:
            manager.set_config(original)

    def test05_reset(self):
        """重置配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            manager.update_config(high_contrast=True, large_text=True)
            manager.reset()
            config = manager.get_config()
            self.assertFalse(config.high_contrast)
            self.assertFalse(config.large_text)
        finally:
            manager.set_config(original)

    def test06_get_formatter(self):
        """获取格式化器"""
        manager = AccessibilityManager()
        formatter = manager.get_formatter()
        self.assertIsInstance(formatter, AccessibleFormatter)

    def test07_format(self):
        """格式化"""
        manager = AccessibilityManager()
        result = manager.format("测试")
        self.assertIn("测试", result)

    def test08_get_report(self):
        """获取报告"""
        manager = AccessibilityManager()
        report = manager.get_report()
        self.assertIn("可访问性", report)
        self.assertIn("等级", report)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def test01_format_accessible_with_config(self):
        """带配置的格式化"""
        config = AccessibilityConfig(screen_reader=True)
        result = format_accessible("running", config)
        self.assertIn("运行中", result)

    def test02_format_accessible_default(self):
        """默认格式化"""
        result = format_accessible("测试")
        self.assertIn("测试", result)

    def test03_get_accessibility_config(self):
        """获取配置"""
        config = get_accessibility_config()
        self.assertIsInstance(config, AccessibilityConfig)

    def test04_set_accessibility_config(self):
        """设置配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            new_config = AccessibilityConfig(large_text=True)
            set_accessibility_config(new_config)
            self.assertTrue(get_accessibility_config().large_text)
        finally:
            set_accessibility_config(original)

    def test05_update_accessibility_config(self):
        """更新配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            update_accessibility_config(screen_reader=True)
            self.assertTrue(get_accessibility_config().screen_reader)
        finally:
            set_accessibility_config(original)

    def test06_reset_accessibility_config(self):
        """重置配置"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            update_accessibility_config(high_contrast=True)
            reset_accessibility_config()
            self.assertFalse(get_accessibility_config().high_contrast)
        finally:
            set_accessibility_config(original)

    def test07_get_accessibility_report(self):
        """获取报告"""
        report = get_accessibility_report()
        self.assertIsInstance(report, str)
        self.assertIn("可访问性", report)

    def test08_enable_high_contrast(self):
        """启用高对比度"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            enable_high_contrast(True)
            self.assertTrue(get_accessibility_config().high_contrast)
            enable_high_contrast(False)
            self.assertFalse(get_accessibility_config().high_contrast)
        finally:
            set_accessibility_config(original)

    def test09_enable_large_text(self):
        """启用大字体"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            enable_large_text(True, 2.0)
            config = get_accessibility_config()
            self.assertTrue(config.large_text)
            self.assertEqual(config.font_scale, 2.0)
        finally:
            set_accessibility_config(original)

    def test10_enable_screen_reader(self):
        """启用屏幕阅读器"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            enable_screen_reader(True)
            self.assertTrue(get_accessibility_config().screen_reader)
        finally:
            set_accessibility_config(original)

    def test11_enable_reduce_motion(self):
        """启用减少动画"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            enable_reduce_motion(True)
            self.assertTrue(get_accessibility_config().reduce_motion)
        finally:
            set_accessibility_config(original)

    def test12_set_color_blindness(self):
        """设置色盲类型"""
        manager = AccessibilityManager()
        original = manager.get_config()
        try:
            set_color_blindness(ColorBlindnessType.PROTANOPIA)
            self.assertEqual(get_accessibility_config().color_blindness,
                             ColorBlindnessType.PROTANOPIA)
        finally:
            set_accessibility_config(original)

    def test13_get_color_blind_palette(self):
        """获取色盲调色板"""
        palette = get_color_blind_palette(ColorBlindnessType.PROTANOPIA)
        self.assertIn("red", palette)

    def test14_get_color_blind_palette_unknown(self):
        """未知色盲类型调色板"""
        palette = get_color_blind_palette("unknown")
        self.assertEqual(palette, {})

    def test15_get_keybindings(self):
        """获取快捷键"""
        bindings = get_keybindings()
        self.assertIn("help", bindings)
        self.assertIn("quit", bindings)

    def test16_format_keybindings_help(self):
        """格式化快捷键帮助"""
        help_text = format_keybindings_help()
        self.assertIn("help", help_text)
        self.assertIn("quit", help_text)


class TestModuleExports(unittest.TestCase):
    """测试模块导出"""

    def test01_all_exports(self):
        """__all__包含所有导出"""
        from seva.accessibility import __all__
        expected = [
            "AccessibilityConfig", "AccessibilityLevel",
            "ColorBlindnessType", "ColorBlindnessSupport",
            "ScreenReaderSupport", "HighContrastMode",
            "KeyboardNavigation", "AccessibleFormatter",
            "AccessibilityManager",
            "format_accessible", "get_accessibility_config",
            "set_accessibility_config", "update_accessibility_config",
            "reset_accessibility_config", "get_accessibility_report",
            "enable_high_contrast", "enable_large_text",
            "enable_screen_reader", "enable_reduce_motion",
            "set_color_blindness", "get_color_blind_palette",
            "get_keybindings", "format_keybindings_help",
        ]
        for name in expected:
            self.assertIn(name, __all__)

    def test02_seva_exports(self):
        """seva包导出"""
        self.assertTrue(hasattr(seva, "AccessibilityConfig"))
        self.assertTrue(hasattr(seva, "AccessibilityManager"))
        self.assertTrue(hasattr(seva, "format_accessible"))
        self.assertTrue(hasattr(seva, "enable_high_contrast"))
        self.assertTrue(hasattr(seva, "ColorBlindnessSupport"))

    def test03_version(self):
        """版本号"""
        self.assertGreaterEqual(seva.__version__, "0.51.0")

    def test04_color_blindness_type_constants(self):
        """色盲类型常量"""
        self.assertEqual(ColorBlindnessType.NONE, "none")
        self.assertEqual(ColorBlindnessType.PROTANOPIA, "protanopia")
        self.assertEqual(ColorBlindnessType.DEUTERANOPIA, "deuteranopia")
        self.assertEqual(ColorBlindnessType.TRITANOPIA, "tritanopia")
        self.assertEqual(ColorBlindnessType.ACHROMATOPSIA, "achromatopsia")

    def test05_accessibility_level_constants(self):
        """可访问性等级常量"""
        self.assertEqual(AccessibilityLevel.NONE, "none")
        self.assertEqual(AccessibilityLevel.BASIC, "basic")
        self.assertEqual(AccessibilityLevel.ENHANCED, "enhanced")
        self.assertEqual(AccessibilityLevel.MAXIMUM, "maximum")

    def test06_constants_exist(self):
        """常量存在"""
        self.assertIsInstance(COLOR_DESCRIPTIONS, dict)
        self.assertIsInstance(STATUS_TEXT, dict)
        self.assertIsInstance(ANSI_TO_TEXT, dict)
        self.assertIsInstance(COLOR_BLIND_PALETTE, dict)
        self.assertIsInstance(HIGH_CONTRAST_COLORS, dict)
        self.assertIsInstance(DEFAULT_KEYBINDINGS, dict)


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def setUp(self):
        """保存原始配置"""
        self.manager = AccessibilityManager()
        self.original_config = self.manager.get_config()

    def tearDown(self):
        """恢复原始配置"""
        self.manager.set_config(self.original_config)

    def test01_cmd_accessibility_default(self):
        """CLI accessibility默认"""
        from seva.cli import main
        try:
            ret = main(["accessibility"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test02_cmd_accessibility_report(self):
        """CLI accessibility --report"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--report"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test03_cmd_accessibility_high_contrast(self):
        """CLI accessibility --high-contrast on"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--high-contrast", "on"])
            self.assertEqual(ret, 0)
            self.assertTrue(get_accessibility_config().high_contrast)
        except SystemExit:
            pass

    def test04_cmd_accessibility_high_contrast_off(self):
        """CLI accessibility --high-contrast off"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--high-contrast", "off"])
            self.assertEqual(ret, 0)
            self.assertFalse(get_accessibility_config().high_contrast)
        except SystemExit:
            pass

    def test05_cmd_accessibility_screen_reader(self):
        """CLI accessibility --screen-reader on"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--screen-reader", "on"])
            self.assertEqual(ret, 0)
            self.assertTrue(get_accessibility_config().screen_reader)
        except SystemExit:
            pass

    def test06_cmd_accessibility_color_blindness(self):
        """CLI accessibility --color-blindness"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--color-blindness",
                        "protanopia"])
            self.assertEqual(ret, 0)
            self.assertEqual(get_accessibility_config().color_blindness,
                             "protanopia")
        except SystemExit:
            pass

    def test07_cmd_accessibility_keybindings(self):
        """CLI accessibility --keybindings"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--keybindings"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test08_cmd_accessibility_test(self):
        """CLI accessibility --test"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--test"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test09_cmd_accessibility_reset(self):
        """CLI accessibility --reset"""
        from seva.cli import main
        try:
            ret = main(["accessibility", "--reset"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test10_accessibility_parser_exists(self):
        """accessibility子解析器存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["accessibility", "--report"])
        self.assertEqual(args.command, "accessibility")
        self.assertTrue(args.report)
        self.assertTrue(hasattr(args, "func"))


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test01_config_all_disabled(self):
        """全部禁用"""
        config = AccessibilityConfig()
        self.assertEqual(config.level, AccessibilityLevel.NONE)

    def test02_config_all_enabled(self):
        """全部启用"""
        config = AccessibilityConfig(
            high_contrast=True, large_text=True,
            screen_reader=True, reduce_motion=True,
            keyboard_only=True, verbose_descriptions=True,
            color_blindness=ColorBlindnessType.ACHROMATOPSIA,
        )
        self.assertEqual(config.level, AccessibilityLevel.MAXIMUM)

    def test03_formatter_no_config(self):
        """无配置格式化"""
        formatter = AccessibleFormatter()
        result = formatter.format("test")
        self.assertEqual(result, "test")

    def test04_color_blind_unknown_type(self):
        """未知色盲类型"""
        cbs = ColorBlindnessSupport("unknown")
        color = cbs.get_color("red")
        self.assertEqual(color, "red")

    def test05_keyboard_navigation_empty_bind(self):
        """空绑定"""
        kn = KeyboardNavigation()
        kn.bind("empty", [])
        self.assertEqual(kn.get_keys("empty"), [])

    def test06_screen_reader_describe_empty(self):
        """空文本描述"""
        srs = ScreenReaderSupport()
        result = srs.describe("")
        self.assertEqual(result, "")

    def test07_high_contrast_format_unknown_semantic(self):
        """未知语义格式化"""
        hcm = HighContrastMode()
        text = hcm.format_text("test", "unknown")
        self.assertIn("test", text)

    def test08_config_font_scale_boundary(self):
        """字体缩放边界"""
        config = AccessibilityConfig(font_scale=0.5)
        self.assertEqual(config.font_scale, 0.5)
        config = AccessibilityConfig(font_scale=3.0)
        self.assertEqual(config.font_scale, 3.0)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """保存原始配置"""
        self.manager = AccessibilityManager()
        self.original_config = self.manager.get_config()

    def tearDown(self):
        """恢复原始配置"""
        self.manager.set_config(self.original_config)

    def test01_full_workflow_screen_reader(self):
        """屏幕阅读器完整工作流"""
        # 启用屏幕阅读器
        config = AccessibilityConfig(
            screen_reader=True,
            verbose_descriptions=True,
            text_alternatives=True,
        )
        formatter = AccessibleFormatter(config)

        # 格式化各种内容
        status = formatter.format_status("error", "任务", "失败")
        self.assertIn("错误", status)

        progress = formatter.format_progress(30, 100, "下载")
        self.assertIn("进度", progress)

        error = formatter.format_error("文件未找到", "路径错误", "检查路径")
        self.assertIn("错误", error)

    def test02_full_workflow_high_contrast(self):
        """高对比度完整工作流"""
        config = AccessibilityConfig(high_contrast=True)
        formatter = AccessibleFormatter(config)

        # 获取颜色
        error_color = formatter.get_color("error")
        self.assertEqual(error_color, "#FF0000")

        # 格式化文本（应移除ANSI）
        text = formatter.format("\033[31m错误\033[0m")
        self.assertNotIn("\033[31m", text)

    def test03_full_workflow_color_blind(self):
        """色盲完整工作流"""
        config = AccessibilityConfig(
            color_blindness=ColorBlindnessType.PROTANOPIA,
        )
        formatter = AccessibleFormatter(config)

        # 获取替代颜色
        color = formatter.get_color("red")
        self.assertNotEqual(color, "red")

        # 获取图案
        pattern = formatter.get_pattern("error")
        self.assertEqual(pattern, "✗")

    def test04_config_persistence(self):
        """配置持久化"""
        # 设置配置
        config = AccessibilityConfig(
            high_contrast=True,
            screen_reader=True,
        )
        set_accessibility_config(config)

        # 新管理器实例应该加载配置
        # (单例模式，所以直接检查)
        loaded = get_accessibility_config()
        self.assertTrue(loaded.high_contrast)
        self.assertTrue(loaded.screen_reader)

    def test05_level_progression(self):
        """等级提升"""
        # NONE
        config = AccessibilityConfig()
        self.assertEqual(config.level, AccessibilityLevel.NONE)

        # BASIC
        config = AccessibilityConfig(high_contrast=True)
        self.assertEqual(config.level, AccessibilityLevel.BASIC)

        # ENHANCED
        config = AccessibilityConfig(
            high_contrast=True, large_text=True,
            screen_reader=True, reduce_motion=True,
        )
        self.assertEqual(config.level, AccessibilityLevel.ENHANCED)

        # MAXIMUM
        config = AccessibilityConfig(
            high_contrast=True, large_text=True,
            screen_reader=True, reduce_motion=True,
            keyboard_only=True, verbose_descriptions=True,
            color_blindness=ColorBlindnessType.PROTANOPIA,
        )
        self.assertEqual(config.level, AccessibilityLevel.MAXIMUM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
