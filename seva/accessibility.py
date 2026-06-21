"""
可访问性与无障碍支持（阶段51）
对应第一性原理：包容 = 系统应对所有用户可用，包括残障人士

提供能力:
    - AccessibilityConfig: 可访问性配置
    - ColorBlindnessSupport: 色盲支持（颜色替代/图案）
    - ScreenReaderSupport: 屏幕阅读器支持（ARIA/描述）
    - HighContrastMode: 高对比度模式
    - KeyboardNavigation: 键盘导航（快捷键映射）
    - AccessibleFormatter: 无障碍格式化器
    - 便捷函数: format_accessible / get_accessibility_report

使用示例:
    from seva import AccessibilityConfig, format_accessible

    config = AccessibilityConfig(high_contrast=True, large_text=True)
    text = format_accessible("错误: 文件未找到", config)
"""
from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


# ============================================================================
# 常量
# ============================================================================

class ColorBlindnessType:
    """色盲类型"""
    NONE = "none"                # 正常
    PROTANOPIA = "protanopia"    # 红色盲
    DEUTERANOPIA = "deuteranopia"  # 绿色盲
    TRITANOPIA = "tritanopia"    # 蓝色盲
    ACHROMATOPSIA = "achromatopsia"  # 全色盲


class AccessibilityLevel:
    """可访问性等级"""
    NONE = "none"        # 无
    BASIC = "basic"      # 基础
    ENHANCED = "enhanced"  # 增强
    MAXIMUM = "maximum"  # 最大


# 配置文件路径
CONFIG_FILE = Path.home() / ".seva" / "accessibility.json"


# 标准颜色到文字描述的映射
COLOR_DESCRIPTIONS = {
    "red": "红色",
    "green": "绿色",
    "blue": "蓝色",
    "yellow": "黄色",
    "orange": "橙色",
    "purple": "紫色",
    "pink": "粉色",
    "brown": "棕色",
    "black": "黑色",
    "white": "白色",
    "gray": "灰色",
    "grey": "灰色",
}

# 状态图标到文字的映射
STATUS_TEXT = {
    "running": "运行中",
    "paused": "已暂停",
    "completed": "已完成",
    "error": "错误",
    "warning": "警告",
    "idle": "空闲",
    "pending": "等待中",
    "success": "成功",
    "fail": "失败",
}

# ANSI颜色代码到文字的映射
ANSI_TO_TEXT = {
    "\033[31m": "[红色]",
    "\033[32m": "[绿色]",
    "\033[33m": "[黄色]",
    "\033[34m": "[蓝色]",
    "\033[35m": "[紫色]",
    "\033[36m": "[青色]",
    "\033[37m": "[白色]",
    "\033[90m": "[灰色]",
    "\033[91m": "[亮红色]",
    "\033[92m": "[亮绿色]",
    "\033[93m": "[亮黄色]",
    "\033[94m": "[亮蓝色]",
    "\033[95m": "[亮紫色]",
    "\033[96m": "[亮青色]",
}

# 色盲友好的颜色替代方案
COLOR_BLIND_PALETTE = {
    ColorBlindnessType.PROTANOPIA: {
        "red": "#D55E00",      # 橙红色
        "green": "#0072B2",    # 蓝色
        "yellow": "#F0E442",   # 黄色
        "blue": "#0072B2",     # 蓝色
        "error": "#D55E00",
        "success": "#0072B2",
        "warning": "#F0E442",
    },
    ColorBlindnessType.DEUTERANOPIA: {
        "red": "#D55E00",
        "green": "#0072B2",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "error": "#D55E00",
        "success": "#0072B2",
        "warning": "#F0E442",
    },
    ColorBlindnessType.TRITANOPIA: {
        "red": "#D55E00",
        "green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "error": "#D55E00",
        "success": "#009E73",
        "warning": "#F0E442",
    },
    ColorBlindnessType.ACHROMATOPSIA: {
        "red": "#FFFFFF",
        "green": "#BBBBBB",
        "yellow": "#888888",
        "blue": "#444444",
        "error": "#FFFFFF",
        "success": "#BBBBBB",
        "warning": "#888888",
    },
}

# 高对比度颜色方案
HIGH_CONTRAST_COLORS = {
    "background": "#000000",
    "foreground": "#FFFFFF",
    "error": "#FF0000",
    "success": "#00FF00",
    "warning": "#FFFF00",
    "info": "#00FFFF",
    "emphasis": "#FFFFFF",
    "border": "#FFFFFF",
}

# 默认键盘快捷键
DEFAULT_KEYBINDINGS = {
    "help": ["?", "h"],
    "quit": ["q", "Ctrl+C"],
    "cancel": ["Escape", "Ctrl+C"],
    "confirm": ["Enter"],
    "next": ["n", "Tab", "ArrowRight"],
    "previous": ["p", "Shift+Tab", "ArrowLeft"],
    "up": ["ArrowUp", "k"],
    "down": ["ArrowDown", "j"],
    "search": ["/", "Ctrl+F"],
    "menu": ["m", "F1"],
    "refresh": ["r", "F5"],
}


# ============================================================================
# AccessibilityConfig - 可访问性配置
# ============================================================================

class AccessibilityConfig:
    """
    可访问性配置

    属性:
        high_contrast: 高对比度模式
        large_text: 大字体模式
        screen_reader: 屏幕阅读器支持
        reduce_motion: 减少动画
        color_blindness: 色盲类型
        keyboard_only: 仅键盘导航
        verbose_descriptions: 详细描述
        text_alternatives: 文字替代
        font_scale: 字体缩放（1.0=正常）
    """

    def __init__(self,
                 high_contrast: bool = False,
                 large_text: bool = False,
                 screen_reader: bool = False,
                 reduce_motion: bool = False,
                 color_blindness: str = ColorBlindnessType.NONE,
                 keyboard_only: bool = False,
                 verbose_descriptions: bool = False,
                 text_alternatives: bool = True,
                 font_scale: float = 1.0) -> None:
        self.high_contrast = high_contrast
        self.large_text = large_text
        self.screen_reader = screen_reader
        self.reduce_motion = reduce_motion
        self.color_blindness = color_blindness
        self.keyboard_only = keyboard_only
        self.verbose_descriptions = verbose_descriptions
        self.text_alternatives = text_alternatives
        self.font_scale = max(0.5, min(3.0, font_scale))

    @property
    def level(self) -> str:
        """可访问性等级"""
        features = sum([
            self.high_contrast, self.large_text, self.screen_reader,
            self.reduce_motion, self.keyboard_only,
            self.verbose_descriptions,
            self.color_blindness != ColorBlindnessType.NONE,
        ])
        if features == 0:
            return AccessibilityLevel.NONE
        elif features <= 2:
            return AccessibilityLevel.BASIC
        elif features <= 4:
            return AccessibilityLevel.ENHANCED
        else:
            return AccessibilityLevel.MAXIMUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_contrast": self.high_contrast,
            "large_text": self.large_text,
            "screen_reader": self.screen_reader,
            "reduce_motion": self.reduce_motion,
            "color_blindness": self.color_blindness,
            "keyboard_only": self.keyboard_only,
            "verbose_descriptions": self.verbose_descriptions,
            "text_alternatives": self.text_alternatives,
            "font_scale": self.font_scale,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessibilityConfig":
        return cls(
            high_contrast=data.get("high_contrast", False),
            large_text=data.get("large_text", False),
            screen_reader=data.get("screen_reader", False),
            reduce_motion=data.get("reduce_motion", False),
            color_blindness=data.get(
                "color_blindness", ColorBlindnessType.NONE),
            keyboard_only=data.get("keyboard_only", False),
            verbose_descriptions=data.get(
                "verbose_descriptions", False),
            text_alternatives=data.get("text_alternatives", True),
            font_scale=data.get("font_scale", 1.0),
        )

    def __repr__(self) -> str:
        return f"AccessibilityConfig(level={self.level!r})"


# ============================================================================
# ColorBlindnessSupport - 色盲支持
# ============================================================================

class ColorBlindnessSupport:
    """
    色盲支持

    提供颜色替代方案和图案描述
    """

    def __init__(self, blindness_type: str = ColorBlindnessType.NONE) -> None:
        self.blindness_type = blindness_type

    def get_color(self, semantic: str) -> str:
        """获取语义颜色的替代色"""
        if self.blindness_type == ColorBlindnessType.NONE:
            return semantic
        palette = COLOR_BLIND_PALETTE.get(self.blindness_type, {})
        return palette.get(semantic, semantic)

    def get_pattern(self, semantic: str) -> str:
        """获取图案描述（用于无颜色环境）"""
        patterns = {
            "error": "✗",
            "success": "✓",
            "warning": "⚠",
            "info": "ℹ",
            "red": "■",
            "green": "●",
            "blue": "▲",
            "yellow": "★",
        }
        return patterns.get(semantic, "?")

    def describe_color(self, color: str) -> str:
        """描述颜色"""
        return COLOR_DESCRIPTIONS.get(color.lower(), color)

    def get_report(self) -> str:
        """获取色盲支持报告"""
        lines = [
            "色盲支持报告",
            "=" * 40,
            f"  色盲类型: {self.blindness_type}",
        ]
        if self.blindness_type != ColorBlindnessType.NONE:
            palette = COLOR_BLIND_PALETTE.get(self.blindness_type, {})
            lines.append("  颜色替代方案:")
            for semantic, color in palette.items():
                lines.append(f"    {semantic}: {color}")
        return "\n".join(lines)


# ============================================================================
# ScreenReaderSupport - 屏幕阅读器支持
# ============================================================================

class ScreenReaderSupport:
    """
    屏幕阅读器支持

    生成ARIA标签和文字描述
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def aria_label(self, element_type: str, text: str,
                   state: str = "") -> str:
        """生成ARIA标签"""
        parts = [text]
        if state:
            parts.append(f"状态: {state}")
        if self.verbose:
            parts.append(f"类型: {element_type}")
        return ", ".join(parts)

    def describe(self, text: str, context: str = "") -> str:
        """生成描述"""
        # 替换图标为文字
        for icon, desc in STATUS_TEXT.items():
            text = text.replace(icon, desc)
        # 替换颜色代码
        for ansi, desc in ANSI_TO_TEXT.items():
            text = text.replace(ansi, desc)
        # 移除其他ANSI代码
        import re
        text = re.sub(r'\033\[[0-9;]*m', '', text)
        if context and self.verbose:
            text = f"[{context}] {text}"
        return text

    def announce(self, message: str, priority: str = "polite") -> str:
        """生成公告"""
        prefix = ""
        if priority == "assertive":
            prefix = "[紧急] "
        return f"{prefix}{message}"

    def describe_progress(self, current: int, total: int,
                          description: str = "") -> str:
        """描述进度"""
        pct = (current / total * 100) if total > 0 else 0
        parts = [f"进度: {current} / {total}, {pct:.0f}%"]
        if description:
            parts.insert(0, description)
        return ", ".join(parts)

    def describe_status(self, status: str, name: str = "",
                        message: str = "") -> str:
        """描述状态"""
        status_text = STATUS_TEXT.get(status, status)
        parts = []
        if name:
            parts.append(name)
        parts.append(f"状态: {status_text}")
        if message:
            parts.append(f"消息: {message}")
        return ", ".join(parts)


# ============================================================================
# HighContrastMode - 高对比度模式
# ============================================================================

class HighContrastMode:
    """
    高对比度模式

    提供高对比度颜色方案
    """

    def __init__(self) -> None:
        self.colors = dict(HIGH_CONTRAST_COLORS)

    def get_color(self, semantic: str) -> str:
        """获取颜色"""
        return self.colors.get(semantic, self.colors["foreground"])

    def format_text(self, text: str, semantic: str = "foreground") -> str:
        """格式化文本（添加颜色）"""
        color = self.get_color(semantic)
        return f"[{color}] {text}"

    def get_contrast_ratio(self, fg: str, bg: str) -> float:
        """计算对比度比率（简化版）"""
        # 简化计算：黑白对比度为21:1
        if (fg.lower() == "#ffffff" and bg.lower() == "#000000") or \
           (fg.lower() == "#000000" and bg.lower() == "#ffffff"):
            return 21.0
        return 7.0  # 默认假设符合WCAG AAA

    def get_report(self) -> str:
        """获取报告"""
        lines = [
            "高对比度模式报告",
            "=" * 40,
            "  颜色方案:",
        ]
        for semantic, color in self.colors.items():
            lines.append(f"    {semantic}: {color}")
        ratio = self.get_contrast_ratio(
            self.colors["foreground"], self.colors["background"])
        lines.append(f"  对比度比率: {ratio:.1f}:1")
        lines.append(f"  WCAG等级: {'AAA' if ratio >= 7 else 'AA'}")
        return "\n".join(lines)


# ============================================================================
# KeyboardNavigation - 键盘导航
# ============================================================================

class KeyboardNavigation:
    """
    键盘导航

    管理快捷键映射
    """

    def __init__(self) -> None:
        self.bindings: Dict[str, List[str]] = dict(DEFAULT_KEYBINDINGS)
        self._lock = threading.Lock()

    def get_keys(self, action: str) -> List[str]:
        """获取动作的快捷键"""
        return self.bindings.get(action, [])

    def get_action(self, key: str) -> Optional[str]:
        """获取快捷键对应的动作"""
        for action, keys in self.bindings.items():
            if key in keys:
                return action
        return None

    def bind(self, action: str, keys: List[str]) -> None:
        """绑定快捷键"""
        with self._lock:
            self.bindings[action] = keys

    def unbind(self, action: str) -> bool:
        """解绑快捷键"""
        with self._lock:
            if action in self.bindings:
                del self.bindings[action]
                return True
            return False

    def list_actions(self) -> Dict[str, List[str]]:
        """列出所有动作"""
        return dict(self.bindings)

    def format_help(self) -> str:
        """格式化帮助"""
        lines = [
            "键盘快捷键",
            "=" * 40,
        ]
        for action, keys in sorted(self.bindings.items()):
            keys_str = " / ".join(keys)
            lines.append(f"  {action:<15} {keys_str}")
        return "\n".join(lines)


# ============================================================================
# AccessibleFormatter - 无障碍格式化器
# ============================================================================

class AccessibleFormatter:
    """
    无障碍格式化器

    根据可访问性配置调整输出格式
    """

    def __init__(self, config: Optional[AccessibilityConfig] = None) -> None:
        self.config = config or AccessibilityConfig()
        self._color_blind = ColorBlindnessSupport(
            self.config.color_blindness)
        self._screen_reader = ScreenReaderSupport(
            self.config.verbose_descriptions)
        self._high_contrast = HighContrastMode()
        self._keyboard = KeyboardNavigation()

    def format(self, text: str, context: str = "") -> str:
        """格式化文本"""
        result = text
        # 屏幕阅读器模式：替换图标和颜色
        if self.config.screen_reader or self.config.text_alternatives:
            result = self._screen_reader.describe(result, context)
        # 高对比度模式：移除原有颜色，添加高对比度标记
        if self.config.high_contrast:
            # 移除ANSI颜色
            import re
            result = re.sub(r'\033\[[0-9;]*m', '', result)
        # 大字体模式：添加标记
        if self.config.large_text:
            result = result  # 实际应用中会调整字体大小
        return result

    def format_status(self, status: str, name: str = "",
                      message: str = "") -> str:
        """格式化状态"""
        if self.config.screen_reader:
            return self._screen_reader.describe_status(
                status, name, message)
        # 普通格式
        from .progress import STATUS_ICONS, STATUS_ICONS_ASCII
        icons = STATUS_ICONS_ASCII if self.config.reduce_motion else \
            STATUS_ICONS
        icon = icons.get(status, "?")
        parts = [f"{icon} {name}"] if name else [icon]
        if message:
            parts.append(f": {message}")
        return "".join(parts)

    def format_progress(self, current: int, total: int,
                        description: str = "") -> str:
        """格式化进度"""
        if self.config.screen_reader:
            return self._screen_reader.describe_progress(
                current, total, description)
        # 普通格式
        pct = (current / total * 100) if total > 0 else 0
        if self.config.reduce_motion:
            return f"{description}: {current}/{total} ({pct:.0f}%)"
        # 进度条
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        return f"{description}: [{bar}] {pct:.0f}%"

    def format_error(self, error: str, cause: str = "",
                     suggestion: str = "") -> str:
        """格式化错误"""
        if self.config.screen_reader:
            parts = [f"错误: {error}"]
            if cause:
                parts.append(f"原因: {cause}")
            if suggestion:
                parts.append(f"建议: {suggestion}")
            return ", ".join(parts)
        # 普通格式
        result = f"✗ 错误: {error}"
        if cause:
            result += f"\n  原因: {cause}"
        if suggestion:
            result += f"\n  建议: {suggestion}"
        return result

    def format_success(self, message: str) -> str:
        """格式化成功"""
        if self.config.screen_reader:
            return f"成功: {message}"
        return f"✓ {message}"

    def format_warning(self, message: str) -> str:
        """格式化警告"""
        if self.config.screen_reader:
            return f"警告: {message}"
        return f"⚠ {message}"

    def format_info(self, message: str) -> str:
        """格式化信息"""
        if self.config.screen_reader:
            return f"信息: {message}"
        return f"ℹ {message}"

    def get_color(self, semantic: str) -> str:
        """获取颜色"""
        if self.config.high_contrast:
            return self._high_contrast.get_color(semantic)
        return self._color_blind.get_color(semantic)

    def get_pattern(self, semantic: str) -> str:
        """获取图案"""
        return self._color_blind.get_pattern(semantic)


# ============================================================================
# AccessibilityManager - 可访问性管理器（单例）
# ============================================================================

class AccessibilityManager:
    """
    可访问性管理器（单例）

    管理全局可访问性配置
    """

    _instance: Optional["AccessibilityManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AccessibilityManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._config = AccessibilityConfig()
        self._formatter = AccessibleFormatter(self._config)
        self._config_lock = threading.Lock()
        self._load()

    def get_config(self) -> AccessibilityConfig:
        """获取配置"""
        with self._config_lock:
            return self._config

    def set_config(self, config: AccessibilityConfig) -> None:
        """设置配置"""
        with self._config_lock:
            self._config = config
            self._formatter = AccessibleFormatter(config)
        self._save()

    def update_config(self, **kwargs) -> None:
        """更新配置"""
        with self._config_lock:
            data = self._config.to_dict()
            data.update(kwargs)
            self._config = AccessibilityConfig.from_dict(data)
            self._formatter = AccessibleFormatter(self._config)
        self._save()

    def reset(self) -> None:
        """重置配置"""
        self.set_config(AccessibilityConfig())

    def get_formatter(self) -> AccessibleFormatter:
        """获取格式化器"""
        return self._formatter

    def format(self, text: str, context: str = "") -> str:
        """格式化文本"""
        return self._formatter.format(text, context)

    def get_report(self) -> str:
        """获取报告"""
        config = self._config.to_dict()
        lines = [
            "SEVA可访问性报告",
            "=" * 50,
            f"  等级: {config['level']}",
            f"  高对比度: {'是' if config['high_contrast'] else '否'}",
            f"  大字体: {'是' if config['large_text'] else '否'}",
            f"  屏幕阅读器: {'是' if config['screen_reader'] else '否'}",
            f"  减少动画: {'是' if config['reduce_motion'] else '否'}",
            f"  色盲类型: {config['color_blindness']}",
            f"  仅键盘: {'是' if config['keyboard_only'] else '否'}",
            f"  详细描述: {'是' if config['verbose_descriptions'] else '否'}",
            f"  文字替代: {'是' if config['text_alternatives'] else '否'}",
            f"  字体缩放: {config['font_scale']:.1f}x",
        ]
        return "\n".join(lines)

    def _load(self) -> None:
        """从文件加载"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                self._config = AccessibilityConfig.from_dict(data)
                self._formatter = AccessibleFormatter(self._config)
        except Exception:
            pass

    def _save(self) -> None:
        """保存到文件"""
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(self._config.to_dict(), f, indent=2,
                          ensure_ascii=False)
        except Exception:
            pass


# ============================================================================
# 便捷函数
# ============================================================================

def format_accessible(text: str,
                      config: Optional[AccessibilityConfig] = None,
                      context: str = "") -> str:
    """无障碍格式化文本"""
    if config:
        formatter = AccessibleFormatter(config)
        return formatter.format(text, context)
    return AccessibilityManager().format(text, context)


def get_accessibility_config() -> AccessibilityConfig:
    """获取可访问性配置"""
    return AccessibilityManager().get_config()


def set_accessibility_config(config: AccessibilityConfig) -> None:
    """设置可访问性配置"""
    AccessibilityManager().set_config(config)


def update_accessibility_config(**kwargs) -> None:
    """更新可访问性配置"""
    AccessibilityManager().update_config(**kwargs)


def reset_accessibility_config() -> None:
    """重置可访问性配置"""
    AccessibilityManager().reset()


def get_accessibility_report() -> str:
    """获取可访问性报告"""
    return AccessibilityManager().get_report()


def enable_high_contrast(enabled: bool = True) -> None:
    """启用/禁用高对比度"""
    update_accessibility_config(high_contrast=enabled)


def enable_large_text(enabled: bool = True,
                      scale: float = 1.5) -> None:
    """启用/禁用大字体"""
    update_accessibility_config(large_text=enabled, font_scale=scale)


def enable_screen_reader(enabled: bool = True) -> None:
    """启用/禁用屏幕阅读器"""
    update_accessibility_config(screen_reader=enabled)


def enable_reduce_motion(enabled: bool = True) -> None:
    """启用/禁用减少动画"""
    update_accessibility_config(reduce_motion=enabled)


def set_color_blindness(blindness_type: str) -> None:
    """设置色盲类型"""
    update_accessibility_config(color_blindness=blindness_type)


def get_color_blind_palette(blindness_type: str) -> Dict[str, str]:
    """获取色盲调色板"""
    return COLOR_BLIND_PALETTE.get(blindness_type, {})


def get_keybindings() -> Dict[str, List[str]]:
    """获取键盘快捷键"""
    return KeyboardNavigation().list_actions()


def format_keybindings_help() -> str:
    """格式化快捷键帮助"""
    return KeyboardNavigation().format_help()


__all__ = [
    "AccessibilityConfig",
    "AccessibilityLevel",
    "ColorBlindnessType",
    "ColorBlindnessSupport",
    "ScreenReaderSupport",
    "HighContrastMode",
    "KeyboardNavigation",
    "AccessibleFormatter",
    "AccessibilityManager",
    "format_accessible",
    "get_accessibility_config",
    "set_accessibility_config",
    "update_accessibility_config",
    "reset_accessibility_config",
    "get_accessibility_report",
    "enable_high_contrast",
    "enable_large_text",
    "enable_screen_reader",
    "enable_reduce_motion",
    "set_color_blindness",
    "get_color_blind_palette",
    "get_keybindings",
    "format_keybindings_help",
    "COLOR_DESCRIPTIONS",
    "STATUS_TEXT",
    "ANSI_TO_TEXT",
    "COLOR_BLIND_PALETTE",
    "HIGH_CONTRAST_COLORS",
    "DEFAULT_KEYBINDINGS",
    "CONFIG_FILE",
]
