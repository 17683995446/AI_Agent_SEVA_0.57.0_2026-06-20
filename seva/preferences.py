"""
用户偏好与个性化设置（阶段45）
对应第一性原理：个性化 = 用户可定制系统行为，符合个人习惯

提供能力:
    - PreferenceSchema: 偏好项定义（键/类型/默认值/分类/描述/可选值）
    - Preferences: 偏好管理器（get/set/reset/导入导出/持久化）
    - Profile: 用户配置文件（命名偏好集合）
    - ProfileManager: 多配置文件管理（创建/删除/切换/列表）
    - 便捷函数: get_preference / set_preference / list_preferences 等

使用示例:
    from seva import Preferences, get_preference, set_preference

    # 获取/设置偏好
    set_preference("display.theme", "dark")
    theme = get_preference("display.theme")

    # 使用配置文件
    from seva import ProfileManager
    pm = ProfileManager()
    pm.create("work")
    pm.switch("work")
"""
from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
PREFS_FILE = SEVA_HOME / "preferences.json"
PROFILES_DIR = SEVA_HOME / "profiles"
CURRENT_PROFILE_FILE = SEVA_HOME / "current_profile"


# ============================================================================
# PreferenceSchema - 偏好项定义
# ============================================================================

class PreferenceItem:
    """单个偏好项定义"""

    def __init__(self, key: str, default: Any, category: str,
                 description: str = "",
                 choices: Optional[List[Any]] = None,
                 value_type: Optional[type] = None,
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None) -> None:
        self.key = key
        self.default = default
        self.category = category
        self.description = description
        self.choices = choices
        self.value_type = value_type or type(default)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> bool:
        """验证值是否合法"""
        # 类型检查
        if not isinstance(value, self.value_type):
            # bool是int的子类，需要特殊处理
            if self.value_type == int and isinstance(value, bool):
                return False
            if self.value_type == bool and isinstance(value, int):
                pass  # 允许int转bool
            elif self.value_type != bool and isinstance(value, bool):
                return False
            else:
                return False
        # 可选值检查
        if self.choices is not None and value not in self.choices:
            return False
        # 范围检查
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "default": self.default,
            "category": self.category,
            "description": self.description,
            "choices": self.choices,
            "value_type": self.value_type.__name__,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


class PreferenceSchema:
    """
    偏好项注册表，定义所有可用偏好

    分类:
        - display: 显示相关（主题、颜色、输出格式等）
        - behavior: 行为相关（自动保存、确认、默认值等）
        - performance: 性能相关（工作线程、缓存、日志级别等）
        - notifications: 通知相关（完成通知、错误通知等）
        - advanced: 高级选项（遥测、自动更新、调试等）
    """

    ITEMS: Dict[str, PreferenceItem] = {}
    CATEGORIES: List[str] = ["display", "behavior", "performance",
                             "notifications", "advanced"]

    @classmethod
    def register(cls, item: PreferenceItem) -> None:
        """注册偏好项"""
        cls.ITEMS[item.key] = item

    @classmethod
    def get(cls, key: str) -> Optional[PreferenceItem]:
        """获取偏好项定义"""
        return cls.ITEMS.get(key)

    @classmethod
    def get_all(cls) -> Dict[str, PreferenceItem]:
        """获取所有偏好项"""
        return cls.ITEMS

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, PreferenceItem]:
        """按分类获取偏好项"""
        return {k: v for k, v in cls.ITEMS.items()
                if v.category == category}

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """获取所有默认值"""
        return {k: v.default for k, v in cls.ITEMS.items()}

    @classmethod
    def get_keys(cls) -> List[str]:
        """获取所有偏好键"""
        return list(cls.ITEMS.keys())

    @classmethod
    def validate_key(cls, key: str) -> bool:
        """验证键是否存在"""
        return key in cls.ITEMS

    @classmethod
    def validate_value(cls, key: str, value: Any) -> bool:
        """验证值是否合法"""
        item = cls.get(key)
        if item is None:
            return False
        return item.validate(value)


# 注册所有偏好项
def _register_default_preferences() -> None:
    """注册默认偏好项"""
    # display 分类
    PreferenceSchema.register(PreferenceItem(
        key="display.theme",
        default="auto",
        category="display",
        description="界面主题",
        choices=["light", "dark", "auto"],
    ))
    PreferenceSchema.register(PreferenceItem(
        key="display.color_scheme",
        default="default",
        category="display",
        description="颜色方案",
        choices=["default", "blue", "green", "purple", "monochrome"],
    ))
    PreferenceSchema.register(PreferenceItem(
        key="display.output_format",
        default="text",
        category="display",
        description="输出格式",
        choices=["text", "json", "table"],
    ))
    PreferenceSchema.register(PreferenceItem(
        key="display.verbose",
        default=False,
        category="display",
        description="详细输出模式",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="display.show_timestamps",
        default=True,
        category="display",
        description="显示时间戳",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="display.locale",
        default="zh_CN",
        category="display",
        description="语言区域",
        choices=["zh_CN", "en_US", "ja_JP"],
    ))

    # behavior 分类
    PreferenceSchema.register(PreferenceItem(
        key="behavior.auto_save",
        default=True,
        category="behavior",
        description="自动保存",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="behavior.confirm_destructive",
        default=True,
        category="behavior",
        description="破坏性操作需确认",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="behavior.default_preset",
        default="easy",
        category="behavior",
        description="默认预设模板",
        choices=["easy", "standard", "research", "production"],
    ))
    PreferenceSchema.register(PreferenceItem(
        key="behavior.default_env_type",
        default="mixed",
        category="behavior",
        description="默认环境类型",
        choices=["blank", "mixed", "drift", "adversarial"],
    ))
    PreferenceSchema.register(PreferenceItem(
        key="behavior.default_steps",
        default=100,
        category="behavior",
        description="默认运行步数",
        value_type=int,
        min_value=1,
        max_value=100000,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="behavior.resume_on_start",
        default=True,
        category="behavior",
        description="启动时恢复上次会话",
        value_type=bool,
    ))

    # performance 分类
    PreferenceSchema.register(PreferenceItem(
        key="performance.max_workers",
        default=4,
        category="performance",
        description="最大工作线程数",
        value_type=int,
        min_value=1,
        max_value=64,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="performance.cache_enabled",
        default=True,
        category="performance",
        description="启用缓存",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="performance.cache_size",
        default=128,
        category="performance",
        description="缓存大小",
        value_type=int,
        min_value=1,
        max_value=10000,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="performance.lazy_loading",
        default=True,
        category="performance",
        description="延迟加载",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="performance.log_level",
        default="INFO",
        category="performance",
        description="日志级别",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    ))

    # notifications 分类
    PreferenceSchema.register(PreferenceItem(
        key="notifications.on_complete",
        default=True,
        category="notifications",
        description="任务完成时通知",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="notifications.on_error",
        default=True,
        category="notifications",
        description="出错时通知",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="notifications.sound_enabled",
        default=False,
        category="notifications",
        description="启用声音",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="notifications.desktop_notifications",
        default=False,
        category="notifications",
        description="桌面通知",
        value_type=bool,
    ))

    # advanced 分类
    PreferenceSchema.register(PreferenceItem(
        key="advanced.telemetry",
        default=False,
        category="advanced",
        description="发送遥测数据",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="advanced.auto_update",
        default=True,
        category="advanced",
        description="自动更新",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="advanced.debug_mode",
        default=False,
        category="advanced",
        description="调试模式",
        value_type=bool,
    ))
    PreferenceSchema.register(PreferenceItem(
        key="advanced.experimental_features",
        default=False,
        category="advanced",
        description="启用实验性功能",
        value_type=bool,
    ))


# 模块加载时注册
_register_default_preferences()


# ============================================================================
# Preferences - 偏好管理器
# ============================================================================

class Preferences:
    """
    用户偏好管理器（线程安全，单例）

    用法:
        prefs = Preferences()
        prefs.set("display.theme", "dark")
        theme = prefs.get("display.theme")
    """

    _instance: Optional["Preferences"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "Preferences":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._values: Dict[str, Any] = {}
        self._data_lock = threading.Lock()
        self._load()

    def get(self, key: str, default: Any = None) -> Any:
        """获取偏好值"""
        with self._data_lock:
            if key in self._values:
                return self._values[key]
            item = PreferenceSchema.get(key)
            if item is not None:
                return item.default
            return default

    def set(self, key: str, value: Any) -> bool:
        """设置偏好值，返回是否成功"""
        if not PreferenceSchema.validate_key(key):
            return False
        if not PreferenceSchema.validate_value(key, value):
            return False
        with self._data_lock:
            self._values[key] = value
        return True

    def reset(self, key: str) -> bool:
        """重置单个偏好为默认值"""
        if not PreferenceSchema.validate_key(key):
            return False
        with self._data_lock:
            self._values.pop(key, None)
        return True

    def reset_all(self) -> None:
        """重置所有偏好为默认值"""
        with self._data_lock:
            self._values.clear()

    def reset_category(self, category: str) -> None:
        """重置指定分类的所有偏好"""
        items = PreferenceSchema.get_by_category(category)
        with self._data_lock:
            for key in items:
                self._values.pop(key, None)

    def has(self, key: str) -> bool:
        """是否设置了指定偏好（非默认值）"""
        with self._data_lock:
            return key in self._values

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（包含默认值）"""
        result = PreferenceSchema.get_defaults()
        with self._data_lock:
            result.update(self._values)
        return result

    def to_dict_custom_only(self) -> Dict[str, Any]:
        """导出为字典（仅自定义值）"""
        with self._data_lock:
            return dict(self._values)

    def from_dict(self, data: Dict[str, Any]) -> int:
        """从字典导入，返回成功设置的项数"""
        count = 0
        for key, value in data.items():
            if self.set(key, value):
                count += 1
        return count

    def get_category(self, category: str) -> Dict[str, Any]:
        """获取指定分类的所有偏好"""
        items = PreferenceSchema.get_by_category(category)
        return {key: self.get(key) for key in items}

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return list(PreferenceSchema.CATEGORIES)

    def save(self, path: Optional[str] = None) -> str:
        """保存到文件"""
        path = path or str(PREFS_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "1.0",
            "timestamp": time.time(),
            "preferences": self.to_dict_custom_only(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def _load(self, path: Optional[str] = None) -> int:
        """从文件加载"""
        path = path or str(PREFS_FILE)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            prefs = data.get("preferences", {})
            return self.from_dict(prefs)
        except (json.JSONDecodeError, IOError):
            return 0

    def load(self, path: Optional[str] = None) -> int:
        """显式从文件加载"""
        return self._load(path)

    def export(self, path: str) -> str:
        """导出到指定文件"""
        return self.save(path)

    def import_from(self, path: str) -> int:
        """从指定文件导入"""
        return self._load(path)

    def get_report(self) -> str:
        """生成偏好报告"""
        lines = ["=" * 60, "用户偏好设置", "=" * 60]
        for category in PreferenceSchema.CATEGORIES:
            items = PreferenceSchema.get_by_category(category)
            if not items:
                continue
            lines.append(f"\n[{category}]")
            for key, item in items.items():
                value = self.get(key)
                default_marker = "" if self.has(key) else " (默认)"
                desc = item.description
                lines.append(f"  {key} = {value}{default_marker}")
                if desc:
                    lines.append(f"    # {desc}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            cls._instance = None


# ============================================================================
# Profile - 用户配置文件
# ============================================================================

class Profile:
    """
    用户配置文件（命名的偏好集合）

    用法:
        profile = Profile("work")
        profile.set("display.theme", "dark")
        profile.save()
    """

    def __init__(self, name: str, preferences: Optional[Dict[str, Any]] = None,
                 created_at: Optional[float] = None,
                 last_used: Optional[float] = None) -> None:
        self.name = name
        self._preferences: Dict[str, Any] = preferences or {}
        self.created_at = created_at or time.time()
        self.last_used = last_used or time.time()

    @property
    def preferences(self) -> Dict[str, Any]:
        return dict(self._preferences)

    def get(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        if not PreferenceSchema.validate_key(key):
            return False
        if not PreferenceSchema.validate_value(key, value):
            return False
        self._preferences[key] = value
        return True

    def remove(self, key: str) -> bool:
        if key in self._preferences:
            del self._preferences[key]
            return True
        return False

    def clear(self) -> None:
        self._preferences.clear()

    def update_last_used(self) -> None:
        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "preferences": dict(self._preferences),
            "created_at": self.created_at,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        return cls(
            name=data["name"],
            preferences=data.get("preferences", {}),
            created_at=data.get("created_at"),
            last_used=data.get("last_used"),
        )

    def get_path(self) -> Path:
        """获取配置文件路径"""
        return PROFILES_DIR / f"{self.name}.json"

    def save(self) -> str:
        """保存配置文件"""
        path = self.get_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    @classmethod
    def load(cls, name: str) -> Optional["Profile"]:
        """加载配置文件"""
        path = PROFILES_DIR / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def delete(self) -> bool:
        """删除配置文件"""
        path = self.get_path()
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self) -> bool:
        """配置文件是否存在"""
        return self.get_path().exists()

    def apply_to(self, prefs: Optional[Preferences] = None) -> int:
        """将配置文件应用到Preferences"""
        prefs = prefs or Preferences()
        count = 0
        for key, value in self._preferences.items():
            if prefs.set(key, value):
                count += 1
        return count

    @classmethod
    def capture_current(cls, name: str,
                        prefs: Optional[Preferences] = None) -> "Profile":
        """从当前Preferences创建配置文件"""
        prefs = prefs or Preferences()
        return cls(name=name, preferences=prefs.to_dict_custom_only())


# ============================================================================
# ProfileManager - 配置文件管理器
# ============================================================================

class ProfileManager:
    """
    多配置文件管理器（单例）

    用法:
        pm = ProfileManager()
        pm.create("work")
        pm.switch("work")
        profiles = pm.list_profiles()
    """

    _instance: Optional["ProfileManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProfileManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._current_profile: Optional[str] = None
        self._load_current()

    def create(self, name: str,
               from_current: bool = False) -> Profile:
        """创建新配置文件"""
        if from_current:
            profile = Profile.capture_current(name)
        else:
            profile = Profile(name=name)
        profile.save()
        return profile

    def delete(self, name: str) -> bool:
        """删除配置文件"""
        if name == self._current_profile:
            return False  # 不能删除当前配置
        profile = Profile.load(name)
        if profile is None:
            return False
        return profile.delete()

    def get(self, name: str) -> Optional[Profile]:
        """获取配置文件"""
        return Profile.load(name)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """列出所有配置文件"""
        profiles = []
        if not PROFILES_DIR.exists():
            return profiles
        for path in PROFILES_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles.append({
                    "name": data["name"],
                    "created_at": data.get("created_at", 0),
                    "last_used": data.get("last_used", 0),
                    "preference_count": len(data.get("preferences", {})),
                    "is_current": data["name"] == self._current_profile,
                })
            except (json.JSONDecodeError, IOError, KeyError):
                continue
        # 按最后使用时间降序
        profiles.sort(key=lambda x: x["last_used"], reverse=True)
        return profiles

    def switch(self, name: str) -> bool:
        """切换到指定配置文件"""
        profile = Profile.load(name)
        if profile is None:
            return False
        profile.update_last_used()
        profile.save()
        profile.apply_to()
        self._current_profile = name
        self._save_current()
        return True

    def get_current(self) -> Optional[str]:
        """获取当前配置文件名"""
        return self._current_profile

    def rename(self, old_name: str, new_name: str) -> bool:
        """重命名配置文件"""
        profile = Profile.load(old_name)
        if profile is None:
            return False
        if Profile.load(new_name) is not None:
            return False  # 新名称已存在
        profile.name = new_name
        profile.save()
        # 删除旧文件
        old_path = PROFILES_DIR / f"{old_name}.json"
        if old_path.exists():
            old_path.unlink()
        if self._current_profile == old_name:
            self._current_profile = new_name
            self._save_current()
        return True

    def duplicate(self, src_name: str, dst_name: str) -> bool:
        """复制配置文件"""
        profile = Profile.load(src_name)
        if profile is None:
            return False
        if Profile.load(dst_name) is not None:
            return False
        new_profile = Profile(
            name=dst_name,
            preferences=profile.preferences,
        )
        new_profile.save()
        return True

    def _load_current(self) -> None:
        """加载当前配置文件名"""
        if CURRENT_PROFILE_FILE.exists():
            try:
                self._current_profile = CURRENT_PROFILE_FILE.read_text(
                    encoding="utf-8").strip()
            except IOError:
                self._current_profile = None

    def _save_current(self) -> None:
        """保存当前配置文件名"""
        CURRENT_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_PROFILE_FILE.write_text(
            self._current_profile or "", encoding="utf-8")

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            cls._instance = None


# ============================================================================
# 便捷函数
# ============================================================================

def get_preference(key: str, default: Any = None) -> Any:
    """获取偏好值"""
    return Preferences().get(key, default)


def set_preference(key: str, value: Any) -> bool:
    """设置偏好值"""
    return Preferences().set(key, value)


def reset_preference(key: str) -> bool:
    """重置偏好为默认值"""
    return Preferences().reset(key)


def reset_all_preferences() -> None:
    """重置所有偏好"""
    Preferences().reset_all()


def list_preferences(category: Optional[str] = None) -> Dict[str, Any]:
    """列出偏好（可按分类）"""
    prefs = Preferences()
    if category:
        return prefs.get_category(category)
    return prefs.to_dict()


def list_preference_schema(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """列出偏好项定义"""
    if category:
        items = PreferenceSchema.get_by_category(category)
    else:
        items = PreferenceSchema.get_all()
    return {k: v.to_dict() for k, v in items.items()}


def save_preferences(path: Optional[str] = None) -> str:
    """保存偏好到文件"""
    return Preferences().save(path)


def load_preferences(path: Optional[str] = None) -> int:
    """从文件加载偏好"""
    return Preferences().load(path)


def export_preferences(path: str) -> str:
    """导出偏好到指定文件"""
    return Preferences().export(path)


def import_preferences(path: str) -> int:
    """从指定文件导入偏好"""
    return Preferences().import_from(path)


def get_preferences_report() -> str:
    """获取偏好报告"""
    return Preferences().get_report()


def create_profile(name: str, from_current: bool = False) -> Profile:
    """创建配置文件"""
    return ProfileManager().create(name, from_current)


def delete_profile(name: str) -> bool:
    """删除配置文件"""
    return ProfileManager().delete(name)


def switch_profile(name: str) -> bool:
    """切换配置文件"""
    return ProfileManager().switch(name)


def list_profiles() -> List[Dict[str, Any]]:
    """列出所有配置文件"""
    return ProfileManager().list_profiles()


def get_current_profile() -> Optional[str]:
    """获取当前配置文件名"""
    return ProfileManager().get_current()


def get_profile(name: str) -> Optional[Profile]:
    """获取配置文件"""
    return ProfileManager().get(name)


__all__ = [
    "PreferenceItem",
    "PreferenceSchema",
    "Preferences",
    "Profile",
    "ProfileManager",
    "get_preference",
    "set_preference",
    "reset_preference",
    "reset_all_preferences",
    "list_preferences",
    "list_preference_schema",
    "save_preferences",
    "load_preferences",
    "export_preferences",
    "import_preferences",
    "get_preferences_report",
    "create_profile",
    "delete_profile",
    "switch_profile",
    "list_profiles",
    "get_current_profile",
    "get_profile",
]
