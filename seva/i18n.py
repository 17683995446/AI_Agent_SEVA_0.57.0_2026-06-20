"""
国际化与多语言支持（阶段47）
对应第一性原理：包容 = 不同语言用户都能无障碍使用

提供能力:
    - I18n: 国际化管理器（翻译/语言切换/回退）
    - TranslationCatalog: 翻译目录（键值对管理）
    - Locale: 区域设置管理
    - 便捷函数: t / gettext / set_language / get_language / list_languages

使用示例:
    from seva import set_language, t

    set_language("en")           # 切换到英文
    print(t("welcome"))          # 输出英文欢迎语
    set_language("zh_CN")        # 切换回中文
    print(t("welcome"))          # 输出中文欢迎语
"""
from __future__ import annotations

import os
import json
import threading
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
LOCALE_FILE = SEVA_HOME / "locale.json"

DEFAULT_LANGUAGE = "zh_CN"
SUPPORTED_LANGUAGES = ["zh_CN", "en_US", "ja_JP"]


# ============================================================================
# TranslationCatalog - 翻译目录
# ============================================================================

class TranslationCatalog:
    """
    翻译目录，存储一个语言的键值对

    用法:
        catalog = TranslationCatalog("zh_CN")
        catalog.set("welcome", "欢迎使用SEVA")
        text = catalog.get("welcome")
    """

    def __init__(self, language: str,
                 translations: Optional[Dict[str, str]] = None) -> None:
        self.language = language
        self._translations: Dict[str, str] = translations or {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """获取翻译"""
        return self._translations.get(key, default)

    def set(self, key: str, value: str) -> None:
        """设置翻译"""
        self._translations[key] = value

    def has(self, key: str) -> bool:
        """是否包含键"""
        return key in self._translations

    def remove(self, key: str) -> bool:
        """删除键"""
        if key in self._translations:
            del self._translations[key]
            return True
        return False

    def keys(self) -> List[str]:
        """所有键"""
        return list(self._translations.keys())

    def update(self, translations: Dict[str, str]) -> int:
        """批量更新，返回更新数量"""
        count = 0
        for k, v in translations.items():
            self._translations[k] = v
            count += 1
        return count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "translations": dict(self._translations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationCatalog":
        return cls(
            language=data["language"],
            translations=data.get("translations", {}),
        )

    def __len__(self) -> int:
        return len(self._translations)

    def __contains__(self, key: str) -> bool:
        return key in self._translations


# ============================================================================
# I18n - 国际化管理器
# ============================================================================

class I18n:
    """
    国际化管理器（单例，线程安全）

    用法:
        i18n = I18n()
        i18n.set_language("en_US")
        text = i18n.t("welcome")
    """

    _instance: Optional["I18n"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "I18n":
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
        self._catalogs: Dict[str, TranslationCatalog] = {}
        self._current_language: str = DEFAULT_LANGUAGE
        self._fallback_language: str = DEFAULT_LANGUAGE
        self._data_lock = threading.Lock()
        self._register_default_catalogs()
        self._load_locale()

    def register_catalog(self, catalog: TranslationCatalog) -> None:
        """注册翻译目录"""
        with self._data_lock:
            self._catalogs[catalog.language] = catalog

    def get_catalog(self, language: str) -> Optional[TranslationCatalog]:
        """获取翻译目录"""
        with self._data_lock:
            return self._catalogs.get(language)

    def set_language(self, language: str) -> bool:
        """设置当前语言"""
        if language not in SUPPORTED_LANGUAGES:
            return False
        with self._data_lock:
            self._current_language = language
        self._save_locale()
        return True

    def get_language(self) -> str:
        """获取当前语言"""
        return self._current_language

    def set_fallback(self, language: str) -> bool:
        """设置回退语言"""
        if language not in SUPPORTED_LANGUAGES:
            return False
        with self._data_lock:
            self._fallback_language = language
        return True

    def get_fallback(self) -> str:
        """获取回退语言"""
        return self._fallback_language

    def t(self, key: str, **kwargs) -> str:
        """
        翻译文本

        支持参数替换: t("hello_name", name="World")
        模板格式: "Hello, {name}!"
        """
        text = self._lookup(key)
        if text is None:
            return key  # 找不到翻译返回键本身
        # 参数替换
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, IndexError):
                pass
        return text

    def _lookup(self, key: str) -> Optional[str]:
        """查找翻译（当前语言 -> 回退语言 -> 默认）"""
        with self._data_lock:
            # 当前语言
            catalog = self._catalogs.get(self._current_language)
            if catalog and catalog.has(key):
                return catalog.get(key)
            # 回退语言
            if self._fallback_language != self._current_language:
                catalog = self._catalogs.get(self._fallback_language)
                if catalog and catalog.has(key):
                    return catalog.get(key)
            # 默认语言
            if DEFAULT_LANGUAGE not in (self._current_language,
                                        self._fallback_language):
                catalog = self._catalogs.get(DEFAULT_LANGUAGE)
                if catalog and catalog.has(key):
                    return catalog.get(key)
        return None

    def list_languages(self) -> List[str]:
        """列出支持的语言"""
        return list(SUPPORTED_LANGUAGES)

    def list_keys(self, language: Optional[str] = None) -> List[str]:
        """列出所有翻译键"""
        lang = language or self._current_language
        catalog = self.get_catalog(lang)
        if catalog:
            return catalog.keys()
        return []

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._data_lock:
            stats = {}
            for lang, catalog in self._catalogs.items():
                stats[lang] = len(catalog)
            return {
                "current_language": self._current_language,
                "fallback_language": self._fallback_language,
                "supported_languages": list(SUPPORTED_LANGUAGES),
                "catalog_sizes": stats,
            }

    def get_report(self) -> str:
        """生成报告"""
        stats = self.get_stats()
        lines = [
            "=" * 60,
            "国际化(i18n)报告",
            "=" * 60,
            f"当前语言: {stats['current_language']}",
            f"回退语言: {stats['fallback_language']}",
            f"支持语言: {', '.join(stats['supported_languages'])}",
            "",
            "翻译目录:",
        ]
        for lang, size in stats["catalog_sizes"].items():
            marker = " (当前)" if lang == stats["current_language"] else ""
            lines.append(f"  {lang}: {size}条翻译{marker}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _load_locale(self) -> None:
        """从文件加载语言设置"""
        if LOCALE_FILE.exists():
            try:
                data = json.loads(LOCALE_FILE.read_text(encoding="utf-8"))
                lang = data.get("language", DEFAULT_LANGUAGE)
                if lang in SUPPORTED_LANGUAGES:
                    self._current_language = lang
                fb = data.get("fallback", DEFAULT_LANGUAGE)
                if fb in SUPPORTED_LANGUAGES:
                    self._fallback_language = fb
            except (json.JSONDecodeError, IOError):
                pass

    def _save_locale(self) -> None:
        """保存语言设置到文件"""
        LOCALE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "language": self._current_language,
            "fallback": self._fallback_language,
        }
        LOCALE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8")

    def _register_default_catalogs(self) -> None:
        """注册默认翻译目录"""
        for lang, translations in _get_default_translations().items():
            catalog = TranslationCatalog(lang, translations)
            self.register_catalog(catalog)

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._lock:
            cls._instance = None


# ============================================================================
# 默认翻译数据
# ============================================================================

def _get_default_translations() -> Dict[str, Dict[str, str]]:
    """获取默认翻译数据"""
    return {
        "zh_CN": {
            # 通用
            "welcome": "欢迎使用SEVA",
            "goodbye": "再见",
            "yes": "是",
            "no": "否",
            "ok": "确定",
            "cancel": "取消",
            "error": "错误",
            "warning": "警告",
            "info": "信息",
            "success": "成功",
            "loading": "加载中...",
            "done": "完成",
            "failed": "失败",
            "retry": "重试",
            "continue": "继续",
            "exit": "退出",
            "help": "帮助",
            "search": "搜索",
            "save": "保存",
            "load": "加载",
            "delete": "删除",
            "create": "创建",
            "edit": "编辑",
            "view": "查看",
            "close": "关闭",
            "next": "下一步",
            "previous": "上一步",
            "back": "返回",
            # 命令
            "cmd.run": "运行Agent",
            "cmd.config": "配置管理",
            "cmd.status": "查看状态",
            "cmd.version": "版本信息",
            "cmd.wizard": "首次运行向导",
            "cmd.tutorial": "交互式教程",
            "cmd.shell": "交互式Shell",
            "cmd.diagnose": "系统诊断",
            "cmd.perf": "性能监控",
            "cmd.prefs": "用户偏好",
            "cmd.help": "帮助系统",
            # 状态
            "status.running": "运行中",
            "status.idle": "空闲",
            "status.error": "错误状态",
            "status.completed": "已完成",
            "status.paused": "已暂停",
            # 消息
            "msg.agent_created": "Agent已创建",
            "msg.agent_running": "Agent运行中...",
            "msg.agent_completed": "Agent运行完成",
            "msg.config_saved": "配置已保存",
            "msg.config_loaded": "配置已加载",
            "msg.session_saved": "会话已保存",
            "msg.session_loaded": "会话已加载",
            "msg.no_data": "暂无数据",
            "msg.not_found": "未找到",
            "msg.invalid_input": "输入无效",
            "msg.operation_success": "操作成功",
            "msg.operation_failed": "操作失败",
            # 提示
            "tip.first_run": "首次运行？试试 seva wizard",
            "tip.need_help": "需要帮助？试试 seva help",
            "tip.learn": "想学习？试试 seva tutorial",
        },
        "en_US": {
            # Common
            "welcome": "Welcome to SEVA",
            "goodbye": "Goodbye",
            "yes": "Yes",
            "no": "No",
            "ok": "OK",
            "cancel": "Cancel",
            "error": "Error",
            "warning": "Warning",
            "info": "Info",
            "success": "Success",
            "loading": "Loading...",
            "done": "Done",
            "failed": "Failed",
            "retry": "Retry",
            "continue": "Continue",
            "exit": "Exit",
            "help": "Help",
            "search": "Search",
            "save": "Save",
            "load": "Load",
            "delete": "Delete",
            "create": "Create",
            "edit": "Edit",
            "view": "View",
            "close": "Close",
            "next": "Next",
            "previous": "Previous",
            "back": "Back",
            # Commands
            "cmd.run": "Run the Agent",
            "cmd.config": "Configuration management",
            "cmd.status": "Show status",
            "cmd.version": "Version info",
            "cmd.wizard": "First-run wizard",
            "cmd.tutorial": "Interactive tutorial",
            "cmd.shell": "Interactive shell",
            "cmd.diagnose": "System diagnostics",
            "cmd.perf": "Performance monitoring",
            "cmd.prefs": "User preferences",
            "cmd.help": "Help system",
            # Status
            "status.running": "Running",
            "status.idle": "Idle",
            "status.error": "Error",
            "status.completed": "Completed",
            "status.paused": "Paused",
            # Messages
            "msg.agent_created": "Agent created",
            "msg.agent_running": "Agent running...",
            "msg.agent_completed": "Agent run completed",
            "msg.config_saved": "Configuration saved",
            "msg.config_loaded": "Configuration loaded",
            "msg.session_saved": "Session saved",
            "msg.session_loaded": "Session loaded",
            "msg.no_data": "No data",
            "msg.not_found": "Not found",
            "msg.invalid_input": "Invalid input",
            "msg.operation_success": "Operation succeeded",
            "msg.operation_failed": "Operation failed",
            # Tips
            "tip.first_run": "First time? Try seva wizard",
            "tip.need_help": "Need help? Try seva help",
            "tip.learn": "Want to learn? Try seva tutorial",
        },
        "ja_JP": {
            # 共通
            "welcome": "SEVAへようこそ",
            "goodbye": "さようなら",
            "yes": "はい",
            "no": "いいえ",
            "ok": "OK",
            "cancel": "キャンセル",
            "error": "エラー",
            "warning": "警告",
            "info": "情報",
            "success": "成功",
            "loading": "読み込み中...",
            "done": "完了",
            "failed": "失敗",
            "retry": "再試行",
            "continue": "続行",
            "exit": "終了",
            "help": "ヘルプ",
            "search": "検索",
            "save": "保存",
            "load": "読み込み",
            "delete": "削除",
            "create": "作成",
            "edit": "編集",
            "view": "表示",
            "close": "閉じる",
            "next": "次へ",
            "previous": "前へ",
            "back": "戻る",
            # コマンド
            "cmd.run": "Agentを実行",
            "cmd.config": "設定管理",
            "cmd.status": "ステータス表示",
            "cmd.version": "バージョン情報",
            "cmd.wizard": "初回ウィザード",
            "cmd.tutorial": "対話型チュートリアル",
            "cmd.shell": "対話型シェル",
            "cmd.diagnose": "システム診断",
            "cmd.perf": "パフォーマンス監視",
            "cmd.prefs": "ユーザー設定",
            "cmd.help": "ヘルプシステム",
            # ステータス
            "status.running": "実行中",
            "status.idle": "アイドル",
            "status.error": "エラー",
            "status.completed": "完了",
            "status.paused": "一時停止",
            # メッセージ
            "msg.agent_created": "Agentが作成されました",
            "msg.agent_running": "Agent実行中...",
            "msg.agent_completed": "Agent実行完了",
            "msg.config_saved": "設定が保存されました",
            "msg.config_loaded": "設定が読み込まれました",
            "msg.session_saved": "セッションが保存されました",
            "msg.session_loaded": "セッションが読み込まれました",
            "msg.no_data": "データなし",
            "msg.not_found": "見つかりません",
            "msg.invalid_input": "無効な入力",
            "msg.operation_success": "操作成功",
            "msg.operation_failed": "操作失敗",
            # ヒント
            "tip.first_run": "初めてですか？seva wizardを試してください",
            "tip.need_help": "ヘルプが必要ですか？seva helpを試してください",
            "tip.learn": "学びたいですか？seva tutorialを試してください",
        },
    }


# ============================================================================
# 便捷函数
# ============================================================================

def t(key: str, **kwargs) -> str:
    """翻译文本"""
    return I18n().t(key, **kwargs)


def gettext(key: str, **kwargs) -> str:
    """翻译文本（gettext别名）"""
    return I18n().t(key, **kwargs)


def set_language(language: str) -> bool:
    """设置当前语言"""
    return I18n().set_language(language)


def get_language() -> str:
    """获取当前语言"""
    return I18n().get_language()


def set_fallback(language: str) -> bool:
    """设置回退语言"""
    return I18n().set_fallback(language)


def get_fallback() -> str:
    """获取回退语言"""
    return I18n().get_fallback()


def list_languages() -> List[str]:
    """列出支持的语言"""
    return I18n().list_languages()


def get_i18n_stats() -> Dict[str, Any]:
    """获取i18n统计"""
    return I18n().get_stats()


def get_i18n_report() -> str:
    """获取i18n报告"""
    return I18n().get_report()


def register_catalog(catalog: TranslationCatalog) -> None:
    """注册翻译目录"""
    I18n().register_catalog(catalog)


def get_catalog(language: str) -> Optional[TranslationCatalog]:
    """获取翻译目录"""
    return I18n().get_catalog(language)


def list_translation_keys(language: Optional[str] = None) -> List[str]:
    """列出翻译键"""
    return I18n().list_keys(language)


__all__ = [
    "TranslationCatalog",
    "I18n",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "t",
    "gettext",
    "set_language",
    "get_language",
    "set_fallback",
    "get_fallback",
    "list_languages",
    "get_i18n_stats",
    "get_i18n_report",
    "register_catalog",
    "get_catalog",
    "list_translation_keys",
]
