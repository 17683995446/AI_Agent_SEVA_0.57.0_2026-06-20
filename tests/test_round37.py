"""
阶段47测试：国际化与多语言支持
验证 TranslationCatalog / I18n 及便捷函数和CLI集成
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import seva
from seva.i18n import (TranslationCatalog, I18n,
                       DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES,
                       t, gettext, set_language, get_language,
                       set_fallback, get_fallback, list_languages,
                       get_i18n_stats, get_i18n_report,
                       register_catalog, get_catalog,
                       list_translation_keys,
                       LOCALE_FILE)


def _reset_i18n():
    """重置I18n单例"""
    I18n.reset_instance()


class TestTranslationCatalog(unittest.TestCase):
    """测试TranslationCatalog"""

    def test01_init_basic(self):
        """基本初始化"""
        cat = TranslationCatalog("zh_CN")
        self.assertEqual(cat.language, "zh_CN")
        self.assertEqual(len(cat), 0)

    def test02_init_with_translations(self):
        """带翻译的初始化"""
        cat = TranslationCatalog("en_US", {"hello": "Hello"})
        self.assertEqual(cat.get("hello"), "Hello")
        self.assertEqual(len(cat), 1)

    def test03_set_and_get(self):
        """set/get"""
        cat = TranslationCatalog("zh_CN")
        cat.set("welcome", "欢迎")
        self.assertEqual(cat.get("welcome"), "欢迎")

    def test04_get_default(self):
        """get with default"""
        cat = TranslationCatalog("zh_CN")
        self.assertEqual(cat.get("nonexistent", "default"), "default")
        self.assertIsNone(cat.get("nonexistent"))

    def test05_has(self):
        """has"""
        cat = TranslationCatalog("zh_CN", {"a": "1"})
        self.assertTrue(cat.has("a"))
        self.assertFalse(cat.has("b"))

    def test06_remove(self):
        """remove"""
        cat = TranslationCatalog("zh_CN", {"a": "1"})
        self.assertTrue(cat.remove("a"))
        self.assertFalse(cat.remove("a"))
        self.assertEqual(len(cat), 0)

    def test07_keys(self):
        """keys"""
        cat = TranslationCatalog("zh_CN", {"a": "1", "b": "2"})
        keys = cat.keys()
        self.assertIn("a", keys)
        self.assertIn("b", keys)

    def test08_update(self):
        """update"""
        cat = TranslationCatalog("zh_CN")
        count = cat.update({"a": "1", "b": "2"})
        self.assertEqual(count, 2)
        self.assertEqual(cat.get("a"), "1")

    def test09_to_dict(self):
        """to_dict"""
        cat = TranslationCatalog("zh_CN", {"a": "1"})
        d = cat.to_dict()
        self.assertEqual(d["language"], "zh_CN")
        self.assertEqual(d["translations"], {"a": "1"})

    def test10_from_dict(self):
        """from_dict"""
        data = {"language": "en_US", "translations": {"x": "y"}}
        cat = TranslationCatalog.from_dict(data)
        self.assertEqual(cat.language, "en_US")
        self.assertEqual(cat.get("x"), "y")

    def test11_contains(self):
        """__contains__"""
        cat = TranslationCatalog("zh_CN", {"a": "1"})
        self.assertIn("a", cat)
        self.assertNotIn("b", cat)

    def test12_len(self):
        """__len__"""
        cat = TranslationCatalog("zh_CN", {"a": "1", "b": "2"})
        self.assertEqual(len(cat), 2)


class TestI18n(unittest.TestCase):
    """测试I18n"""

    def setUp(self):
        _reset_i18n()

    def tearDown(self):
        # 重置回默认语言
        _reset_i18n()
        set_language(DEFAULT_LANGUAGE)

    def test13_singleton(self):
        """单例"""
        i1 = I18n()
        i2 = I18n()
        self.assertIs(i1, i2)

    def test14_default_language(self):
        """默认语言"""
        i18n = I18n()
        self.assertEqual(i18n.get_language(), DEFAULT_LANGUAGE)

    def test15_set_language(self):
        """设置语言"""
        i18n = I18n()
        self.assertTrue(i18n.set_language("en_US"))
        self.assertEqual(i18n.get_language(), "en_US")

    def test16_set_invalid_language(self):
        """设置无效语言"""
        i18n = I18n()
        self.assertFalse(i18n.set_language("invalid"))
        self.assertEqual(i18n.get_language(), DEFAULT_LANGUAGE)

    def test17_set_fallback(self):
        """设置回退语言"""
        i18n = I18n()
        self.assertTrue(i18n.set_fallback("en_US"))
        self.assertEqual(i18n.get_fallback(), "en_US")

    def test18_translate(self):
        """翻译"""
        i18n = I18n()
        i18n.set_language("zh_CN")
        self.assertEqual(i18n.t("welcome"), "欢迎使用SEVA")
        i18n.set_language("en_US")
        self.assertEqual(i18n.t("welcome"), "Welcome to SEVA")

    def test19_translate_with_params(self):
        """带参数翻译"""
        i18n = I18n()
        # 注册带参数的翻译
        cat = i18n.get_catalog("zh_CN")
        if cat:
            cat.set("hello_name", "你好, {name}!")
        result = i18n.t("hello_name", name="世界")
        self.assertEqual(result, "你好, 世界!")

    def test20_translate_missing_key(self):
        """翻译缺失键"""
        i18n = I18n()
        result = i18n.t("nonexistent_key_12345")
        self.assertEqual(result, "nonexistent_key_12345")

    def test21_fallback(self):
        """回退机制"""
        i18n = I18n()
        # 注册一个只有en_US有的键
        cat_en = i18n.get_catalog("en_US")
        if cat_en:
            cat_en.set("english_only", "English only text")
        # 切换到zh_CN
        i18n.set_language("zh_CN")
        i18n.set_fallback("en_US")
        # 应该回退到en_US
        result = i18n.t("english_only")
        self.assertEqual(result, "English only text")

    def test22_list_languages(self):
        """列出语言"""
        i18n = I18n()
        langs = i18n.list_languages()
        self.assertIn("zh_CN", langs)
        self.assertIn("en_US", langs)
        self.assertIn("ja_JP", langs)

    def test23_list_keys(self):
        """列出键"""
        i18n = I18n()
        keys = i18n.list_keys("zh_CN")
        self.assertIsInstance(keys, list)
        self.assertGreater(len(keys), 0)
        self.assertIn("welcome", keys)

    def test24_get_catalog(self):
        """获取目录"""
        i18n = I18n()
        cat = i18n.get_catalog("zh_CN")
        self.assertIsNotNone(cat)
        self.assertEqual(cat.language, "zh_CN")

    def test25_get_catalog_nonexistent(self):
        """获取不存在的目录"""
        i18n = I18n()
        self.assertIsNone(i18n.get_catalog("nonexistent"))

    def test26_register_catalog(self):
        """注册目录"""
        i18n = I18n()
        custom = TranslationCatalog("custom", {"hello": "hi"})
        i18n.register_catalog(custom)
        self.assertIs(i18n.get_catalog("custom"), custom)

    def test27_get_stats(self):
        """获取统计"""
        i18n = I18n()
        stats = i18n.get_stats()
        self.assertIn("current_language", stats)
        self.assertIn("fallback_language", stats)
        self.assertIn("supported_languages", stats)
        self.assertIn("catalog_sizes", stats)

    def test28_get_report(self):
        """获取报告"""
        i18n = I18n()
        report = i18n.get_report()
        self.assertIn("国际化", report)
        self.assertIn("当前语言", report)

    def test29_japanese_translation(self):
        """日语翻译"""
        i18n = I18n()
        i18n.set_language("ja_JP")
        self.assertEqual(i18n.t("welcome"), "SEVAへようこそ")

    def test30_all_languages_have_welcome(self):
        """所有语言都有welcome翻译"""
        i18n = I18n()
        for lang in SUPPORTED_LANGUAGES:
            i18n.set_language(lang)
            text = i18n.t("welcome")
            self.assertNotEqual(text, "welcome",
                                f"语言 {lang} 缺少welcome翻译")


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        _reset_i18n()

    def tearDown(self):
        _reset_i18n()
        set_language(DEFAULT_LANGUAGE)

    def test31_t(self):
        """t函数"""
        set_language("zh_CN")
        self.assertEqual(t("welcome"), "欢迎使用SEVA")
        set_language("en_US")
        self.assertEqual(t("welcome"), "Welcome to SEVA")

    def test32_gettext(self):
        """gettext函数"""
        set_language("zh_CN")
        self.assertEqual(gettext("welcome"), "欢迎使用SEVA")

    def test33_set_language(self):
        """set_language函数"""
        self.assertTrue(set_language("en_US"))
        self.assertEqual(get_language(), "en_US")

    def test34_set_language_invalid(self):
        """set_language无效语言"""
        self.assertFalse(set_language("invalid"))

    def test35_get_language(self):
        """get_language函数"""
        set_language("ja_JP")
        self.assertEqual(get_language(), "ja_JP")

    def test36_set_fallback(self):
        """set_fallback函数"""
        self.assertTrue(set_fallback("en_US"))
        self.assertEqual(get_fallback(), "en_US")

    def test37_list_languages(self):
        """list_languages函数"""
        langs = list_languages()
        self.assertEqual(len(langs), 3)
        self.assertIn("zh_CN", langs)

    def test38_get_i18n_stats(self):
        """get_i18n_stats函数"""
        stats = get_i18n_stats()
        self.assertIn("current_language", stats)

    def test39_get_i18n_report(self):
        """get_i18n_report函数"""
        report = get_i18n_report()
        self.assertIn("国际化", report)

    def test40_register_catalog(self):
        """register_catalog函数"""
        custom = TranslationCatalog("test_lang", {"x": "y"})
        register_catalog(custom)
        self.assertIs(get_catalog("test_lang"), custom)

    def test41_list_translation_keys(self):
        """list_translation_keys函数"""
        keys = list_translation_keys("zh_CN")
        self.assertGreater(len(keys), 0)
        self.assertIn("welcome", keys)


class TestPackageExports(unittest.TestCase):
    """测试包导出"""

    def test42_i18n_exported(self):
        self.assertTrue(hasattr(seva, "I18n"))

    def test43_translation_catalog_exported(self):
        self.assertTrue(hasattr(seva, "TranslationCatalog"))

    def test44_t_exported(self):
        self.assertTrue(hasattr(seva, "t"))

    def test45_set_language_exported(self):
        self.assertTrue(hasattr(seva, "set_language"))

    def test46_gettext_exported(self):
        self.assertTrue(hasattr(seva, "gettext"))

    def test47_version_updated(self):
        self.assertGreaterEqual(seva.__version__, "0.47.0")

    def test48_all_exports(self):
        for name in ["TranslationCatalog", "I18n",
                     "DEFAULT_LANGUAGE", "SUPPORTED_LANGUAGES",
                     "t", "gettext", "set_language", "get_language",
                     "set_fallback", "get_fallback", "list_languages",
                     "get_i18n_stats", "get_i18n_report",
                     "register_catalog", "get_catalog",
                     "list_translation_keys"]:
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
        _reset_i18n()

    def tearDown(self):
        _reset_i18n()
        set_language(DEFAULT_LANGUAGE)

    def test49_cli_i18n_default(self):
        """seva i18n 默认"""
        code = self._run_cli("i18n")
        self.assertEqual(code, 0)

    def test50_cli_i18n_list(self):
        """seva i18n --list"""
        code = self._run_cli("i18n", "--list")
        self.assertEqual(code, 0)

    def test51_cli_i18n_set(self):
        """seva i18n --set"""
        code = self._run_cli("i18n", "--set", "en_US")
        self.assertEqual(code, 0)
        self.assertEqual(get_language(), "en_US")

    def test52_cli_i18n_set_invalid(self):
        """seva i18n --set 无效"""
        code = self._run_cli("i18n", "--set", "invalid")
        self.assertEqual(code, 1)

    def test53_cli_i18n_translate(self):
        """seva i18n --translate"""
        code = self._run_cli("i18n", "--translate", "welcome")
        self.assertEqual(code, 0)

    def test54_cli_i18n_report(self):
        """seva i18n --report"""
        code = self._run_cli("i18n", "--report")
        self.assertEqual(code, 0)

    def test55_cli_i18n_stats(self):
        """seva i18n --stats"""
        code = self._run_cli("i18n", "--stats")
        self.assertEqual(code, 0)

    def test56_cli_i18n_keys(self):
        """seva i18n --keys"""
        code = self._run_cli("i18n", "--keys")
        self.assertEqual(code, 0)


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def setUp(self):
        _reset_i18n()

    def tearDown(self):
        _reset_i18n()
        set_language(DEFAULT_LANGUAGE)

    def test57_language_switching(self):
        """语言切换"""
        # 中文
        set_language("zh_CN")
        self.assertEqual(t("welcome"), "欢迎使用SEVA")
        self.assertEqual(t("cmd.run"), "运行Agent")

        # 英文
        set_language("en_US")
        self.assertEqual(t("welcome"), "Welcome to SEVA")
        self.assertEqual(t("cmd.run"), "Run the Agent")

        # 日文
        set_language("ja_JP")
        self.assertEqual(t("welcome"), "SEVAへようこそ")
        self.assertEqual(t("cmd.run"), "Agentを実行")

    def test58_fallback_chain(self):
        """回退链"""
        i18n = I18n()
        # 添加只有en_US有的键
        cat_en = i18n.get_catalog("en_US")
        cat_en.set("fallback_test", "fallback value")

        # 当前语言zh_CN没有，回退到en_US
        i18n.set_language("zh_CN")
        i18n.set_fallback("en_US")
        self.assertEqual(i18n.t("fallback_test"), "fallback value")

    def test59_parametric_translation(self):
        """参数化翻译"""
        i18n = I18n()
        cat = i18n.get_catalog("zh_CN")
        cat.set("greeting", "你好, {name}! 你有 {count} 条消息。")
        result = i18n.t("greeting", name="张三", count=5)
        self.assertEqual(result, "你好, 张三! 你有 5 条消息。")

    def test60_custom_catalog(self):
        """自定义目录"""
        i18n = I18n()
        custom = TranslationCatalog("custom", {"hello": "custom hello"})
        i18n.register_catalog(custom)
        # 自定义目录不能直接用作当前语言（不在SUPPORTED中）
        # 但可以获取
        self.assertEqual(i18n.get_catalog("custom").get("hello"),
                         "custom hello")

    def test61_all_keys_in_all_languages(self):
        """所有语言都有相同键集合"""
        i18n = I18n()
        zh_keys = set(i18n.list_keys("zh_CN"))
        en_keys = set(i18n.list_keys("en_US"))
        ja_keys = set(i18n.list_keys("ja_JP"))
        # 三个语言的键应该相同
        self.assertEqual(zh_keys, en_keys)
        self.assertEqual(zh_keys, ja_keys)

    def test62_translation_completeness(self):
        """翻译完整性"""
        i18n = I18n()
        for lang in SUPPORTED_LANGUAGES:
            keys = i18n.list_keys(lang)
            self.assertGreater(len(keys), 30,
                               f"语言 {lang} 翻译条目过少")
            for key in keys:
                text = i18n.get_catalog(lang).get(key)
                self.assertIsNotNone(text,
                                     f"语言 {lang} 键 {key} 无翻译")
                self.assertNotEqual(text, "",
                                    f"语言 {lang} 键 {key} 翻译为空")


if __name__ == "__main__":
    unittest.main(verbosity=2)
