"""
阶段46测试：帮助系统与文档集成
验证 HelpSystem / CommandHelp / FAQ / Glossary / Example
及便捷函数和CLI集成
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import seva
from seva.help_system import (HelpTopic, HelpSystem, CommandHelp,
                              CommandHelpRegistry, FAQEntry, FAQRegistry,
                              GlossaryTerm, Glossary, CodeExample,
                              ExampleRegistry,
                              show_help, search_help, list_help_topics,
                              list_help_categories, get_help_report,
                              get_command_help, list_commands,
                              search_faq, list_faqs,
                              get_glossary_term, search_glossary,
                              list_glossary,
                              get_example, list_examples, search_examples,
                              help as help_func)


def _reset_all():
    """重置所有单例"""
    HelpSystem.reset_instance()
    CommandHelpRegistry.reset_instance()
    FAQRegistry.reset_instance()
    Glossary.reset_instance()
    ExampleRegistry.reset_instance()


class TestHelpTopic(unittest.TestCase):
    """测试HelpTopic"""

    def test01_init_basic(self):
        """基本初始化"""
        topic = HelpTopic(
            topic_id="test",
            title="测试主题",
            content="测试内容",
        )
        self.assertEqual(topic.topic_id, "test")
        self.assertEqual(topic.title, "测试主题")
        self.assertEqual(topic.content, "测试内容")
        self.assertEqual(topic.category, "general")
        self.assertEqual(topic.keywords, [])

    def test02_init_full(self):
        """完整初始化"""
        topic = HelpTopic(
            topic_id="test",
            title="测试",
            content="内容",
            category="测试",
            keywords=["a", "b"],
            examples=["ex1"],
            related=["other"],
        )
        self.assertEqual(topic.category, "测试")
        self.assertEqual(topic.keywords, ["a", "b"])
        self.assertEqual(topic.examples, ["ex1"])
        self.assertEqual(topic.related, ["other"])

    def test03_matches_title(self):
        """标题匹配"""
        topic = HelpTopic("t", "Python教程", "内容")
        self.assertTrue(topic.matches("Python"))
        self.assertTrue(topic.matches("教程"))
        self.assertFalse(topic.matches("Java"))

    def test04_matches_content(self):
        """内容匹配"""
        topic = HelpTopic("t", "标题", "这是关于配置的内容")
        self.assertTrue(topic.matches("配置"))

    def test05_matches_keywords(self):
        """关键词匹配"""
        topic = HelpTopic("t", "标题", "内容", keywords=["安装", "install"])
        self.assertTrue(topic.matches("安装"))
        self.assertTrue(topic.matches("install"))

    def test06_relevance_score(self):
        """相关度评分"""
        topic = HelpTopic("t", "Python教程", "Python内容",
                          keywords=["python"])
        # 标题匹配分数最高
        score_title = topic.relevance_score("Python")
        # 内容匹配分数较低
        score_content = topic.relevance_score("内容")
        self.assertGreater(score_title, score_content)

    def test07_to_dict(self):
        """to_dict"""
        topic = HelpTopic("t", "标题", "内容", category="cat",
                          keywords=["k"])
        d = topic.to_dict()
        self.assertEqual(d["topic_id"], "t")
        self.assertEqual(d["title"], "标题")
        self.assertEqual(d["category"], "cat")

    def test08_from_dict(self):
        """from_dict"""
        data = {
            "topic_id": "t",
            "title": "标题",
            "content": "内容",
            "category": "cat",
            "keywords": ["k"],
            "examples": ["e"],
            "related": ["r"],
        }
        topic = HelpTopic.from_dict(data)
        self.assertEqual(topic.topic_id, "t")
        self.assertEqual(topic.title, "标题")
        self.assertEqual(topic.keywords, ["k"])

    def test09_format(self):
        """format"""
        topic = HelpTopic("t", "标题", "内容", examples=["ex"])
        text = topic.format()
        self.assertIn("标题", text)
        self.assertIn("内容", text)
        self.assertIn("ex", text)


class TestHelpSystem(unittest.TestCase):
    """测试HelpSystem"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test10_singleton(self):
        """单例"""
        h1 = HelpSystem()
        h2 = HelpSystem()
        self.assertIs(h1, h2)

    def test11_register_and_get(self):
        """注册和获取"""
        hs = HelpSystem()
        topic = HelpTopic("custom", "自定义", "内容")
        hs.register(topic)
        self.assertIs(hs.get("custom"), topic)

    def test12_get_nonexistent(self):
        """获取不存在的主题"""
        hs = HelpSystem()
        self.assertIsNone(hs.get("nonexistent"))

    def test13_show(self):
        """显示主题"""
        hs = HelpSystem()
        text = hs.show("getting_started")
        self.assertIn("快速开始", text)

    def test14_show_nonexistent(self):
        """显示不存在的主题"""
        hs = HelpSystem()
        text = hs.show("nonexistent")
        self.assertIn("未找到", text)

    def test15_search(self):
        """搜索"""
        hs = HelpSystem()
        results = hs.search("配置")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test16_search_no_results(self):
        """搜索无结果"""
        hs = HelpSystem()
        results = hs.search("xyznotexist12345")
        self.assertEqual(len(results), 0)

    def test17_list_topics(self):
        """列出主题"""
        hs = HelpSystem()
        topics = hs.list_topics()
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

    def test18_list_topics_by_category(self):
        """按分类列出主题"""
        hs = HelpSystem()
        topics = hs.list_topics(category="基础")
        for t in topics:
            self.assertEqual(t["category"], "基础")

    def test19_list_categories(self):
        """列出分类"""
        hs = HelpSystem()
        cats = hs.list_categories()
        self.assertIsInstance(cats, list)
        self.assertGreater(len(cats), 0)

    def test20_get_topics_by_category(self):
        """按分类获取主题"""
        hs = HelpSystem()
        topics = hs.get_topics_by_category("基础")
        self.assertGreater(len(topics), 0)
        for t in topics:
            self.assertEqual(t.category, "基础")

    def test21_get_stats(self):
        """获取统计"""
        hs = HelpSystem()
        stats = hs.get_stats()
        self.assertIn("total_topics", stats)
        self.assertIn("total_categories", stats)
        self.assertGreater(stats["total_topics"], 0)

    def test22_get_report(self):
        """获取报告"""
        hs = HelpSystem()
        report = hs.get_report()
        self.assertIn("SEVA帮助系统", report)
        self.assertIn("主题总数", report)

    def test23_has_default_topics(self):
        """有默认主题"""
        hs = HelpSystem()
        for tid in ["getting_started", "installation", "configuration",
                    "cli", "api", "troubleshooting"]:
            self.assertIsNotNone(hs.get(tid),
                                 f"缺少默认主题: {tid}")


class TestCommandHelp(unittest.TestCase):
    """测试CommandHelp"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test24_init(self):
        """初始化"""
        cmd = CommandHelp(
            command="test",
            description="测试命令",
            usage="seva test [options]",
        )
        self.assertEqual(cmd.command, "test")
        self.assertEqual(cmd.description, "测试命令")
        self.assertEqual(cmd.usage, "seva test [options]")

    def test25_format(self):
        """format"""
        cmd = CommandHelp(
            command="test",
            description="测试",
            usage="seva test",
            examples=["seva test --foo"],
            options={"--foo": "foo选项"},
        )
        text = cmd.format()
        self.assertIn("test", text)
        self.assertIn("测试", text)
        self.assertIn("--foo", text)

    def test26_to_dict(self):
        """to_dict"""
        cmd = CommandHelp("c", "d", "u")
        d = cmd.to_dict()
        self.assertEqual(d["command"], "c")
        self.assertEqual(d["description"], "d")


class TestCommandHelpRegistry(unittest.TestCase):
    """测试CommandHelpRegistry"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test27_singleton(self):
        """单例"""
        r1 = CommandHelpRegistry()
        r2 = CommandHelpRegistry()
        self.assertIs(r1, r2)

    def test28_register_and_get(self):
        """注册和获取"""
        reg = CommandHelpRegistry()
        cmd = CommandHelp("custom", "自定义", "seva custom")
        reg.register(cmd)
        self.assertIs(reg.get("custom"), cmd)

    def test29_list_commands(self):
        """列出命令"""
        reg = CommandHelpRegistry()
        commands = reg.list_commands()
        self.assertIsInstance(commands, list)
        self.assertGreater(len(commands), 0)

    def test30_has_default_commands(self):
        """有默认命令"""
        reg = CommandHelpRegistry()
        for cmd in ["run", "wizard", "tutorial", "perf", "prefs", "help"]:
            self.assertIsNotNone(reg.get(cmd),
                                 f"缺少默认命令: {cmd}")


class TestFAQ(unittest.TestCase):
    """测试FAQ"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test31_faq_entry_init(self):
        """FAQEntry初始化"""
        entry = FAQEntry("问题", "答案", category="测试")
        self.assertEqual(entry.question, "问题")
        self.assertEqual(entry.answer, "答案")
        self.assertEqual(entry.category, "测试")

    def test32_faq_entry_matches(self):
        """FAQEntry匹配"""
        entry = FAQEntry("如何安装", "使用pip安装", keywords=["install"])
        self.assertTrue(entry.matches("安装"))
        self.assertTrue(entry.matches("pip"))
        self.assertFalse(entry.matches("xyz"))

    def test33_faq_entry_format(self):
        """FAQEntry格式化"""
        entry = FAQEntry("问题", "答案")
        text = entry.format()
        self.assertIn("问题", text)
        self.assertIn("答案", text)

    def test34_registry_singleton(self):
        """FAQRegistry单例"""
        r1 = FAQRegistry()
        r2 = FAQRegistry()
        self.assertIs(r1, r2)

    def test35_registry_add_and_search(self):
        """添加和搜索"""
        reg = FAQRegistry()
        entry = FAQEntry("自定义问题", "自定义答案", keywords=["custom"])
        reg.add(entry)
        results = reg.search("自定义")
        self.assertGreater(len(results), 0)

    def test36_registry_list_all(self):
        """列出所有"""
        reg = FAQRegistry()
        entries = reg.list_all()
        self.assertGreater(len(entries), 0)

    def test37_registry_list_by_category(self):
        """按分类列出"""
        reg = FAQRegistry()
        entries = reg.list_all(category="安装")
        for e in entries:
            self.assertEqual(e.category, "安装")

    def test38_registry_list_categories(self):
        """列出分类"""
        reg = FAQRegistry()
        cats = reg.list_categories()
        self.assertGreater(len(cats), 0)

    def test39_has_default_faqs(self):
        """有默认FAQ"""
        reg = FAQRegistry()
        entries = reg.list_all()
        questions = [e.question for e in entries]
        self.assertTrue(any("安装" in q for q in questions))
        self.assertTrue(any("配置" in q for q in questions))


class TestGlossary(unittest.TestCase):
    """测试Glossary"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test40_term_init(self):
        """术语初始化"""
        term = GlossaryTerm("Agent", "代理", "核心")
        self.assertEqual(term.term, "Agent")
        self.assertEqual(term.definition, "代理")
        self.assertEqual(term.category, "核心")

    def test41_term_format(self):
        """术语格式化"""
        term = GlossaryTerm("Agent", "代理")
        text = term.format()
        self.assertIn("Agent", text)
        self.assertIn("代理", text)

    def test42_glossary_singleton(self):
        """Glossary单例"""
        g1 = Glossary()
        g2 = Glossary()
        self.assertIs(g1, g2)

    def test43_glossary_add_and_get(self):
        """添加和获取"""
        g = Glossary()
        term = GlossaryTerm("CustomTerm", "自定义术语")
        g.add(term)
        self.assertIsNotNone(g.get("CustomTerm"))
        # 大小写不敏感
        self.assertIsNotNone(g.get("customterm"))

    def test44_glossary_get_nonexistent(self):
        """获取不存在的术语"""
        g = Glossary()
        self.assertIsNone(g.get("nonexistent"))

    def test45_glossary_search(self):
        """搜索术语"""
        g = Glossary()
        results = g.search("Agent")
        self.assertGreater(len(results), 0)

    def test46_glossary_list_all(self):
        """列出所有术语"""
        g = Glossary()
        terms = g.list_all()
        self.assertGreater(len(terms), 0)

    def test47_glossary_list_by_category(self):
        """按分类列出"""
        g = Glossary()
        terms = g.list_all(category="核心")
        for t in terms:
            self.assertEqual(t.category, "核心")

    def test48_has_default_terms(self):
        """有默认术语"""
        g = Glossary()
        for term in ["Agent", "Cycle", "Perception", "Memory"]:
            self.assertIsNotNone(g.get(term),
                                 f"缺少默认术语: {term}")


class TestExample(unittest.TestCase):
    """测试Example"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test49_example_init(self):
        """示例初始化"""
        ex = CodeExample("test", "测试", "描述", "code")
        self.assertEqual(ex.example_id, "test")
        self.assertEqual(ex.title, "测试")
        self.assertEqual(ex.code, "code")

    def test50_example_format(self):
        """示例格式化"""
        ex = CodeExample("test", "测试", "描述", "print('hello')")
        text = ex.format()
        self.assertIn("测试", text)
        self.assertIn("print('hello')", text)

    def test51_registry_singleton(self):
        """ExampleRegistry单例"""
        r1 = ExampleRegistry()
        r2 = ExampleRegistry()
        self.assertIs(r1, r2)

    def test52_registry_register_and_get(self):
        """注册和获取"""
        reg = ExampleRegistry()
        ex = CodeExample("custom", "自定义", "描述", "code")
        reg.register(ex)
        self.assertIs(reg.get("custom"), ex)

    def test53_registry_list_all(self):
        """列出所有"""
        reg = ExampleRegistry()
        exs = reg.list_all()
        self.assertGreater(len(exs), 0)

    def test54_registry_list_by_category(self):
        """按分类列出"""
        reg = ExampleRegistry()
        exs = reg.list_all(category="基础")
        for e in exs:
            self.assertEqual(e.category, "基础")

    def test55_registry_search(self):
        """搜索"""
        reg = ExampleRegistry()
        results = reg.search("快速")
        self.assertGreater(len(results), 0)

    def test56_has_default_examples(self):
        """有默认示例"""
        reg = ExampleRegistry()
        for eid in ["quickstart", "custom_config", "scenarios"]:
            self.assertIsNotNone(reg.get(eid),
                                 f"缺少默认示例: {eid}")


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test57_show_help(self):
        """show_help"""
        text = show_help("getting_started")
        self.assertIn("快速开始", text)

    def test58_search_help(self):
        """search_help"""
        results = search_help("配置")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test59_list_help_topics(self):
        """list_help_topics"""
        topics = list_help_topics()
        self.assertGreater(len(topics), 0)

    def test60_list_help_topics_by_category(self):
        """按分类列出主题"""
        topics = list_help_topics("基础")
        for t in topics:
            self.assertEqual(t["category"], "基础")

    def test61_list_help_categories(self):
        """list_help_categories"""
        cats = list_help_categories()
        self.assertGreater(len(cats), 0)

    def test62_get_help_report(self):
        """get_help_report"""
        report = get_help_report()
        self.assertIn("SEVA帮助系统", report)

    def test63_get_command_help(self):
        """get_command_help"""
        text = get_command_help("run")
        self.assertIsNotNone(text)
        self.assertIn("run", text)

    def test64_get_command_help_nonexistent(self):
        """获取不存在的命令帮助"""
        text = get_command_help("nonexistent")
        self.assertIsNone(text)

    def test65_list_commands(self):
        """list_commands"""
        commands = list_commands()
        self.assertGreater(len(commands), 0)
        self.assertIn("run", commands)

    def test66_search_faq(self):
        """search_faq"""
        results = search_faq("安装")
        self.assertGreater(len(results), 0)

    def test67_list_faqs(self):
        """list_faqs"""
        faqs = list_faqs()
        self.assertGreater(len(faqs), 0)

    def test68_get_glossary_term(self):
        """get_glossary_term"""
        text = get_glossary_term("Agent")
        self.assertIsNotNone(text)
        self.assertIn("Agent", text)

    def test69_get_glossary_term_nonexistent(self):
        """获取不存在的术语"""
        text = get_glossary_term("nonexistent")
        self.assertIsNone(text)

    def test70_search_glossary(self):
        """search_glossary"""
        results = search_glossary("Agent")
        self.assertGreater(len(results), 0)

    def test71_list_glossary(self):
        """list_glossary"""
        terms = list_glossary()
        self.assertGreater(len(terms), 0)

    def test72_get_example(self):
        """get_example"""
        ex = get_example("quickstart")
        self.assertIsNotNone(ex)
        self.assertEqual(ex.example_id, "quickstart")

    def test73_list_examples(self):
        """list_examples"""
        exs = list_examples()
        self.assertGreater(len(exs), 0)

    def test74_search_examples(self):
        """search_examples"""
        results = search_examples("快速")
        self.assertGreater(len(results), 0)

    def test75_help_function_topic(self):
        """help函数 - 主题"""
        text = help_func(topic="getting_started")
        self.assertIn("快速开始", text)

    def test76_help_function_search(self):
        """help函数 - 搜索"""
        text = help_func(search="配置")
        self.assertIn("配置", text)

    def test77_help_function_list(self):
        """help函数 - 列表"""
        text = help_func(list_all=True)
        self.assertIn("帮助主题", text)

    def test78_help_function_categories(self):
        """help函数 - 分类"""
        text = help_func(categories=True)
        self.assertIn("帮助分类", text)

    def test79_help_function_faq(self):
        """help函数 - FAQ"""
        text = help_func(faq=True)
        self.assertIn("常见问题", text)

    def test80_help_function_glossary(self):
        """help函数 - 术语表"""
        text = help_func(glossary=True)
        self.assertIn("术语表", text)

    def test81_help_function_examples(self):
        """help函数 - 示例"""
        text = help_func(examples=True)
        self.assertIn("代码示例", text)

    def test82_help_function_command(self):
        """help函数 - 命令"""
        text = help_func(command="run")
        self.assertIn("run", text)

    def test83_help_function_default(self):
        """help函数 - 默认"""
        text = help_func()
        self.assertIn("SEVA帮助系统", text)


class TestPackageExports(unittest.TestCase):
    """测试包导出"""

    def test84_help_system_exported(self):
        self.assertTrue(hasattr(seva, "HelpSystem"))

    def test85_help_topic_exported(self):
        self.assertTrue(hasattr(seva, "HelpTopic"))

    def test86_show_help_exported(self):
        self.assertTrue(hasattr(seva, "show_help"))

    def test87_search_help_exported(self):
        self.assertTrue(hasattr(seva, "search_help"))

    def test88_help_exported(self):
        self.assertTrue(hasattr(seva, "help"))

    def test89_version_updated(self):
        self.assertEqual(seva.__version__, "0.46.0")

    def test90_all_exports(self):
        for name in ["HelpTopic", "HelpSystem", "CommandHelp",
                     "CommandHelpRegistry", "FAQEntry", "FAQRegistry",
                     "GlossaryTerm", "Glossary", "CodeExample",
                     "ExampleRegistry",
                     "show_help", "search_help", "list_help_topics",
                     "list_help_categories", "get_help_report",
                     "get_command_help", "list_commands",
                     "search_faq", "list_faqs",
                     "get_glossary_term", "search_glossary",
                     "list_glossary",
                     "get_example", "list_examples", "search_examples",
                     "help"]:
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

    def test91_cli_help_default(self):
        """seva help 默认"""
        code = self._run_cli("help")
        self.assertEqual(code, 0)

    def test92_cli_help_topic(self):
        """seva help <topic>"""
        code = self._run_cli("help", "getting_started")
        self.assertEqual(code, 0)

    def test93_cli_help_search(self):
        """seva help --search"""
        code = self._run_cli("help", "--search", "配置")
        self.assertEqual(code, 0)

    def test94_cli_help_list(self):
        """seva help --list"""
        code = self._run_cli("help", "--list")
        self.assertEqual(code, 0)

    def test95_cli_help_categories(self):
        """seva help --categories"""
        code = self._run_cli("help", "--categories")
        self.assertEqual(code, 0)

    def test96_cli_help_faq(self):
        """seva help --faq"""
        code = self._run_cli("help", "--faq")
        self.assertEqual(code, 0)

    def test97_cli_help_glossary(self):
        """seva help --glossary"""
        code = self._run_cli("help", "--glossary")
        self.assertEqual(code, 0)

    def test98_cli_help_examples(self):
        """seva help --examples"""
        code = self._run_cli("help", "--examples")
        self.assertEqual(code, 0)

    def test99_cli_help_command(self):
        """seva help --command"""
        code = self._run_cli("help", "--command", "run")
        self.assertEqual(code, 0)

    def test100_cli_help_example_id(self):
        """seva help --example-id"""
        code = self._run_cli("help", "--example-id", "quickstart")
        self.assertEqual(code, 0)

    def test101_cli_help_term(self):
        """seva help --term"""
        code = self._run_cli("help", "--term", "Agent")
        self.assertEqual(code, 0)


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test102_full_help_workflow(self):
        """完整帮助工作流"""
        # 1. 查看报告
        report = get_help_report()
        self.assertIn("SEVA帮助系统", report)

        # 2. 列出主题
        topics = list_help_topics()
        self.assertGreater(len(topics), 5)

        # 3. 搜索
        results = search_help("安装")
        self.assertGreater(len(results), 0)

        # 4. 显示主题
        text = show_help("installation")
        self.assertIn("安装", text)

        # 5. 查看命令帮助
        cmd_help = get_command_help("run")
        self.assertIn("run", cmd_help)

        # 6. 查看FAQ
        faqs = list_faqs()
        self.assertGreater(len(faqs), 0)

        # 7. 查看术语
        term = get_glossary_term("Agent")
        self.assertIsNotNone(term)

        # 8. 查看示例
        ex = get_example("quickstart")
        self.assertIsNotNone(ex)

    def test103_search_across_all(self):
        """跨所有资源搜索"""
        # 搜索主题
        topics = search_help("配置")
        self.assertGreater(len(topics), 0)

        # 搜索FAQ
        faqs = search_faq("配置")
        self.assertGreater(len(faqs), 0)

        # 搜索术语
        terms = search_glossary("配置")
        # 术语可能没有"配置"，但函数应该正常工作
        self.assertIsInstance(terms, list)

    def test104_help_function_all_modes(self):
        """help函数所有模式"""
        # 主题模式
        self.assertIn("快速开始", help_func(topic="getting_started"))
        # 搜索模式
        self.assertIn("结果", help_func(search="安装"))
        # 列表模式
        self.assertIn("帮助主题", help_func(list_all=True))
        # 分类模式
        self.assertIn("帮助分类", help_func(categories=True))
        # FAQ模式
        self.assertIn("常见问题", help_func(faq=True))
        # 术语表模式
        self.assertIn("术语表", help_func(glossary=True))
        # 示例模式
        self.assertIn("代码示例", help_func(examples=True))
        # 命令模式
        self.assertIn("run", help_func(command="run"))
        # 默认模式
        self.assertIn("SEVA帮助系统", help_func())

    def test105_custom_topic_registration(self):
        """自定义主题注册"""
        hs = HelpSystem()
        custom = HelpTopic(
            "my_topic",
            "我的主题",
            "自定义内容",
            category="自定义",
            keywords=["my", "custom"],
        )
        hs.register(custom)
        # 能获取
        self.assertIs(hs.get("my_topic"), custom)
        # 能搜索到
        results = hs.search("自定义")
        self.assertTrue(any(r["topic_id"] == "my_topic" for r in results))
        # 能显示
        text = hs.show("my_topic")
        self.assertIn("我的主题", text)

    def test106_all_categories_have_topics(self):
        """所有分类都有主题"""
        hs = HelpSystem()
        cats = hs.list_categories()
        for cat in cats:
            topics = hs.get_topics_by_category(cat)
            self.assertGreater(len(topics), 0,
                               f"分类 {cat} 没有主题")


if __name__ == "__main__":
    unittest.main(verbosity=2)
