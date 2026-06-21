"""
阶段67测试：研究Agent

覆盖：
    - 枚举: ResearchStage/SourceType/ReportFormat
    - 数据类: SourceInfo/ResearchFinding/ResearchReport
    - InfoRetriever: 信息检索器
    - InfoAnalyzer: 信息分析器
    - ReportGenerator: 报告生成器
    - ResearchAgent: 研究Agent
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import time
from unittest.mock import patch, MagicMock

from seva.agents import (
    ResearchStage, SourceType, ReportFormat,
    SourceInfo, ResearchFinding, ResearchReport,
    InfoRetriever, InfoAnalyzer, ReportGenerator,
    ResearchAgentConfig, ResearchAgent,
    get_research_agent, reset_research_agent,
)
from seva.agents import AgentRunResult
from seva.llm import LLMError, reset_llm_client
from seva.tools import reset_tools
from seva.memory_llm import reset_memory_manager


# ============================================================================
# 枚举测试
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_research_stage(self):
        self.assertEqual(ResearchStage.PLANNING.value, "planning")
        self.assertEqual(ResearchStage.RETRIEVING.value, "retrieving")
        self.assertEqual(ResearchStage.COMPLETED.value, "completed")

    def test02_source_type(self):
        self.assertEqual(SourceType.WEB.value, "web")
        self.assertEqual(SourceType.MEMORY.value, "memory")
        self.assertEqual(SourceType.LLM.value, "llm")

    def test03_report_format(self):
        self.assertEqual(ReportFormat.MARKDOWN.value, "markdown")
        self.assertEqual(ReportFormat.JSON.value, "json")
        self.assertEqual(ReportFormat.TEXT.value, "text")


# ============================================================================
# 数据类测试
# ============================================================================

class TestDataClasses(unittest.TestCase):
    """测试数据类"""

    def test01_source_info(self):
        s = SourceInfo(title="test", content="content",
                      relevance_score=0.8, credibility=0.7)
        self.assertEqual(s.title, "test")
        self.assertEqual(s.relevance_score, 0.8)

    def test02_source_info_to_dict(self):
        s = SourceInfo(title="test", content="content")
        d = s.to_dict()
        self.assertEqual(d["title"], "test")

    def test03_research_finding(self):
        f = ResearchFinding(content="发现", confidence=0.8)
        self.assertEqual(f.content, "发现")
        self.assertEqual(f.confidence, 0.8)
        self.assertIsNotNone(f.id)

    def test04_research_finding_to_dict(self):
        f = ResearchFinding(content="test", tags=["tag1"])
        d = f.to_dict()
        self.assertEqual(d["content"], "test")
        self.assertEqual(d["tags"], ["tag1"])

    def test05_research_report(self):
        r = ResearchReport(topic="主题", query="查询")
        self.assertEqual(r.topic, "主题")
        self.assertEqual(r.stage, ResearchStage.PLANNING)
        self.assertFalse(r.is_completed)

    def test06_research_report_completed(self):
        r = ResearchReport(topic="test", stage=ResearchStage.COMPLETED)
        self.assertTrue(r.is_completed)

    def test07_research_report_to_dict(self):
        r = ResearchReport(topic="test", query="q")
        d = r.to_dict()
        self.assertEqual(d["topic"], "test")
        self.assertEqual(d["stage"], "planning")


# ============================================================================
# InfoRetriever测试
# ============================================================================

class TestInfoRetriever(unittest.TestCase):
    """测试信息检索器"""

    def setUp(self):
        reset_memory_manager()

    def tearDown(self):
        reset_memory_manager()

    def test01_creation(self):
        r = InfoRetriever()
        self.assertIsNotNone(r)
        self.assertEqual(r.stats["retrievals"], 0)

    def test02_retrieve_with_mock(self):
        """Mock LLM检索"""
        reset_memory_manager()  # 清空记忆
        r = InfoRetriever()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "sources": [
        {"content": "信息1", "relevance_score": 0.8, "credibility": 0.7},
        {"content": "信息2", "relevance_score": 0.6, "credibility": 0.5}
    ]
}
```'''
        mock_client.chat.return_value = response
        r._client = mock_client

        sources = r.retrieve("查询", "主题")
        self.assertGreater(len(sources), 0)
        # 找到LLM生成的源
        llm_sources = [s for s in sources if s.source_type.value == "llm"]
        self.assertGreater(len(llm_sources), 0)
        self.assertEqual(llm_sources[0].content, "信息1")

    def test03_retrieve_llm_error(self):
        """LLM错误"""
        r = InfoRetriever()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        r._client = mock_client

        sources = r.retrieve("查询")
        # 应该返回空列表(记忆可能也为空)
        self.assertIsInstance(sources, list)

    def test04_get_stats(self):
        r = InfoRetriever()
        stats = r.get_stats()
        self.assertIn("retrievals", stats)


# ============================================================================
# InfoAnalyzer测试
# ============================================================================

class TestInfoAnalyzer(unittest.TestCase):
    """测试信息分析器"""

    def test01_creation(self):
        a = InfoAnalyzer()
        self.assertIsNotNone(a)

    def test02_analyze_with_mock(self):
        """Mock LLM分析"""
        a = InfoAnalyzer()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "findings": [
        {"content": "发现1", "confidence": 0.8, "tags": ["tag1"]},
        {"content": "发现2", "confidence": 0.6, "tags": ["tag2"]}
    ],
    "summary": "分析摘要"
}
```'''
        mock_client.chat.return_value = response
        a._client = mock_client

        sources = [SourceInfo(content="信息1"), SourceInfo(content="信息2")]
        findings, summary = a.analyze("查询", sources)

        self.assertEqual(len(findings), 2)
        self.assertEqual(summary, "分析摘要")
        self.assertEqual(findings[0].content, "发现1")

    def test03_analyze_empty_sources(self):
        """空信息源"""
        a = InfoAnalyzer()
        findings, summary = a.analyze("查询", [])
        self.assertEqual(len(findings), 0)
        self.assertIn("无信息源", summary)

    def test04_analyze_llm_error(self):
        """LLM错误"""
        a = InfoAnalyzer()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        a._client = mock_client

        sources = [SourceInfo(content="test")]
        findings, summary = a.analyze("查询", sources)
        self.assertEqual(len(findings), 0)
        self.assertIn("失败", summary)

    def test05_get_stats(self):
        a = InfoAnalyzer()
        stats = a.get_stats()
        self.assertIn("analyses", stats)


# ============================================================================
# ReportGenerator测试
# ============================================================================

class TestReportGenerator(unittest.TestCase):
    """测试报告生成器"""

    def test01_creation(self):
        g = ReportGenerator()
        self.assertIsNotNone(g)

    def test02_generate_with_mock(self):
        """Mock LLM生成"""
        g = ReportGenerator()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "# 研究报告\n\n这是报告内容。"
        response.usage.total_tokens = 100
        mock_client.chat.return_value = response
        g._client = mock_client

        findings = [ResearchFinding(content="发现1", confidence=0.8)]
        content, tokens = g.generate("主题", "查询", findings)

        self.assertIn("研究报告", content)
        self.assertEqual(tokens, 100)

    def test03_generate_empty_findings(self):
        """空发现"""
        g = ReportGenerator()
        content, tokens = g.generate("主题", "查询", [])
        self.assertIn("无足够信息", content)
        self.assertEqual(tokens, 0)

    def test04_generate_llm_error(self):
        """LLM错误"""
        g = ReportGenerator()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        g._client = mock_client

        findings = [ResearchFinding(content="test")]
        content, tokens = g.generate("主题", "查询", findings)
        self.assertIn("失败", content)
        self.assertEqual(tokens, 0)

    def test05_get_stats(self):
        g = ReportGenerator()
        stats = g.get_stats()
        self.assertIn("reports_generated", stats)


# ============================================================================
# ResearchAgent测试
# ============================================================================

class TestResearchAgent(unittest.TestCase):
    """测试研究Agent"""

    def setUp(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_research_agent()

    def tearDown(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_research_agent()

    def test01_creation(self):
        agent = ResearchAgent()
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.config, ResearchAgentConfig)
        self.assertIsNotNone(agent.retriever)
        self.assertIsNotNone(agent.analyzer)
        self.assertIsNotNone(agent.generator)

    def test02_creation_with_config(self):
        config = ResearchAgentConfig(
            name="custom",
            max_sources=3,
            report_format=ReportFormat.JSON,
        )
        agent = ResearchAgent(config=config)
        self.assertEqual(agent.config.name, "custom")
        self.assertEqual(agent.research_config.max_sources, 3)

    def test03_run_empty_query(self):
        """空查询"""
        agent = ResearchAgent()
        agent.start()
        result = agent.run("")
        self.assertFalse(result.success)
        self.assertIn("空", result.error)

    def test04_run_with_mock(self):
        """Mock完整运行"""
        agent = ResearchAgent()

        # Mock所有LLM调用
        mock_client = MagicMock()
        retrieve_response = MagicMock()
        retrieve_response.content = '''{"sources": [{"content": "信息1", "relevance_score": 0.8}]}'''
        analyze_response = MagicMock()
        analyze_response.content = '''{"findings": [{"content": "发现1", "confidence": 0.8}], "summary": "摘要"}'''
        report_response = MagicMock()
        report_response.content = "# 研究报告\n\n内容"
        report_response.usage.total_tokens = 50
        mock_client.chat.side_effect = [retrieve_response, analyze_response, report_response]
        agent.retriever._client = mock_client
        agent.analyzer._client = mock_client
        agent.generator._client = mock_client

        agent.start()
        result = agent.run("查询", topic="主题")

        self.assertTrue(result.success)
        self.assertIn("研究报告", result.output)
        self.assertEqual(result.tokens_used, 50)

    def test05_research_method(self):
        """research方法"""
        agent = ResearchAgent()

        mock_client = MagicMock()
        retrieve_response = MagicMock()
        retrieve_response.content = '''{"sources": [{"content": "信息1"}]}'''
        analyze_response = MagicMock()
        analyze_response.content = '''{"findings": [{"content": "发现1", "confidence": 0.8}], "summary": "摘要"}'''
        report_response = MagicMock()
        report_response.content = "# 报告"
        report_response.usage.total_tokens = 30
        mock_client.chat.side_effect = [retrieve_response, analyze_response, report_response]
        agent.retriever._client = mock_client
        agent.analyzer._client = mock_client
        agent.generator._client = mock_client

        agent.start()
        report = agent.research("查询", topic="主题")

        self.assertTrue(report.is_completed)
        self.assertEqual(report.topic, "主题")

    def test06_get_stats(self):
        agent = ResearchAgent()
        stats = agent.get_stats()
        self.assertIn("retriever", stats)
        self.assertIn("analyzer", stats)
        self.assertIn("generator", stats)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingletons(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_research_agent()

    def tearDown(self):
        reset_research_agent()

    def test01_singleton(self):
        a1 = get_research_agent()
        a2 = get_research_agent()
        self.assertIs(a1, a2)

    def test02_reset(self):
        a1 = get_research_agent()
        reset_research_agent()
        a2 = get_research_agent()
        self.assertIsNot(a1, a2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_research_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def tearDown(self):
        reset_research_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def test01_full_workflow_mock(self):
        """完整工作流(Mock)"""
        reset_memory_manager()  # 清空记忆
        agent = ResearchAgent()

        mock_client = MagicMock()
        retrieve_response = MagicMock()
        retrieve_response.content = '''{"sources": [{"content": "Python是编程语言", "relevance_score": 0.9}]}'''
        analyze_response = MagicMock()
        analyze_response.content = '''{"findings": [{"content": "Python广泛使用", "confidence": 0.85}], "summary": "Python分析"}'''
        report_response = MagicMock()
        report_response.content = "# Python研究报告\n\nPython是一种流行的编程语言。"
        report_response.usage.total_tokens = 80
        mock_client.chat.side_effect = [retrieve_response, analyze_response, report_response]
        agent.retriever._client = mock_client
        agent.analyzer._client = mock_client
        agent.generator._client = mock_client

        agent.start()
        result = agent.run("Python是什么?", topic="Python")

        self.assertTrue(result.success)
        self.assertIn("Python", result.output)
        # sources_count可能包含记忆源,只检查大于0
        self.assertGreater(result.metadata["sources_count"], 0)
        self.assertEqual(result.metadata["findings_count"], 1)

    def test02_real_llm_research(self):
        """真实LLM研究测试"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        agent = get_research_agent()
        agent.start()
        result = agent.run("什么是人工智能?", topic="人工智能")

        self.assertTrue(result.success)
        self.assertGreater(len(result.output), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
