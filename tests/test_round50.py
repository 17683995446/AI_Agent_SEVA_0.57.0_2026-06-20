"""
阶段60测试：16阶段循环LLM增强

覆盖：
    - LLMStageStatus枚举
    - LLMRole枚举
    - LLMStageResult数据类
    - parse_json_output JSON提取（多种格式）
    - LLMLoop核心（启用/禁用/模板管理/调用/降级/统计/历史）
    - 16阶段默认模板注册
    - 缓存与降级路径
    - 线程安全
    - 集成测试（端到端调用 + 真实API）
"""
import unittest
import os
import json
import time
import threading
from unittest.mock import patch, MagicMock

from seva.llm_loop import (
    LLMStageStatus, LLMRole, LLMStageResult,
    parse_json_output, LLMLoop,
    get_llm_loop, set_llm_loop, reset_llm_loop,
)
from seva.llm import LLMError, get_llm_client, get_llm_stats, reset_llm_client


# ============================================================================
# LLMStageStatus枚举
# ============================================================================

class TestLLMStageStatus(unittest.TestCase):
    """测试LLMStageStatus枚举"""

    def test01_values(self):
        self.assertEqual(LLMStageStatus.SUCCESS.value, "success")
        self.assertEqual(LLMStageStatus.SKIPPED.value, "skipped")
        self.assertEqual(LLMStageStatus.CACHED.value, "cached")
        self.assertEqual(LLMStageStatus.DEGRADED.value, "degraded")
        self.assertEqual(LLMStageStatus.DISABLED.value, "disabled")
        self.assertEqual(LLMStageStatus.ERROR.value, "error")


# ============================================================================
# LLMRole枚举
# ============================================================================

class TestLLMRole(unittest.TestCase):
    """测试LLMRole枚举"""

    def test01_values(self):
        self.assertEqual(LLMRole.ENHANCER.value, "enhancer")
        self.assertEqual(LLMRole.REPLACER.value, "replacer")
        self.assertEqual(LLMRole.EXPLICATOR.value, "explicator")


# ============================================================================
# LLMStageResult
# ============================================================================

class TestLLMStageResult(unittest.TestCase):
    """测试LLMStageResult"""

    def test01_creation(self):
        r = LLMStageResult(stage="test", status=LLMStageStatus.SUCCESS)
        self.assertEqual(r.stage, "test")
        self.assertEqual(r.status, LLMStageStatus.SUCCESS)
        self.assertEqual(r.llm_output, None)
        self.assertEqual(r.total_tokens, 0)
        self.assertGreater(r.timestamp, 0)

    def test02_to_dict(self):
        r = LLMStageResult(
            stage="s1", status=LLMStageStatus.SUCCESS,
            llm_output="hello", parsed_output={"key": "value"},
            total_tokens=100, latency_ms=50.0, degraded=False,
        )
        d = r.to_dict()
        self.assertEqual(d["stage"], "s1")
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["llm_output"], "hello")
        self.assertEqual(d["parsed_output"], {"key": "value"})
        self.assertEqual(d["total_tokens"], 100)
        self.assertEqual(d["latency_ms"], 50.0)


# ============================================================================
# parse_json_output
# ============================================================================

class TestParseJsonOutput(unittest.TestCase):
    """测试parse_json_output"""

    def test01_markdown_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_output(text)
        self.assertEqual(result, {"key": "value"})

    def test02_markdown_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_json_output(text)
        self.assertEqual(result, {"key": "value"})

    def test03_inline_object(self):
        text = 'Output is: {"foo": 1, "bar": 2}'
        result = parse_json_output(text)
        self.assertEqual(result, {"foo": 1, "bar": 2})

    def test04_text_with_object(self):
        text = 'Some text\nThen JSON: {"x": 1}\nMore text'
        result = parse_json_output(text)
        self.assertEqual(result, {"x": 1})

    def test05_array_to_items(self):
        text = '[1, 2, 3]'
        result = parse_json_output(text)
        self.assertEqual(result, {"items": [1, 2, 3]})

    def test06_markdown_array(self):
        text = '```json\n[1, 2, 3]\n```'
        result = parse_json_output(text)
        self.assertEqual(result, {"items": [1, 2, 3]})

    def test07_no_json(self):
        self.assertIsNone(parse_json_output("plain text"))

    def test08_empty(self):
        self.assertIsNone(parse_json_output(""))
        self.assertIsNone(parse_json_output(None))

    def test09_invalid_json(self):
        self.assertIsNone(parse_json_output("{not valid json"))

    def test10_nested_object(self):
        text = '```json\n{"a": {"b": {"c": 1}}}\n```'
        result = parse_json_output(text)
        self.assertEqual(result, {"a": {"b": {"c": 1}}})


# ============================================================================
# LLMLoop
# ============================================================================

class TestLLMLoop(unittest.TestCase):
    """测试LLMLoop核心"""

    def setUp(self):
        # 清理 LLM 客户端避免真实 API 调用
        reset_llm_client()

    def test01_creation_default(self):
        loop = LLMLoop()
        self.assertTrue(loop.enable)
        # 默认应该注册了 16 个阶段模板
        self.assertGreaterEqual(len(loop._templates), 16)
        self.assertIn("perception", loop._templates)
        self.assertIn("goal_generator", loop._templates)
        self.assertIn("metacognition", loop._templates)
        self.assertIn("timing_control", loop._templates)

    def test02_creation_disabled(self):
        loop = LLMLoop(enable=False)
        self.assertFalse(loop.enable)
        self.assertFalse(loop.is_stage_enabled("any_stage"))

    def test03_enable_disable_stages(self):
        loop = LLMLoop()
        self.assertTrue(loop.is_stage_enabled("perception"))
        loop.disable_stage("perception")
        self.assertFalse(loop.is_stage_enabled("perception"))
        loop.enable_stage("perception")
        self.assertTrue(loop.is_stage_enabled("perception"))

    def test04_only_enabled_stages(self):
        loop = LLMLoop(enable_stages=["perception", "goal_generator"])
        self.assertTrue(loop.is_stage_enabled("perception"))
        self.assertFalse(loop.is_stage_enabled("predictor"))
        self.assertFalse(loop.is_stage_enabled("executor"))

    def test05_call_disabled_loop(self):
        loop = LLMLoop(enable=False)
        result = loop.call_stage("perception", inputs={"input": "test"})
        self.assertEqual(result.status, LLMStageStatus.DISABLED)

    def test06_register_template(self):
        loop = LLMLoop()
        from seva.prompts import PromptTemplate, PromptSection
        template = PromptTemplate(
            name="custom_stage",
            sections=[PromptSection.task("处理 {{input}}")],
        )
        loop.register_template("my_stage", template)
        # 应该能查到
        t = loop.get_template("my_stage")
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "custom_stage")

    def test07_get_template_default(self):
        loop = LLMLoop()
        t = loop.get_template("goal_generator")
        self.assertIsNotNone(t)
        self.assertEqual(t.output_format.value, "messages")

    def test08_get_template_unknown(self):
        loop = LLMLoop()
        self.assertIsNone(loop.get_template("nonexistent_stage"))

    def test09_call_stage_disabled_stage(self):
        loop = LLMLoop(enable_stages=["perception"])
        result = loop.call_stage("predictor", inputs={"input": "x"})
        self.assertEqual(result.status, LLMStageStatus.DISABLED)

    def test10_call_with_messages(self):
        loop = LLMLoop()
        messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
        ]
        result = loop.call_with_messages("perception", messages, use_cache=False)
        # 没有API Key时应该降级或失败
        self.assertIn(result.status, [
            LLMStageStatus.SUCCESS, LLMStageStatus.DEGRADED,
            LLMStageStatus.ERROR, LLMStageStatus.DISABLED,
        ])

    def test11_call_fallback_template(self):
        """测试无模板时使用回退模板"""
        loop = LLMLoop()
        # 用一个未注册的stage名
        result = loop.call_stage(
            "very_unknown_stage",
            inputs={"input": "data"},
            use_cache=False,
        )
        # 应该会创建回退模板并尝试调用
        self.assertIn(result.status, [
            LLMStageStatus.SUCCESS, LLMStageStatus.DEGRADED,
            LLMStageStatus.ERROR,
        ])
        # 模板应该被注册
        self.assertIn("very_unknown_stage", loop._templates)

    def test12_history_recorded(self):
        loop = LLMLoop(enable_stages=["perception"])
        loop.call_stage("perception", inputs={"input": "x"}, use_cache=False)
        history = loop.get_history()
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0].stage, "perception")

    def test13_history_filter(self):
        loop = LLMLoop(enable_stages=["perception", "goal_generator"])
        loop.call_stage("perception", inputs={"input": "x"}, use_cache=False)
        loop.call_stage("goal_generator", inputs={"input": "y"}, use_cache=False)
        h_perception = loop.get_history(stage="perception")
        self.assertEqual(len(h_perception), 1)
        self.assertEqual(h_perception[0].stage, "perception")

    def test14_stats(self):
        loop = LLMLoop(enable_stages=["perception"])
        loop.call_stage("perception", inputs={"input": "x"}, use_cache=False)
        stats = loop.get_stats()
        self.assertIn("total_calls", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("avg_latency_ms", stats)
        self.assertGreaterEqual(stats["total_calls"], 1)

    def test15_reset_stats(self):
        loop = LLMLoop(enable_stages=["perception"])
        loop.call_stage("perception", inputs={"input": "x"}, use_cache=False)
        loop.reset_stats()
        stats = loop.get_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(len(loop.history), 0)

    def test16_get_report(self):
        loop = LLMLoop(enable_stages=["perception"])
        report = loop.get_report()
        self.assertIn("stats", report)
        self.assertIn("history_count", report)
        self.assertIn("enabled", report)
        self.assertIn("enabled_stages", report)
        self.assertIn("disabled_stages", report)
        self.assertIn("registered_templates", report)

    def test17_thread_safety(self):
        loop = LLMLoop(enable_stages=["perception", "goal_generator"])
        errors = []

        def worker():
            try:
                for _ in range(10):
                    loop.call_stage("perception", inputs={"input": "x"},
                                   use_cache=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        # 30次调用
        self.assertEqual(len(loop.history), 30)


# ============================================================================
# 全局单例
# ============================================================================

class TestGlobalLLMLoop(unittest.TestCase):
    """测试全局LLM Loop单例"""

    def setUp(self):
        reset_llm_loop()

    def tearDown(self):
        reset_llm_loop()

    def test01_singleton(self):
        loop1 = get_llm_loop()
        loop2 = get_llm_loop()
        self.assertIs(loop1, loop2)

    def test02_set_and_get(self):
        custom = LLMLoop(enable=False)
        set_llm_loop(custom)
        self.assertIs(get_llm_loop(), custom)
        self.assertFalse(get_llm_loop().is_stage_enabled("perception"))

    def test03_reset(self):
        loop1 = get_llm_loop()
        reset_llm_loop()
        loop2 = get_llm_loop()
        self.assertIsNot(loop1, loop2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试：端到端流程"""

    def setUp(self):
        reset_llm_loop()
        reset_llm_client()

    def tearDown(self):
        reset_llm_loop()
        reset_llm_client()

    def test01_no_api_key_fallback(self):
        """无API Key时应该降级而非崩溃"""
        loop = get_llm_loop()
        result = loop.call_stage("goal_generator",
                                  inputs={"input": "测试"},
                                  use_cache=False)
        # 没有 API key 时应该是降级或错误（不崩溃）
        self.assertIn(result.status, [
            LLMStageStatus.DEGRADED, LLMStageStatus.ERROR,
            LLMStageStatus.SUCCESS,
        ])
        # 不应抛异常
        self.assertIsNotNone(result)

    def test02_call_with_messages_no_api(self):
        """无API Key时call_with_messages也应该降级"""
        loop = get_llm_loop()
        result = loop.call_with_messages(
            "perception",
            [{"role": "user", "content": "test"}],
            use_cache=False,
        )
        self.assertIn(result.status, [
            LLMStageStatus.DEGRADED, LLMStageStatus.ERROR,
            LLMStageStatus.SUCCESS,
        ])

    def test03_16_stages_all_have_templates(self):
        """验证16个阶段都有默认模板"""
        loop = LLMLoop()
        stages = [
            "perception", "drive_engine", "goal_generator", "predictor",
            "decision_maker", "safety_ethics", "executor", "error_evaluator",
            "task_update", "skill_update", "coordination", "evolution",
            "optimizer", "memory", "metacognition", "timing_control",
        ]
        for stage in stages:
            t = loop.get_template(stage)
            self.assertIsNotNone(t, f"阶段 {stage} 缺少默认模板")
            # 验证模板有 role + task + output
            section_types = [s.section_type for s in t.sections]
            self.assertIn("role", [st.value for st in section_types])
            self.assertIn("task", [st.value for st in section_types])

    def test04_loop_with_mock_llm(self):
        """使用mock测试LLM调用成功路径"""
        loop = LLMLoop()

        # Mock LLM 客户端
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "test_output", "confidence": 0.9}'
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70
        mock_client.chat.return_value = mock_response

        with patch("seva.llm_loop.get_llm_client", return_value=mock_client):
            result = loop.call_stage("perception",
                                      inputs={"input": "test_data"},
                                      use_cache=False)

        self.assertEqual(result.status, LLMStageStatus.SUCCESS)
        self.assertEqual(result.parsed_output,
                         {"result": "test_output", "confidence": 0.9})
        self.assertEqual(result.total_tokens, 70)
        self.assertGreater(result.latency_ms, 0)

    def test05_loop_with_mock_llm_error(self):
        """使用mock测试LLM失败降级"""
        loop = LLMLoop()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("mock error")

        with patch("seva.llm_loop.get_llm_client", return_value=mock_client):
            result = loop.call_stage("perception",
                                      inputs={"input": "test"},
                                      use_cache=False)

        self.assertEqual(result.status, LLMStageStatus.DEGRADED)
        self.assertTrue(result.degraded)
        self.assertIn("llm_error", result.error)

    def test06_loop_with_real_api(self):
        """真实API调用（需要环境变量）"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY，跳过真实API测试")
        loop = get_llm_loop()
        result = loop.call_stage("perception",
                                  inputs={"input": "你听到了什么声音？"},
                                  use_cache=False)
        self.assertEqual(result.status, LLMStageStatus.SUCCESS)
        self.assertGreater(result.total_tokens, 0)
        self.assertGreater(result.latency_ms, 0)
        if result.parsed_output:
            self.assertIsInstance(result.parsed_output, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
