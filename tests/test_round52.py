"""
阶段62测试：工具调用与Function Calling

覆盖：
    - ParameterType枚举
    - ToolParameter (validate/to_dict/from_dict)
    - ToolDefinition (to_dict/to_openai_schema)
    - ToolCallRequest/ToolCallResult
    - ToolRegistry (register/unregister/get/list/schemas/builtin)
    - ToolExecutor (execute/timeout/error/stats)
    - 内置工具 (calculator/datetime/text_length/text_stats/json_parse)
    - ToolCallLoop (ReAct循环/Mock LLM)
    - 线程安全
    - 集成测试 (真实API)
"""
import unittest
import os
import json
import time
import threading
from unittest.mock import patch, MagicMock

from seva.tools import (
    ParameterType, ToolParameter, ToolDefinition,
    ToolCallRequest, ToolCallStatus, ToolCallResult,
    ToolRegistry, ToolExecutor, ToolCallLoop,
    get_tool_registry, get_tool_executor, reset_tools,
)
from seva.llm import LLMError, reset_llm_client


# ============================================================================
# ParameterType
# ============================================================================

class TestParameterType(unittest.TestCase):
    """测试ParameterType枚举"""

    def test01_values(self):
        self.assertEqual(ParameterType.STRING.value, "string")
        self.assertEqual(ParameterType.NUMBER.value, "number")
        self.assertEqual(ParameterType.INTEGER.value, "integer")
        self.assertEqual(ParameterType.BOOLEAN.value, "boolean")
        self.assertEqual(ParameterType.ARRAY.value, "array")
        self.assertEqual(ParameterType.OBJECT.value, "object")


# ============================================================================
# ToolParameter
# ============================================================================

class TestToolParameter(unittest.TestCase):
    """测试ToolParameter"""

    def test01_creation(self):
        p = ToolParameter(name="x")
        self.assertEqual(p.name, "x")
        self.assertEqual(p.param_type, ParameterType.STRING)
        self.assertTrue(p.required)

    def test02_validate_string(self):
        p = ToolParameter(name="x", param_type=ParameterType.STRING)
        valid, _ = p.validate("hello")
        self.assertTrue(valid)
        valid, _ = p.validate(123)
        self.assertFalse(valid)

    def test03_validate_required(self):
        p = ToolParameter(name="x", required=True)
        valid, err = p.validate(None)
        self.assertFalse(valid)
        self.assertIn("必填", err)

    def test04_validate_optional(self):
        p = ToolParameter(name="x", required=False, default="default")
        valid, _ = p.validate(None)
        self.assertTrue(valid)

    def test05_validate_enum(self):
        p = ToolParameter(name="mode", enum=["a", "b", "c"])
        valid, _ = p.validate("a")
        self.assertTrue(valid)
        valid, err = p.validate("d")
        self.assertFalse(valid)
        self.assertIn("enum", err.lower() + " enum")

    def test06_validate_range(self):
        p = ToolParameter(name="n", param_type=ParameterType.NUMBER,
                         minimum=0, maximum=100)
        valid, _ = p.validate(50)
        self.assertTrue(valid)
        valid, err = p.validate(-1)
        self.assertFalse(valid)
        valid, err = p.validate(101)
        self.assertFalse(valid)

    def test07_validate_integer_bool(self):
        """bool不应被接受为integer"""
        p = ToolParameter(name="n", param_type=ParameterType.INTEGER)
        valid, _ = p.validate(True)
        self.assertFalse(valid)

    def test08_to_dict(self):
        p = ToolParameter(name="x", param_type=ParameterType.NUMBER,
                         description="test", minimum=0)
        d = p.to_dict()
        self.assertEqual(d["name"], "x")
        self.assertEqual(d["type"], "number")
        self.assertEqual(d["minimum"], 0)

    def test09_from_dict(self):
        d = {"name": "y", "type": "boolean", "required": False}
        p = ToolParameter.from_dict(d)
        self.assertEqual(p.name, "y")
        self.assertEqual(p.param_type, ParameterType.BOOLEAN)
        self.assertFalse(p.required)


# ============================================================================
# ToolDefinition
# ============================================================================

class TestToolDefinition(unittest.TestCase):
    """测试ToolDefinition"""

    def test01_creation(self):
        t = ToolDefinition(name="test", description="测试工具")
        self.assertEqual(t.name, "test")
        self.assertEqual(t.parameters, [])
        self.assertEqual(t.category, "general")
        self.assertTrue(t.enabled)

    def test02_to_dict(self):
        t = ToolDefinition(
            name="test",
            description="测试",
            parameters=[ToolParameter(name="x")],
            category="math",
        )
        d = t.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["category"], "math")
        self.assertEqual(len(d["parameters"]), 1)

    def test03_to_openai_schema(self):
        t = ToolDefinition(
            name="calc",
            description="计算器",
            parameters=[
                ToolParameter(name="expr", description="表达式", required=True),
                ToolParameter(name="precision", description="精度",
                             required=False, default=2),
            ],
        )
        schema = t.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calc")
        self.assertIn("expr", schema["function"]["parameters"]["properties"])
        self.assertIn("expr", schema["function"]["parameters"]["required"])
        self.assertNotIn("precision", schema["function"]["parameters"]["required"])


# ============================================================================
# ToolCallRequest/Result
# ============================================================================

class TestToolCallRequestResult(unittest.TestCase):
    """测试ToolCallRequest和ToolCallResult"""

    def test01_request(self):
        req = ToolCallRequest(tool_name="calc", arguments={"x": 1})
        self.assertEqual(req.tool_name, "calc")
        self.assertEqual(req.arguments, {"x": 1})

    def test02_request_to_dict(self):
        req = ToolCallRequest(tool_name="calc", arguments={"x": 1},
                             call_id="c1", reasoning="因为...")
        d = req.to_dict()
        self.assertEqual(d["tool_name"], "calc")
        self.assertEqual(d["call_id"], "c1")

    def test03_result_success(self):
        r = ToolCallResult(call_id="c1", tool_name="calc",
                          status=ToolCallStatus.SUCCESS, result=42)
        self.assertTrue(r.success)
        self.assertEqual(r.result, 42)

    def test04_result_error(self):
        r = ToolCallResult(call_id="c1", tool_name="calc",
                          status=ToolCallStatus.EXECUTION_ERROR,
                          error="something went wrong")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "something went wrong")

    def test05_result_to_dict(self):
        r = ToolCallResult(call_id="c1", tool_name="calc",
                          status=ToolCallStatus.SUCCESS, result={"a": 1})
        d = r.to_dict()
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["result"], {"a": 1})


# ============================================================================
# ToolRegistry
# ============================================================================

class TestToolRegistry(unittest.TestCase):
    """测试ToolRegistry"""

    def test01_creation_with_builtins(self):
        reg = ToolRegistry()
        # 应该有内置工具
        self.assertGreaterEqual(reg.count(), 6)
        self.assertIsNotNone(reg.get("calculator"))
        self.assertIsNotNone(reg.get("datetime"))
        self.assertIsNotNone(reg.get("text_length"))
        self.assertIsNotNone(reg.get("text_stats"))
        self.assertIsNotNone(reg.get("http_get"))
        self.assertIsNotNone(reg.get("json_parse"))

    def test02_register_custom(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="custom",
            description="自定义工具",
            parameters=[ToolParameter(name="x")],
            func=lambda x: {"result": x},
        )
        reg.register(tool)
        self.assertIsNotNone(reg.get("custom"))

    def test03_unregister(self):
        reg = ToolRegistry()
        tool = ToolDefinition(name="temp", description="临时")
        reg.register(tool)
        self.assertTrue(reg.unregister("temp"))
        self.assertIsNone(reg.get("temp"))
        self.assertFalse(reg.unregister("temp"))

    def test04_list(self):
        reg = ToolRegistry()
        names = reg.list()
        self.assertIn("calculator", names)
        # 按类别
        math_tools = reg.list(category="math")
        self.assertIn("calculator", math_tools)
        self.assertNotIn("datetime", math_tools)

    def test05_to_openai_schemas(self):
        reg = ToolRegistry()
        schemas = reg.to_openai_schemas()
        self.assertGreater(len(schemas), 0)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("calculator", names)

    def test06_disabled_tool(self):
        reg = ToolRegistry()
        tool = ToolDefinition(name="disabled", description="禁用", enabled=False)
        reg.register(tool)
        self.assertNotIn("disabled", reg.list())
        self.assertIn("disabled", [t.name for t in reg.list_all()])


# ============================================================================
# ToolExecutor
# ============================================================================

class TestToolExecutor(unittest.TestCase):
    """测试ToolExecutor"""

    def setUp(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    def test01_execute_calculator(self):
        req = ToolCallRequest(tool_name="calculator",
                             arguments={"expression": "2+3*4"})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["result"], 14)

    def test02_execute_calculator_sqrt(self):
        req = ToolCallRequest(tool_name="calculator",
                             arguments={"expression": "sqrt(16)"})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["result"], 4.0)

    def test03_execute_datetime(self):
        req = ToolCallRequest(tool_name="datetime", arguments={})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertIn("formatted", result.result)

    def test04_execute_text_length(self):
        req = ToolCallRequest(tool_name="text_length",
                             arguments={"text": "hello world", "mode": "words"})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["length"], 2)

    def test05_execute_text_stats(self):
        req = ToolCallRequest(tool_name="text_stats",
                             arguments={"text": "Hello world. How are you?"})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["chars"], 25)
        self.assertEqual(result.result["sentences"], 2)

    def test06_execute_json_parse(self):
        req = ToolCallRequest(tool_name="json_parse",
                             arguments={"text": '{"a": 1, "b": [2, 3]}'})
        result = self.executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["parsed"]["a"], 1)

    def test07_not_found(self):
        req = ToolCallRequest(tool_name="nonexistent", arguments={})
        result = self.executor.execute(req)
        self.assertEqual(result.status, ToolCallStatus.NOT_FOUND)

    def test08_validation_error(self):
        req = ToolCallRequest(tool_name="calculator", arguments={})
        result = self.executor.execute(req)
        self.assertEqual(result.status, ToolCallStatus.VALIDATION_ERROR)

    def test09_execution_error(self):
        req = ToolCallRequest(tool_name="calculator",
                             arguments={"expression": "1/0"})
        result = self.executor.execute(req)
        self.assertEqual(result.status, ToolCallStatus.EXECUTION_ERROR)

    def test10_timeout(self):
        """测试超时"""
        # 注册一个会阻塞的工具
        def slow_func(seconds):
            time.sleep(seconds)
            return {"slept": seconds}

        self.registry.register(ToolDefinition(
            name="slow",
            description="慢工具",
            parameters=[ToolParameter(name="seconds",
                                     param_type=ParameterType.NUMBER)],
            func=slow_func,
            timeout=0.5,
        ))
        req = ToolCallRequest(tool_name="slow", arguments={"seconds": 2})
        result = self.executor.execute(req)
        self.assertEqual(result.status, ToolCallStatus.TIMEOUT)

    def test11_stats(self):
        req = ToolCallRequest(tool_name="calculator",
                             arguments={"expression": "1+1"})
        self.executor.execute(req)
        stats = self.executor.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["success_calls"], 1)
        self.assertGreater(stats["avg_latency_ms"], 0)


# ============================================================================
# ToolCallLoop
# ============================================================================

class TestToolCallLoop(unittest.TestCase):
    """测试ToolCallLoop"""

    def setUp(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    def test01_creation(self):
        loop = ToolCallLoop(self.registry, self.executor)
        self.assertEqual(loop.max_iterations, 5)

    def test02_no_tools(self):
        """无工具时返回错误"""
        empty_reg = ToolRegistry()
        empty_reg.clear()
        loop = ToolCallLoop(empty_reg)
        result = loop.run("hello")
        self.assertEqual(result["answer"], "没有可用的工具")

    def test03_with_mock_llm_tool_call(self):
        """Mock LLM返回工具调用"""
        loop = ToolCallLoop(self.registry, self.executor)

        mock_client = MagicMock()
        # 第一次返回工具调用
        call1 = MagicMock()
        call1.content = '{"tool": "calculator", "arguments": {"expression": "2+2"}, "reasoning": "需要计算"}'
        call1.usage.total_tokens = 50
        # 第二次返回最终答案
        call2 = MagicMock()
        call2.content = "2+2的结果是4"
        call2.usage.total_tokens = 30
        mock_client.chat.side_effect = [call1, call2]

        with patch("seva.tools.get_llm_client", return_value=mock_client):
            result = loop.run("计算2+2")

        self.assertEqual(result["iterations"], 1)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["answer"], "2+2的结果是4")
        self.assertEqual(result["total_tokens"], 80)

    def test04_with_mock_llm_direct_answer(self):
        """Mock LLM直接回答(不调用工具)"""
        loop = ToolCallLoop(self.registry, self.executor)

        mock_client = MagicMock()
        call1 = MagicMock()
        call1.content = "你好！我是助手。"
        call1.usage.total_tokens = 20
        mock_client.chat.side_effect = [call1]

        with patch("seva.tools.get_llm_client", return_value=mock_client):
            result = loop.run("你好")

        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["answer"], "你好！我是助手。")

    def test05_with_mock_llm_error(self):
        """Mock LLM错误"""
        loop = ToolCallLoop(self.registry, self.executor)

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("mock error")

        with patch("seva.tools.get_llm_client", return_value=mock_client):
            result = loop.run("hello")

        self.assertIn("LLM调用失败", result["answer"])

    def test06_max_iterations(self):
        """测试最大轮次限制"""
        loop = ToolCallLoop(self.registry, self.executor, max_iterations=2)

        mock_client = MagicMock()
        # 每次都返回工具调用(永不停止)
        call = MagicMock()
        call.content = '{"tool": "calculator", "arguments": {"expression": "1+1"}}'
        call.usage.total_tokens = 10
        mock_client.chat.side_effect = [call] * 5

        with patch("seva.tools.get_llm_client", return_value=mock_client):
            result = loop.run("计算")

        self.assertEqual(result["iterations"], 2)


# ============================================================================
# 全局单例
# ============================================================================

class TestGlobalTools(unittest.TestCase):
    """测试全局工具"""

    def setUp(self):
        reset_tools()

    def tearDown(self):
        reset_tools()

    def test01_registry_singleton(self):
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        self.assertIs(r1, r2)

    def test02_executor_singleton(self):
        e1 = get_tool_executor()
        e2 = get_tool_executor()
        self.assertIs(e1, e2)

    def test03_reset(self):
        r1 = get_tool_registry()
        reset_tools()
        r2 = get_tool_registry()
        self.assertIsNot(r1, r2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_tools()
        reset_llm_client()

    def tearDown(self):
        reset_tools()
        reset_llm_client()

    def test01_full_workflow(self):
        """完整工作流: 注册→执行→统计"""
        reg = get_tool_registry()
        executor = get_tool_executor()

        # 执行内置工具
        req = ToolCallRequest(tool_name="calculator",
                             arguments={"expression": "sqrt(144) + 1"})
        result = executor.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.result["result"], 13.0)

        # 检查统计
        stats = executor.get_stats()
        self.assertEqual(stats["total_calls"], 1)

    def test02_real_llm_tool_call(self):
        """真实LLM工具调用(需要API key)"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        reg = get_tool_registry()
        executor = get_tool_executor()
        loop = ToolCallLoop(reg, executor, max_iterations=3)

        result = loop.run("请计算 sqrt(256) 的值", use_cache=False)
        self.assertGreater(result["iterations"], 0)
        self.assertGreater(len(result["tool_calls"]), 0)
        # 工具应该成功执行
        for tc in result["tool_calls"]:
            self.assertEqual(tc["status"], "success")


if __name__ == "__main__":
    unittest.main(verbosity=2)
