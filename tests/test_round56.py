"""
阶段66测试：编程Agent

覆盖：
    - 枚举: CodeLanguage/ExecutionStatus
    - 数据类: CodeBlock/ExecutionResult/ReviewResult
    - CodeSandbox: 代码执行沙箱
    - CodeGenerator: LLM驱动代码生成
    - CodeReviewer: LLM驱动代码审查
    - CodeDebugger: LLM驱动代码调试
    - CodeAgent: 编程Agent
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import time
from unittest.mock import patch, MagicMock

from seva.agents import (
    CodeLanguage, ExecutionStatus, CodeBlock, ExecutionResult, ReviewResult,
    CodeSandbox, CodeGenerator, CodeReviewer, CodeDebugger,
    CodeAgentConfig, CodeAgent,
    get_code_agent, reset_code_agent,
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

    def test01_code_language(self):
        self.assertEqual(CodeLanguage.PYTHON.value, "python")
        self.assertEqual(CodeLanguage.JAVASCRIPT.value, "javascript")
        self.assertEqual(CodeLanguage.SHELL.value, "shell")

    def test02_execution_status(self):
        self.assertEqual(ExecutionStatus.SUCCESS.value, "success")
        self.assertEqual(ExecutionStatus.FAILED.value, "failed")
        self.assertEqual(ExecutionStatus.TIMEOUT.value, "timeout")
        self.assertEqual(ExecutionStatus.BLOCKED.value, "blocked")


# ============================================================================
# 数据类测试
# ============================================================================

class TestDataClasses(unittest.TestCase):
    """测试数据类"""

    def test01_code_block(self):
        cb = CodeBlock(code="print('hello')", language=CodeLanguage.PYTHON)
        self.assertEqual(cb.code, "print('hello')")
        self.assertEqual(cb.language, CodeLanguage.PYTHON)

    def test02_code_block_to_dict(self):
        cb = CodeBlock(code="x=1", description="test")
        d = cb.to_dict()
        self.assertEqual(d["code"], "x=1")
        self.assertEqual(d["description"], "test")

    def test03_code_block_from_dict(self):
        d = {"code": "y=2", "language": "python"}
        cb = CodeBlock.from_dict(d)
        self.assertEqual(cb.code, "y=2")
        self.assertEqual(cb.language, CodeLanguage.PYTHON)

    def test04_execution_result(self):
        r = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="output")
        self.assertTrue(r.success)
        self.assertEqual(r.stdout, "output")

    def test05_execution_result_failed(self):
        r = ExecutionResult(status=ExecutionStatus.FAILED, stderr="error")
        self.assertFalse(r.success)

    def test06_execution_result_to_dict(self):
        r = ExecutionResult(status=ExecutionStatus.SUCCESS, exit_code=0)
        d = r.to_dict()
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["exit_code"], 0)

    def test07_review_result(self):
        r = ReviewResult(approved=True, score=0.8, summary="good")
        self.assertTrue(r.approved)
        self.assertEqual(r.score, 0.8)

    def test08_review_result_to_dict(self):
        r = ReviewResult(approved=False, issues=["issue1"])
        d = r.to_dict()
        self.assertFalse(d["approved"])
        self.assertEqual(d["issues"], ["issue1"])


# ============================================================================
# CodeSandbox测试
# ============================================================================

class TestCodeSandbox(unittest.TestCase):
    """测试代码沙箱"""

    def setUp(self):
        self.sandbox = CodeSandbox(timeout=5.0)

    def test01_execute_python_success(self):
        """执行Python成功"""
        result = self.sandbox.execute("print('hello world')")
        self.assertTrue(result.success)
        self.assertIn("hello world", result.stdout)

    def test02_execute_python_with_output(self):
        """执行带计算"""
        result = self.sandbox.execute("print(2 + 3)")
        self.assertTrue(result.success)
        self.assertIn("5", result.stdout)

    def test03_execute_python_failed(self):
        """执行Python失败"""
        result = self.sandbox.execute("print(undefined_var)")
        self.assertFalse(result.success)
        self.assertGreater(len(result.stderr), 0)

    def test04_execute_python_syntax_error(self):
        """语法错误"""
        result = self.sandbox.execute("print('unterminated")
        self.assertFalse(result.success)

    def test05_execute_timeout(self):
        """超时测试"""
        sandbox = CodeSandbox(timeout=0.5)
        result = sandbox.execute("import time; time.sleep(2)")
        self.assertEqual(result.status, ExecutionStatus.TIMEOUT)

    def test06_execute_blocked_dangerous(self):
        """危险操作被阻止"""
        result = self.sandbox.execute("import os; os.system('ls')")
        self.assertEqual(result.status, ExecutionStatus.BLOCKED)

    def test07_execute_shell(self):
        """执行Shell"""
        result = self.sandbox.execute("echo hello", language=CodeLanguage.SHELL)
        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test08_execute_with_stdin(self):
        """带标准输入"""
        code = "x = input(); print(f'got: {x}')"
        result = self.sandbox.execute(code, stdin="test_input")
        self.assertTrue(result.success)
        self.assertIn("test_input", result.stdout)

    def test09_execute_multiline(self):
        """多行代码"""
        code = """
x = 10
y = 20
print(f'sum: {x + y}')
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("sum: 30", result.stdout)

    def test10_get_stats(self):
        """统计"""
        self.sandbox.execute("print('test')")
        stats = self.sandbox.get_stats()
        self.assertEqual(stats["executions"], 1)
        self.assertEqual(stats["successful"], 1)

    def test11_unsupported_language(self):
        """不支持的语言"""
        result = self.sandbox.execute("code", language=CodeLanguage.SQL)
        self.assertEqual(result.status, ExecutionStatus.ERROR)


# ============================================================================
# CodeGenerator测试
# ============================================================================

class TestCodeGenerator(unittest.TestCase):
    """测试代码生成器"""

    def test01_creation(self):
        gen = CodeGenerator()
        self.assertIsNotNone(gen)
        self.assertEqual(gen.stats["generated"], 0)

    def test02_generate_with_mock(self):
        """Mock LLM生成"""
        gen = CodeGenerator()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "code": "print('hello')",
    "description": "打印hello"
}
```'''
        mock_client.chat.return_value = response
        gen._client = mock_client

        code_block, error = gen.generate("打印hello")
        self.assertIsNotNone(code_block)
        self.assertIsNone(error)
        self.assertEqual(code_block.code, "print('hello')")

    def test03_generate_llm_error(self):
        """LLM错误"""
        gen = CodeGenerator()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("API错误")
        gen._client = mock_client

        code_block, error = gen.generate("test")
        self.assertIsNone(code_block)
        self.assertIn("API错误", error)

    def test04_generate_invalid_json(self):
        """无效JSON"""
        gen = CodeGenerator()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "这不是JSON"
        mock_client.chat.return_value = response
        gen._client = mock_client

        code_block, error = gen.generate("test")
        self.assertIsNone(code_block)
        self.assertEqual(gen.stats["failures"], 1)

    def test05_get_stats(self):
        gen = CodeGenerator()
        stats = gen.get_stats()
        self.assertIn("generated", stats)


# ============================================================================
# CodeReviewer测试
# ============================================================================

class TestCodeReviewer(unittest.TestCase):
    """测试代码审查器"""

    def test01_creation(self):
        rev = CodeReviewer()
        self.assertIsNotNone(rev)

    def test02_review_with_mock(self):
        """Mock LLM审查"""
        rev = CodeReviewer()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "approved": true,
    "score": 0.8,
    "issues": [],
    "suggestions": ["可以添加注释"],
    "summary": "代码质量良好"
}
```'''
        mock_client.chat.return_value = response
        rev._client = mock_client

        code_block = CodeBlock(code="print('hello')")
        result = rev.review(code_block)
        self.assertTrue(result.approved)
        self.assertEqual(result.score, 0.8)

    def test03_review_llm_error(self):
        """LLM错误回退"""
        rev = CodeReviewer()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        rev._client = mock_client

        code_block = CodeBlock(code="print('hello')")
        result = rev.review(code_block)
        # 应该回退到规则审查
        self.assertTrue(result.approved)

    def test04_review_dangerous_code(self):
        """危险代码回退审查"""
        rev = CodeReviewer()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        rev._client = mock_client

        code_block = CodeBlock(code="eval('1+1')")
        result = rev.review(code_block)
        self.assertFalse(result.approved)
        self.assertGreater(len(result.issues), 0)

    def test05_get_stats(self):
        rev = CodeReviewer()
        stats = rev.get_stats()
        self.assertIn("reviewed", stats)


# ============================================================================
# CodeDebugger测试
# ============================================================================

class TestCodeDebugger(unittest.TestCase):
    """测试代码调试器"""

    def test01_creation(self):
        deb = CodeDebugger()
        self.assertIsNotNone(deb)

    def test02_debug_with_mock(self):
        """Mock LLM调试"""
        deb = CodeDebugger()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "fixed_code": "print('fixed')",
    "explanation": "修复了语法错误",
    "changes": ["添加引号"]
}
```'''
        mock_client.chat.return_value = response
        deb._client = mock_client

        code_block = CodeBlock(code="print('broken")
        fixed, error = deb.debug(code_block, "SyntaxError")
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed.code, "print('fixed')")

    def test03_debug_llm_error(self):
        """LLM错误"""
        deb = CodeDebugger()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("error")
        deb._client = mock_client

        code_block = CodeBlock(code="broken")
        fixed, error = deb.debug(code_block, "error")
        self.assertIsNone(fixed)
        self.assertIn("LLM错误", error)

    def test04_get_stats(self):
        deb = CodeDebugger()
        stats = deb.get_stats()
        self.assertIn("debugged", stats)


# ============================================================================
# CodeAgent测试
# ============================================================================

class TestCodeAgent(unittest.TestCase):
    """测试编程Agent"""

    def setUp(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_code_agent()

    def tearDown(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_code_agent()

    def test01_creation(self):
        agent = CodeAgent()
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.config, CodeAgentConfig)
        self.assertIsNotNone(agent.sandbox)
        self.assertIsNotNone(agent.generator)
        self.assertIsNotNone(agent.reviewer)
        self.assertIsNotNone(agent.debugger)

    def test02_creation_with_config(self):
        config = CodeAgentConfig(
            name="custom",
            sandbox_timeout=5.0,
            max_debug_iterations=2,
        )
        agent = CodeAgent(config=config)
        self.assertEqual(agent.config.name, "custom")
        self.assertEqual(agent.code_config.sandbox_timeout, 5.0)

    def test03_execute_code_directly(self):
        """直接执行代码"""
        agent = CodeAgent()
        result = agent.execute_code("print('hello')")
        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test04_generate_code_with_mock(self):
        """Mock生成代码"""
        agent = CodeAgent()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{"code": "print('generated')", "description": "test"}
```'''
        mock_client.chat.return_value = response
        agent.generator._client = mock_client

        code_block, error = agent.generate_code("打印hello")
        self.assertIsNotNone(code_block)
        self.assertEqual(code_block.code, "print('generated')")

    def test05_run_with_mock(self):
        """Mock完整运行"""
        agent = CodeAgent()

        # Mock生成器
        mock_client = MagicMock()
        gen_response = MagicMock()
        gen_response.content = '''{"code": "print('hello')", "description": "test"}'''
        review_response = MagicMock()
        review_response.content = '''{"approved": true, "score": 0.8, "issues": [], "suggestions": [], "summary": "good"}'''
        mock_client.chat.side_effect = [gen_response, review_response]
        agent.generator._client = mock_client
        agent.reviewer._client = mock_client

        agent.start()
        result = agent.run("打印hello")
        self.assertTrue(result.success)
        self.assertIn("print('hello')", result.output)

    def test06_run_empty_requirement(self):
        """空需求"""
        agent = CodeAgent()
        agent.start()
        result = agent.run("")
        self.assertFalse(result.success)
        self.assertIn("空", result.error)

    def test07_run_with_execution_failure(self):
        """执行失败"""
        agent = CodeAgent()

        # Mock生成器返回有bug的代码
        mock_client = MagicMock()
        gen_response = MagicMock()
        gen_response.content = '''{"code": "print(undefined_var)", "description": "buggy"}'''
        # Mock调试器返回修复后的代码
        debug_response = MagicMock()
        debug_response.content = '''{"fixed_code": "print('fixed')", "explanation": "fix"}'''
        mock_client.chat.side_effect = [gen_response, debug_response, debug_response]
        agent.generator._client = mock_client
        agent.debugger._client = mock_client

        config = CodeAgentConfig(enable_review=False, max_debug_iterations=1)
        agent.config = config

        agent.start()
        result = agent.run("test", execute=True)
        # 应该经过调试后成功
        self.assertTrue(result.success)

    def test08_get_stats(self):
        agent = CodeAgent()
        stats = agent.get_stats()
        self.assertIn("sandbox", stats)
        self.assertIn("generator", stats)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingletons(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_code_agent()

    def tearDown(self):
        reset_code_agent()

    def test01_singleton(self):
        a1 = get_code_agent()
        a2 = get_code_agent()
        self.assertIs(a1, a2)

    def test02_reset(self):
        a1 = get_code_agent()
        reset_code_agent()
        a2 = get_code_agent()
        self.assertIsNot(a1, a2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_code_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def tearDown(self):
        reset_code_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def test01_full_workflow_mock(self):
        """完整工作流(Mock)"""
        agent = CodeAgent()

        # Mock生成器
        mock_client = MagicMock()
        gen_response = MagicMock()
        gen_response.content = '''{"code": "x = 10\\ny = 20\\nprint(f'sum: {x+y}')", "description": "计算两数之和"}'''
        review_response = MagicMock()
        review_response.content = '''{"approved": true, "score": 0.9, "issues": [], "suggestions": [], "summary": "good"}'''
        mock_client.chat.side_effect = [gen_response, review_response]
        agent.generator._client = mock_client
        agent.reviewer._client = mock_client

        agent.start()
        result = agent.run("计算10和20的和")

        self.assertTrue(result.success)
        self.assertIn("sum: 30", result.output)
        self.assertIn("执行成功", result.output)

    def test02_real_llm_code_generation(self):
        """真实LLM代码生成"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        agent = get_code_agent()
        agent.start()
        result = agent.run("写一个Python函数计算斐波那契数列第n项")

        self.assertTrue(result.success)
        self.assertIn("python", result.output.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
