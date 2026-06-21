"""
阶段64测试：Agent基类与生命周期管理

覆盖：
    - 枚举: AgentState/AgentEventType
    - 数据类: AgentConfig/AgentStats/AgentContext/AgentRunResult/AgentEvent
    - BaseAgent (生命周期/状态机/事件/钩子/上下文/记忆)
    - AgentRunner (注册/注销/启动/停止/事件/报告)
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import json
import time
import threading
from unittest.mock import patch, MagicMock

from seva.agents import (
    BaseAgent, AgentConfig, AgentState, AgentRunner,
    AgentEvent, AgentEventType, AgentStats, AgentContext, AgentRunResult,
    get_agent_runner, reset_agent_runner,
)
from seva.agents.base import _STATE_TRANSITIONS
from seva.llm import LLMError, reset_llm_client
from seva.tools import reset_tools
from seva.memory_llm import reset_memory_manager, MemoryType
from seva.planner import reset_planner, reset_reflector, reset_decision_maker, reset_autonomous_agent


# ============================================================================
# 测试用具体Agent
# ============================================================================

class EchoAgent(BaseAgent):
    """回显Agent - 用于测试"""

    def _run_impl(self, input, **kwargs):
        # 模拟处理
        output = f"Echo: {input}" if isinstance(input, str) else f"Echo: {input}"
        return AgentRunResult(
            agent_id=self.agent_id,
            session_id=self.context.session_id,
            success=True,
            output=output,
            tokens_used=10,
            latency_ms=1.0,
        )


class FailingAgent(BaseAgent):
    """总是失败的Agent"""

    def _run_impl(self, input, **kwargs):
        raise RuntimeError("故意失败")


class SlowAgent(BaseAgent):
    """慢速Agent"""

    def _run_impl(self, input, **kwargs):
        time.sleep(0.1)
        return AgentRunResult(
            agent_id=self.agent_id,
            session_id=self.context.session_id,
            success=True,
            output="done",
        )


# ============================================================================
# 枚举测试
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_agent_state_values(self):
        self.assertEqual(AgentState.CREATED.value, "created")
        self.assertEqual(AgentState.CONFIGURED.value, "configured")
        self.assertEqual(AgentState.RUNNING.value, "running")
        self.assertEqual(AgentState.PAUSED.value, "paused")
        self.assertEqual(AgentState.STOPPED.value, "stopped")
        self.assertEqual(AgentState.TERMINATED.value, "terminated")
        self.assertEqual(AgentState.ERROR.value, "error")

    def test02_agent_event_types(self):
        self.assertEqual(AgentEventType.STATE_CHANGE.value, "state_change")
        self.assertEqual(AgentEventType.MESSAGE.value, "message")
        self.assertEqual(AgentEventType.TOOL_CALL.value, "tool_call")
        self.assertEqual(AgentEventType.ERROR.value, "error")
        self.assertEqual(AgentEventType.LIFECYCLE.value, "lifecycle")

    def test03_state_transitions(self):
        """测试状态转换图"""
        # CREATED可以转到CONFIGURED
        self.assertIn(AgentState.CONFIGURED, _STATE_TRANSITIONS[AgentState.CREATED])
        # CONFIGURED可以转到RUNNING
        self.assertIn(AgentState.RUNNING, _STATE_TRANSITIONS[AgentState.CONFIGURED])
        # RUNNING可以转到PAUSED
        self.assertIn(AgentState.PAUSED, _STATE_TRANSITIONS[AgentState.RUNNING])
        # PAUSED可以转到RUNNING
        self.assertIn(AgentState.RUNNING, _STATE_TRANSITIONS[AgentState.PAUSED])
        # TERMINATED是终态
        self.assertEqual(_STATE_TRANSITIONS[AgentState.TERMINATED], set())


# ============================================================================
# AgentConfig测试
# ============================================================================

class TestAgentConfig(unittest.TestCase):
    """测试AgentConfig"""

    def test01_creation(self):
        c = AgentConfig(name="test", description="测试")
        self.assertEqual(c.name, "test")
        self.assertEqual(c.description, "测试")
        self.assertEqual(c.temperature, 0.7)
        self.assertTrue(c.enable_tools)

    def test02_to_dict(self):
        c = AgentConfig(name="test", model="gpt-4", max_tokens=2048)
        d = c.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["max_tokens"], 2048)

    def test03_from_dict(self):
        d = {"name": "from_dict", "temperature": 0.5, "enable_memory": False}
        c = AgentConfig.from_dict(d)
        self.assertEqual(c.name, "from_dict")
        self.assertEqual(c.temperature, 0.5)
        self.assertFalse(c.enable_memory)

    def test04_roundtrip(self):
        c1 = AgentConfig(name="roundtrip", description="测试",
                        model="test-model", max_tokens=512)
        d = c1.to_dict()
        c2 = AgentConfig.from_dict(d)
        self.assertEqual(c2.name, "roundtrip")
        self.assertEqual(c2.model, "test-model")
        self.assertEqual(c2.max_tokens, 512)


# ============================================================================
# AgentStats/Context/RunResult测试
# ============================================================================

class TestStatsContextResult(unittest.TestCase):
    """测试统计/上下文/结果"""

    def test01_stats_creation(self):
        s = AgentStats()
        self.assertEqual(s.total_runs, 0)
        self.assertEqual(s.successful_runs, 0)

    def test02_stats_to_dict(self):
        s = AgentStats(total_runs=10, successful_runs=8)
        d = s.to_dict()
        self.assertEqual(d["total_runs"], 10)
        self.assertEqual(d["successful_runs"], 8)
        self.assertEqual(d["success_rate"], 0.8)

    def test03_context_creation(self):
        ctx = AgentContext(agent_id="a1")
        self.assertEqual(ctx.agent_id, "a1")
        self.assertEqual(len(ctx.messages), 0)

    def test04_context_messages(self):
        ctx = AgentContext(agent_id="a1")
        ctx.add_user_message("hello")
        ctx.add_assistant_message("hi")
        self.assertEqual(len(ctx.messages), 2)
        self.assertEqual(ctx.messages[0]["role"], "user")
        self.assertEqual(ctx.messages[1]["role"], "assistant")

    def test05_context_variables(self):
        ctx = AgentContext(agent_id="a1")
        ctx.set_variable("key", "value")
        self.assertEqual(ctx.get_variable("key"), "value")
        self.assertIsNone(ctx.get_variable("nonexistent"))

    def test06_context_clear(self):
        ctx = AgentContext(agent_id="a1")
        ctx.add_user_message("test")
        ctx.set_variable("k", "v")
        ctx.clear_messages()
        self.assertEqual(len(ctx.messages), 0)
        # 变量不被清空
        self.assertEqual(ctx.get_variable("k"), "v")

    def test07_context_get_messages_limit(self):
        ctx = AgentContext(agent_id="a1")
        for i in range(10):
            ctx.add_user_message(f"msg{i}")
        msgs = ctx.get_messages(max_count=3)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[-1]["content"], "msg9")

    def test08_run_result(self):
        r = AgentRunResult(
            agent_id="a1", session_id="s1", success=True,
            output="done", tokens_used=50,
        )
        self.assertTrue(r.success)
        self.assertEqual(r.output, "done")

    def test09_run_result_to_dict(self):
        r = AgentRunResult(agent_id="a1", session_id="s1", success=False,
                          error="bad")
        d = r.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["error"], "bad")

    def test10_agent_event(self):
        e = AgentEvent(agent_id="a1", event_type=AgentEventType.STATE_CHANGE,
                      data={"from": "created", "to": "configured"})
        d = e.to_dict()
        self.assertEqual(d["agent_id"], "a1")
        self.assertEqual(d["event_type"], "state_change")
        self.assertEqual(d["data"]["from"], "created")


# ============================================================================
# BaseAgent生命周期测试
# ============================================================================

class TestBaseAgentLifecycle(unittest.TestCase):
    """测试BaseAgent生命周期"""

    def setUp(self):
        reset_tools()
        reset_memory_manager()
        reset_planner()
        reset_reflector()
        reset_decision_maker()
        reset_autonomous_agent()

    def tearDown(self):
        reset_tools()
        reset_memory_manager()
        reset_planner()
        reset_reflector()
        reset_decision_maker()
        reset_autonomous_agent()

    def test01_creation(self):
        agent = BaseAgent()
        self.assertEqual(agent.state, AgentState.CREATED)
        self.assertIsNotNone(agent.agent_id)
        self.assertIsInstance(agent.config, AgentConfig)

    def test02_creation_with_config(self):
        config = AgentConfig(name="custom", description="自定义")
        agent = BaseAgent(config=config)
        self.assertEqual(agent.config.name, "custom")
        self.assertEqual(agent.config.description, "自定义")

    def test03_configure(self):
        agent = BaseAgent()
        self.assertTrue(agent.configure())
        self.assertEqual(agent.state, AgentState.CONFIGURED)

    def test04_start(self):
        agent = BaseAgent()
        agent.configure()
        self.assertTrue(agent.start())
        self.assertEqual(agent.state, AgentState.RUNNING)

    def test05_start_without_configure(self):
        """未配置直接启动会自动配置"""
        agent = BaseAgent()
        self.assertTrue(agent.start())
        self.assertEqual(agent.state, AgentState.RUNNING)

    def test06_pause_resume(self):
        agent = BaseAgent()
        agent.start()
        self.assertTrue(agent.pause())
        self.assertEqual(agent.state, AgentState.PAUSED)
        self.assertTrue(agent.resume())
        self.assertEqual(agent.state, AgentState.RUNNING)

    def test07_stop(self):
        agent = BaseAgent()
        agent.start()
        self.assertTrue(agent.stop())
        self.assertEqual(agent.state, AgentState.STOPPED)

    def test08_terminate(self):
        agent = BaseAgent()
        agent.start()
        self.assertTrue(agent.terminate())
        self.assertEqual(agent.state, AgentState.TERMINATED)

    def test09_terminated_cannot_restart(self):
        """终止后不能重启"""
        agent = BaseAgent()
        agent.start()
        agent.terminate()
        self.assertFalse(agent.start())
        self.assertEqual(agent.state, AgentState.TERMINATED)

    def test10_reset(self):
        agent = BaseAgent()
        agent.start()
        agent.stop()
        self.assertTrue(agent.reset())
        self.assertEqual(agent.state, AgentState.CONFIGURED)

    def test11_illegal_transition(self):
        """非法状态转换"""
        agent = BaseAgent()
        # CREATED不能直接到RUNNING
        self.assertFalse(agent._transition_to(AgentState.RUNNING))
        self.assertEqual(agent.state, AgentState.CREATED)

    def test12_can_transition_to(self):
        agent = BaseAgent()
        self.assertTrue(agent.can_transition_to(AgentState.CONFIGURED))
        self.assertFalse(agent.can_transition_to(AgentState.RUNNING))


# ============================================================================
# BaseAgent运行测试
# ============================================================================

class TestBaseAgentRun(unittest.TestCase):
    """测试BaseAgent运行"""

    def setUp(self):
        reset_tools()
        reset_memory_manager()

    def tearDown(self):
        reset_tools()
        reset_memory_manager()

    def test01_run_echo_agent(self):
        """测试EchoAgent运行"""
        agent = EchoAgent()
        result = agent.run("hello")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Echo: hello")
        self.assertEqual(agent.stats.total_runs, 1)
        self.assertEqual(agent.stats.successful_runs, 1)

    def test02_run_auto_start(self):
        """运行时自动启动"""
        agent = EchoAgent()
        self.assertEqual(agent.state, AgentState.CREATED)
        agent.run("test")
        self.assertEqual(agent.state, AgentState.RUNNING)

    def test03_run_failing_agent(self):
        """测试失败Agent"""
        agent = FailingAgent()
        result = agent.run("test")
        self.assertFalse(result.success)
        self.assertIn("故意失败", result.error)
        self.assertEqual(agent.state, AgentState.ERROR)
        self.assertEqual(agent.stats.errors, 1)

    def test04_run_multiple(self):
        """多次运行"""
        agent = EchoAgent()
        for i in range(5):
            agent.run(f"msg{i}")
        self.assertEqual(agent.stats.total_runs, 5)
        self.assertEqual(agent.stats.successful_runs, 5)
        stats = agent.get_stats()
        self.assertEqual(stats["success_rate"], 1.0)

    def test05_run_stats(self):
        """运行统计"""
        agent = EchoAgent()
        agent.run("test1")
        agent.run("test2")
        stats = agent.get_stats()
        self.assertEqual(stats["total_runs"], 2)
        self.assertEqual(stats["successful_runs"], 2)
        self.assertGreater(stats["total_latency_ms"], 0)


# ============================================================================
# BaseAgent事件系统测试
# ============================================================================

class TestBaseAgentEvents(unittest.TestCase):
    """测试BaseAgent事件系统"""

    def test01_event_listener(self):
        """测试事件监听"""
        agent = BaseAgent()
        events = []
        agent.add_event_listener(lambda e: events.append(e))
        agent.configure()
        # 应该收到状态转换事件
        state_events = [e for e in events
                       if e.event_type == AgentEventType.STATE_CHANGE]
        self.assertGreater(len(state_events), 0)

    def test02_remove_event_listener(self):
        """测试移除监听器"""
        agent = BaseAgent()
        events = []
        listener = lambda e: events.append(e)
        agent.add_event_listener(listener)
        agent.remove_event_listener(listener)
        agent.configure()
        self.assertEqual(len(events), 0)

    def test03_error_event(self):
        """测试错误事件"""
        agent = FailingAgent()
        events = []
        agent.add_event_listener(lambda e: events.append(e))
        agent.run("test")
        error_events = [e for e in events
                       if e.event_type == AgentEventType.ERROR]
        self.assertGreater(len(error_events), 0)


# ============================================================================
# BaseAgent钩子测试
# ============================================================================

class TestBaseAgentHooks(unittest.TestCase):
    """测试BaseAgent钩子"""

    def test01_before_after_run_hooks(self):
        """测试运行前后钩子"""
        agent = EchoAgent()
        calls = []
        agent.add_hook("before_run", lambda a, **kw: calls.append("before"))
        agent.add_hook("after_run", lambda a, **kw: calls.append("after"))
        agent.run("test")
        self.assertIn("before", calls)
        self.assertIn("after", calls)
        self.assertLess(calls.index("before"), calls.index("after"))

    def test02_lifecycle_hooks(self):
        """测试生命周期钩子"""
        agent = BaseAgent()
        calls = []
        agent.add_hook("before_configure", lambda a, **kw: calls.append("before_cfg"))
        agent.add_hook("after_configure", lambda a, **kw: calls.append("after_cfg"))
        agent.configure()
        self.assertIn("before_cfg", calls)
        self.assertIn("after_cfg", calls)

    def test03_error_hook(self):
        """测试错误钩子"""
        agent = FailingAgent()
        errors = []
        agent.add_hook("on_error", lambda a, **kw: errors.append(kw.get("error")))
        agent.run("test")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)

    def test04_remove_hook(self):
        """测试移除钩子"""
        agent = EchoAgent()
        calls = []
        def hook(a, **kw):
            calls.append("called")
        agent.add_hook("before_run", hook)
        agent.remove_hook("before_run", hook)
        agent.run("test")
        self.assertEqual(len(calls), 0)


# ============================================================================
# BaseAgent上下文与记忆测试
# ============================================================================

class TestBaseAgentContextMemory(unittest.TestCase):
    """测试上下文与记忆"""

    def setUp(self):
        reset_memory_manager()

    def tearDown(self):
        reset_memory_manager()

    def test01_add_message(self):
        agent = BaseAgent()
        agent.add_message("user", "hello")
        agent.add_message("assistant", "hi")
        self.assertEqual(len(agent.context.messages), 2)

    def test02_clear_context(self):
        agent = BaseAgent()
        agent.add_message("user", "test")
        agent.set_variable("k", "v")
        agent.clear_context()
        self.assertEqual(len(agent.context.messages), 0)

    def test03_set_get_variable(self):
        agent = BaseAgent()
        agent.set_variable("key", "value")
        self.assertEqual(agent.get_variable("key"), "value")
        self.assertIsNone(agent.get_variable("nonexistent"))

    def test04_remember_recall(self):
        """测试记忆存储与检索"""
        agent = BaseAgent()
        agent.remember("重要信息: SEVA是自驱永续进化系统",
                      MemoryType.SEMANTIC, importance=0.9)
        results = agent.recall("SEVA")
        self.assertGreater(len(results), 0)

    def test05_remember_disabled(self):
        """禁用记忆时不存储"""
        config = AgentConfig(enable_memory=False)
        agent = BaseAgent(config=config)
        agent.remember("test", MemoryType.EPISODIC)
        results = agent.recall("test")
        self.assertEqual(len(results), 0)


# ============================================================================
# BaseAgent状态查询测试
# ============================================================================

class TestBaseAgentStateQueries(unittest.TestCase):
    """测试状态查询"""

    def test01_is_running(self):
        agent = BaseAgent()
        self.assertFalse(agent.is_running())
        agent.start()
        self.assertTrue(agent.is_running())

    def test02_is_paused(self):
        agent = BaseAgent()
        agent.start()
        agent.pause()
        self.assertTrue(agent.is_paused())

    def test03_is_stopped(self):
        agent = BaseAgent()
        agent.start()
        agent.stop()
        self.assertTrue(agent.is_stopped())

    def test04_is_terminated(self):
        agent = BaseAgent()
        agent.start()
        agent.terminate()
        self.assertTrue(agent.is_terminated())

    def test05_get_info(self):
        agent = BaseAgent(config=AgentConfig(name="test"))
        info = agent.get_info()
        self.assertEqual(info["name"], "test")
        self.assertEqual(info["state"], "created")
        self.assertIn("stats", info)
        self.assertIn("config", info)

    def test06_to_dict(self):
        agent = BaseAgent()
        d = agent.to_dict()
        self.assertIn("agent_id", d)
        self.assertIn("config", d)
        self.assertIn("state", d)


# ============================================================================
# AgentRunner测试
# ============================================================================

class TestAgentRunner(unittest.TestCase):
    """测试AgentRunner"""

    def setUp(self):
        reset_agent_runner()

    def tearDown(self):
        reset_agent_runner()

    def test01_creation(self):
        runner = AgentRunner()
        self.assertEqual(runner.count(), 0)

    def test02_register_unregister(self):
        runner = AgentRunner()
        agent = BaseAgent()
        self.assertTrue(runner.register(agent))
        self.assertEqual(runner.count(), 1)
        self.assertTrue(runner.unregister(agent.agent_id))
        self.assertEqual(runner.count(), 0)

    def test03_get_agent(self):
        runner = AgentRunner()
        agent = BaseAgent()
        runner.register(agent)
        self.assertIs(runner.get(agent.agent_id), agent)
        self.assertIsNone(runner.get("nonexistent"))

    def test04_list_agents(self):
        runner = AgentRunner()
        a1 = BaseAgent()
        a2 = BaseAgent()
        runner.register(a1)
        runner.register(a2)
        agents = runner.list_agents()
        self.assertEqual(len(agents), 2)
        self.assertIn(a1.agent_id, agents)

    def test05_list_by_state(self):
        runner = AgentRunner()
        a1 = BaseAgent()
        a2 = BaseAgent()
        runner.register(a1)
        runner.register(a2)
        a1.start()
        running = runner.list_agents(state=AgentState.RUNNING)
        self.assertEqual(len(running), 1)
        self.assertIn(a1.agent_id, running)

    def test06_start_stop_all(self):
        runner = AgentRunner()
        a1 = BaseAgent()
        a2 = BaseAgent()
        runner.register(a1)
        runner.register(a2)
        results = runner.start_all()
        self.assertTrue(all(results.values()))
        self.assertEqual(a1.state, AgentState.RUNNING)
        self.assertEqual(a2.state, AgentState.RUNNING)
        results = runner.stop_all()
        self.assertTrue(all(results.values()))

    def test07_terminate_all(self):
        runner = AgentRunner()
        a1 = BaseAgent()
        runner.register(a1)
        a1.start()
        runner.terminate_all()
        self.assertTrue(a1.is_terminated())

    def test08_events(self):
        """测试事件收集"""
        runner = AgentRunner()
        agent = BaseAgent()
        runner.register(agent)
        agent.configure()
        events = runner.get_events(agent_id=agent.agent_id)
        self.assertGreater(len(events), 0)

    def test09_filter_events(self):
        """测试事件过滤"""
        runner = AgentRunner()
        agent = BaseAgent()
        runner.register(agent)
        agent.configure()
        agent.start()
        state_events = runner.get_events(
            agent_id=agent.agent_id,
            event_type=AgentEventType.STATE_CHANGE,
        )
        self.assertGreater(len(state_events), 0)
        for e in state_events:
            self.assertEqual(e.event_type, AgentEventType.STATE_CHANGE)

    def test10_clear_events(self):
        runner = AgentRunner()
        agent = BaseAgent()
        runner.register(agent)
        agent.configure()
        runner.clear_events()
        self.assertEqual(len(runner.get_events()), 0)

    def test11_get_report(self):
        runner = AgentRunner()
        a1 = BaseAgent()
        a2 = BaseAgent()
        runner.register(a1)
        runner.register(a2)
        a1.start()
        report = runner.get_report()
        self.assertEqual(report["total_agents"], 2)
        self.assertIn("running", report["states"])
        self.assertEqual(len(report["agents"]), 2)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingletons(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_agent_runner()

    def tearDown(self):
        reset_agent_runner()

    def test01_runner_singleton(self):
        r1 = get_agent_runner()
        r2 = get_agent_runner()
        self.assertIs(r1, r2)

    def test02_reset_runner(self):
        r1 = get_agent_runner()
        reset_agent_runner()
        r2 = get_agent_runner()
        self.assertIsNot(r1, r2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_agent_runner()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def tearDown(self):
        reset_agent_runner()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def test01_full_lifecycle(self):
        """完整生命周期测试"""
        agent = EchoAgent(config=AgentConfig(name="lifecycle_test"))
        runner = get_agent_runner()
        runner.register(agent)

        # 配置→启动→运行→暂停→恢复→停止
        self.assertTrue(agent.configure())
        self.assertTrue(agent.start())
        result = agent.run("test")
        self.assertTrue(result.success)
        self.assertTrue(agent.pause())
        self.assertTrue(agent.resume())
        result = agent.run("test2")
        self.assertTrue(result.success)
        self.assertTrue(agent.stop())

        # 检查统计
        stats = agent.get_stats()
        self.assertEqual(stats["total_runs"], 2)
        self.assertEqual(stats["successful_runs"], 2)

    def test02_multiple_agents(self):
        """多Agent管理"""
        runner = get_agent_runner()
        a1 = EchoAgent(config=AgentConfig(name="agent1"))
        a2 = EchoAgent(config=AgentConfig(name="agent2"))
        runner.register(a1)
        runner.register(a2)

        runner.start_all()
        r1 = a1.run("hello1")
        r2 = a2.run("hello2")
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)

        report = runner.get_report()
        self.assertEqual(report["total_agents"], 2)

    def test03_with_hooks_and_events(self):
        """钩子与事件集成"""
        agent = EchoAgent()
        runner = get_agent_runner()
        runner.register(agent)

        hook_calls = []
        agent.add_hook("before_run", lambda a, **kw: hook_calls.append("before"))
        agent.add_hook("after_run", lambda a, **kw: hook_calls.append("after"))

        agent.start()
        agent.run("test")

        self.assertEqual(len(hook_calls), 2)
        events = runner.get_events(agent_id=agent.agent_id)
        self.assertGreater(len(events), 0)

    def test04_real_llm_agent(self):
        """真实LLM Agent测试"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        from seva.llm import Message
        from seva.agents import BaseAgent, AgentConfig, AgentRunResult

        # 创建一个使用真实LLM的Agent
        class LLMAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                client = self.llm_client
                original_cache = client.use_cache
                client.use_cache = False
                try:
                    response = client.chat(
                        [Message.user(str(input))],
                        max_tokens=self.config.max_tokens,
                    )
                finally:
                    client.use_cache = original_cache
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                    output=response.content,
                    tokens_used=response.usage.total_tokens,
                )

        agent = LLMAgent(config=AgentConfig(name="llm_agent", max_tokens=100))
        agent.start()
        result = agent.run("说一个字: 好")
        self.assertTrue(result.success)
        self.assertGreater(len(result.output), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
