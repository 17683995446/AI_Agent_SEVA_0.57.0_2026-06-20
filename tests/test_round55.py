"""
阶段65测试：对话Agent

覆盖：
    - Session/SessionStatus: 会话数据类
    - SessionManager: 会话管理器
    - ConversationConfig: 对话配置
    - ConversationAgent: 对话Agent
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import json
import time
from unittest.mock import patch, MagicMock

from seva.agents import (
    Session, SessionStatus, SessionManager,
    ConversationConfig, ConversationAgent,
    get_conversation_agent, reset_conversation_agent,
)
from seva.agents import AgentRunResult
from seva.llm import LLMError, reset_llm_client
from seva.tools import reset_tools
from seva.memory_llm import reset_memory_manager


# ============================================================================
# Session测试
# ============================================================================

class TestSession(unittest.TestCase):
    """测试Session"""

    def test01_creation(self):
        s = Session()
        self.assertIsNotNone(s.session_id)
        self.assertEqual(s.status, SessionStatus.ACTIVE)
        self.assertEqual(s.message_count, 0)

    def test02_add_messages(self):
        s = Session()
        s.add_user_message("hello", tokens=5)
        s.add_assistant_message("hi", tokens=3)
        self.assertEqual(s.message_count, 2)
        self.assertEqual(s.total_tokens, 8)

    def test03_get_messages(self):
        s = Session()
        s.add_system_message("system")
        s.add_user_message("u1")
        s.add_assistant_message("a1")
        msgs = s.get_messages()
        self.assertEqual(len(msgs), 3)

    def test04_get_messages_no_system(self):
        s = Session()
        s.add_system_message("system")
        s.add_user_message("u1")
        msgs = s.get_messages(include_system=False)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test05_get_recent_messages(self):
        s = Session()
        for i in range(10):
            s.add_user_message(f"msg{i}")
        recent = s.get_recent_messages(count=3)
        self.assertEqual(len(recent), 3)

    def test06_clear_messages(self):
        s = Session()
        s.add_user_message("test")
        s.clear_messages()
        self.assertEqual(s.message_count, 0)
        self.assertEqual(s.total_tokens, 0)

    def test07_close(self):
        s = Session()
        s.close()
        self.assertEqual(s.status, SessionStatus.CLOSED)
        self.assertIsNotNone(s.closed_at)

    def test08_archive(self):
        s = Session()
        s.archive()
        self.assertEqual(s.status, SessionStatus.ARCHIVED)

    def test09_to_dict(self):
        s = Session(title="test")
        s.add_user_message("hello")
        d = s.to_dict()
        self.assertEqual(d["title"], "test")
        self.assertEqual(d["message_count"], 1)

    def test10_from_dict(self):
        d = {"session_id": "s1", "title": "from dict", "message_count": 5}
        s = Session.from_dict(d)
        self.assertEqual(s.session_id, "s1")
        self.assertEqual(s.title, "from dict")
        self.assertEqual(s.message_count, 5)

    def test11_roundtrip(self):
        s1 = Session(title="roundtrip")
        s1.add_user_message("test")
        d = s1.to_dict()
        s2 = Session.from_dict(d)
        self.assertEqual(s2.title, "roundtrip")
        self.assertEqual(s2.message_count, 1)


# ============================================================================
# SessionManager测试
# ============================================================================

class TestSessionManager(unittest.TestCase):
    """测试SessionManager"""

    def test01_creation(self):
        sm = SessionManager()
        self.assertEqual(sm.count(), 0)

    def test02_create_session(self):
        sm = SessionManager()
        s = sm.create_session(user_id="u1", title="test")
        self.assertIsNotNone(s)
        self.assertEqual(sm.count(), 1)
        self.assertEqual(s.user_id, "u1")

    def test03_get_session(self):
        sm = SessionManager()
        s = sm.create_session()
        retrieved = sm.get_session(s.session_id)
        self.assertIs(retrieved, s)
        self.assertIsNone(sm.get_session("nonexistent"))

    def test04_close_session(self):
        sm = SessionManager()
        s = sm.create_session()
        self.assertTrue(sm.close_session(s.session_id))
        self.assertEqual(s.status, SessionStatus.CLOSED)
        # 不能重复关闭
        self.assertFalse(sm.close_session(s.session_id))

    def test05_archive_session(self):
        sm = SessionManager()
        s = sm.create_session()
        self.assertTrue(sm.archive_session(s.session_id))
        self.assertEqual(s.status, SessionStatus.ARCHIVED)

    def test06_delete_session(self):
        sm = SessionManager()
        s = sm.create_session()
        self.assertTrue(sm.delete_session(s.session_id))
        self.assertEqual(sm.count(), 0)
        self.assertFalse(sm.delete_session(s.session_id))

    def test07_list_sessions(self):
        sm = SessionManager()
        s1 = sm.create_session(user_id="u1")
        s2 = sm.create_session(user_id="u2")
        all_sessions = sm.list_sessions()
        self.assertEqual(len(all_sessions), 2)
        u1_sessions = sm.list_sessions(user_id="u1")
        self.assertEqual(len(u1_sessions), 1)

    def test08_list_by_status(self):
        sm = SessionManager()
        s1 = sm.create_session()
        s2 = sm.create_session()
        sm.close_session(s2.session_id)
        active = sm.list_sessions(status=SessionStatus.ACTIVE)
        self.assertEqual(len(active), 1)

    def test09_max_sessions(self):
        """测试最大会话数限制"""
        sm = SessionManager(max_sessions=3)
        s1 = sm.create_session()
        s2 = sm.create_session()
        s3 = sm.create_session()
        # 第4个应该触发清理
        s4 = sm.create_session()
        self.assertLessEqual(sm.count(), 3)

    def test10_get_stats(self):
        sm = SessionManager()
        sm.create_session()
        sm.create_session()
        stats = sm.get_stats()
        self.assertEqual(stats["sessions_created"], 2)
        self.assertEqual(stats["active_sessions"], 2)

    def test11_clear_all(self):
        sm = SessionManager()
        sm.create_session()
        sm.create_session()
        sm.clear_all()
        self.assertEqual(sm.count(), 0)


# ============================================================================
# ConversationConfig测试
# ============================================================================

class TestConversationConfig(unittest.TestCase):
    """测试ConversationConfig"""

    def test01_creation(self):
        c = ConversationConfig()
        self.assertEqual(c.name, "conversation_agent")
        self.assertTrue(c.enable_context_compression)
        self.assertTrue(c.enable_memory_integration)

    def test02_custom_config(self):
        c = ConversationConfig(
            name="custom",
            max_history_messages=10,
            context_compression_threshold=8,
        )
        self.assertEqual(c.name, "custom")
        self.assertEqual(c.max_history_messages, 10)
        self.assertEqual(c.context_compression_threshold, 8)


# ============================================================================
# ConversationAgent测试
# ============================================================================

class TestConversationAgent(unittest.TestCase):
    """测试ConversationAgent"""

    def setUp(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_conversation_agent()

    def tearDown(self):
        reset_tools()
        reset_memory_manager()
        reset_llm_client()
        reset_conversation_agent()

    def test01_creation(self):
        agent = ConversationAgent()
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.config, ConversationConfig)
        self.assertIsNotNone(agent.session_manager)

    def test02_creation_with_config(self):
        config = ConversationConfig(name="custom", max_history_messages=5)
        agent = ConversationAgent(config=config)
        self.assertEqual(agent.config.name, "custom")
        self.assertEqual(agent.conv_config.max_history_messages, 5)

    def test03_new_session(self):
        agent = ConversationAgent()
        session = agent.new_session(user_id="u1", title="test")
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, "u1")
        self.assertIs(agent.get_current_session(), session)

    def test04_switch_session(self):
        agent = ConversationAgent()
        s1 = agent.new_session(title="s1")
        s2 = agent.new_session(title="s2")
        self.assertIs(agent.get_current_session(), s2)
        self.assertTrue(agent.switch_session(s1.session_id))
        self.assertIs(agent.get_current_session(), s1)

    def test05_close_session(self):
        agent = ConversationAgent()
        session = agent.new_session()
        self.assertTrue(agent.close_session())
        self.assertIsNone(agent.get_current_session())

    def test06_list_sessions(self):
        agent = ConversationAgent()
        agent.new_session(title="s1")
        agent.new_session(title="s2")
        sessions = agent.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test07_get_session_history(self):
        agent = ConversationAgent()
        session = agent.new_session()
        session.add_user_message("hello")
        session.add_assistant_message("hi")
        history = agent.get_session_history(session.session_id)
        self.assertEqual(len(history), 2)

    def test08_chat_with_mock_llm(self):
        """Mock LLM对话"""
        agent = ConversationAgent()

        # Mock LLM客户端
        mock_client = MagicMock()
        response = MagicMock()
        response.content = "你好！我是AI助手。"
        response.usage.total_tokens = 20
        mock_client.chat.return_value = response
        agent._llm_client = mock_client

        agent.start()
        result = agent.chat("你好")

        self.assertTrue(result.success)
        self.assertEqual(result.output, "你好！我是AI助手。")
        self.assertEqual(result.tokens_used, 20)

        # 验证会话有消息
        session = agent.get_current_session()
        self.assertEqual(session.message_count, 2)  # user + assistant

    def test09_chat_multiple_turns(self):
        """多轮对话"""
        agent = ConversationAgent()

        mock_client = MagicMock()
        responses = [
            MagicMock(content="第一轮回答", usage=MagicMock(total_tokens=10)),
            MagicMock(content="第二轮回答", usage=MagicMock(total_tokens=15)),
        ]
        mock_client.chat.side_effect = responses
        agent._llm_client = mock_client

        agent.start()
        r1 = agent.chat("问题1")
        r2 = agent.chat("问题2")

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        session = agent.get_current_session()
        self.assertEqual(session.message_count, 4)  # 2轮 × 2消息

    def test10_chat_llm_error(self):
        """LLM错误处理"""
        agent = ConversationAgent()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("API错误")
        agent._llm_client = mock_client

        agent.start()
        result = agent.chat("test")
        self.assertFalse(result.success)
        self.assertIn("API错误", result.error)

    def test11_chat_empty_message(self):
        """空消息处理"""
        agent = ConversationAgent()
        agent.start()
        result = agent.chat("")
        self.assertFalse(result.success)
        self.assertIn("空", result.error)

    def test12_chat_with_specific_session(self):
        """指定会话对话"""
        agent = ConversationAgent()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "回答"
        response.usage.total_tokens = 10
        mock_client.chat.return_value = response
        agent._llm_client = mock_client

        session = agent.new_session(title="specific")
        agent.start()
        result = agent.chat("test", session_id=session.session_id)
        self.assertTrue(result.success)
        # 应该使用了同一个会话
        self.assertEqual(agent.get_current_session().session_id, session.session_id)

    def test13_context_compression(self):
        """测试上下文压缩"""
        config = ConversationConfig(
            context_compression_threshold=4,
            max_history_messages=10,
        )
        agent = ConversationAgent(config=config)

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "回答"
        response.usage.total_tokens = 5
        mock_client.chat.return_value = response
        agent._llm_client = mock_client

        agent.start()
        # 进行5轮对话(超过阈值4)
        for i in range(5):
            agent.chat(f"问题{i}")

        session = agent.get_current_session()
        # 应该触发了压缩
        self.assertLess(session.message_count, 10)

    def test14_memory_integration_disabled(self):
        """禁用记忆整合"""
        config = ConversationConfig(enable_memory_integration=False)
        agent = ConversationAgent(config=config)

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "回答"
        response.usage.total_tokens = 5
        mock_client.chat.return_value = response
        agent._llm_client = mock_client

        agent.start()
        agent.chat("test")
        # 不应该报错
        self.assertTrue(True)

    def test15_chat_with_tools(self):
        """带工具的对话"""
        agent = ConversationAgent(config=ConversationConfig(enable_tools=True))

        # Mock tool_loop
        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "计算结果是4",
            "tool_calls": [{"tool_name": "calculator", "status": "success"}],
            "iterations": 1,
            "total_tokens": 30,
        }
        agent._tool_loop = mock_loop

        agent.start()
        result = agent.chat("计算2+2", use_tools=True)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "计算结果是4")
        self.assertEqual(len(result.tool_calls), 1)

    def test16_get_info(self):
        agent = ConversationAgent()
        info = agent.get_info()
        self.assertEqual(info["name"], "conversation_agent")


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingletons(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_conversation_agent()

    def tearDown(self):
        reset_conversation_agent()

    def test01_singleton(self):
        a1 = get_conversation_agent()
        a2 = get_conversation_agent()
        self.assertIs(a1, a2)

    def test02_reset(self):
        a1 = get_conversation_agent()
        reset_conversation_agent()
        a2 = get_conversation_agent()
        self.assertIsNot(a1, a2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_conversation_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def tearDown(self):
        reset_conversation_agent()
        reset_tools()
        reset_memory_manager()
        reset_llm_client()

    def test01_full_conversation(self):
        """完整对话流程"""
        agent = ConversationAgent()

        mock_client = MagicMock()
        responses = [
            MagicMock(content="你好！", usage=MagicMock(total_tokens=10)),
            MagicMock(content="我是AI助手", usage=MagicMock(total_tokens=15)),
            MagicMock(content="再见！", usage=MagicMock(total_tokens=5)),
        ]
        mock_client.chat.side_effect = responses
        agent._llm_client = mock_client

        agent.start()

        # 创建会话
        session = agent.new_session(user_id="user1", title="测试对话")

        # 多轮对话
        r1 = agent.chat("你好", session_id=session.session_id)
        r2 = agent.chat("你是谁", session_id=session.session_id)
        r3 = agent.chat("再见", session_id=session.session_id)

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        self.assertTrue(r3.success)

        # 验证会话历史
        history = agent.get_session_history(session.session_id)
        self.assertEqual(len(history), 6)  # 3轮 × 2消息

        # 验证统计
        stats = agent.get_stats()
        self.assertEqual(stats["total_runs"], 3)
        self.assertEqual(stats["successful_runs"], 3)

    def test02_multiple_sessions(self):
        """多会话管理"""
        agent = ConversationAgent()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "回答"
        response.usage.total_tokens = 10
        mock_client.chat.return_value = response
        agent._llm_client = mock_client

        agent.start()

        # 创建多个会话
        s1 = agent.new_session(user_id="u1", title="会话1")
        s2 = agent.new_session(user_id="u2", title="会话2")

        # 在不同会话中对话
        agent.chat("msg1", session_id=s1.session_id)
        agent.chat("msg2", session_id=s2.session_id)

        # 验证会话独立
        h1 = agent.get_session_history(s1.session_id)
        h2 = agent.get_session_history(s2.session_id)
        self.assertEqual(len(h1), 2)
        self.assertEqual(len(h2), 2)

    def test03_real_llm_conversation(self):
        """真实LLM对话测试"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        agent = ConversationAgent(config=ConversationConfig(max_tokens=100))
        agent.start()

        r1 = agent.chat("你好,请用一句话介绍自己")
        self.assertTrue(r1.success)
        self.assertGreater(len(r1.output), 0)

        r2 = agent.chat("我刚才问了什么?")
        self.assertTrue(r2.success)
        # 应该能记住上文
        self.assertGreater(len(r2.output), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
