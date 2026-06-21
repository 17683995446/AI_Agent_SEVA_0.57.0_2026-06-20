"""
阶段58测试：LLM客户端与API对接
"""
import unittest
import os
import time
import json
import threading
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

from seva.llm import (
    Role, Message, TokenUsage, ChatRequest, ChatResponse, StreamChunk,
    LLMError, RateLimitError, QuotaExhaustedError, AuthenticationError,
    ContentFilterError, ModelOverloadedError, NetworkError, ContextLengthError,
    RetryConfig, RateLimiter, ResponseCache, LLMClient, LLMSession,
    get_llm_client, set_llm_client, chat, stream_chat,
    get_llm_stats, reset_llm_client,
)


# ============================================================================
# 测试枚举与数据结构
# ============================================================================

class TestRole(unittest.TestCase):
    """测试角色枚举"""

    def test01_values(self):
        self.assertEqual(Role.SYSTEM.value, "system")
        self.assertEqual(Role.USER.value, "user")
        self.assertEqual(Role.ASSISTANT.value, "assistant")
        self.assertEqual(Role.TOOL.value, "tool")


class TestMessage(unittest.TestCase):
    """测试消息"""

    def test01_creation(self):
        msg = Message(role=Role.USER, content="你好")
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "你好")
        self.assertIsNone(msg.name)
        self.assertIsNone(msg.tool_call_id)
        self.assertIsNone(msg.tool_calls)

    def test02_class_methods(self):
        sys_msg = Message.system("你是助手")
        self.assertEqual(sys_msg.role, Role.SYSTEM)
        user_msg = Message.user("你好")
        self.assertEqual(user_msg.role, Role.USER)
        assistant_msg = Message.assistant("您好")
        self.assertEqual(assistant_msg.role, Role.ASSISTANT)
        tool_msg = Message.tool("结果", "call_123")
        self.assertEqual(tool_msg.role, Role.TOOL)
        self.assertEqual(tool_msg.tool_call_id, "call_123")

    def test03_to_dict(self):
        msg = Message(role=Role.USER, content="hi", name="alice")
        d = msg.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "hi")
        self.assertEqual(d["name"], "alice")

    def test04_repr(self):
        msg = Message.user("a" * 100)
        r = repr(msg)
        self.assertIn("user", r)
        self.assertIn("...", r)  # 长内容截断


class TestTokenUsage(unittest.TestCase):
    """测试Token使用统计"""

    def test01_creation(self):
        usage = TokenUsage()
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)

    def test02_add(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        a.add(b)
        self.assertEqual(a.prompt_tokens, 30)
        self.assertEqual(a.completion_tokens, 15)
        self.assertEqual(a.total_tokens, 45)

    def test03_to_from_dict(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        d = usage.to_dict()
        self.assertEqual(d["prompt_tokens"], 10)

        usage2 = TokenUsage.from_dict(d)
        self.assertEqual(usage2.prompt_tokens, 10)
        self.assertEqual(usage2.total_tokens, 15)

    def test04_repr(self):
        usage = TokenUsage(total_tokens=100)
        r = repr(usage)
        self.assertIn("100", r)


class TestChatRequest(unittest.TestCase):
    """测试聊天请求"""

    def test01_minimal(self):
        req = ChatRequest(messages=[Message.user("hi")])
        d = req.to_dict()
        self.assertEqual(d["model"], "minimax-m2.5")
        self.assertEqual(d["temperature"], 0.7)
        self.assertEqual(d["top_p"], 1.0)
        self.assertFalse(d["stream"])
        self.assertEqual(len(d["messages"]), 1)

    def test02_full(self):
        req = ChatRequest(
            messages=[Message.user("hi")],
            model="custom-model",
            temperature=0.5,
            max_tokens=100,
            stop=["END"],
            stream=True,
            response_format={"type": "json_object"},
            seed=42,
        )
        d = req.to_dict()
        self.assertEqual(d["model"], "custom-model")
        self.assertEqual(d["temperature"], 0.5)
        self.assertEqual(d["max_tokens"], 100)
        self.assertEqual(d["stop"], ["END"])
        self.assertTrue(d["stream"])
        self.assertEqual(d["response_format"]["type"], "json_object")
        self.assertEqual(d["seed"], 42)


class TestChatResponse(unittest.TestCase):
    """测试聊天响应"""

    def test01_creation(self):
        resp = ChatResponse(content="你好")
        self.assertEqual(resp.content, "你好")
        self.assertEqual(resp.role, "assistant")
        self.assertEqual(resp.finish_reason, "stop")

    def test02_to_dict(self):
        resp = ChatResponse(content="hi", usage=TokenUsage(total_tokens=10))
        d = resp.to_dict()
        self.assertEqual(d["content"], "hi")
        self.assertEqual(d["usage"]["total_tokens"], 10)


# ============================================================================
# 测试LLMError
# ============================================================================

class TestLLMError(unittest.TestCase):
    """测试LLM错误类"""

    def test01_base_error(self):
        err = LLMError("测试错误", status_code=500, retryable=True)
        self.assertEqual(str(err), "测试错误")
        self.assertEqual(err.status_code, 500)
        self.assertTrue(err.retryable)

    def test02_rate_limit_error(self):
        err = RateLimitError("限流", retry_after=2.0)
        self.assertEqual(err.status_code, 429)
        self.assertTrue(err.retryable)
        self.assertEqual(err.retry_after, 2.0)

    def test03_quota_error(self):
        err = QuotaExhaustedError("余额不足")
        self.assertEqual(err.status_code, 402)
        self.assertFalse(err.retryable)

    def test04_auth_error(self):
        err = AuthenticationError("认证失败")
        self.assertEqual(err.status_code, 401)
        self.assertFalse(err.retryable)

    def test05_content_filter(self):
        err = ContentFilterError("内容违规")
        self.assertFalse(err.retryable)

    def test06_overloaded(self):
        err = ModelOverloadedError("过载", status_code=503)
        self.assertEqual(err.status_code, 503)
        self.assertTrue(err.retryable)

    def test07_network(self):
        err = NetworkError("网络错误")
        self.assertTrue(err.retryable)

    def test08_context_length(self):
        err = ContextLengthError("上下文超限")
        self.assertFalse(err.retryable)

    def test09_repr(self):
        err = LLMError("错误", status_code=500)
        r = repr(err)
        self.assertIn("LLMError", r)
        self.assertIn("500", r)


# ============================================================================
# 测试RetryConfig
# ============================================================================

class TestRetryConfig(unittest.TestCase):
    """测试重试配置"""

    def test01_defaults(self):
        config = RetryConfig()
        self.assertEqual(config.max_retries, 3)

    def test02_exponential_delay(self):
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        self.assertAlmostEqual(config.get_delay(1), 1.0)
        self.assertAlmostEqual(config.get_delay(2), 2.0)
        self.assertAlmostEqual(config.get_delay(3), 4.0)

    def test03_max_delay_cap(self):
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter=False)
        self.assertLessEqual(config.get_delay(10), 5.0)

    def test04_to_dict(self):
        config = RetryConfig()
        d = config.to_dict()
        self.assertEqual(d["max_retries"], 3)


# ============================================================================
# 测试RateLimiter
# ============================================================================

class TestRateLimiter(unittest.TestCase):
    """测试限流器"""

    def setUp(self):
        self.limiter = RateLimiter(rpm=60, rps=2)

    def test01_initial(self):
        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["rpm_limit"], 60)
        self.assertEqual(stats["rps_limit"], 2)

    def test02_acquire(self):
        self.assertTrue(self.limiter.acquire(timeout=1.0))
        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_requests"], 1)

    def test03_multiple_acquires(self):
        for _ in range(3):
            self.assertTrue(self.limiter.acquire(timeout=1.0))
        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_requests"], 3)

    def test04_reset(self):
        self.limiter.acquire()
        self.limiter.reset()
        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["throttled_count"], 0)


# ============================================================================
# 测试ResponseCache
# ============================================================================

class TestResponseCache(unittest.TestCase):
    """测试响应缓存"""

    def setUp(self):
        self.cache = ResponseCache(max_size=10, ttl=60.0)

    def test01_initial(self):
        stats = self.cache.get_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

    def test02_put_and_get(self):
        req = ChatRequest(messages=[Message.user("hi")])
        resp = ChatResponse(content="hello")
        self.cache.put(req, resp)
        cached = self.cache.get(req)
        self.assertEqual(cached.content, "hello")

    def test03_miss(self):
        req = ChatRequest(messages=[Message.user("hi")])
        cached = self.cache.get(req)
        self.assertIsNone(cached)
        stats = self.cache.get_stats()
        self.assertEqual(stats["misses"], 1)

    def test04_different_requests_different_keys(self):
        req1 = ChatRequest(messages=[Message.user("hi1")])
        req2 = ChatRequest(messages=[Message.user("hi2")])
        self.cache.put(req1, ChatResponse(content="response1"))
        self.cache.put(req2, ChatResponse(content="response2"))
        self.assertEqual(self.cache.get(req1).content, "response1")
        self.assertEqual(self.cache.get(req2).content, "response2")

    def test05_cache_disabled_ttl_zero(self):
        cache = ResponseCache(max_size=10, ttl=0)
        req = ChatRequest(messages=[Message.user("hi")])
        cache.put(req, ChatResponse(content="hello"))
        self.assertIsNone(cache.get(req))

    def test06_lru_eviction(self):
        cache = ResponseCache(max_size=2, ttl=60.0)
        req1 = ChatRequest(messages=[Message.user("1")])
        req2 = ChatRequest(messages=[Message.user("2")])
        req3 = ChatRequest(messages=[Message.user("3")])
        cache.put(req1, ChatResponse(content="1"))
        time.sleep(0.01)
        cache.put(req2, ChatResponse(content="2"))
        time.sleep(0.01)
        cache.put(req3, ChatResponse(content="3"))
        # req1应该被淘汰
        self.assertIsNone(cache.get(req1))
        self.assertIsNotNone(cache.get(req2))
        self.assertIsNotNone(cache.get(req3))

    def test07_clear(self):
        req = ChatRequest(messages=[Message.user("hi")])
        self.cache.put(req, ChatResponse(content="hello"))
        self.cache.clear()
        self.assertIsNone(self.cache.get(req))

    def test08_stats_hit_rate(self):
        req = ChatRequest(messages=[Message.user("hi")])
        self.cache.get(req)  # miss
        self.cache.put(req, ChatResponse(content="hi"))
        self.cache.get(req)  # hit
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)


# ============================================================================
# 测试LLMClient（使用Mock，不调用真实API）
# ============================================================================

class TestLLMClient(unittest.TestCase):
    """测试LLM客户端"""

    def setUp(self):
        # 使用Mock API Key以避免真实调用
        self.client = LLMClient(
            api_key="test-key",
            model="test-model",
            timeout=5.0,
        )

    def test01_creation(self):
        self.assertEqual(self.client.api_key, "test-key")
        self.assertEqual(self.client.model, "test-model")
        self.assertEqual(self.client.timeout, 5.0)

    def test02_env_var_api_key(self):
        with patch.dict(os.environ, {"SEVA_LLM_API_KEY": "env-key"}):
            client = LLMClient()
            self.assertEqual(client.api_key, "env-key")

    def test03_normalize_messages_dict(self):
        messages = [{"role": "user", "content": "hi"}]
        norm = self.client._normalize_messages(messages)
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0].role, Role.USER)

    def test04_normalize_messages_mixed(self):
        messages = [Message.user("hi"), {"role": "assistant", "content": "hello"}]
        norm = self.client._normalize_messages(messages)
        self.assertEqual(len(norm), 2)

    def test05_normalize_messages_invalid(self):
        with self.assertRaises(TypeError):
            self.client._normalize_messages(["invalid"])

    def test06_map_http_error_rate_limit(self):
        err = self.client._map_http_error(429, {"error": {"message": "rate limited"}})
        self.assertIsInstance(err, RateLimitError)

    def test07_map_http_error_auth(self):
        err = self.client._map_http_error(401, {"error": {"message": "auth failed"}})
        self.assertIsInstance(err, AuthenticationError)

    def test08_map_http_error_quota(self):
        err = self.client._map_http_error(402, {"error": {"message": "no quota"}})
        self.assertIsInstance(err, QuotaExhaustedError)

    def test09_map_http_error_overloaded(self):
        err = self.client._map_http_error(503, {"error": {"message": "overloaded"}})
        self.assertIsInstance(err, ModelOverloadedError)

    def test10_map_http_error_context_length(self):
        err = self.client._map_http_error(400, {
            "error": {"message": "context too long", "type": "context_length"}
        })
        self.assertIsInstance(err, ContextLengthError)

    def test11_map_http_error_unknown_5xx(self):
        err = self.client._map_http_error(500, {"error": {"message": "internal"}})
        self.assertIsInstance(err, LLMError)
        self.assertTrue(err.retryable)

    def test12_map_http_error_4xx_not_retryable(self):
        err = self.client._map_http_error(400, {"error": {"message": "bad request"}})
        self.assertIsInstance(err, LLMError)
        self.assertFalse(err.retryable)

    def test13_parse_response(self):
        data = {
            "model": "test-model",
            "choices": [{
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        resp = self.client._parse_response(data)
        self.assertEqual(resp.content, "hello")
        self.assertEqual(resp.usage.total_tokens, 15)
        self.assertEqual(resp.finish_reason, "stop")

    def test14_parse_response_no_choices(self):
        with self.assertRaises(LLMError):
            self.client._parse_response({"choices": []})

    def test15_stats(self):
        stats = self.client.get_stats()
        self.assertEqual(stats["model"], "test-model")
        self.assertEqual(stats["total_requests"], 0)
        self.assertIn("rate_limiter", stats)

    def test16_repr(self):
        r = repr(self.client)
        self.assertIn("LLMClient", r)
        self.assertIn("test-model", r)

    def test17_chat_with_cache(self):
        cache = ResponseCache(ttl=60.0)
        client = LLMClient(
            api_key="test-key",
            model="test-model",
            use_cache=True,
            cache=cache,
        )

        req = ChatRequest(messages=[Message.user("hi")], model="test-model")
        expected = ChatResponse(content="cached response")
        cache.put(req, expected)

        response = client.chat([Message.user("hi")])
        self.assertEqual(response.content, "cached response")
        # 不应发起新请求
        self.assertEqual(client._total_requests, 0)


# ============================================================================
# 测试LLMSession
# ============================================================================

class TestLLMSession(unittest.TestCase):
    """测试LLM会话管理"""

    def setUp(self):
        # 使用Mock客户端
        self.mock_client = MagicMock(spec=LLMClient)
        self.mock_client.chat.return_value = ChatResponse(
            content="hello response",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        self.session = LLMSession(self.mock_client, system_prompt="你是助手", max_messages=10)

    def test01_creation_with_system_prompt(self):
        self.assertEqual(self.session.system_prompt, "你是助手")
        messages = self.session.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, Role.SYSTEM)

    def test02_creation_no_system_prompt(self):
        session = LLMSession(self.mock_client)
        self.assertIsNone(session.system_prompt)
        self.assertEqual(len(session.get_messages()), 0)

    def test03_add_messages(self):
        self.session.add_user_message("hi")
        self.session.add_assistant_message("hello")
        messages = self.session.get_messages()
        self.assertEqual(len(messages), 3)  # system + user + assistant

    def test04_chat(self):
        response = self.session.chat("你好")
        self.assertEqual(response.content, "hello response")
        messages = self.session.get_messages()
        self.assertEqual(len(messages), 3)  # system + user(你好) + assistant(hello response)

    def test05_max_messages_trim(self):
        # max_messages=10, 包含1个system，所以保留9个其他消息
        self.session.max_messages = 5
        for i in range(10):
            self.session.add_user_message(f"msg {i}")
        messages = self.session.get_messages()
        self.assertLessEqual(len(messages), 5)

    def test06_clear_preserves_system_prompt(self):
        self.session.add_user_message("hi")
        self.session.clear()
        messages = self.session.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, Role.SYSTEM)

    def test07_clear_no_system_prompt(self):
        session = LLMSession(self.mock_client)
        session.add_user_message("hi")
        session.clear()
        self.assertEqual(len(session.get_messages()), 0)

    def test08_get_usage(self):
        self.session.chat("hi")
        self.session.chat("hi again")
        usage = self.session.get_usage()
        self.assertEqual(usage.total_tokens, 30)  # 2次 * 15tokens

    def test09_get_info(self):
        info = self.session.get_info()
        self.assertEqual(info["message_count"], 1)
        self.assertEqual(info["system_prompt"], "你是助手")
        self.assertEqual(info["max_messages"], 10)

    def test10_repr(self):
        r = repr(self.session)
        self.assertIn("LLMSession", r)
        self.assertIn("messages=1", r)


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        reset_llm_client()

    def tearDown(self):
        reset_llm_client()

    def test01_get_llm_client(self):
        client = get_llm_client()
        self.assertIsInstance(client, LLMClient)

    def test02_singleton(self):
        c1 = get_llm_client()
        c2 = get_llm_client()
        self.assertIs(c1, c2)

    def test03_set_llm_client(self):
        custom = LLMClient(api_key="custom", model="custom-model")
        set_llm_client(custom)
        self.assertIs(get_llm_client(), custom)
        self.assertEqual(get_llm_client().model, "custom-model")

    def test04_get_llm_stats(self):
        stats = get_llm_stats()
        self.assertIn("total_requests", stats)
        self.assertIn("rate_limiter", stats)

    def test05_reset_llm_client(self):
        c1 = get_llm_client()
        reset_llm_client()
        c2 = get_llm_client()
        self.assertIsNot(c1, c2)

    def test06_chat_function_with_cache(self):
        cache = ResponseCache(ttl=60.0)
        client = LLMClient(
            api_key="test",
            model="minimax-m2.5",
            use_cache=True,
            cache=cache,
        )
        set_llm_client(client)

        req = ChatRequest(messages=[Message.user("hi")], model="minimax-m2.5")
        expected = ChatResponse(content="cached")
        cache.put(req, expected)

        response = chat([Message.user("hi")])
        self.assertEqual(response.content, "cached")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_llm_client()

    def tearDown(self):
        reset_llm_client()

    def test01_full_chat_workflow(self):
        """完整聊天工作流（使用Mock）"""
        client = LLMClient(
            api_key="test",
            model="test-model",
            timeout=5.0,
        )
        session = LLMSession(client, system_prompt="你是助手")

        # Mock chat
        client.chat = MagicMock(return_value=ChatResponse(
            content="你好！",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ))

        response = session.chat("你好")
        self.assertEqual(response.content, "你好！")
        self.assertGreater(session.get_usage().total_tokens, 0)

    def test02_retry_on_transient_error(self):
        """瞬态错误重试"""
        client = LLMClient(
            api_key="test",
            model="test",
            retry_config=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )

        call_count = [0]
        def mock_execute(req):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ModelOverloadedError("overloaded")
            return ChatResponse(content="success")

        client._execute = mock_execute
        response = client.chat([Message.user("hi")])
        self.assertEqual(response.content, "success")
        self.assertEqual(call_count[0], 2)

    def test03_no_retry_on_auth_error(self):
        """认证错误不重试"""
        client = LLMClient(
            api_key="test",
            model="test",
            retry_config=RetryConfig(max_retries=3, initial_delay=0.001, jitter=False),
        )

        call_count = [0]
        def mock_execute(req):
            call_count[0] += 1
            raise AuthenticationError("auth failed")

        client._execute = mock_execute
        with self.assertRaises(AuthenticationError):
            client.chat([Message.user("hi")])
        # 只调用一次
        self.assertEqual(call_count[0], 1)

    def test04_cache_hits(self):
        """缓存命中"""
        cache = ResponseCache(ttl=60.0)
        client = LLMClient(
            api_key="test",
            model="test",
            use_cache=True,
            cache=cache,
        )

        req = ChatRequest(messages=[Message.user("same")], model="test")
        cache.put(req, ChatResponse(content="cached_response"))

        response1 = client.chat([Message.user("same")])
        response2 = client.chat([Message.user("same")])
        self.assertEqual(response1.content, "cached_response")
        self.assertEqual(response2.content, "cached_response")
        self.assertEqual(client._total_requests, 0)  # 都走缓存

    def test05_thread_safety(self):
        """线程安全"""
        client = LLMClient(api_key="test", model="test")
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    stats = client.get_stats()
                    results.append(stats)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
