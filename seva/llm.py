"""
LLM客户端与API对接（阶段58）
对应第一性原理：LLM是认知引擎；客户端 = 协议适配器 + 稳定性保障 + 成本控制

提供能力:
    - Message/Role: 消息与角色数据结构
    - ChatRequest/ChatResponse: 请求与响应
    - TokenUsage: Token使用统计
    - LLMError系列: 错误类型（RateLimitError/QuotaExhaustedError/ContentFilterError等）
    - RetryConfig: 重试配置
    - RateLimiter: 限流器（令牌桶）
    - LLMClient: 兼容OpenAI协议的HTTP客户端（流式/非流式）
    - ResponseCache: 响应缓存
    - LLMSession: 会话管理（多轮对话、token统计）
    - 便捷函数: chat / stream_chat / get_llm_client 等

使用示例:
    from seva.llm import LLMClient, Message, Role

    client = LLMClient(api_key="sk-xxx",
                       base_url="https://openapi.cloud-ai.cn/v1")
    response = client.chat([
        Message(role=Role.SYSTEM, content="你是助手"),
        Message(role=Role.USER, content="你好"),
    ])
    print(response.content)
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import hashlib
import threading
import functools
import gzip
import io
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator, Generator
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin
from collections import deque

# http客户端（urllib标准库，避免依赖requests）
import urllib.request
import urllib.error
import ssl


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
LLM_CACHE_DIR = SEVA_HOME / "llm_cache"

# 默认配置
DEFAULT_BASE_URL = "https://openapi.cloud-ai.cn/v1"
DEFAULT_MODEL = "minimax-m2.5"
DEFAULT_TIMEOUT = 60.0
DEFAULT_STREAM_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_RPM = 60  # 每分钟请求数
DEFAULT_RATE_LIMIT_RPS = 10  # 每秒请求数


# ============================================================================
# Role - 消息角色枚举
# ============================================================================

class Role(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ============================================================================
# Message - 消息
# ============================================================================

@dataclass
class Message:
    """聊天消息"""
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为API请求格式"""
        data: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data

    @classmethod
    def system(cls, content: str) -> "Message":
        """系统消息"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """用户消息"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """助手消息"""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        """工具消息"""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role.value!r}, content={preview!r})"


# ============================================================================
# TokenUsage - Token使用统计
# ============================================================================

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        """累加"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )

    def __repr__(self) -> str:
        return f"TokenUsage(total={self.total_tokens})"


# ============================================================================
# ChatRequest - 聊天请求
# ============================================================================

@dataclass
class ChatRequest:
    """聊天请求"""
    messages: List[Message]
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为API请求体"""
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.stop is not None:
            data["stop"] = self.stop
        if self.response_format is not None:
            data["response_format"] = self.response_format
        if self.seed is not None:
            data["seed"] = self.seed
        if self.user is not None:
            data["user"] = self.user
        if self.tools is not None:
            data["tools"] = self.tools
        if self.tool_choice is not None:
            data["tool_choice"] = self.tool_choice
        return data


# ============================================================================
# ChatResponse - 聊天响应
# ============================================================================

@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    role: str = "assistant"
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    tool_calls: Optional[List[Dict[str, Any]]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created: float = field(default_factory=time.time)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "request_id": self.request_id,
            "created": self.created,
        }

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"ChatResponse(content={preview!r}, usage={self.usage.total_tokens}tok)"


# ============================================================================
# StreamChunk - 流式响应片段
# ============================================================================

@dataclass
class StreamChunk:
    """流式响应片段"""
    content: str
    finish_reason: Optional[str] = None
    delta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"StreamChunk(content={self.content[:30]!r}...)"


# ============================================================================
# LLMError - LLM错误类
# ============================================================================

class LLMError(Exception):
    """LLM基础错误"""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 error_type: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"LLMError(type={self.error_type}, status={self.status_code}, msg={self.message!r})"


class RateLimitError(LLMError):
    """限流错误（429）"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, status_code=429, error_type="rate_limit", retryable=True)
        self.retry_after = retry_after


class QuotaExhaustedError(LLMError):
    """配额耗尽错误（402/配额）"""

    def __init__(self, message: str):
        super().__init__(message, status_code=402, error_type="quota_exhausted", retryable=False)


class AuthenticationError(LLMError):
    """认证错误（401）"""

    def __init__(self, message: str):
        super().__init__(message, status_code=401, error_type="authentication", retryable=False)


class ContentFilterError(LLMError):
    """内容过滤错误（内容安全）"""

    def __init__(self, message: str):
        super().__init__(message, error_type="content_filter", retryable=False)


class ModelOverloadedError(LLMError):
    """模型过载（503/504）"""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message, status_code=status_code,
                        error_type="model_overloaded", retryable=True)


class NetworkError(LLMError):
    """网络错误"""

    def __init__(self, message: str):
        super().__init__(message, error_type="network", retryable=True)


class ContextLengthError(LLMError):
    """上下文长度超限错误"""

    def __init__(self, message: str):
        super().__init__(message, error_type="context_length", retryable=False)


# ============================================================================
# RetryConfig - 重试配置
# ============================================================================

@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """计算第attempt次重试的延迟"""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            jitter_amount = delay * 0.25
            delay += (hash((attempt, time.time())) % 100 / 100.0 - 0.5) * 2 * jitter_amount
        return max(0.0, delay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
        }


# ============================================================================
# RateLimiter - 令牌桶限流器
# ============================================================================

class RateLimiter:
    """基于滑动窗口的限流器"""

    def __init__(self, rpm: int = DEFAULT_RATE_LIMIT_RPM,
                 rps: int = DEFAULT_RATE_LIMIT_RPS):
        """
        Args:
            rpm: 每分钟最大请求数
            rps: 每秒最大请求数
        """
        self.rpm = max(1, rpm)
        self.rps = max(1, rps)
        self._request_times: deque = deque()
        self._lock = threading.Lock()
        self._total_requests = 0
        self._throttled_count = 0

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        获取许可

        Returns:
            是否成功获取
        """
        start = time.time()
        while True:
            with self._lock:
                now = time.time()
                # 清理过期的请求时间
                while self._request_times and (now - self._request_times[0]) > 60.0:
                    self._request_times.popleft()

                # 检查速率
                if len(self._request_times) >= self.rpm:
                    # 超RPM限制
                    self._throttled_count += 1
                elif self._has_recent_request(now):
                    # 1秒内有请求
                    self._throttled_count += 1
                else:
                    # 允许请求
                    self._request_times.append(now)
                    self._total_requests += 1
                    return True

            if time.time() - start > timeout:
                return False
            time.sleep(0.05)

    def _has_recent_request(self, now: float) -> bool:
        """检查1秒内是否有请求"""
        if not self._request_times:
            return False
        return (now - self._request_times[-1]) < (1.0 / self.rps)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "rpm_limit": self.rpm,
                "rps_limit": self.rps,
                "total_requests": self._total_requests,
                "throttled_count": self._throttled_count,
                "current_window": len(self._request_times),
            }

    def reset(self) -> None:
        """重置"""
        with self._lock:
            self._request_times.clear()
            self._total_requests = 0
            self._throttled_count = 0


# ============================================================================
# ResponseCache - 响应缓存
# ============================================================================

class ResponseCache:
    """基于prompt hash的响应缓存"""

    def __init__(self, max_size: int = 1000, ttl: float = 3600.0,
                 persist: bool = False):
        """
        Args:
            max_size: 最大缓存条目
            ttl: 过期时间（秒）
            persist: 是否持久化到磁盘
        """
        self.max_size = max(1, max_size)
        self.ttl = max(0.0, ttl)
        self.persist = persist
        self._cache: Dict[str, Tuple[float, ChatResponse]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        if self.persist:
            self._load()

    def _make_key(self, request: ChatRequest) -> str:
        """生成缓存key"""
        data = {
            "model": request.model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, request: ChatRequest) -> Optional[ChatResponse]:
        """获取缓存"""
        if self.ttl == 0:
            return None
        key = self._make_key(request)
        with self._lock:
            if key in self._cache:
                timestamp, response = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return response
                else:
                    del self._cache[key]
            self._misses += 1
        return None

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        """放入缓存"""
        if self.ttl == 0:
            return
        key = self._make_key(request)
        with self._lock:
            # LRU: 超过max_size时删除最早的
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), response)
            if self.persist:
                self._save()

    def clear(self) -> None:
        """清空"""
        with self._lock:
            self._cache.clear()
            if self.persist:
                self._save()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / (self._hits + self._misses)
                             if (self._hits + self._misses) > 0 else 0.0),
            }

    def _load(self) -> None:
        """从磁盘加载"""
        cache_file = LLM_CACHE_DIR / "cache.json"
        if not cache_file.exists():
            return
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                for key, value in data.items():
                    ts, resp_dict = value
                    if time.time() - ts < self.ttl:
                        self._cache[key] = (ts, ChatResponse(
                            content=resp_dict["content"],
                            role=resp_dict.get("role", "assistant"),
                            model=resp_dict.get("model", ""),
                            usage=TokenUsage.from_dict(resp_dict.get("usage", {})),
                            finish_reason=resp_dict.get("finish_reason", "stop"),
                        ))
        except Exception:
            pass

    def _save(self) -> None:
        """保存到磁盘"""
        LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = LLM_CACHE_DIR / "cache.json"
        try:
            data = {
                key: (ts, {"content": resp.content, "role": resp.role,
                          "model": resp.model, "usage": resp.usage.to_dict(),
                          "finish_reason": resp.finish_reason})
                for key, (ts, resp) in self._cache.items()
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass


# ============================================================================
# LLMClient - LLM客户端
# ============================================================================

class LLMClient:
    """
    兼容OpenAI协议的LLM客户端

    使用标准库urllib实现，无外部HTTP依赖
    """

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 timeout: float = DEFAULT_TIMEOUT,
                 retry_config: Optional[RetryConfig] = None,
                 rate_limiter: Optional[RateLimiter] = None,
                 cache: Optional[ResponseCache] = None,
                 use_cache: bool = False,
                 proxy: Optional[str] = None):
        """
        Args:
            api_key: API密钥（默认从环境变量SEVA_LLM_API_KEY读取）
            base_url: API基础URL
            model: 默认模型
            timeout: 请求超时（秒）
            retry_config: 重试配置
            rate_limiter: 限流器
            cache: 响应缓存
            use_cache: 是否启用缓存
            proxy: HTTP代理URL
        """
        self.api_key = api_key or os.environ.get("SEVA_LLM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache or ResponseCache()
        self.use_cache = use_cache
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        # 统计
        self._total_requests = 0
        self._total_errors = 0
        self._total_retries = 0
        self._total_tokens = 0
        self._lock = threading.RLock()

        # User-Agent
        self._user_agent = f"SEVA/0.58.0 (Python/{sys.version.split()[0]})"

    def _get_ssl_context(self) -> ssl.SSLContext:
        """获取SSL上下文"""
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx

    def _build_request(self, endpoint: str, body: Dict[str, Any]) -> urllib.request.Request:
        """构建HTTP请求"""
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("User-Agent", self._user_agent)
        request.add_header("Accept-Encoding", "gzip")
        # 请求ID用于追踪
        request.add_header("X-Request-ID", str(uuid.uuid4()))
        return request

    def _make_http_request(self, request: urllib.request.Request,
                           timeout: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
        """执行HTTP请求"""
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self._get_ssl_context())
        )

        # 配置代理
        if self.proxy:
            proxy_handler = urllib.request.ProxyHandler({
                "http": self.proxy,
                "https": self.proxy,
            })
            opener = urllib.request.build_opener(
                proxy_handler,
                urllib.request.HTTPSHandler(context=self._get_ssl_context()),
            )

        try:
            with opener.open(request, timeout=timeout or self.timeout) as response:
                raw_data = response.read()
                # 处理gzip压缩
                if response.headers.get("Content-Encoding") == "gzip":
                    raw_data = gzip.decompress(raw_data)
                return response.status, json.loads(raw_data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 读取错误响应
            try:
                raw_data = e.read()
                if e.headers.get("Content-Encoding") == "gzip":
                    raw_data = gzip.decompress(raw_data)
                error_body = json.loads(raw_data.decode("utf-8"))
            except Exception:
                error_body = {"error": {"message": str(e)}}
            raise self._map_http_error(e.code, error_body)
        except urllib.error.URLError as e:
            raise NetworkError(f"网络错误: {e.reason}")
        except (TimeoutError, OSError) as e:
            raise NetworkError(f"连接错误: {e}")

    def _map_http_error(self, status_code: int,
                        body: Dict[str, Any]) -> LLMError:
        """映射HTTP错误为LLMError"""
        error = body.get("error", {})
        message = error.get("message", f"HTTP {status_code}")
        error_type = error.get("type", "")

        if status_code == 429 or "rate" in error_type.lower():
            return RateLimitError(message)
        elif status_code == 401 or "auth" in error_type.lower():
            return AuthenticationError(message)
        elif status_code == 402 or "quota" in error_type.lower() or "billing" in error_type.lower():
            return QuotaExhaustedError(message)
        elif status_code in (503, 504) or "overloaded" in error_type.lower():
            return ModelOverloadedError(message, status_code=status_code)
        elif status_code == 400 and ("context" in error_type.lower() or "length" in error_type.lower()):
            return ContextLengthError(message)
        elif "content" in error_type.lower() or "filter" in error_type.lower():
            return ContentFilterError(message)
        else:
            retryable = 500 <= status_code < 600
            return LLMError(message, status_code=status_code,
                          error_type=error_type or "unknown",
                          retryable=retryable)

    def chat(self, messages: Union[List[Message], List[Dict[str, str]]],
             **kwargs) -> ChatResponse:
        """
        同步聊天调用

        Args:
            messages: 消息列表
            **kwargs: 覆盖默认参数（model/temperature等）

        Returns:
            聊天响应
        """
        # 标准化消息
        norm_messages = self._normalize_messages(messages)

        # 构建请求
        request = ChatRequest(messages=norm_messages, **kwargs)
        if request.model == DEFAULT_MODEL and "model" not in kwargs:
            request.model = self.model

        # 缓存命中
        if self.use_cache and not request.stream:
            cached = self.cache.get(request)
            if cached:
                return cached

        # 限流
        if not self.rate_limiter.acquire():
            raise RateLimitError("限流器超时")

        # 重试执行
        response = self._execute_with_retry(request, stream=False)
        assert isinstance(response, ChatResponse)

        # 缓存
        if self.use_cache:
            self.cache.put(request, response)

        return response

    def stream_chat(self, messages: Union[List[Message], List[Dict[str, str]]],
                    **kwargs) -> Iterator[StreamChunk]:
        """
        流式聊天调用

        Yields:
            StreamChunk对象
        """
        norm_messages = self._normalize_messages(messages)
        request = ChatRequest(messages=norm_messages, stream=True, **kwargs)
        if request.model == DEFAULT_MODEL and "model" not in kwargs:
            request.model = self.model

        if not self.rate_limiter.acquire():
            raise RateLimitError("限流器超时")

        yield from self._execute_with_retry(request, stream=True)

    def _normalize_messages(self, messages: Union[List[Message], List[Dict[str, str]]]
                            ) -> List[Message]:
        """标准化消息格式"""
        norm: List[Message] = []
        for m in messages:
            if isinstance(m, Message):
                norm.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                norm.append(Message(role=Role(role), content=content))
            else:
                raise TypeError(f"不支持的消息类型: {type(m)}")
        return norm

    def _execute_with_retry(self, request: ChatRequest,
                            stream: bool) -> Union[ChatResponse, Iterator[StreamChunk]]:
        """带重试执行"""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_config.max_retries + 2):
            try:
                if stream:
                    return list(self._execute_stream(request))
                else:
                    response = self._execute(request)
                    self._record_success(response.usage.total_tokens)
                    return response
            except LLMError as e:
                last_error = e
                with self._lock:
                    self._total_errors += 1
                if not e.retryable:
                    raise
                if attempt > self.retry_config.max_retries:
                    raise
                with self._lock:
                    self._total_retries += 1
                delay = self.retry_config.get_delay(attempt)
                time.sleep(delay)
            except Exception as e:
                last_error = NetworkError(str(e))
                if attempt > self.retry_config.max_retries:
                    raise last_error
                with self._lock:
                    self._total_retries += 1
                delay = self.retry_config.get_delay(attempt)
                time.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("重试逻辑异常")

    def _execute(self, request: ChatRequest) -> ChatResponse:
        """执行单次非流式请求"""
        body = request.to_dict()
        body["stream"] = False
        http_request = self._build_request("chat/completions", body)

        with self._lock:
            self._total_requests += 1

        status_code, response_data = self._make_http_request(http_request)
        return self._parse_response(response_data)

    def _execute_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """执行流式请求"""
        body = request.to_dict()
        body["stream"] = True
        http_request = self._build_request("chat/completions", body)

        with self._lock:
            self._total_requests += 1

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self._get_ssl_context())
        )

        with opener.open(http_request, timeout=DEFAULT_STREAM_TIMEOUT) as response:
            accumulated_content = ""
            final_chunk_data: Dict[str, Any] = {}

            for line in response:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # 去掉 "data: " 前缀
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                final_chunk_data = chunk

                # 提取delta
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish_reason = choices[0].get("finish_reason")

                accumulated_content += content
                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    delta=delta,
                )

        # 记录token使用
        if final_chunk_data and "usage" in final_chunk_data:
            self._record_success(final_chunk_data["usage"].get("total_tokens", 0))

    def _parse_response(self, data: Dict[str, Any]) -> ChatResponse:
        """解析API响应"""
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("响应中没有choices", error_type="invalid_response", retryable=False)

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
        finish_reason = first_choice.get("finish_reason", "stop")
        role = message.get("role", "assistant")

        usage_data = data.get("usage", {})
        usage = TokenUsage.from_dict(usage_data)

        return ChatResponse(
            content=content,
            role=role,
            model=data.get("model", ""),
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=data,
        )

    def _record_success(self, tokens: int) -> None:
        """记录成功调用"""
        with self._lock:
            self._total_tokens += tokens

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计"""
        with self._lock:
            stats = {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "total_retries": self._total_retries,
                "total_tokens": self._total_tokens,
                "model": self.model,
                "base_url": self.base_url,
            }
        stats["rate_limiter"] = self.rate_limiter.get_stats()
        stats["cache"] = self.cache.get_stats() if self.use_cache else None
        return stats

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, base_url={self.base_url!r})"


# ============================================================================
# LLMSession - 多轮对话会话
# ============================================================================

class LLMSession:
    """多轮对话会话管理"""

    def __init__(self, client: LLMClient, system_prompt: Optional[str] = None,
                 max_messages: int = 50):
        self.client = client
        self.system_prompt = system_prompt
        self.max_messages = max(1, max_messages)
        self._messages: List[Message] = []
        if system_prompt:
            self._messages.append(Message.system(system_prompt))
        self._total_usage = TokenUsage()
        self._created = time.time()
        self._lock = threading.RLock()

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        with self._lock:
            self._messages.append(Message.user(content))
            self._trim()

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        with self._lock:
            self._messages.append(Message.assistant(content))
            self._trim()

    def add_system_message(self, content: str) -> None:
        """添加系统消息"""
        with self._lock:
            # 系统消息通常放在开头
            self._messages.insert(0, Message.system(content))
            self._trim()

    def chat(self, user_input: str, **kwargs) -> ChatResponse:
        """发送用户输入并获取响应"""
        with self._lock:
            self.add_user_message(user_input)
            messages = list(self._messages)

        response = self.client.chat(messages, **kwargs)
        with self._lock:
            self.add_assistant_message(response.content)
            self._total_usage.add(response.usage)
        return response

    def _trim(self) -> None:
        """修剪消息列表"""
        if len(self._messages) <= self.max_messages:
            return
        # 保留第一条系统消息，删除中间最旧的消息
        if self._messages and self._messages[0].role == Role.SYSTEM:
            system = self._messages[0]
            other = self._messages[1:]
            keep_count = self.max_messages - 1
            if len(other) > keep_count:
                other = other[-keep_count:]
            self._messages = [system] + other
        else:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> List[Message]:
        """获取消息列表（副本）"""
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        """清空会话"""
        with self._lock:
            self._messages.clear()
            if self.system_prompt:
                self._messages.append(Message.system(self.system_prompt))

    def get_usage(self) -> TokenUsage:
        """获取总token使用"""
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._total_usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens,
                total_tokens=self._total_usage.total_tokens,
            )

    def get_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        with self._lock:
            return {
                "message_count": len(self._messages),
                "max_messages": self.max_messages,
                "system_prompt": self.system_prompt,
                "total_usage": self._total_usage.to_dict(),
                "duration": time.time() - self._created,
            }

    def __repr__(self) -> str:
        return f"LLMSession(messages={len(self._messages)}, usage={self._total_usage.total_tokens}tok)"


# ============================================================================
# 便捷函数
# ============================================================================

_default_client: Optional[LLMClient] = None
_default_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """获取默认LLM客户端（单例）"""
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                _default_client = LLMClient()
    return _default_client


def set_llm_client(client: LLMClient) -> None:
    """设置默认LLM客户端"""
    global _default_client
    with _default_lock:
        _default_client = client


def chat(messages: Union[List[Message], List[Dict[str, str]]],
         **kwargs) -> ChatResponse:
    """便捷聊天函数"""
    return get_llm_client().chat(messages, **kwargs)


def stream_chat(messages: Union[List[Message], List[Dict[str, str]]],
                **kwargs) -> Iterator[StreamChunk]:
    """便捷流式聊天函数"""
    return get_llm_client().stream_chat(messages, **kwargs)


def get_llm_stats() -> Dict[str, Any]:
    """获取LLM客户端统计"""
    return get_llm_client().get_stats()


def reset_llm_client() -> None:
    """重置默认LLM客户端"""
    global _default_client
    with _default_lock:
        _default_client = None
