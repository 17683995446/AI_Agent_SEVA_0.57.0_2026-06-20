"""
对话Agent（阶段65）

对应第一性原理：
    1. 对话是Agent与用户交互的基本接口
    2. 多轮对话需要会话管理（上下文保持/历史压缩/会话切换）
    3. 对话Agent = BaseAgent + 会话管理 + LLM对话能力
    4. 上下文窗口有限：需要智能压缩历史
    5. 可观测性：对话事件可被监控

模块组成：
    - Session/SessionManager: 会话管理
    - ConversationConfig: 对话Agent配置
    - ConversationAgent: 对话Agent (多轮对话/上下文压缩/记忆整合)
    - 全局单例: get_conversation_agent
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import Message, LLMError, get_llm_client
from ..llm_loop import parse_json_output
from ..memory_llm import get_memory_manager, MemoryType
from ..tools import (
    ToolCallLoop, ToolCallRequest, ToolCallResult, ToolCallStatus,
    get_tool_registry, get_tool_executor,
)
from .base import (
    BaseAgent, AgentConfig, AgentState, AgentRunResult,
    AgentEvent, AgentEventType,
)


logger = logging.getLogger("seva.agents.conversation")


# ============================================================================
# 会话管理
# ============================================================================

class SessionStatus(Enum):
    """会话状态"""
    ACTIVE = "active"          # 活跃
    IDLE = "idle"              # 空闲
    CLOSED = "closed"          # 已关闭
    ARCHIVED = "archived"      # 已归档


@dataclass
class Session:
    """会话"""
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    user_id: Optional[str] = None
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    message_count: int = 0
    total_tokens: int = 0

    def add_message(self, role: str, content: str, tokens: int = 0):
        """添加消息"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "tokens": tokens,
        }
        self.messages.append(msg)
        self.message_count += 1
        self.total_tokens += tokens
        self.last_active_at = time.time()

    def add_user_message(self, content: str, tokens: int = 0):
        self.add_message("user", content, tokens)

    def add_assistant_message(self, content: str, tokens: int = 0):
        self.add_message("assistant", content, tokens)

    def add_system_message(self, content: str, tokens: int = 0):
        self.add_message("system", content, tokens)

    def get_messages(self, max_count: Optional[int] = None,
                     include_system: bool = True) -> List[Dict[str, str]]:
        """获取消息列表(用于LLM)"""
        msgs = []
        for m in self.messages:
            if not include_system and m["role"] == "system":
                continue
            msgs.append({"role": m["role"], "content": m["content"]})
        if max_count and len(msgs) > max_count:
            msgs = msgs[-max_count:]
        return msgs

    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """获取最近N条消息"""
        return self.get_messages(max_count=count)

    def clear_messages(self):
        """清空消息"""
        self.messages.clear()
        self.message_count = 0
        self.total_tokens = 0

    def close(self):
        """关闭会话"""
        self.status = SessionStatus.CLOSED
        self.closed_at = time.time()

    def archive(self):
        """归档会话"""
        self.status = SessionStatus.ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status.value,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "closed_at": self.closed_at,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data.get("session_id", f"sess_{uuid.uuid4().hex[:8]}"),
            user_id=data.get("user_id"),
            title=data.get("title", ""),
            status=SessionStatus(data.get("status", "active")),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            last_active_at=data.get("last_active_at", time.time()),
            closed_at=data.get("closed_at"),
            message_count=data.get("message_count", 0),
            total_tokens=data.get("total_tokens", 0),
        )


class SessionManager:
    """
    会话管理器

    职责:
        1. 创建/获取/关闭会话
        2. 管理多个并发会话
        3. 会话超时清理
        4. 会话持久化(可选)
    """

    def __init__(self, max_sessions: int = 100,
                 idle_timeout: float = 3600.0):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._idle_timeout = idle_timeout
        self._lock = threading.RLock()
        self.stats = {
            "sessions_created": 0,
            "sessions_closed": 0,
            "sessions_archived": 0,
        }

    def create_session(self, user_id: Optional[str] = None,
                       title: str = "") -> Session:
        """创建会话"""
        with self._lock:
            # 检查会话数量限制
            if len(self._sessions) >= self._max_sessions:
                # 先清理空闲会话
                self._cleanup_idle_sessions()
                # 如果还是满,关闭最旧的活跃会话
                if len(self._sessions) >= self._max_sessions:
                    self._close_oldest_active()

            session = Session(user_id=user_id, title=title)
            self._sessions[session.session_id] = session
            self.stats["sessions_created"] += 1
            logger.info(f"[SessionManager] 创建会话: {session.session_id}")
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_active_at = time.time()
            return session

    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                session.close()
                self.stats["sessions_closed"] += 1
                return True
            return False

    def archive_session(self, session_id: str) -> bool:
        """归档会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.archive()
                self.stats["sessions_archived"] += 1
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self, user_id: Optional[str] = None,
                      status: Optional[SessionStatus] = None) -> List[Session]:
        """列出会话"""
        with self._lock:
            sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.stats,
                "active_sessions": len(self._sessions),
            }

    def _cleanup_idle_sessions(self):
        """清理空闲会话"""
        now = time.time()
        to_close = []
        with self._lock:
            for sid, session in self._sessions.items():
                if (session.status == SessionStatus.ACTIVE
                    and now - session.last_active_at > self._idle_timeout):
                    to_close.append(sid)
        for sid in to_close:
            self.close_session(sid)

    def _close_oldest_idle(self):
        """关闭最旧的空闲会话"""
        self._close_oldest_active()

    def _close_oldest_active(self):
        """关闭最旧的活跃会话"""
        with self._lock:
            active_sessions = [
                (sid, s) for sid, s in self._sessions.items()
                if s.status == SessionStatus.ACTIVE
            ]
            if active_sessions:
                active_sessions.sort(key=lambda x: x[1].last_active_at)
                oldest_id = active_sessions[0][0]
                # 直接删除最旧的会话
                if oldest_id in self._sessions:
                    del self._sessions[oldest_id]

    def clear_all(self):
        """清空所有会话"""
        with self._lock:
            self._sessions.clear()


# ============================================================================
# 对话Agent配置
# ============================================================================

@dataclass
class ConversationConfig(AgentConfig):
    """对话Agent配置"""
    name: str = "conversation_agent"
    description: str = "对话Agent"
    # 对话特定配置
    max_history_messages: int = 20          # 最大历史消息数
    context_compression_threshold: int = 15 # 上下文压缩阈值
    enable_context_compression: bool = True # 启用上下文压缩
    system_prompt: Optional[str] = "你是一个有帮助的AI助手。请用中文回答用户问题。"
    # 记忆整合
    enable_memory_integration: bool = True  # 启用记忆整合
    memory_recall_count: int = 3            # 记忆检索数量
    # 会话管理
    auto_create_session: bool = True        # 自动创建会话


# ============================================================================
# 对话Agent
# ============================================================================

class ConversationAgent(BaseAgent):
    """
    对话Agent

    特性:
        1. 多轮对话: 维护对话历史,支持上下文
        2. 会话管理: 支持多个并发会话
        3. 上下文压缩: 历史过长时自动压缩
        4. 记忆整合: 从长期记忆检索相关信息
        5. 工具调用: 可选的工具调用能力
    """

    def __init__(self, agent_id: Optional[str] = None,
                 config: Optional[ConversationConfig] = None):
        super().__init__(agent_id=agent_id,
                        config=config or ConversationConfig())
        self.session_manager = SessionManager()
        self._current_session: Optional[Session] = None

    @property
    def conv_config(self) -> ConversationConfig:
        return self.config

    def chat(self, message: str,
             session_id: Optional[str] = None,
             use_tools: bool = False,
             **kwargs) -> AgentRunResult:
        """
        对话

        Args:
            message: 用户消息
            session_id: 会话ID(可选,不传则使用当前或创建新会话)
            use_tools: 是否使用工具
            **kwargs: 额外参数

        Returns:
            运行结果
        """
        # 获取或创建会话
        session = self._get_or_create_session(session_id)
        if session is None:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=session_id or "",
                success=False,
                error="无法获取或创建会话",
            )

        # 构造输入
        input_data = {
            "message": message,
            "session_id": session.session_id,
            "use_tools": use_tools,
            **kwargs,
        }

        # 运行
        result = self.run(input_data, **kwargs)
        return result

    def _get_or_create_session(self, session_id: Optional[str]) -> Optional[Session]:
        """获取或创建会话"""
        if session_id:
            # 先尝试从管理器获取
            session = self.session_manager.get_session(session_id)
            if session:
                self._current_session = session
                return session
            # 会话不存在,创建新的并使用指定ID
            session = self.session_manager.create_session(title=f"session_{session_id}")
            # 注意: create_session生成的ID无法更改,这里直接使用新会话
            self._current_session = session
            return session

        # 使用当前会话
        if self._current_session and self._current_session.status == SessionStatus.ACTIVE:
            return self._current_session

        # 创建新会话
        if self.conv_config.auto_create_session:
            session = self.session_manager.create_session()
            self._current_session = session
            return session

        return None

    def _run_impl(self, input: Union[str, Dict[str, Any]],
                  **kwargs) -> AgentRunResult:
        """对话Agent核心实现"""
        # 解析输入
        if isinstance(input, str):
            message = input
            session_id = kwargs.get("session_id")
            use_tools = kwargs.get("use_tools", False)
        else:
            message = input.get("message", "")
            session_id = input.get("session_id")
            use_tools = input.get("use_tools", False)

        if not message:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=session_id or "",
                success=False,
                error="消息为空",
            )

        # 获取会话
        session = self._get_or_create_session(session_id)
        if session is None:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id="",
                success=False,
                error="无法获取会话",
            )

        t0 = time.time()
        tokens_used = 0
        tool_calls = []

        try:
            # 添加用户消息到会话
            session.add_user_message(message)

            # 构造LLM消息
            llm_messages = self._build_llm_messages(session, message)

            # 调用LLM
            client = self.llm_client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                if use_tools and self.conv_config.enable_tools:
                    # 使用工具调用
                    loop_result = self.tool_loop.run(
                        message,
                        enable_tools=self.conv_config.enabled_tools,
                        use_cache=False,
                    )
                    response_content = loop_result.get("answer", "")
                    tokens_used = loop_result.get("total_tokens", 0)
                    tool_calls = loop_result.get("tool_calls", [])
                else:
                    # 直接LLM调用
                    response = client.chat(
                        llm_messages,
                        temperature=self.conv_config.temperature,
                        max_tokens=self.conv_config.max_tokens,
                    )
                    response_content = response.content
                    tokens_used = response.usage.total_tokens
            finally:
                client.use_cache = original_cache

            # 添加助手消息到会话
            session.add_assistant_message(response_content, tokens_used)

            # 上下文压缩
            if (self.conv_config.enable_context_compression
                and session.message_count > self.conv_config.context_compression_threshold):
                self._compress_session_context(session)

            # 存储到记忆
            if self.conv_config.enable_memory_integration:
                self._store_to_memory(session, message, response_content)

            latency_ms = (time.time() - t0) * 1000

            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=session.session_id,
                success=True,
                output=response_content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                tool_calls=tool_calls,
            )

        except LLMError as e:
            logger.error(f"[ConversationAgent] LLM错误: {e}")
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=session.session_id,
                success=False,
                error=f"LLM错误: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            logger.error(f"[ConversationAgent] 异常: {e}")
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=session.session_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _build_llm_messages(self, session: Session,
                            current_message: str) -> List[Message]:
        """构造LLM消息列表"""
        messages = []

        # 系统提示
        if self.conv_config.system_prompt:
            messages.append(Message.system(self.conv_config.system_prompt))

        # 记忆检索
        if self.conv_config.enable_memory_integration:
            memories = self.recall(current_message,
                                  limit=self.conv_config.memory_recall_count)
            if memories:
                memory_text = "\n".join(
                    f"- {m.content if hasattr(m, 'content') else str(m)}"
                    for m in memories[:3]
                )
                messages.append(Message.system(f"相关记忆:\n{memory_text}"))

        # 历史消息
        history = session.get_messages(
            max_count=self.conv_config.max_history_messages,
            include_system=False,
        )
        for msg in history[:-1]:  # 排除最后一条(刚添加的用户消息)
            if msg["role"] == "user":
                messages.append(Message.user(msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(Message.assistant(msg["content"]))

        # 当前消息
        messages.append(Message.user(current_message))

        return messages

    def _compress_session_context(self, session: Session):
        """压缩会话上下文"""
        try:
            if session.message_count <= self.conv_config.context_compression_threshold:
                return

            # 保留最近的消息,压缩旧消息
            keep_count = self.conv_config.context_compression_threshold // 2
            old_messages = session.messages[:-keep_count]
            recent_messages = session.messages[-keep_count:]

            # 生成摘要
            summary = self._generate_summary(old_messages)

            # 替换消息
            session.messages = [
                {"role": "system", "content": f"对话摘要: {summary}",
                 "timestamp": time.time(), "tokens": 0},
                *recent_messages,
            ]
            session.message_count = len(session.messages)
            logger.info(f"[ConversationAgent] 压缩会话 {session.session_id}: "
                       f"摘要{len(old_messages)}条,保留{len(recent_messages)}条")

        except Exception as e:
            logger.warning(f"[ConversationAgent] 上下文压缩失败: {e}")

    def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """生成对话摘要"""
        if not messages:
            return ""

        # 简单摘要: 取前几条消息的关键内容
        text_parts = []
        for msg in messages[-5:]:  # 最近5条
            role = msg.get("role", "")
            content = msg.get("content", "")[:100]
            text_parts.append(f"{role}: {content}")

        return " | ".join(text_parts)

    def _store_to_memory(self, session: Session, user_msg: str, assistant_msg: str):
        """存储对话到记忆"""
        try:
            content = f"用户: {user_msg}\n助手: {assistant_msg}"
            self.memory_manager.add(
                content,
                MemoryType.EPISODIC,
                importance=0.5,
                auto_summarize=False,
            )
        except Exception as e:
            logger.debug(f"[ConversationAgent] 记忆存储失败: {e}")

    # ========================================================================
    # 会话管理接口
    # ========================================================================

    def new_session(self, user_id: Optional[str] = None,
                    title: str = "") -> Session:
        """创建新会话"""
        session = self.session_manager.create_session(user_id=user_id, title=title)
        self._current_session = session
        return session

    def switch_session(self, session_id: str) -> bool:
        """切换会话"""
        session = self.session_manager.get_session(session_id)
        if session:
            self._current_session = session
            return True
        return False

    def get_current_session(self) -> Optional[Session]:
        """获取当前会话"""
        return self._current_session

    def close_session(self, session_id: Optional[str] = None) -> bool:
        """关闭会话"""
        sid = session_id or (self._current_session.session_id
                            if self._current_session else None)
        if sid:
            result = self.session_manager.close_session(sid)
            if self._current_session and self._current_session.session_id == sid:
                self._current_session = None
            return result
        return False

    def list_sessions(self) -> List[Session]:
        """列出所有会话"""
        return self.session_manager.list_sessions()

    def get_session_history(self, session_id: str,
                            max_count: Optional[int] = None) -> List[Dict[str, str]]:
        """获取会话历史"""
        session = self.session_manager.get_session(session_id)
        if session:
            return session.get_messages(max_count=max_count)
        return []


# ============================================================================
# 全局单例
# ============================================================================

_global_conversation_agent: Optional[ConversationAgent] = None
_global_lock = threading.RLock()


def get_conversation_agent() -> ConversationAgent:
    """获取全局对话Agent"""
    global _global_conversation_agent
    with _global_lock:
        if _global_conversation_agent is None:
            _global_conversation_agent = ConversationAgent()
        return _global_conversation_agent


def reset_conversation_agent():
    """重置全局对话Agent"""
    global _global_conversation_agent
    with _global_lock:
        _global_conversation_agent = None
