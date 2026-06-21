"""
研究Agent（阶段67）

对应第一性原理：
    1. 研究 = 信息检索 + 信息分析 + 报告生成的流程
    2. 分阶段进行：先广度收集，再深度分析，最后结构化输出
    3. 质量优先：每个阶段都有质量检查
    4. 可追溯：所有信息来源可追溯
    5. 增量改进：根据反馈调整研究策略

模块组成：
    - ResearchStage: 研究阶段枚举
    - SourceInfo/ResearchFinding: 信息源与发现
    - ResearchReport: 研究报告
    - InfoRetriever: 信息检索器 (LLM驱动)
    - InfoAnalyzer: 信息分析器 (LLM驱动)
    - ReportGenerator: 报告生成器 (LLM驱动)
    - ResearchAgent: 研究Agent
    - 全局单例: get_research_agent
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
from .base import (
    BaseAgent, AgentConfig, AgentState, AgentRunResult,
    AgentEvent, AgentEventType,
)


logger = logging.getLogger("seva.agents.research")


# ============================================================================
# 枚举
# ============================================================================

class ResearchStage(Enum):
    """研究阶段"""
    PLANNING = "planning"          # 规划研究
    RETRIEVING = "retrieving"      # 检索信息
    ANALYZING = "analyzing"        # 分析信息
    SYNTHESIZING = "synthesizing"  # 综合信息
    REPORTING = "reporting"        # 生成报告
    COMPLETED = "completed"        # 完成
    FAILED = "failed"              # 失败


class SourceType(Enum):
    """信息源类型"""
    WEB = "web"                    # 网络来源
    KNOWLEDGE = "knowledge"        # 知识库
    MEMORY = "memory"              # 记忆系统
    LLM = "llm"                    # LLM生成
    USER = "user"                  # 用户提供
    UNKNOWN = "unknown"            # 未知来源


class ReportFormat(Enum):
    """报告格式"""
    MARKDOWN = "markdown"          # Markdown
    JSON = "json"                  # JSON
    TEXT = "text"                  # 纯文本
    STRUCTURED = "structured"      # 结构化


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class SourceInfo:
    """信息源"""
    url: str = ""
    title: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    content: str = ""
    relevance_score: float = 0.0  # 0-1
    credibility: float = 0.0      # 0-1
    retrieved_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "source_type": self.source_type.value,
            "content": self.content[:500],  # 限制长度
            "relevance_score": self.relevance_score,
            "credibility": self.credibility,
            "retrieved_at": self.retrieved_at,
        }


@dataclass
class ResearchFinding:
    """研究发现"""
    id: str = field(default_factory=lambda: f"finding_{uuid.uuid4().hex[:8]}")
    content: str = ""
    sources: List[SourceInfo] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass
class ResearchReport:
    """研究报告"""
    id: str = field(default_factory=lambda: f"report_{uuid.uuid4().hex[:8]}")
    topic: str = ""
    query: str = ""
    stage: ResearchStage = ResearchStage.PLANNING
    findings: List[ResearchFinding] = field(default_factory=list)
    sources: List[SourceInfo] = field(default_factory=list)
    summary: str = ""
    content: str = ""
    format: ReportFormat = ReportFormat.MARKDOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "query": self.query,
            "stage": self.stage.value,
            "findings": [f.to_dict() for f in self.findings],
            "sources": [s.to_dict() for s in self.sources],
            "summary": self.summary,
            "content": self.content,
            "format": self.format.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "tokens_used": self.tokens_used,
        }

    @property
    def is_completed(self) -> bool:
        return self.stage == ResearchStage.COMPLETED


# ============================================================================
# 信息检索器
# ============================================================================

class InfoRetriever:
    """
    信息检索器

    职责:
        1. 从记忆系统检索相关信息
        2. 使用LLM生成候选信息
        3. 评估信息相关性
    """

    RETRIEVE_PROMPT = """你是信息检索专家。请根据查询生成相关信息。

查询: {query}
主题: {topic}

请返回JSON:
{{
    "sources": [
        {{
            "content": "信息内容",
            "relevance_score": 0.8,
            "credibility": 0.7,
            "tags": ["标签1", "标签2"]
        }}
    ]
}}

只返回JSON。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"retrievals": 0, "sources_found": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def retrieve(self, query: str, topic: str = "",
                 max_sources: int = 5) -> List[SourceInfo]:
        """检索信息"""
        with self._lock:
            self.stats["retrievals"] += 1

        sources = []

        # 1. 从记忆系统检索
        try:
            manager = get_memory_manager()
            memories = manager.retrieve(query, limit=max_sources // 2)
            for mem in memories:
                content = mem.content if hasattr(mem, "content") else str(mem)
                sources.append(SourceInfo(
                    title="记忆",
                    source_type=SourceType.MEMORY,
                    content=content,
                    relevance_score=0.7,
                    credibility=0.8,
                ))
        except Exception as e:
            logger.debug(f"[Retriever] 记忆检索失败: {e}")

        # 2. 使用LLM生成信息
        try:
            prompt = self.RETRIEVE_PROMPT.format(query=query, topic=topic)
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是信息检索专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.3,
                    max_tokens=768,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if parsed and "sources" in parsed:
                for src_data in parsed["sources"]:
                    sources.append(SourceInfo(
                        title="LLM生成",
                        source_type=SourceType.LLM,
                        content=src_data.get("content", ""),
                        relevance_score=float(src_data.get("relevance_score", 0.5)),
                        credibility=float(src_data.get("credibility", 0.5)),
                    ))
        except LLMError as e:
            logger.warning(f"[Retriever] LLM检索失败: {e}")
        except Exception as e:
            logger.error(f"[Retriever] 检索异常: {e}")

        # 限制数量
        sources = sources[:max_sources]
        with self._lock:
            self.stats["sources_found"] += len(sources)
        return sources

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 信息分析器
# ============================================================================

class InfoAnalyzer:
    """
    信息分析器

    职责:
        1. 分析信息源,提取关键发现
        2. 评估发现的可信度
        3. 识别信息间的关联
    """

    ANALYZE_PROMPT = """请分析以下信息,提取关键发现。

查询: {query}
信息源:
{sources}

请返回JSON:
{{
    "findings": [
        {{
            "content": "发现内容",
            "confidence": 0.8,
            "tags": ["标签"]
        }}
    ],
    "summary": "整体分析"
}}

只返回JSON。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"analyses": 0, "findings_extracted": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def analyze(self, query: str,
                sources: List[SourceInfo]) -> Tuple[List[ResearchFinding], str]:
        """
        分析信息源

        Returns:
            (发现列表, 摘要)
        """
        with self._lock:
            self.stats["analyses"] += 1

        if not sources:
            return [], "无信息源可分析"

        # 格式化信息源
        sources_text = "\n".join(
            f"[{i+1}] {s.content[:200]}"
            for i, s in enumerate(sources)
        )

        prompt = self.ANALYZE_PROMPT.format(query=query, sources=sources_text)

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是信息分析专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.3,
                    max_tokens=768,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed:
                return [], "分析失败"

            findings = []
            for f_data in parsed.get("findings", []):
                finding = ResearchFinding(
                    content=f_data.get("content", ""),
                    sources=sources,
                    confidence=float(f_data.get("confidence", 0.5)),
                    tags=f_data.get("tags", []),
                )
                findings.append(finding)

            summary = parsed.get("summary", "")
            with self._lock:
                self.stats["findings_extracted"] += len(findings)
            return findings, summary

        except LLMError as e:
            logger.warning(f"[Analyzer] LLM分析失败: {e}")
            return [], f"分析失败: {e}"
        except Exception as e:
            logger.error(f"[Analyzer] 异常: {e}")
            return [], f"异常: {e}"

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """
    报告生成器

    职责:
        1. 根据发现生成结构化报告
        2. 支持多种格式 (Markdown/JSON/Text)
        3. 包含来源引用
    """

    REPORT_PROMPT = """请根据研究发现生成研究报告。

主题: {topic}
查询: {query}

研究发现:
{findings}

请生成{format}格式的报告,包含:
1. 摘要
2. 主要发现
3. 详细分析
4. 结论

直接返回报告内容,不要JSON包装。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"reports_generated": 0, "tokens_used": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def generate(self, topic: str, query: str,
                 findings: List[ResearchFinding],
                 format: ReportFormat = ReportFormat.MARKDOWN) -> Tuple[str, int]:
        """
        生成报告

        Returns:
            (报告内容, tokens使用量)
        """
        with self._lock:
            self.stats["reports_generated"] += 1

        if not findings:
            return f"# {topic}\n\n无足够信息生成报告。", 0

        # 格式化发现
        findings_text = "\n".join(
            f"- [{f.confidence:.1f}] {f.content}"
            for f in findings
        )

        format_str = format.value
        prompt = self.REPORT_PROMPT.format(
            topic=topic, query=query,
            findings=findings_text, format=format_str,
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是研究报告撰写专家。"),
                     Message.user(prompt)],
                    temperature=0.4,
                    max_tokens=1024,
                )
            finally:
                client.use_cache = original_cache

            tokens = response.usage.total_tokens
            with self._lock:
                self.stats["tokens_used"] += tokens
            return response.content, tokens

        except LLMError as e:
            logger.warning(f"[Generator] LLM生成失败: {e}")
            return f"# {topic}\n\n报告生成失败: {e}", 0
        except Exception as e:
            logger.error(f"[Generator] 异常: {e}")
            return f"# {topic}\n\n报告生成异常: {e}", 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 研究Agent配置
# ============================================================================

@dataclass
class ResearchAgentConfig(AgentConfig):
    """研究Agent配置"""
    name: str = "research_agent"
    description: str = "研究Agent"
    max_sources: int = 5              # 最大信息源数
    max_findings: int = 10            # 最大发现数
    report_format: ReportFormat = ReportFormat.MARKDOWN
    enable_memory_search: bool = True # 启用记忆检索
    min_confidence: float = 0.3       # 最低置信度
    system_prompt: Optional[str] = "你是研究专家,帮助用户进行信息检索、分析和报告生成。"


# ============================================================================
# 研究Agent
# ============================================================================

class ResearchAgent(BaseAgent):
    """
    研究Agent

    工作流:
        1. PLANNING: 理解查询,规划研究
        2. RETRIEVING: 检索信息 (InfoRetriever)
        3. ANALYZING: 分析信息 (InfoAnalyzer)
        4. REPORTING: 生成报告 (ReportGenerator)
        5. COMPLETED: 返回结果
    """

    def __init__(self, agent_id: Optional[str] = None,
                 config: Optional[ResearchAgentConfig] = None):
        super().__init__(agent_id=agent_id,
                        config=config or ResearchAgentConfig())
        self.retriever = InfoRetriever()
        self.analyzer = InfoAnalyzer()
        self.generator = ReportGenerator()

    @property
    def research_config(self) -> ResearchAgentConfig:
        return self.config

    def research(self, query: str,
                 topic: Optional[str] = None) -> ResearchReport:
        """执行研究"""
        input_data = {
            "query": query,
            "topic": topic or query,
        }
        result = self.run(input_data)
        # 从结果中恢复报告
        if result.success and result.metadata.get("report"):
            return ResearchReport(
                topic=topic or query,
                query=query,
                stage=ResearchStage.COMPLETED,
                content=result.output,
                completed_at=time.time(),
                tokens_used=result.tokens_used,
            )
        return ResearchReport(
            topic=topic or query,
            query=query,
            stage=ResearchStage.FAILED,
            content=result.error or "研究失败",
        )

    def _run_impl(self, input: Union[str, Dict[str, Any]],
                  **kwargs) -> AgentRunResult:
        """研究Agent核心实现"""
        # 解析输入
        if isinstance(input, str):
            query = input
            topic = kwargs.get("topic", query)
        else:
            query = input.get("query", "")
            topic = input.get("topic", query)

        if not query:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error="查询为空",
            )

        t0 = time.time()
        tokens_used = 0
        report = ResearchReport(topic=topic, query=query)

        try:
            # 阶段1: 检索信息
            report.stage = ResearchStage.RETRIEVING
            logger.info(f"[ResearchAgent] 检索信息: {query}")
            sources = self.retriever.retrieve(
                query, topic,
                max_sources=self.research_config.max_sources,
            )
            report.sources = sources

            if not sources:
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=False,
                    error="未检索到相关信息",
                    latency_ms=(time.time() - t0) * 1000,
                )

            # 阶段2: 分析信息
            report.stage = ResearchStage.ANALYZING
            logger.info(f"[ResearchAgent] 分析 {len(sources)} 个信息源")
            findings, summary = self.analyzer.analyze(query, sources)
            report.findings = findings[:self.research_config.max_findings]
            report.summary = summary

            # 过滤低置信度发现
            report.findings = [
                f for f in report.findings
                if f.confidence >= self.research_config.min_confidence
            ]

            if not report.findings:
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=False,
                    error="未提取到有效发现",
                    latency_ms=(time.time() - t0) * 1000,
                )

            # 阶段3: 生成报告
            report.stage = ResearchStage.REPORTING
            logger.info(f"[ResearchAgent] 生成报告")
            content, tokens = self.generator.generate(
                topic, query, report.findings,
                format=self.research_config.report_format,
            )
            report.content = content
            report.tokens_used = tokens
            tokens_used = tokens

            # 阶段4: 完成
            report.stage = ResearchStage.COMPLETED
            report.completed_at = time.time()

            # 存储到记忆
            if self.research_config.enable_memory_search:
                try:
                    manager = get_memory_manager()
                    manager.add(
                        f"研究: {topic}\n摘要: {summary}",
                        MemoryType.SEMANTIC,
                        importance=0.7,
                        auto_summarize=False,
                    )
                except Exception as e:
                    logger.debug(f"[ResearchAgent] 记忆存储失败: {e}")

            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=True,
                output=content,
                tokens_used=tokens_used,
                latency_ms=(time.time() - t0) * 1000,
                metadata={
                    "report": report.to_dict(),
                    "sources_count": len(sources),
                    "findings_count": len(report.findings),
                },
            )

        except Exception as e:
            logger.error(f"[ResearchAgent] 异常: {e}")
            report.stage = ResearchStage.FAILED
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取详细统计"""
        base_stats = super().get_stats()
        base_stats.update({
            "retriever": self.retriever.get_stats(),
            "analyzer": self.analyzer.get_stats(),
            "generator": self.generator.get_stats(),
        })
        return base_stats


# ============================================================================
# 全局单例
# ============================================================================

_global_research_agent: Optional[ResearchAgent] = None
_global_lock = threading.RLock()


def get_research_agent() -> ResearchAgent:
    """获取全局研究Agent"""
    global _global_research_agent
    with _global_lock:
        if _global_research_agent is None:
            _global_research_agent = ResearchAgent()
        return _global_research_agent


def reset_research_agent():
    """重置全局研究Agent"""
    global _global_research_agent
    with _global_lock:
        _global_research_agent = None
