"""
帮助系统与文档集成（阶段46）
对应第一性原理：可发现 = 用户能快速找到所需帮助和文档

提供能力:
    - HelpTopic: 帮助主题（标题/内容/分类/关键词）
    - HelpSystem: 帮助系统（主题管理/搜索/分类浏览）
    - CommandHelp: 命令帮助（用法/参数/示例）
    - FAQ: 常见问题
    - Glossary: 术语表
    - Example: 代码示例
    - 便捷函数: help / search_help / show_help / list_help_topics 等

使用示例:
    from seva import show_help, search_help, list_help_topics

    show_help("getting_started")  # 显示主题
    results = search_help("配置")  # 搜索
    topics = list_help_topics()    # 列出主题
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SEVA_HOME = Path.home() / ".seva"
HELP_CACHE_FILE = SEVA_HOME / "help_cache.json"


# ============================================================================
# HelpTopic - 帮助主题
# ============================================================================

class HelpTopic:
    """帮助主题"""

    def __init__(self, topic_id: str, title: str, content: str,
                 category: str = "general",
                 keywords: Optional[List[str]] = None,
                 examples: Optional[List[str]] = None,
                 related: Optional[List[str]] = None) -> None:
        self.topic_id = topic_id
        self.title = title
        self.content = content
        self.category = category
        self.keywords = keywords or []
        self.examples = examples or []
        self.related = related or []

    def matches(self, query: str) -> bool:
        """检查是否匹配查询"""
        query_lower = query.lower()
        if query_lower in self.title.lower():
            return True
        if query_lower in self.content.lower():
            return True
        for kw in self.keywords:
            if query_lower in kw.lower():
                return True
        return False

    def relevance_score(self, query: str) -> int:
        """计算相关度分数"""
        score = 0
        query_lower = query.lower()
        # 标题匹配权重最高
        if query_lower in self.title.lower():
            score += 10
        # 关键词匹配
        for kw in self.keywords:
            if query_lower in kw.lower():
                score += 5
            if kw.lower() in query_lower:
                score += 3
        # 内容匹配
        if query_lower in self.content.lower():
            score += 2
        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "keywords": self.keywords,
            "examples": self.examples,
            "related": self.related,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HelpTopic":
        return cls(
            topic_id=data["topic_id"],
            title=data["title"],
            content=data["content"],
            category=data.get("category", "general"),
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
            related=data.get("related", []),
        )

    def format(self) -> str:
        """格式化为可读文本"""
        lines = [
            "=" * 60,
            f"{self.title}",
            f"(主题ID: {self.topic_id} | 分类: {self.category})",
            "=" * 60,
            "",
            self.content,
        ]
        if self.examples:
            lines.append("")
            lines.append("示例:")
            for ex in self.examples:
                lines.append(f"  $ {ex}")
        if self.related:
            lines.append("")
            lines.append("相关主题:")
            for r in self.related:
                lines.append(f"  - {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# HelpSystem - 帮助系统
# ============================================================================

class HelpSystem:
    """
    帮助系统（单例）

    用法:
        hs = HelpSystem()
        hs.show("getting_started")
        results = hs.search("配置")
    """

    _instance: Optional["HelpSystem"] = None

    def __new__(cls) -> "HelpSystem":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._topics: Dict[str, HelpTopic] = {}
        self._register_default_topics()

    def register(self, topic: HelpTopic) -> None:
        """注册主题"""
        self._topics[topic.topic_id] = topic

    def get(self, topic_id: str) -> Optional[HelpTopic]:
        """获取主题"""
        return self._topics.get(topic_id)

    def show(self, topic_id: str) -> str:
        """显示主题内容"""
        topic = self.get(topic_id)
        if topic is None:
            return f"未找到主题: {topic_id}\n使用 list_topics() 查看可用主题"
        return topic.format()

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索主题"""
        results = []
        for topic in self._topics.values():
            if topic.matches(query):
                score = topic.relevance_score(query)
                results.append({
                    "topic_id": topic.topic_id,
                    "title": topic.title,
                    "category": topic.category,
                    "score": score,
                })
        # 按分数降序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def list_topics(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出主题"""
        topics = []
        for topic in self._topics.values():
            if category is None or topic.category == category:
                topics.append({
                    "topic_id": topic.topic_id,
                    "title": topic.title,
                    "category": topic.category,
                })
        # 按分类和标题排序
        topics.sort(key=lambda x: (x["category"], x["title"]))
        return topics

    def list_categories(self) -> List[str]:
        """列出所有分类"""
        cats = set()
        for topic in self._topics.values():
            cats.add(topic.category)
        return sorted(cats)

    def get_topics_by_category(self, category: str) -> List[HelpTopic]:
        """按分类获取主题"""
        return [t for t in self._topics.values() if t.category == category]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        categories = self.list_categories()
        return {
            "total_topics": len(self._topics),
            "total_categories": len(categories),
            "categories": categories,
        }

    def get_report(self) -> str:
        """生成帮助系统报告"""
        stats = self.get_stats()
        lines = [
            "=" * 60,
            "SEVA帮助系统",
            "=" * 60,
            f"主题总数: {stats['total_topics']}",
            f"分类数: {stats['total_categories']}",
            "",
            "分类:",
        ]
        for cat in stats["categories"]:
            topics = self.get_topics_by_category(cat)
            lines.append(f"\n  [{cat}] ({len(topics)}个主题)")
            for t in topics:
                lines.append(f"    - {t.topic_id}: {t.title}")
        lines.append("")
        lines.append("使用 show_help(topic_id) 查看主题内容")
        lines.append("使用 search_help(query) 搜索主题")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _register_default_topics(self) -> None:
        """注册默认主题"""
        for topic in _get_default_topics():
            self.register(topic)

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        cls._instance = None


# ============================================================================
# CommandHelp - 命令帮助
# ============================================================================

class CommandHelp:
    """命令帮助信息"""

    def __init__(self, command: str, description: str,
                 usage: str, examples: Optional[List[str]] = None,
                 options: Optional[Dict[str, str]] = None,
                 see_also: Optional[List[str]] = None) -> None:
        self.command = command
        self.description = description
        self.usage = usage
        self.examples = examples or []
        self.options = options or {}
        self.see_also = see_also or []

    def format(self) -> str:
        """格式化"""
        lines = [
            "=" * 60,
            f"命令: {self.command}",
            "=" * 60,
            "",
            f"描述: {self.description}",
            "",
            f"用法: {self.usage}",
        ]
        if self.options:
            lines.append("")
            lines.append("选项:")
            for opt, desc in self.options.items():
                lines.append(f"  {opt:<20} {desc}")
        if self.examples:
            lines.append("")
            lines.append("示例:")
            for ex in self.examples:
                lines.append(f"  $ {ex}")
        if self.see_also:
            lines.append("")
            lines.append("参见:")
            for s in self.see_also:
                lines.append(f"  - {s}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "description": self.description,
            "usage": self.usage,
            "examples": self.examples,
            "options": self.options,
            "see_also": self.see_also,
        }


class CommandHelpRegistry:
    """命令帮助注册表（单例）"""

    _instance: Optional["CommandHelpRegistry"] = None

    def __new__(cls) -> "CommandHelpRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._commands: Dict[str, CommandHelp] = {}
        self._register_default_commands()

    def register(self, help_info: CommandHelp) -> None:
        self._commands[help_info.command] = help_info

    def get(self, command: str) -> Optional[CommandHelp]:
        return self._commands.get(command)

    def list_commands(self) -> List[str]:
        return sorted(self._commands.keys())

    def get_all(self) -> Dict[str, CommandHelp]:
        return dict(self._commands)

    def _register_default_commands(self) -> None:
        for cmd in _get_default_commands():
            self.register(cmd)

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ============================================================================
# FAQ - 常见问题
# ============================================================================

class FAQEntry:
    """FAQ条目"""

    def __init__(self, question: str, answer: str,
                 category: str = "general",
                 keywords: Optional[List[str]] = None) -> None:
        self.question = question
        self.answer = answer
        self.category = category
        self.keywords = keywords or []

    def matches(self, query: str) -> bool:
        query_lower = query.lower()
        if query_lower in self.question.lower():
            return True
        if query_lower in self.answer.lower():
            return True
        for kw in self.keywords:
            if query_lower in kw.lower():
                return True
        return False

    def format(self) -> str:
        lines = [
            "=" * 60,
            f"Q: {self.question}",
            "=" * 60,
            "",
            f"A: {self.answer}",
        ]
        if self.keywords:
            lines.append("")
            lines.append(f"关键词: {', '.join(self.keywords)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "keywords": self.keywords,
        }


class FAQRegistry:
    """FAQ注册表（单例）"""

    _instance: Optional["FAQRegistry"] = None

    def __new__(cls) -> "FAQRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._entries: List[FAQEntry] = []
        self._register_default_faqs()

    def add(self, entry: FAQEntry) -> None:
        self._entries.append(entry)

    def search(self, query: str, limit: int = 10) -> List[FAQEntry]:
        results = [e for e in self._entries if e.matches(query)]
        return results[:limit]

    def list_all(self, category: Optional[str] = None) -> List[FAQEntry]:
        if category:
            return [e for e in self._entries if e.category == category]
        return list(self._entries)

    def list_categories(self) -> List[str]:
        return sorted(set(e.category for e in self._entries))

    def _register_default_faqs(self) -> None:
        for faq in _get_default_faqs():
            self.add(faq)

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ============================================================================
# Glossary - 术语表
# ============================================================================

class GlossaryTerm:
    """术语条目"""

    def __init__(self, term: str, definition: str,
                 category: str = "general") -> None:
        self.term = term
        self.definition = definition
        self.category = category

    def format(self) -> str:
        return f"{self.term}: {self.definition}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "category": self.category,
        }


class Glossary:
    """术语表（单例）"""

    _instance: Optional["Glossary"] = None

    def __new__(cls) -> "Glossary":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._terms: Dict[str, GlossaryTerm] = {}
        self._register_default_terms()

    def add(self, term: GlossaryTerm) -> None:
        self._terms[term.term.lower()] = term

    def get(self, term: str) -> Optional[GlossaryTerm]:
        return self._terms.get(term.lower())

    def search(self, query: str, limit: int = 10) -> List[GlossaryTerm]:
        query_lower = query.lower()
        results = []
        for term in self._terms.values():
            if query_lower in term.term.lower() or \
               query_lower in term.definition.lower():
                results.append(term)
        return results[:limit]

    def list_all(self, category: Optional[str] = None) -> List[GlossaryTerm]:
        if category:
            return [t for t in self._terms.values() if t.category == category]
        return list(self._terms.values())

    def list_categories(self) -> List[str]:
        return sorted(set(t.category for t in self._terms.values()))

    def _register_default_terms(self) -> None:
        for term in _get_default_terms():
            self.add(term)

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ============================================================================
# Example - 代码示例
# ============================================================================

class CodeExample:
    """代码示例"""

    def __init__(self, example_id: str, title: str, description: str,
                 code: str, category: str = "general",
                 language: str = "python") -> None:
        self.example_id = example_id
        self.title = title
        self.description = description
        self.code = code
        self.category = category
        self.language = language

    def format(self) -> str:
        lines = [
            "=" * 60,
            f"示例: {self.title}",
            f"(ID: {self.example_id} | 分类: {self.category})",
            "=" * 60,
            "",
            self.description,
            "",
            f"```{self.language}",
            self.code,
            "```",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "title": self.title,
            "description": self.description,
            "code": self.code,
            "category": self.category,
            "language": self.language,
        }


class ExampleRegistry:
    """示例注册表（单例）"""

    _instance: Optional["ExampleRegistry"] = None

    def __new__(cls) -> "ExampleRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._examples: Dict[str, CodeExample] = {}
        self._register_default_examples()

    def register(self, example: CodeExample) -> None:
        self._examples[example.example_id] = example

    def get(self, example_id: str) -> Optional[CodeExample]:
        return self._examples.get(example_id)

    def list_all(self, category: Optional[str] = None) -> List[CodeExample]:
        if category:
            return [e for e in self._examples.values()
                    if e.category == category]
        return list(self._examples.values())

    def list_categories(self) -> List[str]:
        return sorted(set(e.category for e in self._examples.values()))

    def search(self, query: str, limit: int = 10) -> List[CodeExample]:
        query_lower = query.lower()
        results = []
        for ex in self._examples.values():
            if (query_lower in ex.title.lower() or
                query_lower in ex.description.lower()):
                results.append(ex)
        return results[:limit]

    def _register_default_examples(self) -> None:
        for ex in _get_default_examples():
            self.register(ex)

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ============================================================================
# 默认数据
# ============================================================================

def _get_default_topics() -> List[HelpTopic]:
    """获取默认帮助主题"""
    return [
        HelpTopic(
            topic_id="getting_started",
            title="快速开始",
            category="基础",
            content=(
                "SEVA（自驱永续进化预测修正Agent系统）是一个自进化AI Agent框架。\n\n"
                "三步快速开始:\n"
                "1. 安装: pip install seva-agent\n"
                "2. 首次运行: seva wizard\n"
                "3. 启动运行: seva run --steps 100\n\n"
                "或使用Python API:\n"
                "    from seva import quickstart\n"
                "    agent, stats = quickstart(steps=100)"
            ),
            keywords=["安装", "开始", "入门", "install", "start"],
            examples=[
                "seva wizard",
                "seva run --steps 100 --preset easy",
                "python -c 'from seva import quickstart; quickstart(10)'",
            ],
            related=["installation", "configuration", "cli"],
        ),
        HelpTopic(
            topic_id="installation",
            title="安装指南",
            category="基础",
            content=(
                "SEVA支持多种安装方式:\n\n"
                "1. pip安装（推荐）:\n"
                "    pip install seva-agent\n\n"
                "2. 从源码安装:\n"
                "    git clone <repo>\n"
                "    cd seva-agent\n"
                "    pip install -e .\n\n"
                "3. 使用安装脚本:\n"
                "    ./install.sh\n\n"
                "依赖:\n"
                "    - Python 3.8+\n"
                "    - numpy\n"
                "    - pyyaml"
            ),
            keywords=["安装", "install", "pip", "源码"],
            related=["getting_started"],
        ),
        HelpTopic(
            topic_id="configuration",
            title="配置指南",
            category="基础",
            content=(
                "SEVA支持多种配置方式:\n\n"
                "1. 预设模板:\n"
                "    easy - 快速测试\n"
                "    standard - 标准运行\n"
                "    research - 研究用途\n"
                "    production - 生产环境\n\n"
                "2. 配置文件 (YAML):\n"
                "    seva config init  # 生成配置文件\n"
                "    seva run --config my_config.yaml\n\n"
                "3. 用户偏好:\n"
                "    seva prefs --set display.theme=dark\n\n"
                "4. 配置文件管理:\n"
                "    seva prefs --profile create work --from-current"
            ),
            keywords=["配置", "config", "预设", "preset", "yaml"],
            examples=[
                "seva config init",
                "seva run --config config.yaml",
                "seva prefs --set behavior.default_steps=200",
            ],
            related=["preferences", "cli"],
        ),
        HelpTopic(
            topic_id="cli",
            title="命令行工具",
            category="基础",
            content=(
                "SEVA提供丰富的CLI命令:\n\n"
                "核心命令:\n"
                "    seva run          运行Agent\n"
                "    seva status       查看状态\n"
                "    seva version      版本信息\n\n"
                "配置命令:\n"
                "    seva config       配置管理\n"
                "    seva prefs        用户偏好\n"
                "    seva wizard       首次运行向导\n\n"
                "工具命令:\n"
                "    seva shell        交互式Shell\n"
                "    seva dashboard    Web Dashboard\n"
                "    seva diagnose     系统诊断\n"
                "    seva perf         性能监控\n\n"
                "学习命令:\n"
                "    seva tutorial     交互式教程\n"
                "    seva completion   Shell补全\n"
                "    seva help         帮助系统"
            ),
            keywords=["命令", "cli", "command", "命令行"],
            related=["getting_started", "configuration"],
        ),
        HelpTopic(
            topic_id="api",
            title="Python API",
            category="开发",
            content=(
                "SEVA提供完整的Python API:\n\n"
                "快速启动:\n"
                "    from seva import quickstart\n"
                "    agent, stats = quickstart(steps=100)\n\n"
                "自定义配置:\n"
                "    from seva import SEVAFactory, PresetConfigs\n"
                "    agent = SEVAFactory.create(preset='easy')\n\n"
                "场景运行:\n"
                "    from seva import run_scenario, list_scenarios\n"
                "    result = run_scenario('quick_validation')\n\n"
                "会话管理:\n"
                "    from seva import save_session, load_session\n"
                "    save_session(agent, 'session.json')"
            ),
            keywords=["api", "python", "编程", "开发"],
            examples=[
                "from seva import quickstart; quickstart(10)",
                "from seva import SEVAFactory; SEVAFactory.create('easy')",
            ],
            related=["getting_started", "scenarios"],
        ),
        HelpTopic(
            topic_id="scenarios",
            title="预设场景",
            category="开发",
            content=(
                "SEVA内置多种预设场景:\n\n"
                "    quick_validation - 快速验证\n"
                "    standard_run - 标准运行\n"
                "    drift_test - 漂移测试\n"
                "    adversarial_test - 对抗测试\n\n"
                "使用场景:\n"
                "    seva scenarios list          # 列出场景\n"
                "    seva scenario run quick_validation  # 运行场景\n"
                "    seva scenario compare a b    # 对比场景"
            ),
            keywords=["场景", "scenario", "预设", "preset"],
            examples=[
                "seva scenarios list",
                "seva scenario run quick_validation",
            ],
            related=["api", "cli"],
        ),
        HelpTopic(
            topic_id="dashboard",
            title="Web Dashboard",
            category="工具",
            content=(
                "SEVA提供Web Dashboard可视化:\n\n"
                "启动Dashboard:\n"
                "    seva dashboard --port 8080\n\n"
                "或在Python中:\n"
                "    from seva import start_dashboard\n"
                "    start_dashboard(agent, port=8080)\n\n"
                "功能:\n"
                "    - 实时监控Agent状态\n"
                "    - 查看运行指标\n"
                "    - 导出报告"
            ),
            keywords=["dashboard", "web", "可视化", "监控"],
            related=["monitoring", "visualization"],
        ),
        HelpTopic(
            topic_id="shell",
            title="交互式Shell",
            category="工具",
            content=(
                "SEVA提供交互式Shell:\n\n"
                "启动Shell:\n"
                "    seva shell\n\n"
                "Shell命令:\n"
                "    help              显示帮助\n"
                "    run <steps>       运行指定步数\n"
                "    status            查看状态\n"
                "    config show       显示配置\n"
                "    save <file>       保存会话\n"
                "    load <file>       加载会话\n"
                "    exit              退出"
            ),
            keywords=["shell", "repl", "交互", "interactive"],
            related=["cli", "persistence"],
        ),
        HelpTopic(
            topic_id="persistence",
            title="会话持久化",
            category="工具",
            content=(
                "SEVA支持会话保存和恢复:\n\n"
                "保存会话:\n"
                "    seva session save --output my_session.json\n\n"
                "加载会话:\n"
                "    seva session load --input my_session.json\n\n"
                "检查点:\n"
                "    seva session checkpoint --steps 50\n\n"
                "Python API:\n"
                "    from seva import save_session, load_session\n"
                "    save_session(agent, 'session.json')\n"
                "    agent = load_session('session.json')"
            ),
            keywords=["持久化", "保存", "加载", "session", "检查点"],
            related=["shell", "api"],
        ),
        HelpTopic(
            topic_id="wizard",
            title="首次运行向导",
            category="引导",
            content=(
                "首次运行向导帮助新用户快速配置:\n\n"
                "启动向导:\n"
                "    seva wizard\n\n"
                "向导选项:\n"
                "    - 预设模板 (easy/standard/research/production)\n"
                "    - 环境类型 (mixed/blank/drift/adversarial)\n"
                "    - 运行步数 (10/50/100/500)\n\n"
                "静默模式:\n"
                "    seva wizard --silent --preset easy --steps 10\n\n"
                "重置向导:\n"
                "    seva wizard --reset"
            ),
            keywords=["向导", "wizard", "首次", "引导"],
            related=["getting_started", "tutorial"],
        ),
        HelpTopic(
            topic_id="tutorial",
            title="交互式教程",
            category="引导",
            content=(
                "SEVA提供3级交互式教程:\n\n"
                "    beginner    - 初学者教程 (5步)\n"
                "    intermediate - 中级教程 (5步)\n"
                "    advanced    - 高级教程 (5步)\n\n"
                "开始教程:\n"
                "    seva tutorial beginner\n\n"
                "列出教程:\n"
                "    seva tutorial --list\n\n"
                "查看进度:\n"
                "    seva tutorial beginner --progress\n\n"
                "重置进度:\n"
                "    seva tutorial beginner --reset"
            ),
            keywords=["教程", "tutorial", "学习", "教学"],
            related=["wizard", "getting_started"],
        ),
        HelpTopic(
            topic_id="performance",
            title="性能优化",
            category="高级",
            content=(
                "SEVA提供性能监控和优化工具:\n\n"
                "性能监控:\n"
                "    seva perf --startup    # 测量启动时间\n"
                "    seva perf --optimize   # 优化建议\n"
                "    seva perf --stats      # 性能统计\n\n"
                "Python API:\n"
                "    from seva import PerformanceMonitor\n"
                "    with PerformanceMonitor().measure('op'):\n"
                "        do_something()\n\n"
                "    from seva import LRUCache\n"
                "    cache = LRUCache(maxsize=100)"
            ),
            keywords=["性能", "performance", "优化", "监控", "缓存"],
            related=["preferences", "api"],
        ),
        HelpTopic(
            topic_id="preferences",
            title="用户偏好",
            category="高级",
            content=(
                "SEVA支持个性化设置:\n\n"
                "设置偏好:\n"
                "    seva prefs --set display.theme=dark\n\n"
                "查看偏好:\n"
                "    seva prefs --list\n"
                "    seva prefs --get display.theme\n\n"
                "配置文件:\n"
                "    seva prefs --profile create work --from-current\n"
                "    seva prefs --profile switch work\n\n"
                "分类:\n"
                "    display       - 显示设置\n"
                "    behavior      - 行为设置\n"
                "    performance   - 性能设置\n"
                "    notifications - 通知设置\n"
                "    advanced      - 高级设置"
            ),
            keywords=["偏好", "preferences", "个性化", "设置"],
            related=["configuration", "performance"],
        ),
        HelpTopic(
            topic_id="troubleshooting",
            title="故障排除",
            category="高级",
            content=(
                "常见问题排查:\n\n"
                "1. 系统诊断:\n"
                "    seva diagnose\n\n"
                "2. 查看错误码:\n"
                "    seva errors\n\n"
                "3. 查看日志:\n"
                "    seva logs show\n\n"
                "4. 性能问题:\n"
                "    seva perf --optimize\n\n"
                "5. 重置系统:\n"
                "    seva wizard --reset\n"
                "    seva prefs --reset all"
            ),
            keywords=["故障", "排除", "troubleshoot", "诊断", "错误"],
            related=["diagnostics", "performance"],
        ),
    ]


def _get_default_commands() -> List[CommandHelp]:
    """获取默认命令帮助"""
    return [
        CommandHelp(
            command="run",
            description="启动Agent运行",
            usage="seva run [options]",
            examples=[
                "seva run --steps 100",
                "seva run --preset easy --env-type mixed",
                "seva run --config my_config.yaml",
            ],
            options={
                "--steps": "运行步数 (默认: 100)",
                "--preset": "预设模板 (easy/standard/research/production)",
                "--env-type": "环境类型 (blank/mixed/drift/adversarial)",
                "--seed": "随机种子",
                "--config": "配置文件路径",
                "--output": "结果保存路径",
                "--verbose": "详细输出",
            },
            see_also=["config", "status", "scenarios"],
        ),
        CommandHelp(
            command="wizard",
            description="首次运行向导",
            usage="seva wizard [options]",
            examples=[
                "seva wizard",
                "seva wizard --silent --preset easy",
                "seva wizard --reset",
                "seva wizard --status",
            ],
            options={
                "--silent": "静默模式",
                "--preset": "预设模板",
                "--env-type": "环境类型",
                "--steps": "运行步数",
                "--reset": "重置向导状态",
                "--status": "查看向导状态",
                "--tips": "显示使用提示",
            },
            see_also=["tutorial", "prefs"],
        ),
        CommandHelp(
            command="tutorial",
            description="交互式教程",
            usage="seva tutorial [tutorial_id] [options]",
            examples=[
                "seva tutorial beginner",
                "seva tutorial --list",
                "seva tutorial beginner --progress",
                "seva tutorial beginner --reset",
            ],
            options={
                "tutorial_id": "教程ID (beginner/intermediate/advanced)",
                "--list": "列出所有教程",
                "--silent": "静默模式",
                "--no-resume": "不恢复进度",
                "--reset": "重置进度",
                "--progress": "查看进度",
            },
            see_also=["wizard", "completion"],
        ),
        CommandHelp(
            command="completion",
            description="生成Shell补全脚本",
            usage="seva completion [options]",
            examples=[
                "seva completion --shell bash",
                "seva completion --shell zsh --output ~/.zsh/completion/_seva",
                "seva completion --suggest 'wiz'",
                "seva completion --list",
            ],
            options={
                "--shell": "Shell类型 (bash/zsh)",
                "--output": "输出文件路径",
                "--suggest": "拼写纠错建议",
                "--list": "列出所有命令",
                "--help-cmd": "获取命令帮助",
            },
            see_also=["tutorial", "help"],
        ),
        CommandHelp(
            command="perf",
            description="性能监控与优化",
            usage="seva perf [options]",
            examples=[
                "seva perf --startup",
                "seva perf --optimize",
                "seva perf --stats",
                "seva perf --save perf.json",
            ],
            options={
                "--startup": "测量启动时间",
                "--optimize": "优化Agent配置",
                "--stats": "查看详细统计",
                "--report": "查看性能报告",
                "--save": "保存性能数据",
                "--reset": "重置监控数据",
                "--preset": "预设模板",
                "--env-type": "环境类型",
            },
            see_also=["prefs", "diagnose"],
        ),
        CommandHelp(
            command="prefs",
            description="用户偏好与个性化设置",
            usage="seva prefs [options]",
            examples=[
                "seva prefs --list",
                "seva prefs --set display.theme=dark",
                "seva prefs --get display.theme",
                "seva prefs --profile create work --from-current",
                "seva prefs --profile switch work",
            ],
            options={
                "--get": "获取偏好值",
                "--set": "设置偏好值 (key=value)",
                "--reset": "重置偏好 (key 或 all)",
                "--list": "列出所有偏好",
                "--schema": "列出偏好定义",
                "--category": "按分类筛选",
                "--export": "导出偏好到文件",
                "--import-file": "从文件导入偏好",
                "--profile": "配置文件操作",
            },
            see_also=["config", "perf"],
        ),
        CommandHelp(
            command="help",
            description="帮助系统",
            usage="seva help [topic] [options]",
            examples=[
                "seva help",
                "seva help getting_started",
                "seva help --search 配置",
                "seva help --list",
                "seva help --faq",
            ],
            options={
                "topic": "主题ID",
                "--search": "搜索主题",
                "--list": "列出所有主题",
                "--categories": "列出分类",
                "--faq": "常见问题",
                "--glossary": "术语表",
                "--examples": "代码示例",
                "--command": "命令帮助",
            },
            see_also=["tutorial", "wizard"],
        ),
        CommandHelp(
            command="config",
            description="配置管理",
            usage="seva config <action> [options]",
            examples=[
                "seva config init",
                "seva config show",
                "seva config validate --file config.yaml",
            ],
            options={
                "init": "生成配置文件",
                "show": "显示当前配置",
                "validate": "校验配置文件",
            },
            see_also=["prefs", "run"],
        ),
        CommandHelp(
            command="shell",
            description="启动交互式Shell",
            usage="seva shell [options]",
            examples=[
                "seva shell",
                "seva shell --preset standard",
            ],
            options={
                "--preset": "预设模板",
                "--env-type": "环境类型",
                "--seed": "随机种子",
            },
            see_also=["exec", "run"],
        ),
        CommandHelp(
            command="diagnose",
            description="诊断系统",
            usage="seva diagnose [options]",
            examples=[
                "seva diagnose",
                "seva diagnose --steps 20",
                "seva diagnose --output report.json",
            ],
            options={
                "--steps": "诊断前运行步数",
                "--preset": "预设模板",
                "--output": "诊断报告保存路径",
            },
            see_also=["errors", "perf"],
        ),
    ]


def _get_default_faqs() -> List[FAQEntry]:
    """获取默认FAQ"""
    return [
        FAQEntry(
            question="如何安装SEVA？",
            answer=(
                "使用pip安装: pip install seva-agent\n"
                "或从源码安装: git clone <repo> && pip install -e .\n"
                "安装后运行 seva wizard 进行首次配置。"
            ),
            category="安装",
            keywords=["安装", "install", "pip"],
        ),
        FAQEntry(
            question="如何配置SEVA？",
            answer=(
                "三种方式:\n"
                "1. 首次运行向导: seva wizard\n"
                "2. 配置文件: seva config init && seva run --config config.yaml\n"
                "3. 用户偏好: seva prefs --set <key>=<value>"
            ),
            category="配置",
            keywords=["配置", "config", "设置"],
        ),
        FAQEntry(
            question="如何运行第一次模拟？",
            answer=(
                "最简单的方式:\n"
                "  seva run --steps 100 --preset easy\n\n"
                "或使用Python API:\n"
                "  from seva import quickstart\n"
                "  agent, stats = quickstart(steps=100)"
            ),
            category="使用",
            keywords=["运行", "模拟", "开始"],
        ),
        FAQEntry(
            question="如何保存和加载会话？",
            answer=(
                "保存: seva session save --output session.json\n"
                "加载: seva session load --input session.json\n"
                "检查点: seva session checkpoint --steps 50\n\n"
                "Python API:\n"
                "  from seva import save_session, load_session\n"
                "  save_session(agent, 'session.json')"
            ),
            category="使用",
            keywords=["保存", "加载", "会话", "session"],
        ),
        FAQEntry(
            question="如何使用Web Dashboard？",
            answer=(
                "启动: seva dashboard --port 8080\n"
                "然后访问 http://localhost:8080\n\n"
                "Python API:\n"
                "  from seva import start_dashboard\n"
                "  start_dashboard(agent, port=8080)"
            ),
            category="使用",
            keywords=["dashboard", "web", "可视化"],
        ),
        FAQEntry(
            question="如何生成报告？",
            answer=(
                "导出HTML报告: seva visualize --format html --output report.html\n"
                "生成图表: seva visualize --charts ./charts/\n\n"
                "Python API:\n"
                "  from seva import visualize, generate_charts\n"
                "  visualize(agent, 'html', 'report.html')"
            ),
            category="使用",
            keywords=["报告", "report", "导出", "可视化"],
        ),
        FAQEntry(
            question="遇到错误如何排查？",
            answer=(
                "1. 运行诊断: seva diagnose\n"
                "2. 查看错误码: seva errors\n"
                "3. 查看日志: seva logs show\n"
                "4. 性能优化: seva perf --optimize\n"
                "5. 查看帮助: seva help troubleshooting"
            ),
            category="故障排除",
            keywords=["错误", "排查", "诊断", "troubleshoot"],
        ),
        FAQEntry(
            question="如何启用Shell补全？",
            answer=(
                "Bash: seva completion --shell bash --output ~/.bash_completion.d/seva\n"
                "Zsh: seva completion --shell zsh --output ~/.zsh/completion/_seva\n"
                "然后重启Shell或 source 配置文件。"
            ),
            category="使用",
            keywords=["补全", "completion", "shell", "bash", "zsh"],
        ),
        FAQEntry(
            question="如何学习使用SEVA？",
            answer=(
                "1. 首次运行向导: seva wizard\n"
                "2. 交互式教程: seva tutorial beginner\n"
                "3. 查看帮助: seva help\n"
                "4. 查看示例: seva help --examples"
            ),
            category="学习",
            keywords=["学习", "教程", "tutorial", "入门"],
        ),
        FAQEntry(
            question="如何个性化设置？",
            answer=(
                "设置偏好: seva prefs --set display.theme=dark\n"
                "创建配置文件: seva prefs --profile create work --from-current\n"
                "切换配置: seva prefs --profile switch work\n"
                "查看所有偏好: seva prefs --list"
            ),
            category="配置",
            keywords=["偏好", "个性化", "设置", "profile"],
        ),
    ]


def _get_default_terms() -> List[GlossaryTerm]:
    """获取默认术语"""
    return [
        GlossaryTerm("Agent", "自驱永续进化Agent，系统的核心实体", "核心"),
        GlossaryTerm("Cycle", "一个完整的16阶段处理周期", "核心"),
        GlossaryTerm("Perception", "感知阶段，从环境获取输入", "阶段"),
        GlossaryTerm("DriveEngine", "驱动引擎，产生内在动机", "阶段"),
        GlossaryTerm("GoalGenerator", "目标生成器，产生具体目标", "阶段"),
        GlossaryTerm("Predictor", "预测器，预测未来状态", "阶段"),
        GlossaryTerm("DecisionMaker", "决策器，选择最佳行动", "阶段"),
        GlossaryTerm("SafetyEthics", "安全伦理模块，约束行动", "阶段"),
        GlossaryTerm("Executor", "执行器，执行决策", "阶段"),
        GlossaryTerm("ErrorEvaluator", "误差评估器，评估预测误差", "阶段"),
        GlossaryTerm("TaskManager", "任务管理器，管理任务队列", "阶段"),
        GlossaryTerm("SkillLibrary", "技能库，存储已学技能", "阶段"),
        GlossaryTerm("Evolution", "进化模块，调整系统参数", "阶段"),
        GlossaryTerm("Optimizer", "优化器，优化系统性能", "阶段"),
        GlossaryTerm("Memory", "记忆系统，存储经验", "阶段"),
        GlossaryTerm("Metacognition", "元认知，监控系统状态", "阶段"),
        GlossaryTerm("TimingControl", "时序控制，协调各阶段", "阶段"),
        GlossaryTerm("Environment", "环境，提供输入和反馈", "核心"),
        GlossaryTerm("Preset", "预设模板，快速配置系统", "配置"),
        GlossaryTerm("Scenario", "场景，预定义的运行配置", "配置"),
        GlossaryTerm("Session", "会话，可保存和恢复的运行状态", "持久化"),
        GlossaryTerm("Checkpoint", "检查点，会话的快照", "持久化"),
        GlossaryTerm("Dashboard", "Web仪表盘，可视化监控", "工具"),
        GlossaryTerm("Wizard", "向导，引导式配置", "工具"),
        GlossaryTerm("Tutorial", "教程，交互式学习", "工具"),
    ]


def _get_default_examples() -> List[CodeExample]:
    """获取默认代码示例"""
    return [
        CodeExample(
            example_id="quickstart",
            title="快速启动",
            description="最简单的启动方式，一行代码运行Agent",
            code=(
                "from seva import quickstart\n"
                "\n"
                "# 运行100步，返回agent和统计信息\n"
                "agent, stats = quickstart(steps=100)\n"
                "\n"
                "print(f\"完成步数: {stats['steps_completed']}\")\n"
                "print(f\"总奖励: {stats['total_real_reward']:.4f}\")"
            ),
            category="基础",
        ),
        CodeExample(
            example_id="custom_config",
            title="自定义配置",
            description="使用预设模板和自定义参数创建Agent",
            code=(
                "from seva import SEVAFactory, run\n"
                "\n"
                "# 使用预设模板创建\n"
                "agent = SEVAFactory.create(\n"
                "    preset='standard',\n"
                "    env_type='drift',\n"
                "    seed=42\n"
                ")\n"
                "\n"
                "# 运行\n"
                "stats = run(agent, steps=200, verbose=True)"
            ),
            category="基础",
        ),
        CodeExample(
            example_id="scenarios",
            title="运行场景",
            description="运行预定义场景并对比结果",
            code=(
                "from seva import run_scenario, list_scenarios, compare_scenarios\n"
                "\n"
                "# 列出所有场景\n"
                "print(list_scenarios())\n"
                "\n"
                "# 运行场景\n"
                "result = run_scenario('quick_validation')\n"
                "\n"
                "# 对比多个场景\n"
                "comparison = compare_scenarios('quick_validation', 'standard_run')"
            ),
            category="基础",
        ),
        CodeExample(
            example_id="monitoring",
            title="监控运行",
            description="使用监控和日志系统",
            code=(
                "from seva import SEVAFactory, run, setup_logging\n"
                "from seva import MetricsCollector, export_metrics\n"
                "\n"
                "# 设置日志\n"
                "setup_logging(level='INFO')\n"
                "\n"
                "# 创建并运行\n"
                "agent = SEVAFactory.create(preset='easy')\n"
                "stats = run(agent, steps=100)\n"
                "\n"
                "# 导出指标\n"
                "export_metrics(agent, 'metrics.json')"
            ),
            category="高级",
        ),
        CodeExample(
            example_id="persistence",
            title="会话持久化",
            description="保存和恢复Agent会话",
            code=(
                "from seva import SEVAFactory, run, save_session, load_session\n"
                "\n"
                "# 创建并运行\n"
                "agent = SEVAFactory.create(preset='easy')\n"
                "run(agent, steps=50)\n"
                "\n"
                "# 保存会话\n"
                "save_session(agent, 'my_session.json')\n"
                "\n"
                "# 加载会话\n"
                "agent2 = load_session('my_session.json')\n"
                "run(agent2, steps=50)"
            ),
            category="高级",
        ),
        CodeExample(
            example_id="performance",
            title="性能监控",
            description="使用性能监控工具",
            code=(
                "from seva import PerformanceMonitor, LRUCache, benchmark\n"
                "\n"
                "# 监控操作\n"
                "monitor = PerformanceMonitor()\n"
                "with monitor.measure('my_operation'):\n"
                "    do_something()\n"
                "\n"
                "print(monitor.get_report())\n"
                "\n"
                "# 使用缓存\n"
                "cache = LRUCache(maxsize=100)\n"
                "value = cache.get_or_compute('key', expensive_func)"
            ),
            category="高级",
        ),
        CodeExample(
            example_id="preferences",
            title="用户偏好",
            description="设置和管理用户偏好",
            code=(
                "from seva import (set_preference, get_preference,\n"
                "                  create_profile, switch_profile)\n"
                "\n"
                "# 设置偏好\n"
                "set_preference('display.theme', 'dark')\n"
                "set_preference('behavior.default_steps', 200)\n"
                "\n"
                "# 获取偏好\n"
                "theme = get_preference('display.theme')\n"
                "\n"
                "# 配置文件管理\n"
                "create_profile('work', from_current=True)\n"
                "switch_profile('work')"
            ),
            category="高级",
        ),
        CodeExample(
            example_id="visualization",
            title="可视化与报告",
            description="生成可视化报告和图表",
            code=(
                "from seva import SEVAFactory, run, visualize, generate_charts\n"
                "\n"
                "agent = SEVAFactory.create(preset='standard')\n"
                "run(agent, steps=200)\n"
                "\n"
                "# 导出HTML报告\n"
                "visualize(agent, 'html', 'report.html')\n"
                "\n"
                "# 生成SVG图表\n"
                "generate_charts(agent, './charts/')"
            ),
            category="高级",
        ),
    ]


# ============================================================================
# 便捷函数
# ============================================================================

def show_help(topic_id: str) -> str:
    """显示帮助主题"""
    return HelpSystem().show(topic_id)


def search_help(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """搜索帮助主题"""
    return HelpSystem().search(query, limit)


def list_help_topics(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出帮助主题"""
    return HelpSystem().list_topics(category)


def list_help_categories() -> List[str]:
    """列出帮助分类"""
    return HelpSystem().list_categories()


def get_help_report() -> str:
    """获取帮助系统报告"""
    return HelpSystem().get_report()


def get_command_help(command: str) -> Optional[str]:
    """获取命令帮助"""
    cmd = CommandHelpRegistry().get(command)
    return cmd.format() if cmd else None


def list_commands() -> List[str]:
    """列出所有命令"""
    return CommandHelpRegistry().list_commands()


def search_faq(query: str, limit: int = 10) -> List[FAQEntry]:
    """搜索FAQ"""
    return FAQRegistry().search(query, limit)


def list_faqs(category: Optional[str] = None) -> List[FAQEntry]:
    """列出FAQ"""
    return FAQRegistry().list_all(category)


def get_glossary_term(term: str) -> Optional[str]:
    """获取术语定义"""
    t = Glossary().get(term)
    return t.format() if t else None


def search_glossary(query: str, limit: int = 10) -> List[GlossaryTerm]:
    """搜索术语"""
    return Glossary().search(query, limit)


def list_glossary(category: Optional[str] = None) -> List[GlossaryTerm]:
    """列出术语"""
    return Glossary().list_all(category)


def get_example(example_id: str) -> Optional[CodeExample]:
    """获取代码示例"""
    return ExampleRegistry().get(example_id)


def list_examples(category: Optional[str] = None) -> List[CodeExample]:
    """列出代码示例"""
    return ExampleRegistry().list_all(category)


def search_examples(query: str, limit: int = 10) -> List[CodeExample]:
    """搜索代码示例"""
    return ExampleRegistry().search(query, limit)


def help(topic: Optional[str] = None,
         search: Optional[str] = None,
         list_all: bool = False,
         categories: bool = False,
         faq: bool = False,
         glossary: bool = False,
         examples: bool = False,
         command: Optional[str] = None) -> str:
    """
    统一帮助入口

    Args:
        topic: 主题ID
        search: 搜索查询
        list_all: 列出所有主题
        categories: 列出分类
        faq: 显示FAQ
        glossary: 显示术语表
        examples: 显示示例
        command: 命令名

    Returns:
        帮助文本
    """
    if topic:
        return show_help(topic)
    if search:
        results = search_help(search)
        if not results:
            return f"未找到匹配 '{search}' 的主题"
        lines = [f"搜索 '{search}' 的结果 ({len(results)}个):", "=" * 60]
        for r in results:
            lines.append(f"  [{r['category']}] {r['topic_id']}: {r['title']}"
                         f" (相关度: {r['score']})")
        return "\n".join(lines)
    if list_all:
        topics = list_help_topics()
        lines = [f"所有帮助主题 ({len(topics)}个):", "=" * 60]
        current_cat = None
        for t in topics:
            if t["category"] != current_cat:
                current_cat = t["category"]
                lines.append(f"\n[{current_cat}]")
            lines.append(f"  {t['topic_id']}: {t['title']}")
        return "\n".join(lines)
    if categories:
        cats = list_help_categories()
        lines = ["帮助分类:", "=" * 60]
        for cat in cats:
            count = len(HelpSystem().get_topics_by_category(cat))
            lines.append(f"  {cat} ({count}个主题)")
        return "\n".join(lines)
    if faq:
        faqs = list_faqs()
        lines = [f"常见问题 ({len(faqs)}个):", "=" * 60]
        for i, f in enumerate(faqs, 1):
            lines.append(f"\n{i}. {f.question}")
            lines.append(f"   {f.answer[:100]}...")
        return "\n".join(lines)
    if glossary:
        terms = list_glossary()
        lines = [f"术语表 ({len(terms)}个):", "=" * 60]
        for t in sorted(terms, key=lambda x: x.term):
            lines.append(f"  {t.term}: {t.definition}")
        return "\n".join(lines)
    if examples:
        exs = list_examples()
        lines = [f"代码示例 ({len(exs)}个):", "=" * 60]
        current_cat = None
        for ex in exs:
            if ex.category != current_cat:
                current_cat = ex.category
                lines.append(f"\n[{current_cat}]")
            lines.append(f"  {ex.example_id}: {ex.title}")
        return "\n".join(lines)
    if command:
        cmd_help = get_command_help(command)
        return cmd_help or f"未找到命令: {command}"
    # 默认显示报告
    return get_help_report()


__all__ = [
    "HelpTopic",
    "HelpSystem",
    "CommandHelp",
    "CommandHelpRegistry",
    "FAQEntry",
    "FAQRegistry",
    "GlossaryTerm",
    "Glossary",
    "CodeExample",
    "ExampleRegistry",
    "show_help",
    "search_help",
    "list_help_topics",
    "list_help_categories",
    "get_help_report",
    "get_command_help",
    "list_commands",
    "search_faq",
    "list_faqs",
    "get_glossary_term",
    "search_glossary",
    "list_glossary",
    "get_example",
    "list_examples",
    "search_examples",
    "help",
]
