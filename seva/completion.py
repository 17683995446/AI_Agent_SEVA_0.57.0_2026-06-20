"""
阶段42: 智能命令补全与提示

提供命令行智能补全、模糊匹配、拼写纠错功能，减少用户记忆负担。

核心能力:
    1. 命令补全: 前缀匹配、模糊匹配
    2. 拼写纠错: Levenshtein距离算法推荐相似命令
    3. 参数提示: 命令参数自动补全
    4. 命令历史: 记录和搜索历史命令
    5. Shell集成: Bash/Zsh补全脚本生成

快速启动:
    from seva import CommandCompleter, suggest_command

    completer = CommandCompleter()
    suggestions = completer.complete("ru")  # ["run", "report"]
    suggestions = completer.suggest("rn")   # ["run"] (拼写纠错)

    # 便捷函数
    suggestions = suggest_command("stat")
"""
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
# Levenshtein距离算法
# =====================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的Levenshtein编辑距离

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        编辑距离（插入/删除/替换的最小操作数）
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_score(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度分数（0-1，1为完全相同）

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        相似度分数
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


# =====================================================================
# 命令补全器
# =====================================================================

class CommandCompleter:
    """
    命令补全器

    提供命令前缀匹配、模糊匹配、拼写纠错功能。

    Example:
        >>> completer = CommandCompleter()
        >>> completer.complete("ru")  # ["run", "report"]
        >>> completer.suggest("stat")  # ["status"]
    """

    # SEVA所有命令及其参数
    COMMANDS = {
        "run": {
            "description": "启动Agent运行",
            "args": ["--steps", "--preset", "--env-type", "--seed",
                     "--verbose", "--output", "--config"],
        },
        "config": {
            "description": "配置管理",
            "subcommands": ["init", "show", "validate", "save", "load",
                            "wizard", "status"],
        },
        "status": {
            "description": "查看系统状态",
            "args": ["--preset", "--steps"],
        },
        "version": {
            "description": "版本信息",
            "args": [],
        },
        "presets": {
            "description": "列出预设模板",
            "args": [],
        },
        "demo": {
            "description": "运行演示",
            "args": ["--steps", "--preset"],
        },
        "scenarios": {
            "description": "列出/运行预设场景",
            "subcommands": ["list", "run", "compare", "run-all"],
        },
        "scenario": {
            "description": "场景管理",
            "subcommands": ["run", "compare", "run-all", "list"],
        },
        "shell": {
            "description": "启动交互式Shell",
            "args": [],
        },
        "exec": {
            "description": "执行单条Shell命令",
            "args": [],
        },
        "web": {
            "description": "启动Web Dashboard",
            "args": ["--host", "--port", "--preset", "--steps"],
        },
        "report": {
            "description": "生成报告",
            "args": ["--output", "--preset", "--steps", "--format"],
        },
        "export": {
            "description": "导出多格式报告",
            "args": ["--format", "--output", "--preset", "--steps"],
        },
        "charts": {
            "description": "生成SVG图表",
            "args": ["--output-dir", "--preset", "--steps"],
        },
        "compare": {
            "description": "比较多个运行",
            "args": ["--steps", "--format", "--output"],
        },
        "session": {
            "description": "会话持久化管理",
            "subcommands": ["save", "load", "list", "info", "delete",
                            "checkpoint"],
        },
        "wizard": {
            "description": "首次运行向导",
            "args": ["--silent", "--preset", "--env-type", "--steps",
                     "--reset", "--status", "--tips"],
        },
        "diagnose": {
            "description": "诊断系统",
            "args": ["--steps", "--preset", "--output"],
        },
        "errors": {
            "description": "列出所有错误码",
            "args": [],
        },
        "monitor": {
            "description": "实时监控",
            "args": ["--preset", "--steps", "--interval"],
        },
        "logs": {
            "description": "日志管理",
            "args": ["--level", "--tail", "--follow"],
        },
    }

    # 补全建议的最大数量
    MAX_SUGGESTIONS = 5
    # 拼写纠错的最小相似度阈值
    MIN_SIMILARITY = 0.5

    def __init__(self):
        """初始化补全器"""
        self._commands = list(self.COMMANDS.keys())

    def complete(self, prefix: str) -> List[str]:
        """
        前缀补全

        Args:
            prefix: 命令前缀

        Returns:
            匹配的命令列表（按字母排序）

        Example:
            >>> completer = CommandCompleter()
            >>> completer.complete("ru")  # ["run"]
            >>> completer.complete("s")   # ["scenarios", "session", ...]
        """
        if not prefix:
            return sorted(self._commands)

        prefix_lower = prefix.lower()
        matches = [cmd for cmd in self._commands
                   if cmd.lower().startswith(prefix_lower)]
        return sorted(matches)

    def fuzzy_match(self, query: str) -> List[str]:
        """
        模糊匹配（包含关系）

        Args:
            query: 查询字符串

        Returns:
            匹配的命令列表

        Example:
            >>> completer.fuzzy_match("ess")  # ["session", "presets"]
        """
        if not query:
            return []

        query_lower = query.lower()
        matches = [cmd for cmd in self._commands
                   if query_lower in cmd.lower()]
        return sorted(matches)

    def suggest(self, query: str,
                min_similarity: Optional[float] = None) -> List[str]:
        """
        拼写纠错建议

        使用Levenshtein距离找到最相似的命令

        Args:
            query: 用户输入（可能拼写错误）
            min_similarity: 最小相似度阈值（默认使用MIN_SIMILARITY）

        Returns:
            建议的命令列表（按相似度降序）

        Example:
            >>> completer.suggest("stat")   # ["status"]
            >>> completer.suggest("ruun")   # ["run"]
        """
        if not query:
            return []

        threshold = min_similarity if min_similarity is not None \
            else self.MIN_SIMILARITY

        scored = []
        for cmd in self._commands:
            score = similarity_score(query, cmd)
            if score >= threshold:
                scored.append((cmd, score))

        # 按相似度降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cmd for cmd, _ in scored[:self.MAX_SUGGESTIONS]]

    def get_command_info(self, command: str) -> Optional[Dict]:
        """
        获取命令详细信息

        Args:
            command: 命令名

        Returns:
            命令信息字典，包含description/args/subcommands
        """
        return self.COMMANDS.get(command)

    def complete_args(self, command: str, prefix: str = "") -> List[str]:
        """
        补全命令参数

        Args:
            command: 命令名
            prefix: 参数前缀

        Returns:
            匹配的参数列表
        """
        info = self.get_command_info(command)
        if not info:
            return []

        args = info.get("args", [])
        if not prefix:
            return sorted(args)

        prefix_lower = prefix.lower()
        return sorted([a for a in args if a.lower().startswith(prefix_lower)])

    def complete_subcommands(self, command: str,
                             prefix: str = "") -> List[str]:
        """
        补全子命令

        Args:
            command: 命令名
            prefix: 子命令前缀

        Returns:
            匹配的子命令列表
        """
        info = self.get_command_info(command)
        if not info:
            return []

        subcommands = info.get("subcommands", [])
        if not prefix:
            return sorted(subcommands)

        prefix_lower = prefix.lower()
        return sorted([s for s in subcommands
                       if s.lower().startswith(prefix_lower)])

    def get_all_commands(self) -> List[str]:
        """获取所有命令列表"""
        return sorted(self._commands)

    def format_suggestions(self, query: str,
                           suggestions: List[str]) -> str:
        """
        格式化建议输出

        Args:
            query: 原始输入
            suggestions: 建议列表

        Returns:
            格式化的提示字符串
        """
        if not suggestions:
            return f"未找到与 '{query}' 匹配的命令"

        lines = [f"您是不是想输入:"]
        for cmd in suggestions:
            info = self.get_command_info(cmd)
            desc = info["description"] if info else ""
            lines.append(f"  {cmd:<15} {desc}")
        return "\n".join(lines)


# =====================================================================
# 命令历史管理
# =====================================================================

class CommandHistory:
    """
    命令历史管理

    记录用户执行的命令，支持搜索和回放。

    Example:
        >>> history = CommandHistory()
        >>> history.add("run --steps 100")
        >>> history.search("run")  # ["run --steps 100"]
    """

    MAX_HISTORY = 1000

    def __init__(self, history_file: Optional[str] = None):
        """
        初始化命令历史

        Args:
            history_file: 历史文件路径（可选，默认~/.seva/history.json）
        """
        if history_file is None:
            history_file = os.path.join(os.path.expanduser("~/.seva"),
                                        "history.json")
        self.history_file = history_file
        self._history: List[Dict] = []
        self._load()

    def add(self, command: str) -> None:
        """
        添加命令到历史

        Args:
            command: 命令字符串
        """
        if not command.strip():
            return

        entry = {
            "command": command.strip(),
            "timestamp": time.time(),
        }
        self._history.append(entry)

        # 限制历史长度
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

        self._save()

    def search(self, query: str) -> List[str]:
        """
        搜索历史命令

        Args:
            query: 搜索关键词

        Returns:
            匹配的命令列表（按时间倒序）
        """
        query_lower = query.lower()
        matches = [e["command"] for e in reversed(self._history)
                   if query_lower in e["command"].lower()]
        return matches

    def get_recent(self, n: int = 10) -> List[str]:
        """
        获取最近的N条命令

        Args:
            n: 数量

        Returns:
            命令列表
        """
        return [e["command"] for e in reversed(self._history[-n:])]

    def clear(self) -> None:
        """清空历史"""
        self._history = []
        self._save()

    def size(self) -> int:
        """获取历史记录数"""
        return len(self._history)

    def _load(self) -> None:
        """加载历史"""
        if not os.path.exists(self.history_file):
            self._history = []
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        except Exception:
            self._history = []

    def _save(self) -> None:
        """保存历史"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# =====================================================================
# Shell补全脚本生成器
# =====================================================================

class ShellCompletionGenerator:
    """
    Shell补全脚本生成器

    生成Bash/Zsh的命令补全脚本。

    Example:
        >>> gen = ShellCompletionGenerator()
        >>> gen.generate_bash()  # 生成bash补全脚本
        >>> gen.generate_zsh()   # 生成zsh补全脚本
    """

    def __init__(self):
        self.completer = CommandCompleter()

    def generate_bash(self) -> str:
        """生成Bash补全脚本"""
        commands = self.completer.get_all_commands()
        commands_str = " ".join(commands)

        script = f"""# SEVA Agent Bash补全脚本
# 安装: source <(seva completion bash) 或添加到 ~/.bashrc

_seva_completions() {{
    local cur prev commands
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    commands="{commands_str}"

    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=($(compgen -W "$commands" -- "$cur"))
    elif [ $COMP_CWORD -ge 2 ]; then
        case "${{COMP_WORDS[1]}}" in
            config|scenario|scenarios|session)
                local subcmds=""
                case "${{COMP_WORDS[1]}}" in
                    config) subcmds="init show validate save load wizard status" ;;
                    scenario|scenarios) subcmds="list run compare run-all" ;;
                    session) subcmds="save load list info delete checkpoint" ;;
                esac
                COMPREPLY=($(compgen -W "$subcmds" -- "$cur"))
                ;;
            run|status|demo|web|report|export|charts|compare|diagnose|monitor)
                local opts="--steps --preset --env-type --seed --verbose --output --config --format --host --port --output-dir --interval --level --tail --follow"
                COMPREPLY=($(compgen -W "$opts" -- "$cur"))
                ;;
            wizard)
                local opts="--silent --preset --env-type --steps --reset --status --tips"
                COMPREPLY=($(compgen -W "$opts" -- "$cur"))
                ;;
        esac
    fi
}}

complete -F _seva_completions seva
"""
        return script

    def generate_zsh(self) -> str:
        """生成Zsh补全脚本"""
        commands = self.completer.get_all_commands()
        commands_str = " ".join(commands)

        script = """#compdef seva
# SEVA Agent Zsh补全脚本
# 安装: source <(seva completion zsh) 或添加到 ~/.zshrc

_seva() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \\
        "1:command:(""" + commands_str + """)" \\
        "*::arg:->args"

    case $state in
      args)
        case ${line[1]} in
            config)
                _arguments "1:subcommand:(init show validate save load wizard status)"
                ;;
            scenario|scenarios)
                _arguments "1:subcommand:(list run compare run-all)"
                ;;
            session)
                _arguments "1:subcommand:(save load list info delete checkpoint)"
                ;;
            run|status|demo|web|report|export|charts|compare|diagnose|monitor)
                _arguments "--steps[步数]" "--preset[预设]" "--env-type[环境类型]" "--format[格式]" "--output[输出]"
                ;;
            wizard)
                _arguments "--silent[静默]" "--preset[预设]" "--env-type[环境]" "--steps[步数]" "--reset[重置]" "--status[状态]" "--tips[提示]"
                ;;
        esac
      ;;
    esac
}

_seva "$@"
"""
        return script

    def save_bash(self, file_path: str) -> str:
        """保存Bash补全脚本"""
        script = self.generate_bash()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script)
        return file_path

    def save_zsh(self, file_path: str) -> str:
        """保存Zsh补全脚本"""
        script = self.generate_zsh()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script)
        return file_path


# =====================================================================
# 便捷函数
# =====================================================================

def suggest_command(query: str) -> List[str]:
    """
    命令建议（拼写纠错）

    Args:
        query: 用户输入

    Returns:
        建议的命令列表

    Example:
        >>> suggest_command("stat")  # ["status"]
        >>> suggest_command("ruun")  # ["run"]
    """
    completer = CommandCompleter()
    return completer.suggest(query)


def complete_command(prefix: str) -> List[str]:
    """
    命令补全

    Args:
        prefix: 命令前缀

    Returns:
        匹配的命令列表

    Example:
        >>> complete_command("ru")  # ["run"]
        >>> complete_command("s")   # ["scenarios", "session", ...]
    """
    completer = CommandCompleter()
    return completer.complete(prefix)


def did_you_mean(query: str) -> str:
    """
    "您是不是想输入..."提示

    Args:
        query: 用户输入

    Returns:
        格式化的提示字符串

    Example:
        >>> print(did_you_mean("stat"))
        您是不是想输入:
          status          查看系统状态
    """
    completer = CommandCompleter()
    suggestions = completer.suggest(query)
    return completer.format_suggestions(query, suggestions)


def get_command_help(command: str) -> Optional[str]:
    """
    获取命令帮助信息

    Args:
        command: 命令名

    Returns:
        帮助字符串，命令不存在则返回None
    """
    completer = CommandCompleter()
    info = completer.get_command_info(command)
    if not info:
        return None

    lines = [f"命令: {command}"]
    lines.append(f"描述: {info['description']}")

    if info.get("args"):
        lines.append(f"参数: {', '.join(info['args'])}")
    if info.get("subcommands"):
        lines.append(f"子命令: {', '.join(info['subcommands'])}")

    return "\n".join(lines)
