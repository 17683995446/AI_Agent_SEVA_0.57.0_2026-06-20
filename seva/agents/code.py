"""
编程Agent（阶段66）

对应第一性原理：
    1. 编程Agent = 代码生成 + 代码执行 + 代码调试的闭环
    2. 安全第一：代码执行必须在沙箱中,限制权限与资源
    3. 反馈驱动：执行结果反馈给LLM进行调试
    4. 增量改进：每次修复一个问题,逐步完善代码
    5. 可观测性：所有代码生成/执行/调试过程可被监控

模块组成：
    - CodeLanguage: 支持的编程语言枚举
    - CodeBlock/ExecutionResult: 代码块与执行结果
    - CodeSandbox: 代码执行沙箱 (subprocess隔离/超时/资源限制)
    - CodeGenerator: LLM驱动代码生成
    - CodeReviewer: LLM驱动代码审查
    - CodeDebugger: LLM驱动代码调试
    - CodeAgent: 编程Agent (生成→执行→调试闭环)
    - 全局单例: get_code_agent
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import uuid
import logging
import subprocess
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import Message, LLMError, get_llm_client
from ..llm_loop import parse_json_output
from .base import (
    BaseAgent, AgentConfig, AgentState, AgentRunResult,
    AgentEvent, AgentEventType,
)


logger = logging.getLogger("seva.agents.code")


# ============================================================================
# 枚举
# ============================================================================

class CodeLanguage(Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SHELL = "shell"
    SQL = "sql"
    JSON = "json"
    TEXT = "text"


class ExecutionStatus(Enum):
    """执行状态"""
    SUCCESS = "success"          # 成功
    FAILED = "failed"            # 失败(代码错误)
    TIMEOUT = "timeout"          # 超时
    ERROR = "error"              # 系统错误
    BLOCKED = "blocked"          # 被阻止(安全限制)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class CodeBlock:
    """代码块"""
    code: str = ""
    language: CodeLanguage = CodeLanguage.PYTHON
    filename: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "language": self.language.value,
            "filename": self.filename,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeBlock":
        return cls(
            code=data.get("code", ""),
            language=CodeLanguage(data.get("language", "python")),
            filename=data.get("filename"),
            description=data.get("description", ""),
        )


@dataclass
class ExecutionResult:
    """执行结果"""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    latency_ms: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "latency_ms": self.latency_ms,
            "output_files": self.output_files,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


@dataclass
class ReviewResult:
    """代码审查结果"""
    approved: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-1
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "score": self.score,
            "summary": self.summary,
        }


# ============================================================================
# 代码沙箱
# ============================================================================

class CodeSandbox:
    """
    代码执行沙箱

    安全措施:
        1. subprocess隔离: 子进程执行,不影响主进程
        2. 超时限制: 防止死循环
        3. 工作目录: 临时目录,执行后清理
        4. 环境变量: 限制敏感环境变量
        5. 禁用危险操作: 阻止网络/文件系统访问(可选)
    """

    # 危险导入关键词
    DANGEROUS_IMPORTS = [
        "os.system", "subprocess.call", "subprocess.run",
        "subprocess.Popen", "os.popen", "os.exec",
        "shutil.rmtree", "os.remove", "os.unlink",
    ]

    def __init__(self, work_dir: Optional[str] = None,
                 timeout: float = 10.0,
                 max_output_size: int = 10000,
                 enable_network: bool = False):
        self._work_dir = work_dir or tempfile.gettempdir()
        self._timeout = timeout
        self._max_output_size = max_output_size
        self._enable_network = enable_network
        self._lock = threading.RLock()
        self.stats = {
            "executions": 0,
            "successful": 0,
            "failed": 0,
            "blocked": 0,
            "timeouts": 0,
        }

    def execute(self, code: str,
                language: CodeLanguage = CodeLanguage.PYTHON,
                stdin: Optional[str] = None) -> ExecutionResult:
        """
        执行代码

        Args:
            code: 代码字符串
            language: 编程语言
            stdin: 标准输入

        Returns:
            执行结果
        """
        with self._lock:
            self.stats["executions"] += 1

        t0 = time.time()

        # 安全检查
        if not self._enable_network:
            blocked = self._check_safety(code)
            if blocked:
                with self._lock:
                    self.stats["blocked"] += 1
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    error=f"代码包含危险操作: {blocked}",
                    latency_ms=(time.time() - t0) * 1000,
                )

        # 创建临时工作目录
        with tempfile.TemporaryDirectory(dir=self._work_dir) as tmpdir:
            try:
                result = self._execute_in_tmpdir(code, language, stdin, tmpdir)
                result.latency_ms = (time.time() - t0) * 1000

                with self._lock:
                    if result.success:
                        self.stats["successful"] += 1
                    elif result.status == ExecutionStatus.TIMEOUT:
                        self.stats["timeouts"] += 1
                    else:
                        self.stats["failed"] += 1

                return result

            except Exception as e:
                logger.error(f"[Sandbox] 执行异常: {e}")
                with self._lock:
                    self.stats["failed"] += 1
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"{type(e).__name__}: {e}",
                    latency_ms=(time.time() - t0) * 1000,
                )

    def _check_safety(self, code: str) -> Optional[str]:
        """安全检查,返回被阻止的危险操作"""
        for dangerous in self.DANGEROUS_IMPORTS:
            if dangerous in code:
                return dangerous
        return None

    def _execute_in_tmpdir(self, code: str, language: CodeLanguage,
                           stdin: Optional[str], tmpdir: str) -> ExecutionResult:
        """在临时目录中执行代码"""
        if language == CodeLanguage.PYTHON:
            return self._execute_python(code, stdin, tmpdir)
        elif language == CodeLanguage.SHELL:
            return self._execute_shell(code, stdin, tmpdir)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._execute_javascript(code, stdin, tmpdir)
        else:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"不支持的语言: {language.value}",
            )

    def _execute_python(self, code: str, stdin: Optional[str],
                        tmpdir: str) -> ExecutionResult:
        """执行Python代码"""
        # 写入临时文件
        code_file = os.path.join(tmpdir, "code.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        # 构造命令
        cmd = [sys.executable, code_file]

        # 限制环境变量
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "PYTHONPATH": tmpdir,
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        }

        try:
            result = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=tmpdir,
                env=env,
            )

            stdout = result.stdout[:self._max_output_size]
            stderr = result.stderr[:self._max_output_size]

            if result.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result.returncode,
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result.returncode,
                )

        except subprocess.TimeoutExpired as e:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                error=f"执行超时({self._timeout}秒)",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
            )

    def _execute_shell(self, code: str, stdin: Optional[str],
                       tmpdir: str) -> ExecutionResult:
        """执行Shell代码"""
        try:
            result = subprocess.run(
                code,
                shell=True,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=tmpdir,
            )
            stdout = result.stdout[:self._max_output_size]
            stderr = result.stderr[:self._max_output_size]

            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED
            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"执行超时({self._timeout}秒)",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
            )

    def _execute_javascript(self, code: str, stdin: Optional[str],
                            tmpdir: str) -> ExecutionResult:
        """执行JavaScript代码(需要node)"""
        code_file = os.path.join(tmpdir, "code.js")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["node", code_file],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=tmpdir,
            )
            stdout = result.stdout[:self._max_output_size]
            stderr = result.stderr[:self._max_output_size]

            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED
            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
            )
        except FileNotFoundError:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error="Node.js未安装",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"执行超时({self._timeout}秒)",
            )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 代码生成器
# ============================================================================

class CodeGenerator:
    """LLM驱动的代码生成器"""

    GENERATE_PROMPT = """你是代码生成专家。请根据需求生成代码。

需求: {requirement}
语言: {language}
{constraints}

请返回JSON格式:
{{
    "code": "代码内容",
    "description": "代码说明",
    "test_cases": ["测试用例1", "测试用例2"]
}}

只返回JSON,不要其他内容。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"generated": 0, "failures": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def generate(self, requirement: str,
                 language: CodeLanguage = CodeLanguage.PYTHON,
                 constraints: Optional[str] = None) -> Tuple[Optional[CodeBlock], Optional[str]]:
        """
        生成代码

        Returns:
            (代码块, 错误信息)
        """
        prompt = self.GENERATE_PROMPT.format(
            requirement=requirement,
            language=language.value,
            constraints=f"约束: {constraints}" if constraints else "",
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是代码生成专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.2,
                    max_tokens=1024,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed or "code" not in parsed:
                with self._lock:
                    self.stats["failures"] += 1
                return None, "LLM返回无效JSON"

            code_block = CodeBlock(
                code=parsed["code"],
                language=language,
                description=parsed.get("description", ""),
            )

            with self._lock:
                self.stats["generated"] += 1
            return code_block, None

        except LLMError as e:
            with self._lock:
                self.stats["failures"] += 1
            return None, f"LLM错误: {e}"
        except Exception as e:
            with self._lock:
                self.stats["failures"] += 1
            return None, f"{type(e).__name__}: {e}"

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 代码审查器
# ============================================================================

class CodeReviewer:
    """LLM驱动的代码审查器"""

    REVIEW_PROMPT = """请审查以下代码,评估质量并识别问题。

代码:
```
{code}
```

语言: {language}

请返回JSON:
{{
    "approved": true/false,
    "score": 0.8,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "summary": "总体评价"
}}

只返回JSON。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"reviewed": 0, "approved": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def review(self, code_block: CodeBlock) -> ReviewResult:
        """审查代码"""
        prompt = self.REVIEW_PROMPT.format(
            code=code_block.code,
            language=code_block.language.value,
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是代码审查专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.3,
                    max_tokens=512,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed:
                return self._fallback_review(code_block)

            with self._lock:
                self.stats["reviewed"] += 1
                if parsed.get("approved", False):
                    self.stats["approved"] += 1

            return ReviewResult(
                approved=bool(parsed.get("approved", False)),
                issues=parsed.get("issues", []),
                suggestions=parsed.get("suggestions", []),
                score=float(parsed.get("score", 0.0)),
                summary=parsed.get("summary", ""),
            )

        except LLMError as e:
            logger.warning(f"[Reviewer] LLM错误: {e}")
            return self._fallback_review(code_block)
        except Exception as e:
            logger.error(f"[Reviewer] 异常: {e}")
            return self._fallback_review(code_block)

    def _fallback_review(self, code_block: CodeBlock) -> ReviewResult:
        """回退审查: 基于规则的简单评估"""
        issues = []
        if not code_block.code.strip():
            issues.append("代码为空")
        if "eval(" in code_block.code:
            issues.append("使用eval()不安全")
        if "exec(" in code_block.code:
            issues.append("使用exec()不安全")

        return ReviewResult(
            approved=len(issues) == 0,
            issues=issues,
            score=0.5 if not issues else 0.2,
            summary="基于规则的简单审查",
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 代码调试器
# ============================================================================

class CodeDebugger:
    """LLM驱动的代码调试器"""

    DEBUG_PROMPT = """请调试以下代码,修复错误。

原始代码:
```
{code}
```

错误信息:
{error}

请返回JSON:
{{
    "fixed_code": "修复后的代码",
    "explanation": "修复说明",
    "changes": ["修改1", "修改2"]
}}

只返回JSON。"""

    def __init__(self, llm_client=None):
        self._client = llm_client
        self._lock = threading.RLock()
        self.stats = {"debugged": 0, "fixed": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def debug(self, code_block: CodeBlock,
              error: str) -> Tuple[Optional[CodeBlock], Optional[str]]:
        """
        调试代码

        Returns:
            (修复后的代码块, 错误信息)
        """
        prompt = self.DEBUG_PROMPT.format(
            code=code_block.code,
            error=error,
        )

        try:
            client = self.client
            original_cache = client.use_cache
            client.use_cache = False
            try:
                response = client.chat(
                    [Message.system("你是代码调试专家,只返回JSON。"),
                     Message.user(prompt)],
                    temperature=0.2,
                    max_tokens=1024,
                )
            finally:
                client.use_cache = original_cache

            parsed = parse_json_output(response.content)
            if not parsed or "fixed_code" not in parsed:
                with self._lock:
                    self.stats["debugged"] += 1
                return None, "LLM返回无效JSON"

            fixed = CodeBlock(
                code=parsed["fixed_code"],
                language=code_block.language,
                description=parsed.get("explanation", ""),
            )

            with self._lock:
                self.stats["debugged"] += 1
                self.stats["fixed"] += 1
            return fixed, None

        except LLMError as e:
            with self._lock:
                self.stats["debugged"] += 1
            return None, f"LLM错误: {e}"
        except Exception as e:
            with self._lock:
                self.stats["debugged"] += 1
            return None, f"{type(e).__name__}: {e}"

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)


# ============================================================================
# 编程Agent配置
# ============================================================================

@dataclass
class CodeAgentConfig(AgentConfig):
    """编程Agent配置"""
    name: str = "code_agent"
    description: str = "编程Agent"
    # 沙箱配置
    sandbox_timeout: float = 10.0
    sandbox_max_output: int = 10000
    enable_network: bool = False
    # 调试配置
    max_debug_iterations: int = 3
    enable_review: bool = True
    min_review_score: float = 0.5
    # 系统提示
    system_prompt: Optional[str] = "你是编程专家,帮助用户生成、执行和调试代码。"


# ============================================================================
# 编程Agent
# ============================================================================

class CodeAgent(BaseAgent):
    """
    编程Agent

    工作流:
        1. 理解需求
        2. 生成代码 (CodeGenerator)
        3. 审查代码 (CodeReviewer, 可选)
        4. 执行代码 (CodeSandbox)
        5. 如失败,调试修复 (CodeDebugger)
        6. 重复4-5直到成功或达到最大迭代
        7. 返回结果
    """

    def __init__(self, agent_id: Optional[str] = None,
                 config: Optional[CodeAgentConfig] = None):
        super().__init__(agent_id=agent_id,
                        config=config or CodeAgentConfig())
        self.sandbox = CodeSandbox(
            timeout=self.config.sandbox_timeout,
            max_output_size=self.config.sandbox_max_output,
            enable_network=self.config.enable_network,
        )
        self.generator = CodeGenerator()
        self.reviewer = CodeReviewer()
        self.debugger = CodeDebugger()

    @property
    def code_config(self) -> CodeAgentConfig:
        return self.config

    def generate_code(self, requirement: str,
                      language: CodeLanguage = CodeLanguage.PYTHON,
                      constraints: Optional[str] = None) -> Tuple[Optional[CodeBlock], Optional[str]]:
        """生成代码"""
        return self.generator.generate(requirement, language, constraints)

    def execute_code(self, code: str,
                     language: CodeLanguage = CodeLanguage.PYTHON,
                     stdin: Optional[str] = None) -> ExecutionResult:
        """执行代码"""
        return self.sandbox.execute(code, language, stdin)

    def review_code(self, code_block: CodeBlock) -> ReviewResult:
        """审查代码"""
        return self.reviewer.review(code_block)

    def debug_code(self, code_block: CodeBlock,
                   error: str) -> Tuple[Optional[CodeBlock], Optional[str]]:
        """调试代码"""
        return self.debugger.debug(code_block, error)

    def _run_impl(self, input: Union[str, Dict[str, Any]],
                  **kwargs) -> AgentRunResult:
        """编程Agent核心实现"""
        # 解析输入
        if isinstance(input, str):
            requirement = input
            language = CodeLanguage(kwargs.get("language", "python"))
            execute = kwargs.get("execute", True)
            constraints = kwargs.get("constraints")
        else:
            requirement = input.get("requirement", "")
            language = CodeLanguage(input.get("language", "python"))
            execute = input.get("execute", True)
            constraints = input.get("constraints")

        if not requirement:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error="需求为空",
            )

        t0 = time.time()
        tokens_used = 0

        # 1. 生成代码
        code_block, error = self.generator.generate(
            requirement, language, constraints
        )
        if code_block is None:
            return AgentRunResult(
                agent_id=self.agent_id,
                session_id=self.context.session_id,
                success=False,
                error=f"代码生成失败: {error}",
                latency_ms=(time.time() - t0) * 1000,
            )

        # 2. 审查代码(可选)
        review_result = None
        if self.code_config.enable_review:
            review_result = self.reviewer.review(code_block)
            if review_result.score < self.code_config.min_review_score:
                logger.warning(f"[CodeAgent] 代码审查分数低: {review_result.score}")

        # 3. 执行代码
        execution_result = None
        debug_iterations = 0
        current_code = code_block

        if execute:
            while debug_iterations <= self.code_config.max_debug_iterations:
                execution_result = self.sandbox.execute(
                    current_code.code, current_code.language
                )

                if execution_result.success:
                    break

                # 执行失败,尝试调试
                if debug_iterations >= self.code_config.max_debug_iterations:
                    break

                error_msg = execution_result.stderr or execution_result.error or "未知错误"
                logger.info(f"[CodeAgent] 调试迭代 {debug_iterations + 1}")

                fixed_block, debug_error = self.debugger.debug(
                    current_code, error_msg
                )
                if fixed_block is None:
                    break

                current_code = fixed_block
                debug_iterations += 1

        # 构造输出
        output_parts = [f"生成的代码 ({language.value}):"]
        output_parts.append(f"```{language.value}")
        output_parts.append(current_code.code)
        output_parts.append("```")

        if review_result:
            output_parts.append(f"\n审查结果: {'通过' if review_result.approved else '未通过'} "
                               f"(分数: {review_result.score:.1f})")
            if review_result.issues:
                output_parts.append(f"问题: {', '.join(review_result.issues)}")

        if execution_result:
            if execution_result.success:
                output_parts.append(f"\n执行成功:")
                if execution_result.stdout:
                    output_parts.append(f"输出:\n{execution_result.stdout}")
            else:
                output_parts.append(f"\n执行失败: {execution_result.error or execution_result.stderr}")

        if debug_iterations > 0:
            output_parts.append(f"\n调试迭代: {debug_iterations}次")

        return AgentRunResult(
            agent_id=self.agent_id,
            session_id=self.context.session_id,
            success=execution_result.success if execution_result else True,
            output="\n".join(output_parts),
            tokens_used=tokens_used,
            latency_ms=(time.time() - t0) * 1000,
            metadata={
                "code": current_code.to_dict(),
                "review": review_result.to_dict() if review_result else None,
                "execution": execution_result.to_dict() if execution_result else None,
                "debug_iterations": debug_iterations,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取详细统计"""
        base_stats = super().get_stats()
        base_stats.update({
            "sandbox": self.sandbox.get_stats(),
            "generator": self.generator.get_stats(),
            "reviewer": self.reviewer.get_stats(),
            "debugger": self.debugger.get_stats(),
        })
        return base_stats


# ============================================================================
# 全局单例
# ============================================================================

_global_code_agent: Optional[CodeAgent] = None
_global_lock = threading.RLock()


def get_code_agent() -> CodeAgent:
    """获取全局编程Agent"""
    global _global_code_agent
    with _global_lock:
        if _global_code_agent is None:
            _global_code_agent = CodeAgent()
        return _global_code_agent


def reset_code_agent():
    """重置全局编程Agent"""
    global _global_code_agent
    with _global_lock:
        _global_code_agent = None
