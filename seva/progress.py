"""
进度反馈与状态指示（阶段49）
对应第一性原理：透明 = 用户始终知道系统在做什么

提供能力:
    - ProgressBar: 进度条（百分比/ETA/速率）
    - Spinner: 旋转动画（不确定进度的等待）
    - StatusIndicator: 状态指示器（运行/暂停/完成/错误）
    - TaskProgress: 任务进度跟踪（多任务并行）
    - 便捷函数: progress_bar / spinner / status / track_progress

使用示例:
    from seva import ProgressBar, track_progress

    # 进度条
    with ProgressBar("处理中", total=100) as bar:
        for i in range(100):
            do_something()
            bar.update(1)

    # 跟踪迭代
    for item in track_progress(items, "处理"):
        process(item)
"""
from __future__ import annotations

import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Iterator, Iterable
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_CHARS_ASCII = ["|", "/", "-", "\\"]

STATUS_ICONS = {
    "running": "▶",
    "paused": "⏸",
    "completed": "✓",
    "error": "✗",
    "warning": "⚠",
    "idle": "○",
    "pending": "…",
}

STATUS_ICONS_ASCII = {
    "running": ">",
    "paused": "||",
    "completed": "OK",
    "error": "X",
    "warning": "!",
    "idle": "o",
    "pending": ".",
}


# ============================================================================
# ProgressBar - 进度条
# ============================================================================

class ProgressBar:
    """
    进度条

    用法:
        with ProgressBar("下载", total=1000) as bar:
            for i in range(1000):
                download_chunk()
                bar.update(1)

        # 或手动控制
        bar = ProgressBar("处理", total=100)
        bar.start()
        for i in range(100):
            bar.update(1)
        bar.finish()
    """

    def __init__(self, description: str = "",
                 total: int = 100,
                 width: int = 30,
                 output: Any = None,
                 use_unicode: bool = True) -> None:
        self.description = description
        self.total = max(total, 1)
        self.width = width
        self.output = output or sys.stderr
        self.use_unicode = use_unicode
        self._current = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        return self._current

    @property
    def percentage(self) -> float:
        return (self._current / self.total) * 100

    @property
    def is_complete(self) -> bool:
        return self._current >= self.total

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    @property
    def eta(self) -> Optional[float]:
        """预估剩余时间（秒）"""
        if self._current == 0 or self._start_time is None:
            return None
        elapsed = time.time() - self._start_time
        rate = self._current / elapsed
        if rate == 0:
            return None
        remaining = self.total - self._current
        return remaining / rate

    @property
    def rate(self) -> float:
        """速率（项/秒）"""
        if self._start_time is None or self._current == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._current / elapsed

    def start(self) -> None:
        """开始进度条"""
        self._start_time = time.time()
        self._running = True
        self._render()

    def update(self, n: int = 1) -> None:
        """更新进度"""
        with self._lock:
            self._current = min(self._current + n, self.total)
            if self._running:
                self._render()

    def set_progress(self, current: int) -> None:
        """直接设置进度"""
        with self._lock:
            self._current = max(0, min(current, self.total))
            if self._running:
                self._render()

    def finish(self) -> None:
        """完成进度条"""
        with self._lock:
            self._current = self.total
            self._end_time = time.time()
            self._running = False
            self._render()
            self._write("\n")

    def _render(self) -> None:
        """渲染进度条"""
        pct = self.percentage
        filled = int(self.width * self._current / self.total)
        empty = self.width - filled

        if self.use_unicode:
            bar_char = "█"
            empty_char = "░"
        else:
            bar_char = "#"
            empty_char = "-"

        bar = bar_char * filled + empty_char * empty

        # ETA
        eta_str = ""
        if self.eta is not None and not self.is_complete:
            eta_str = f" ETA: {self._format_time(self.eta)}"

        # 速率
        rate_str = ""
        if self.rate > 0:
            rate_str = f" {self.rate:.1f}it/s"

        line = f"\r{self.description}: [{bar}] {pct:5.1f}%{rate_str}{eta_str}"
        self._write(line)

    def _write(self, text: str) -> None:
        """写入输出"""
        try:
            self.output.write(text)
            self.output.flush()
        except (IOError, ValueError):
            pass

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def __enter__(self) -> "ProgressBar":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.finish()
        else:
            self._running = False
            self._write("\n")


# ============================================================================
# Spinner - 旋转动画
# ============================================================================

class Spinner:
    """
    旋转动画（用于不确定进度的等待）

    用法:
        with Spinner("加载中..."):
            do_something_slow()

        # 或手动控制
        spinner = Spinner("处理中")
        spinner.start()
        do_something()
        spinner.succeed("完成")
    """

    def __init__(self, message: str = "",
                 output: Any = None,
                 use_unicode: bool = True,
                 interval: float = 0.1) -> None:
        self.message = message
        self.output = output or sys.stderr
        self.use_unicode = use_unicode
        self.interval = interval
        self._chars = SPINNER_CHARS if use_unicode else SPINNER_CHARS_ASCII
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._char_index = 0

    def start(self) -> None:
        """开始旋转"""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, final_message: str = "") -> None:
        """停止旋转"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # 清除行
        self._write("\r" + " " * (len(self.message) + 10) + "\r")
        if final_message:
            self._write(final_message + "\n")

    def succeed(self, message: str = "") -> None:
        """成功停止"""
        icon = "✓" if self.use_unicode else "OK"
        msg = message or self.message
        self.stop(f"{icon} {msg}")

    def fail(self, message: str = "") -> None:
        """失败停止"""
        icon = "✗" if self.use_unicode else "X"
        msg = message or self.message
        self.stop(f"{icon} {msg}")

    def warn(self, message: str = "") -> None:
        """警告停止"""
        icon = "⚠" if self.use_unicode else "!"
        msg = message or self.message
        self.stop(f"{icon} {msg}")

    def _spin(self) -> None:
        """旋转线程"""
        while self._running:
            char = self._chars[self._char_index % len(self._chars)]
            self._write(f"\r{char} {self.message}")
            self._char_index += 1
            time.sleep(self.interval)

    def _write(self, text: str) -> None:
        try:
            self.output.write(text)
            self.output.flush()
        except (IOError, ValueError):
            pass

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.succeed()
        else:
            self.fail()


# ============================================================================
# StatusIndicator - 状态指示器
# ============================================================================

class StatusIndicator:
    """
    状态指示器

    用法:
        status = StatusIndicator("数据处理")
        status.set_running()
        do_work()
        status.set_completed()
        print(status.format())
    """

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    WARNING = "warning"
    IDLE = "idle"
    PENDING = "pending"

    def __init__(self, name: str = "",
                 initial_status: str = "idle",
                 use_unicode: bool = True) -> None:
        self.name = name
        self._status = initial_status
        self._use_unicode = use_unicode
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._message: str = ""
        self._progress: Optional[float] = None  # 0-100

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == self.RUNNING

    @property
    def is_completed(self) -> bool:
        return self._status == self.COMPLETED

    @property
    def is_error(self) -> bool:
        return self._status == self.ERROR

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def set_running(self, message: str = "") -> None:
        self._status = self.RUNNING
        self._start_time = time.time()
        self._end_time = None
        self._message = message

    def set_paused(self, message: str = "") -> None:
        self._status = self.PAUSED
        self._message = message

    def set_completed(self, message: str = "") -> None:
        self._status = self.COMPLETED
        self._end_time = time.time()
        self._message = message

    def set_error(self, message: str = "") -> None:
        self._status = self.ERROR
        self._end_time = time.time()
        self._message = message

    def set_warning(self, message: str = "") -> None:
        self._status = self.WARNING
        self._message = message

    def set_idle(self, message: str = "") -> None:
        self._status = self.IDLE
        self._message = message

    def set_pending(self, message: str = "") -> None:
        self._status = self.PENDING
        self._message = message

    def set_progress(self, progress: float) -> None:
        """设置进度（0-100）"""
        self._progress = max(0, min(100, progress))

    def get_icon(self) -> str:
        """获取状态图标"""
        icons = STATUS_ICONS if self._use_unicode else STATUS_ICONS_ASCII
        return icons.get(self._status, "?")

    def format(self) -> str:
        """格式化状态"""
        icon = self.get_icon()
        parts = [f"{icon} {self.name}"]
        if self._message:
            parts.append(f": {self._message}")
        if self._progress is not None:
            parts.append(f" ({self._progress:.1f}%)")
        if self._start_time:
            parts.append(f" [{self._format_time(self.elapsed_time)}]")
        return "".join(parts)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self._status,
            "message": self._message,
            "progress": self._progress,
            "elapsed_time": self.elapsed_time,
        }


# ============================================================================
# TaskProgress - 任务进度跟踪
# ============================================================================

class TaskProgress:
    """
    多任务进度跟踪

    用法:
        tracker = TaskProgress()
        task_id = tracker.add_task("下载数据", total=100)
        tracker.update_task(task_id, 50)
        tracker.complete_task(task_id)
        print(tracker.format())
    """

    def __init__(self, use_unicode: bool = True) -> None:
        self._use_unicode = use_unicode
        self._tasks: Dict[int, StatusIndicator] = {}
        self._task_totals: Dict[int, int] = {}
        self._task_currents: Dict[int, int] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def add_task(self, name: str, total: int = 100) -> int:
        """添加任务，返回任务ID"""
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            indicator = StatusIndicator(name, use_unicode=self._use_unicode)
            indicator.set_pending()
            self._tasks[task_id] = indicator
            self._task_totals[task_id] = total
            self._task_currents[task_id] = 0
            return task_id

    def start_task(self, task_id: int) -> None:
        """开始任务"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].set_running()

    def update_task(self, task_id: int, increment: int = 1) -> None:
        """更新任务进度"""
        with self._lock:
            if task_id not in self._tasks:
                return
            self._task_currents[task_id] = min(
                self._task_currents[task_id] + increment,
                self._task_totals[task_id])
            current = self._task_currents[task_id]
            total = self._task_totals[task_id]
            progress = (current / total) * 100 if total > 0 else 0
            self._tasks[task_id].set_progress(progress)
            if self._tasks[task_id].status == StatusIndicator.PENDING:
                self._tasks[task_id].set_running()

    def complete_task(self, task_id: int, message: str = "") -> None:
        """完成任务"""
        with self._lock:
            if task_id in self._tasks:
                self._task_currents[task_id] = self._task_totals[task_id]
                self._tasks[task_id].set_progress(100)
                self._tasks[task_id].set_completed(message)

    def fail_task(self, task_id: int, message: str = "") -> None:
        """任务失败"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].set_error(message)

    def get_task(self, task_id: int) -> Optional[StatusIndicator]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[int, StatusIndicator]:
        """获取所有任务"""
        return dict(self._tasks)

    @property
    def total_tasks(self) -> int:
        return len(self._tasks)

    @property
    def completed_tasks(self) -> int:
        return sum(1 for t in self._tasks.values()
                   if t.is_completed)

    @property
    def overall_progress(self) -> float:
        """总体进度（0-100）"""
        if not self._tasks:
            return 0.0
        total_progress = sum(
            (self._task_currents.get(tid, 0) / self._task_totals.get(tid, 1))
            for tid in self._tasks
        )
        return (total_progress / len(self._tasks)) * 100

    def format(self) -> str:
        """格式化所有任务状态"""
        lines = [f"任务进度 ({self.completed_tasks}/{self.total_tasks}完成, "
                 f"{self.overall_progress:.1f}%)", "=" * 60]
        for task_id, indicator in self._tasks.items():
            lines.append(f"  {indicator.format()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "overall_progress": self.overall_progress,
            "tasks": {str(tid): ind.to_dict()
                      for tid, ind in self._tasks.items()},
        }


# ============================================================================
# 便捷函数
# ============================================================================

def progress_bar(description: str, total: int,
                 use_unicode: bool = True) -> ProgressBar:
    """创建进度条"""
    return ProgressBar(description, total, use_unicode=use_unicode)


def spinner(message: str = "",
            use_unicode: bool = True) -> Spinner:
    """创建旋转动画"""
    return Spinner(message, use_unicode=use_unicode)


def status(name: str, initial_status: str = "idle",
           use_unicode: bool = True) -> StatusIndicator:
    """创建状态指示器"""
    return StatusIndicator(name, initial_status, use_unicode)


def track_progress(iterable: Iterable, description: str = "",
                   total: Optional[int] = None,
                   use_unicode: bool = True) -> Iterator:
    """
    跟踪迭代进度

    用法:
        for item in track_progress(items, "处理"):
            process(item)
    """
    items = list(iterable) if total is None else iterable
    if total is None:
        total = len(items)

    bar = ProgressBar(description, total, use_unicode=use_unicode)
    bar.start()
    try:
        for item in items:
            yield item
            bar.update(1)
    finally:
        bar.finish()


def track_progress_async(iterable: Iterable, description: str = "",
                         total: Optional[int] = None) -> Iterator:
    """跟踪迭代进度（ASCII模式，适用于无Unicode环境）"""
    return track_progress(iterable, description, total, use_unicode=False)


def format_status(status_str: str, use_unicode: bool = True) -> str:
    """格式化状态字符串"""
    icons = STATUS_ICONS if use_unicode else STATUS_ICONS_ASCII
    return icons.get(status_str, "?")


def get_progress_report(tracker: TaskProgress) -> str:
    """获取进度报告"""
    return tracker.format()


__all__ = [
    "ProgressBar",
    "Spinner",
    "StatusIndicator",
    "TaskProgress",
    "SPINNER_CHARS",
    "SPINNER_CHARS_ASCII",
    "STATUS_ICONS",
    "STATUS_ICONS_ASCII",
    "progress_bar",
    "spinner",
    "status",
    "track_progress",
    "track_progress_async",
    "format_status",
    "get_progress_report",
]
