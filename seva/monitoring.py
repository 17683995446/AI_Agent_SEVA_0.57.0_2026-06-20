"""
日志可视化与监控模块（阶段32）
对应第一性原理：易用 = 运行状态可见

提供：
- StructuredLogger: 结构化日志(JSON格式可选)
- MetricsCollector: 运行时指标采集与聚合
- Dashboard: 实时仪表盘(终端输出)
- LogFormatter: 日志格式化器(支持text/json两种格式)
- monitor(): 监控装饰器

使用示例:
    from seva.monitoring import StructuredLogger, Dashboard, monitor

    # 结构化日志
    logger = StructuredLogger("my_app", format="json")
    logger.info("启动", extra={"preset": "easy"})

    # 仪表盘
    dashboard = Dashboard(agent)
    dashboard.render()

    # 监控装饰器
    @monitor
    def my_function():
        ...
"""
import sys
import os
import time
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict


class StructuredLogger:
    """
    结构化日志器
    支持text和json两种格式
    """
    def __init__(self, name: str, format: str = "text",
                 level: int = logging.INFO,
                 output_file: str = None):
        """
        参数:
            name: 日志器名称
            format: 格式 (text/json)
            level: 日志级别
            output_file: 输出文件路径(可选)
        """
        self.name = name
        self.format = format
        self.level = level
        self.output_file = output_file
        self._records: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

        # 配置底层logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        # 避免重复添加handler
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(file_handler)

    def _log(self, level: int, level_name: str, message: str,
             extra: Dict = None):
        """记录日志"""
        record = {
            "timestamp": time.time(),
            "level": level_name,
            "logger": self.name,
            "message": message,
        }
        if extra:
            record.update(extra)

        with self._lock:
            self._records.append(record)

        if level >= self.level:
            if self.format == "json":
                output = json.dumps(record, ensure_ascii=False,
                                    default=str)
            else:
                # text格式
                ts = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(record["timestamp"]))
                extra_str = ""
                if extra:
                    extra_str = " " + " ".join(
                        f"{k}={v}" for k, v in extra.items()
                        if k not in ("timestamp", "level", "logger", "message")
                    )
                output = f"{ts} [{level_name}] {self.name}: {message}{extra_str}"
            self._logger.log(level, output)

    def debug(self, message: str, extra: Dict = None):
        self._log(logging.DEBUG, "DEBUG", message, extra)

    def info(self, message: str, extra: Dict = None):
        self._log(logging.INFO, "INFO", message, extra)

    def warning(self, message: str, extra: Dict = None):
        self._log(logging.WARNING, "WARNING", message, extra)

    def error(self, message: str, extra: Dict = None):
        self._log(logging.ERROR, "ERROR", message, extra)

    def critical(self, message: str, extra: Dict = None):
        self._log(logging.CRITICAL, "CRITICAL", message, extra)

    def get_records(self, level: str = None, limit: int = 100) -> List[Dict]:
        """获取日志记录"""
        with self._lock:
            records = list(self._records)
        if level:
            records = [r for r in records if r["level"] == level]
        return records[-limit:]

    def clear(self):
        """清空记录"""
        with self._lock:
            self._records.clear()


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    指标采集器
    采集和聚合运行时指标
    """
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        self._lock = threading.Lock()

    def record(self, name: str, value: float, tags: Dict = None):
        """记录指标"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        with self._lock:
            self._metrics[name].append(point)

    def get_metric(self, name: str, limit: int = 100) -> List[MetricPoint]:
        """获取指标历史"""
        with self._lock:
            points = list(self._metrics.get(name, []))
        return points[-limit:]

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        """获取最新指标值"""
        with self._lock:
            points = self._metrics.get(name, [])
            return points[-1] if points else None

    def get_stats(self, name: str, window: int = 100) -> Dict[str, float]:
        """获取指标统计"""
        points = self.get_metric(name, window)
        if not points:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "last": 0}
        values = [p.value for p in points]
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }

    def list_metrics(self) -> List[str]:
        """列出所有指标名"""
        with self._lock:
            return list(self._metrics.keys())

    def clear(self):
        """清空所有指标"""
        with self._lock:
            self._metrics.clear()


class Dashboard:
    """
    实时仪表盘
    在终端输出Agent运行状态
    """
    def __init__(self, agent=None, metrics_collector: MetricsCollector = None):
        self.agent = agent
        self.collector = metrics_collector or MetricsCollector()
        self._last_render = 0

    def collect_from_agent(self):
        """从Agent采集指标"""
        if not self.agent:
            return

        metrics = self.agent.get_metrics()
        self.collector.record("cycle_count", metrics["cycle_count"])
        self.collector.record("total_reward", metrics["total_reward"])
        self.collector.record("avg_cycle_time",
                            metrics["avg_cycle_time"] * 1000)  # ms
        self.collector.record("avg_prediction_error",
                            metrics["avg_prediction_error"])
        self.collector.record("avg_surprise", metrics["avg_surprise"])

        # 任务/技能/进化指标
        task_stats = metrics.get("task_stats", {})
        skill_stats = metrics.get("skill_stats", {})
        evo_stats = metrics.get("evolution_stats", {})
        self.collector.record("tasks_completed",
                            task_stats.get("total_completed", 0))
        self.collector.record("skills_total",
                            skill_stats.get("total_skills", 0))
        self.collector.record("evolution_evaluations",
                            evo_stats.get("total_evaluations", 0))

    def render(self, compact: bool = False) -> str:
        """
        渲染仪表盘
        返回格式化的仪表盘字符串
        """
        self.collect_from_agent()
        lines = []
        lines.append("="*60)
        lines.append("SEVA运行时仪表盘")
        lines.append("="*60)

        if not self.agent:
            lines.append("  (未连接Agent)")
            return "\n".join(lines)

        metrics = self.agent.get_metrics()
        status = "正常" if metrics["mode"] == "normal" else f"⚠ {metrics['mode']}"

        lines.append(f"  周期数:         {metrics['cycle_count']}")
        lines.append(f"  运行模式:       {status}")
        lines.append(f"  总奖励:         {metrics['total_reward']:.4f}")
        lines.append(f"  平均周期时间:   {metrics['avg_cycle_time']*1000:.2f}ms")
        lines.append(f"  平均预测误差:   {metrics['avg_prediction_error']:.4f}")
        lines.append(f"  平均惊奇值:     {metrics['avg_surprise']:.4f}")

        if not compact:
            # 详细信息
            task_stats = metrics.get("task_stats", {})
            skill_stats = metrics.get("skill_stats", {})
            evo_stats = metrics.get("evolution_stats", {})
            coord_stats = metrics.get("coord_stats", {})

            lines.append("")
            lines.append("  [任务系统]")
            lines.append(f"    活跃任务:     {task_stats.get('active_tasks', 0)}")
            lines.append(f"    已完成:       {task_stats.get('total_completed', 0)}")
            lines.append(f"    已失败:       {task_stats.get('total_failed', 0)}")

            lines.append("  [技能系统]")
            lines.append(f"    技能总数:     {skill_stats.get('total_skills', 0)}")
            lines.append(f"    可靠技能:     {skill_stats.get('reliable_skills', 0)}")
            lines.append(f"    总使用次数:   {skill_stats.get('total_uses', 0)}")

            lines.append("  [多Agent协调]")
            lines.append(f"    总Agent数:    {coord_stats.get('total_agents', 0)}")
            lines.append(f"    可用Agent:    {coord_stats.get('available_agents', 0)}")
            lines.append(f"    已分配任务:   {coord_stats.get('total_tasks_distributed', 0)}")

            lines.append("  [进化系统]")
            lines.append(f"    评估次数:     {evo_stats.get('total_evaluations', 0)}")
            lines.append(f"    当前难度:     {evo_stats.get('current_difficulty', 'EASY')}")
            lines.append(f"    误差趋势:     {evo_stats.get('error_trend', 'stable')}")
            lines.append(f"    收敛速度:     {evo_stats.get('convergence_rate', 0):.6f}")
            lines.append(f"    技能获取率:   {evo_stats.get('skill_acquisition_rate', 0):.6f}")

            # 动作分布
            action_dist = metrics.get("action_distribution", {})
            if action_dist:
                lines.append("  [动作分布]")
                for action, count in sorted(action_dist.items(),
                                           key=lambda x: -x[1]):
                    lines.append(f"    {action}: {count}")

        lines.append("="*60)
        self._last_render = time.time()
        return "\n".join(lines)

    def print(self, compact: bool = False):
        """打印仪表盘"""
        print(self.render(compact=compact))

    def get_trend(self, metric_name: str, window: int = 20) -> str:
        """
        获取指标趋势
        返回趋势字符串: ↑/↓/→
        """
        points = self.collector.get_metric(metric_name, window)
        if len(points) < 2:
            return "→"
        first_half = points[:len(points)//2]
        second_half = points[len(points)//2:]
        avg_first = sum(p.value for p in first_half) / len(first_half)
        avg_second = sum(p.value for p in second_half) / len(second_half)

        if avg_second > avg_first * 1.05:
            return "↑"
        elif avg_second < avg_first * 0.95:
            return "↓"
        else:
            return "→"


def monitor(func: Callable) -> Callable:
    """
    监控装饰器
    自动记录函数执行时间和异常
    """
    def wrapper(*args, **kwargs):
        logger = StructuredLogger(f"monitor.{func.__name__}")
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            logger.info(f"{func.__name__}完成",
                       extra={"elapsed_ms": round(elapsed*1000, 2),
                              "status": "success"})
            return result
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"{func.__name__}异常: {e}",
                        extra={"elapsed_ms": round(elapsed*1000, 2),
                               "status": "error",
                               "error_type": type(e).__name__})
            raise
    return wrapper


def setup_logging(level: str = "INFO", format: str = "text",
                  output_file: str = None) -> StructuredLogger:
    """
    全局日志配置
    便捷函数

    参数:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        format: 格式 (text/json)
        output_file: 输出文件路径(可选)

    示例:
        from seva.monitoring import setup_logging
        setup_logging(level="INFO", format="json", output_file="seva.log")
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    return StructuredLogger("seva", format=format,
                           level=level_map.get(level, logging.INFO),
                           output_file=output_file)


def export_metrics(collector: MetricsCollector, output_file: str,
                   format: str = "json"):
    """
    导出指标到文件
    """
    data = {}
    for name in collector.list_metrics():
        points = collector.get_metric(name, limit=10000)
        data[name] = [
            {"timestamp": p.timestamp, "value": p.value, "tags": p.tags}
            for p in points
        ]

    with open(output_file, "w") as f:
        if format == "json":
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        else:
            for name, points in data.items():
                f.write(f"# {name}\n")
                for p in points:
                    f.write(f"{p['timestamp']},{p['value']}\n")
