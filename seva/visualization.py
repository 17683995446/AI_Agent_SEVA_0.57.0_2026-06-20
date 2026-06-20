"""
阶段38: 结果可视化与报告导出

提供丰富的可视化图表和多格式报告导出功能，帮助用户理解、分享和分析Agent执行结果。

核心能力:
    1. SVG图表生成（纯Python实现，无外部依赖）
       - 趋势图: 预测误差/惊奇值/周期时间随周期变化
       - 分布图: 动作类型分布
       - 柱状图: 技能统计、任务统计
    2. 多格式报告导出
       - HTML: 富文本报告（含SVG图表、CSS样式）
       - JSON: 机器可读结构化数据
       - CSV: 电子表格友好格式
       - Markdown: 文档友好格式
       - TXT: 纯文本摘要
    3. 多次运行比较
       - 横向对比多个Agent的关键指标

快速启动:
    from seva import quickstart, ResultVisualizer, export_report

    agent, stats = quickstart(steps=50)
    viz = ResultVisualizer(agent)

    # 导出多格式报告
    viz.export("html", "report.html")
    viz.export("markdown", "report.md")
    viz.export("json", "report.json")

    # 生成SVG图表
    viz.generate_all_charts("./charts/")

    # 便捷函数
    export_report(agent, "csv", "metrics.csv")
"""
import os
import time
import json
import csv
import io
from typing import Any, Dict, List, Optional, Union, Tuple

from .core import SelfEvolvingAgentCore
from .api import get_status
from .diagnostics import Diagnoser


# =====================================================================
# SVG 图表生成器（纯Python实现，无外部依赖）
# =====================================================================

class SVGChartGenerator:
    """SVG图表生成器 - 纯Python实现，生成内联SVG"""

    # 配色方案
    COLORS = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
        "#9c755f", "#bab0ac",
    ]
    BG_COLOR = "#ffffff"
    GRID_COLOR = "#e0e0e0"
    TEXT_COLOR = "#333333"
    AXIS_COLOR = "#666666"

    @staticmethod
    def _escape_xml(text: str) -> str:
        """转义XML特殊字符"""
        if not isinstance(text, str):
            text = str(text)
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;"))

    @classmethod
    def line_chart(cls, data: List[Tuple[str, float]],
                   title: str = "",
                   xlabel: str = "",
                   ylabel: str = "",
                   width: int = 600,
                   height: int = 300) -> str:
        """
        生成折线图SVG

        Args:
            data: [(label, value), ...] 数据点列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            width: SVG宽度
            height: SVG高度

        Returns:
            SVG字符串
        """
        if not data:
            return cls._empty_chart(title, width, height, "暂无数据")

        # 布局参数
        margin_left = 60
        margin_right = 20
        margin_top = 40
        margin_bottom = 50
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = [d[1] for d in data]
        n = len(values)
        val_min = min(values) if values else 0
        val_max = max(values) if values else 1
        if val_max == val_min:
            val_max = val_min + 1
        val_range = val_max - val_min

        # 坐标映射
        def x_pos(i):
            if n <= 1:
                return margin_left + plot_w / 2
            return margin_left + (i / (n - 1)) * plot_w

        def y_pos(v):
            return margin_top + plot_h - ((v - val_min) / val_range) * plot_h

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{cls.BG_COLOR}"/>',
        ]

        # 标题
        if title:
            svg_parts.append(
                f'<text x="{width/2}" y="20" text-anchor="middle" '
                f'font-size="14" font-weight="bold" fill="{cls.TEXT_COLOR}">'
                f'{cls._escape_xml(title)}</text>'
            )

        # 网格线（5条水平线）
        for i in range(5):
            y = margin_top + (i / 4) * plot_h
            val = val_max - (i / 4) * val_range
            svg_parts.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" '
                f'x2="{margin_left + plot_w}" y2="{y:.1f}" '
                f'stroke="{cls.GRID_COLOR}" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{margin_left - 5}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-size="10" fill="{cls.AXIS_COLOR}">'
                f'{val:.4f}</text>'
            )

        # X轴
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{margin_top + plot_h}" '
            f'x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" '
            f'stroke="{cls.AXIS_COLOR}" stroke-width="1.5"/>'
        )
        # Y轴
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + plot_h}" '
            f'stroke="{cls.AXIS_COLOR}" stroke-width="1.5"/>'
        )

        # X轴标签（最多显示8个）
        label_step = max(1, n // 8)
        for i in range(0, n, label_step):
            x = x_pos(i)
            label = data[i][0] if len(data[i][0]) < 10 else data[i][0][:8] + "..."
            svg_parts.append(
                f'<text x="{x:.1f}" y="{margin_top + plot_h + 15}" '
                f'text-anchor="middle" font-size="9" fill="{cls.AXIS_COLOR}">'
                f'{cls._escape_xml(label)}</text>'
            )

        # 轴标签
        if xlabel:
            svg_parts.append(
                f'<text x="{margin_left + plot_w/2}" y="{height - 5}" '
                f'text-anchor="middle" font-size="11" fill="{cls.TEXT_COLOR}">'
                f'{cls._escape_xml(xlabel)}</text>'
            )
        if ylabel:
            svg_parts.append(
                f'<text x="15" y="{margin_top + plot_h/2}" '
                f'text-anchor="middle" font-size="11" fill="{cls.TEXT_COLOR}" '
                f'transform="rotate(-90, 15, {margin_top + plot_h/2})">'
                f'{cls._escape_xml(ylabel)}</text>'
            )

        # 折线
        points_str = " ".join(
            f"{x_pos(i):.1f},{y_pos(values[i]):.1f}" for i in range(n)
        )
        svg_parts.append(
            f'<polyline points="{points_str}" fill="none" '
            f'stroke="{cls.COLORS[0]}" stroke-width="2"/>'
        )

        # 数据点
        for i in range(n):
            x = x_pos(i)
            y = y_pos(values[i])
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                f'fill="{cls.COLORS[0]}"/>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def bar_chart(cls, data: List[Tuple[str, float]],
                  title: str = "",
                  ylabel: str = "",
                  width: int = 600,
                  height: int = 300,
                  color_idx: int = 0) -> str:
        """
        生成柱状图SVG

        Args:
            data: [(label, value), ...] 数据点列表
            title: 图表标题
            ylabel: Y轴标签
            width: SVG宽度
            height: SVG高度
            color_idx: 颜色索引

        Returns:
            SVG字符串
        """
        if not data:
            return cls._empty_chart(title, width, height, "暂无数据")

        margin_left = 60
        margin_right = 20
        margin_top = 40
        margin_bottom = 60
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = [d[1] for d in data]
        labels = [d[0] for d in data]
        n = len(values)
        val_max = max(values) if values else 1
        if val_max <= 0:
            val_max = 1

        bar_width = plot_w / n * 0.7
        bar_gap = plot_w / n * 0.3

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{cls.BG_COLOR}"/>',
        ]

        if title:
            svg_parts.append(
                f'<text x="{width/2}" y="20" text-anchor="middle" '
                f'font-size="14" font-weight="bold" fill="{cls.TEXT_COLOR}">'
                f'{cls._escape_xml(title)}</text>'
            )

        # 网格线
        for i in range(5):
            y = margin_top + (i / 4) * plot_h
            val = val_max - (i / 4) * val_max
            svg_parts.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" '
                f'x2="{margin_left + plot_w}" y2="{y:.1f}" '
                f'stroke="{cls.GRID_COLOR}" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{margin_left - 5}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-size="10" fill="{cls.AXIS_COLOR}">'
                f'{val:.1f}</text>'
            )

        # 轴
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{margin_top + plot_h}" '
            f'x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" '
            f'stroke="{cls.AXIS_COLOR}" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + plot_h}" '
            f'stroke="{cls.AXIS_COLOR}" stroke-width="1.5"/>'
        )

        # 柱子
        for i in range(n):
            bar_h = (values[i] / val_max) * plot_h
            x = margin_left + i * (bar_width + bar_gap) + bar_gap / 2
            y = margin_top + plot_h - bar_h
            color = cls.COLORS[i % len(cls.COLORS)]
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}" rx="2"/>'
            )
            # 数值标签
            svg_parts.append(
                f'<text x="{x + bar_width/2:.1f}" y="{y - 5:.1f}" '
                f'text-anchor="middle" font-size="10" fill="{cls.TEXT_COLOR}">'
                f'{values[i]:.1f}</text>'
            )
            # X轴标签
            label = labels[i] if len(labels[i]) < 12 else labels[i][:10] + "..."
            svg_parts.append(
                f'<text x="{x + bar_width/2:.1f}" y="{margin_top + plot_h + 15}" '
                f'text-anchor="middle" font-size="9" fill="{cls.AXIS_COLOR}">'
                f'{cls._escape_xml(label)}</text>'
            )

        if ylabel:
            svg_parts.append(
                f'<text x="15" y="{margin_top + plot_h/2}" '
                f'text-anchor="middle" font-size="11" fill="{cls.TEXT_COLOR}" '
                f'transform="rotate(-90, 15, {margin_top + plot_h/2})">'
                f'{cls._escape_xml(ylabel)}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def pie_chart(cls, data: List[Tuple[str, float]],
                  title: str = "",
                  width: int = 400,
                  height: int = 300) -> str:
        """
        生成饼图SVG

        Args:
            data: [(label, value), ...] 数据点列表
            title: 图表标题
            width: SVG宽度
            height: SVG高度

        Returns:
            SVG字符串
        """
        if not data:
            return cls._empty_chart(title, width, height, "暂无数据")

        total = sum(v for _, v in data)
        if total <= 0:
            return cls._empty_chart(title, width, height, "数据总和为0")

        cx = width / 2
        cy = height / 2 + 10
        radius = min(width, height) / 3

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{cls.BG_COLOR}"/>',
        ]

        if title:
            svg_parts.append(
                f'<text x="{width/2}" y="20" text-anchor="middle" '
                f'font-size="14" font-weight="bold" fill="{cls.TEXT_COLOR}">'
                f'{cls._escape_xml(title)}</text>'
            )

        # 绘制扇形
        start_angle = -90  # 从12点钟方向开始
        for i, (label, value) in enumerate(data):
            angle = (value / total) * 360
            end_angle = start_angle + angle
            color = cls.COLORS[i % len(cls.COLORS)]

            # 计算扇形路径
            x1 = cx + radius * _cos_deg(start_angle)
            y1 = cy + radius * _sin_deg(start_angle)
            x2 = cx + radius * _cos_deg(end_angle)
            y2 = cy + radius * _sin_deg(end_angle)

            large_arc = 1 if angle > 180 else 0
            path = (f"M {cx:.1f} {cy:.1f} "
                    f"L {x1:.1f} {y1:.1f} "
                    f"A {radius:.1f} {radius:.1f} 0 {large_arc} 1 "
                    f"{x2:.1f} {y2:.1f} Z")

            svg_parts.append(
                f'<path d="{path}" fill="{color}" stroke="{cls.BG_COLOR}" '
                f'stroke-width="1.5"/>'
            )

            # 标签（百分比）
            mid_angle = start_angle + angle / 2
            label_r = radius * 0.6
            lx = cx + label_r * _cos_deg(mid_angle)
            ly = cy + label_r * _sin_deg(mid_angle)
            pct = (value / total) * 100
            if pct >= 5:  # 只显示>=5%的标签
                svg_parts.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                    f'dominant-baseline="middle" font-size="10" '
                    f'fill="white" font-weight="bold">{pct:.0f}%</text>'
                )

            start_angle = end_angle

        # 图例
        legend_x = width - 120
        legend_y = 40
        for i, (label, value) in enumerate(data):
            color = cls.COLORS[i % len(cls.COLORS)]
            ly = legend_y + i * 18
            pct = (value / total) * 100
            svg_parts.append(
                f'<rect x="{legend_x}" y="{ly}" width="12" height="12" '
                f'fill="{color}"/>'
            )
            svg_parts.append(
                f'<text x="{legend_x + 16}" y="{ly + 10}" font-size="10" '
                f'fill="{cls.TEXT_COLOR}">'
                f'{cls._escape_xml(label)} ({pct:.1f}%)</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def _empty_chart(cls, title: str, width: int, height: int,
                     message: str = "暂无数据") -> str:
        """生成空图表占位符"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="{cls.BG_COLOR}"/>'
            f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" '
            f'font-weight="bold" fill="{cls.TEXT_COLOR}">'
            f'{cls._escape_xml(title)}</text>'
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
            f'font-size="12" fill="{cls.AXIS_COLOR}">'
            f'{cls._escape_xml(message)}</text>'
            f'</svg>'
        )


def _cos_deg(deg: float) -> float:
    """角度余弦（输入为度）"""
    import math
    return math.cos(math.radians(deg))


def _sin_deg(deg: float) -> float:
    """角度正弦（输入为度）"""
    import math
    return math.sin(math.radians(deg))


# =====================================================================
# 结果可视化器
# =====================================================================

class ResultVisualizer:
    """
    结果可视化与报告导出引擎

    提供SVG图表生成和多格式报告导出功能。

    支持的导出格式: html, json, csv, markdown, txt
    支持的图表类型: trend, distribution, timing, skills, tasks, evolution
    """

    SUPPORTED_FORMATS = ["html", "json", "csv", "markdown", "txt"]
    SUPPORTED_CHARTS = ["trend", "distribution", "timing", "skills",
                        "tasks", "evolution"]

    def __init__(self, agent: Optional[SelfEvolvingAgentCore] = None):
        """
        初始化可视化器

        Args:
            agent: SEVA Agent实例（可选，可在export时指定）
        """
        self.agent = agent

    def set_agent(self, agent: SelfEvolvingAgentCore) -> None:
        """设置Agent实例"""
        self.agent = agent

    def get_summary(self) -> Dict[str, Any]:
        """
        获取完整摘要数据

        Returns:
            包含状态、指标、诊断的完整摘要字典
        """
        if self.agent is None:
            raise ValueError("未设置Agent实例，请先调用set_agent()或在构造时传入agent")

        status = get_status(self.agent)
        metrics = self.agent.get_metrics()
        diagnosis = Diagnoser(self.agent).diagnose()

        return {
            "timestamp": time.time(),
            "version": _get_version(),
            "status": status,
            "metrics": metrics,
            "diagnosis": diagnosis,
        }

    # ================ 图表生成 ================

    def generate_chart(self, chart_type: str,
                       output_file: Optional[str] = None) -> str:
        """
        生成单个SVG图表

        Args:
            chart_type: 图表类型 (trend/distribution/timing/skills/tasks/evolution)
            output_file: 输出文件路径（可选，为None则仅返回SVG字符串）

        Returns:
            SVG字符串
        """
        if chart_type not in self.SUPPORTED_CHARTS:
            raise ValueError(
                f"不支持的图表类型: {chart_type}，"
                f"支持: {self.SUPPORTED_CHARTS}"
            )

        if self.agent is None:
            raise ValueError("未设置Agent实例")

        method = getattr(self, f"_chart_{chart_type}")
        svg = method()

        if output_file:
            self._write_file(output_file, svg)

        return svg

    def generate_all_charts(self,
                            output_dir: str = ".") -> List[str]:
        """
        生成所有SVG图表

        Args:
            output_dir: 输出目录

        Returns:
            生成的文件路径列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        generated = []
        for chart_type in self.SUPPORTED_CHARTS:
            filename = os.path.join(output_dir, f"chart_{chart_type}.svg")
            try:
                self.generate_chart(chart_type, filename)
                generated.append(filename)
            except Exception:
                # 单个图表失败不影响其他
                pass

        return generated

    def _chart_trend(self) -> str:
        """趋势图: 预测误差随周期变化"""
        errors = self.agent.metrics.get("prediction_errors", [])
        if not errors:
            return SVGChartGenerator._empty_chart(
                "预测误差趋势", 600, 300, "暂无数据")

        # 采样最多100个点
        step = max(1, len(errors) // 100)
        sampled = [(str(i * step), errors[i * step])
                   for i in range(0, len(errors), step)]
        if len(sampled) > 100:
            sampled = sampled[:100]

        return SVGChartGenerator.line_chart(
            sampled, title="预测误差趋势",
            xlabel="周期", ylabel="预测误差",
            width=600, height=300
        )

    def _chart_distribution(self) -> str:
        """分布图: 动作类型分布（饼图）"""
        dist = self.agent.metrics.get("action_distribution", {})
        if not dist:
            # 从metrics列表构建
            action_types = self.agent.metrics.get("action_types", [])
            if action_types:
                from collections import Counter
                dist = dict(Counter(action_types))
            else:
                return SVGChartGenerator._empty_chart(
                    "动作分布", 400, 300, "暂无数据")

        data = [(k, float(v)) for k, v in dist.items()]
        return SVGChartGenerator.pie_chart(
            data, title="动作类型分布", width=400, height=300
        )

    def _chart_timing(self) -> str:
        """时序图: 周期执行时间"""
        times = self.agent.metrics.get("cycle_times", [])
        if not times:
            return SVGChartGenerator._empty_chart(
                "周期执行时间", 600, 300, "暂无数据")

        step = max(1, len(times) // 100)
        sampled = [(str(i * step), times[i * step] * 1000)
                   for i in range(0, len(times), step)]
        if len(sampled) > 100:
            sampled = sampled[:100]

        return SVGChartGenerator.line_chart(
            sampled, title="周期执行时间趋势",
            xlabel="周期", ylabel="时间(ms)",
            width=600, height=300
        )

    def _chart_skills(self) -> str:
        """技能统计柱状图"""
        metrics = self.agent.get_metrics()
        skill_stats = metrics.get("skill_stats", {})
        data = [
            ("技能总数", float(skill_stats.get("total_skills", 0))),
            ("可靠技能", float(skill_stats.get("reliable_skills", 0))),
            ("总使用次数", float(skill_stats.get("total_uses", 0))),
            ("总成功次数", float(skill_stats.get("total_successes", 0))),
        ]
        return SVGChartGenerator.bar_chart(
            data, title="技能统计", ylabel="数量",
            width=600, height=300
        )

    def _chart_tasks(self) -> str:
        """任务统计柱状图"""
        metrics = self.agent.get_metrics()
        task_stats = metrics.get("task_stats", {})
        data = [
            ("活跃任务", float(task_stats.get("active_tasks", 0))),
            ("已创建", float(task_stats.get("total_created", 0))),
            ("已完成", float(task_stats.get("total_completed", 0))),
            ("已失败", float(task_stats.get("total_failed", 0))),
        ]
        return SVGChartGenerator.bar_chart(
            data, title="任务统计", ylabel="数量",
            width=600, height=300
        )

    def _chart_evolution(self) -> str:
        """进化统计柱状图"""
        metrics = self.agent.get_metrics()
        evo_stats = metrics.get("evolution_stats", {})
        data = [
            ("评估次数", float(evo_stats.get("total_evaluations", 0))),
            ("调整次数", float(evo_stats.get("total_adjustments", 0))),
            ("进化历史", float(evo_stats.get("evolution_history_size", 0))),
        ]
        success_rate = evo_stats.get("success_rate", 0)
        if success_rate:
            data.append(("成功率(%)", float(success_rate) * 100))

        return SVGChartGenerator.bar_chart(
            data, title="进化统计", ylabel="数值",
            width=600, height=300
        )

    # ================ 报告导出 ================

    def export(self, format: str = "html",
               output_file: Optional[str] = None) -> str:
        """
        导出报告

        Args:
            format: 导出格式 (html/json/csv/markdown/txt)
            output_file: 输出文件路径（可选）

        Returns:
            报告内容字符串
        """
        format = format.lower().strip()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的格式: {format}，支持: {self.SUPPORTED_FORMATS}")

        method = getattr(self, f"_export_{format}")
        content = method()

        if output_file:
            self._write_file(output_file, content)

        return content

    def _export_html(self) -> str:
        """生成HTML报告（含SVG图表）"""
        summary = self.get_summary()
        status = summary["status"]
        metrics = summary["metrics"]
        diagnosis = summary["diagnosis"]

        # 生成所有图表
        charts_svg = {}
        for chart_type in self.SUPPORTED_CHARTS:
            try:
                method = getattr(self, f"_chart_{chart_type}")
                charts_svg[chart_type] = method()
            except Exception:
                charts_svg[chart_type] = SVGChartGenerator._empty_chart(
                    chart_type, 600, 300, "生成失败")

        # 构建HTML
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, '
            'initial-scale=1.0">',
            f"<title>SEVA Agent 报告</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; "
            "margin: 20px; background: #f5f5f5; color: #333; }",
            ".container { max-width: 1000px; margin: 0 auto; }",
            ".card { background: white; border-radius: 8px; padding: 20px; "
            "margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".card h2 { margin-top: 0; color: #4e79a7; "
            "border-bottom: 2px solid #4e79a7; padding-bottom: 10px; }",
            "table { width: 100%; border-collapse: collapse; }",
            "th, td { padding: 8px 12px; text-align: left; "
            "border-bottom: 1px solid #eee; }",
            "th { background: #f8f8f8; font-weight: 600; }",
            ".badge { display: inline-block; padding: 3px 10px; "
            "border-radius: 12px; font-size: 12px; font-weight: bold; }",
            ".badge-healthy { background: #d4edda; color: #155724; }",
            ".badge-error { background: #f8d7da; color: #721c24; }",
            ".badge-warning { background: #fff3cd; color: #856404; }",
            ".chart-grid { display: grid; grid-template-columns: 1fr 1fr; "
            "gap: 20px; }",
            ".chart-item { text-align: center; }",
            ".issue-item { padding: 8px; margin: 5px 0; "
            "border-radius: 4px; }",
            ".issue-error { background: #f8d7da; }",
            ".issue-warning { background: #fff3cd; }",
            ".issue-info { background: #d1ecf1; }",
            ".header { text-align: center; margin-bottom: 30px; }",
            ".header h1 { color: #4e79a7; margin-bottom: 5px; }",
            ".header p { color: #666; margin: 5px 0; }",
            "</style>",
            "</head>",
            "<body>",
            '<div class="container">',
        ]

        # 头部
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(summary["timestamp"]))
        html_parts.append(
            f'<div class="header">'
            f'<h1>SEVA Agent 执行报告</h1>'
            f'<p>生成时间: {ts_str}</p>'
            f'<p>版本: {summary["version"]}</p>'
            f'</div>'
        )

        # 状态摘要
        healthy_badge = ("badge-healthy" if status["healthy"]
                         else "badge-error")
        healthy_text = "健康" if status["healthy"] else "异常"
        html_parts.append(
            f'<div class="card">'
            f'<h2>状态摘要</h2>'
            f'<table>'
            f'<tr><th>指标</th><th>值</th></tr>'
            f'<tr><td>周期数</td><td>{status["cycle_count"]}</td></tr>'
            f'<tr><td>运行模式</td><td>{status["mode"]}</td></tr>'
            f'<tr><td>健康状态</td>'
            f'<td><span class="badge {healthy_badge}">{healthy_text}</span>'
            f'</td></tr>'
            f'<tr><td>总奖励</td><td>{status["total_reward"]:.2f}</td></tr>'
            f'<tr><td>平均周期时间</td>'
            f'<td>{status["avg_cycle_time_ms"]:.2f} ms</td></tr>'
            f'<tr><td>任务完成</td><td>{status["tasks_completed"]}</td></tr>'
            f'<tr><td>技能总数</td><td>{status["skills_total"]}</td></tr>'
            f'<tr><td>进化评估</td>'
            f'<td>{status["evolution_evaluations"]}</td></tr>'
            f'<tr><td>当前难度</td><td>{status["current_difficulty"]}</td></tr>'
            f'</table>'
            f'</div>'
        )

        # 诊断报告
        diag_healthy = ("badge-healthy" if diagnosis["healthy"]
                        else "badge-error")
        diag_text = "健康" if diagnosis["healthy"] else "存在问题"
        html_parts.append(
            f'<div class="card">'
            f'<h2>诊断报告</h2>'
            f'<p><span class="badge {diag_healthy}">{diag_text}</span></p>'
            f'<p>问题数: {diagnosis["issues_count"]} | '
            f'警告数: {diagnosis["warnings_count"]}</p>'
        )

        # 问题详情
        if diagnosis["issues"]:
            html_parts.append("<h3>问题</h3>")
            for issue in diagnosis["issues"]:
                html_parts.append(
                    f'<div class="issue-item issue-error">'
                    f'<strong>{issue.get("code", "")}</strong>: '
                    f'{issue.get("message", "")}<br>'
                    f'<em>建议: {issue.get("suggestion", "")}</em>'
                    f'</div>'
                )

        if diagnosis["warnings"]:
            html_parts.append("<h3>警告</h3>")
            for warn in diagnosis["warnings"]:
                html_parts.append(
                    f'<div class="issue-item issue-warning">'
                    f'<strong>{warn.get("code", "")}</strong>: '
                    f'{warn.get("message", "")}<br>'
                    f'<em>建议: {warn.get("suggestion", "")}</em>'
                    f'</div>'
                )

        if diagnosis["info"]:
            html_parts.append("<h3>信息</h3>")
            for info in diagnosis["info"]:
                msg = info.get("message", str(info))
                html_parts.append(
                    f'<div class="issue-item issue-info">{msg}</div>'
                )

        html_parts.append("</div>")  # card

        # 图表
        html_parts.append('<div class="card"><h2>可视化图表</h2>')
        html_parts.append('<div class="chart-grid">')

        chart_titles = {
            "trend": "预测误差趋势",
            "distribution": "动作分布",
            "timing": "周期时间",
            "skills": "技能统计",
            "tasks": "任务统计",
            "evolution": "进化统计",
        }

        for chart_type in self.SUPPORTED_CHARTS:
            title = chart_titles.get(chart_type, chart_type)
            svg = charts_svg.get(chart_type, "")
            html_parts.append(
                f'<div class="chart-item"><h3>{title}</h3>{svg}</div>'
            )

        html_parts.append('</div></div>')  # chart-grid, card

        # 详细指标
        html_parts.append(
            f'<div class="card">'
            f'<h2>详细指标</h2>'
            f'<table>'
            f'<tr><th>指标</th><th>值</th></tr>'
            f'<tr><td>平均预测误差</td>'
            f'<td>{metrics.get("avg_prediction_error", 0):.4f}</td></tr>'
            f'<tr><td>平均惊奇值</td>'
            f'<td>{metrics.get("avg_surprise", 0):.4f}</td></tr>'
            f'<tr><td>平均周期时间</td>'
            f'<td>{metrics.get("avg_cycle_time", 0):.4f} s</td></tr>'
            f'</table>'
            f'</div>'
        )

        html_parts.append("</div>")  # container
        html_parts.append("</body></html>")

        return "\n".join(html_parts)

    def _export_json(self) -> str:
        """生成JSON报告"""
        summary = self.get_summary()
        return json.dumps(summary, indent=2, ensure_ascii=False,
                          default=str)

    def _export_csv(self) -> str:
        """生成CSV报告"""
        summary = self.get_summary()
        status = summary["status"]
        metrics = summary["metrics"]
        diagnosis = summary["diagnosis"]

        output = io.StringIO()
        writer = csv.writer(output)

        # 状态摘要
        writer.writerow(["=== 状态摘要 ==="])
        writer.writerow(["指标", "值"])
        for key, value in status.items():
            writer.writerow([key, value])
        writer.writerow([])

        # 指标
        writer.writerow(["=== 详细指标 ==="])
        writer.writerow(["指标", "值"])
        writer.writerow(["平均预测误差",
                         metrics.get("avg_prediction_error", 0)])
        writer.writerow(["平均惊奇值", metrics.get("avg_surprise", 0)])
        writer.writerow(["平均周期时间",
                         metrics.get("avg_cycle_time", 0)])
        writer.writerow([])

        # 动作分布
        writer.writerow(["=== 动作分布 ==="])
        writer.writerow(["动作类型", "次数"])
        for action, count in metrics.get("action_distribution", {}).items():
            writer.writerow([action, count])
        writer.writerow([])

        # 任务统计
        task_stats = metrics.get("task_stats", {})
        writer.writerow(["=== 任务统计 ==="])
        writer.writerow(["指标", "值"])
        for key, value in task_stats.items():
            if not isinstance(value, (dict, list)):
                writer.writerow([key, value])
        writer.writerow([])

        # 技能统计
        skill_stats = metrics.get("skill_stats", {})
        writer.writerow(["=== 技能统计 ==="])
        writer.writerow(["指标", "值"])
        for key, value in skill_stats.items():
            if not isinstance(value, (dict, list)):
                writer.writerow([key, value])
        writer.writerow([])

        # 诊断
        writer.writerow(["=== 诊断报告 ==="])
        writer.writerow(["类型", "代码", "消息", "建议"])
        for issue in diagnosis.get("issues", []):
            writer.writerow([
                "问题", issue.get("code", ""),
                issue.get("message", ""),
                issue.get("suggestion", "")
            ])
        for warn in diagnosis.get("warnings", []):
            writer.writerow([
                "警告", warn.get("code", ""),
                warn.get("message", ""),
                warn.get("suggestion", "")
            ])
        for info in diagnosis.get("info", []):
            writer.writerow([
                "信息", info.get("code", ""),
                info.get("message", ""),
                info.get("suggestion", "")
            ])

        return output.getvalue()

    def _export_markdown(self) -> str:
        """生成Markdown报告"""
        summary = self.get_summary()
        status = summary["status"]
        metrics = summary["metrics"]
        diagnosis = summary["diagnosis"]

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(summary["timestamp"]))

        lines = [
            "# SEVA Agent 执行报告",
            "",
            f"> 生成时间: {ts_str}  ",
            f"> 版本: {summary['version']}",
            "",
            "## 状态摘要",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 周期数 | {status['cycle_count']} |",
            f"| 运行模式 | {status['mode']} |",
            f"| 健康状态 | {'✅ 健康' if status['healthy'] else '❌ 异常'} |",
            f"| 总奖励 | {status['total_reward']:.2f} |",
            f"| 平均周期时间 | {status['avg_cycle_time_ms']:.2f} ms |",
            f"| 任务完成 | {status['tasks_completed']} |",
            f"| 技能总数 | {status['skills_total']} |",
            f"| 进化评估 | {status['evolution_evaluations']} |",
            f"| 当前难度 | {status['current_difficulty']} |",
            "",
            "## 诊断报告",
            "",
            f"**健康状态**: {'✅ 健康' if diagnosis['healthy'] else '❌ 存在问题'}  ",
            f"**问题数**: {diagnosis['issues_count']}  ",
            f"**警告数**: {diagnosis['warnings_count']}",
            "",
        ]

        if diagnosis["issues"]:
            lines.append("### 问题")
            lines.append("")
            for issue in diagnosis["issues"]:
                lines.append(
                    f"- **{issue.get('code', '')}**: "
                    f"{issue.get('message', '')}"
                )
                if issue.get("suggestion"):
                    lines.append(f"  - 建议: {issue['suggestion']}")
            lines.append("")

        if diagnosis["warnings"]:
            lines.append("### 警告")
            lines.append("")
            for warn in diagnosis["warnings"]:
                lines.append(
                    f"- **{warn.get('code', '')}**: "
                    f"{warn.get('message', '')}"
                )
                if warn.get("suggestion"):
                    lines.append(f"  - 建议: {warn['suggestion']}")
            lines.append("")

        if diagnosis["info"]:
            lines.append("### 信息")
            lines.append("")
            for info in diagnosis["info"]:
                lines.append(f"- {info.get('message', str(info))}")
            lines.append("")

        lines.extend([
            "## 详细指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 平均预测误差 | {metrics.get('avg_prediction_error', 0):.4f} |",
            f"| 平均惊奇值 | {metrics.get('avg_surprise', 0):.4f} |",
            f"| 平均周期时间 | {metrics.get('avg_cycle_time', 0):.4f} s |",
            "",
        ])

        # 动作分布
        action_dist = metrics.get("action_distribution", {})
        if action_dist:
            lines.append("## 动作分布")
            lines.append("")
            lines.append("| 动作类型 | 次数 |")
            lines.append("|----------|------|")
            for action, count in action_dist.items():
                lines.append(f"| {action} | {count} |")
            lines.append("")

        # 任务统计
        task_stats = metrics.get("task_stats", {})
        if task_stats:
            lines.append("## 任务统计")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in task_stats.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        # 技能统计
        skill_stats = metrics.get("skill_stats", {})
        if skill_stats:
            lines.append("## 技能统计")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in skill_stats.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        # 进化统计
        evo_stats = metrics.get("evolution_stats", {})
        if evo_stats:
            lines.append("## 进化统计")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in evo_stats.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        return "\n".join(lines)

    def _export_txt(self) -> str:
        """生成纯文本报告"""
        summary = self.get_summary()
        status = summary["status"]
        metrics = summary["metrics"]
        diagnosis = summary["diagnosis"]

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(summary["timestamp"]))

        lines = [
            "=" * 60,
            "  SEVA Agent 执行报告",
            "=" * 60,
            f"  生成时间: {ts_str}",
            f"  版本: {summary['version']}",
            "",
            "-" * 60,
            "  状态摘要",
            "-" * 60,
            f"  周期数:       {status['cycle_count']}",
            f"  运行模式:     {status['mode']}",
            f"  健康状态:     {'健康' if status['healthy'] else '异常'}",
            f"  总奖励:       {status['total_reward']:.2f}",
            f"  平均周期时间: {status['avg_cycle_time_ms']:.2f} ms",
            f"  任务完成:     {status['tasks_completed']}",
            f"  技能总数:     {status['skills_total']}",
            f"  进化评估:     {status['evolution_evaluations']}",
            f"  当前难度:     {status['current_difficulty']}",
            "",
            "-" * 60,
            "  诊断报告",
            "-" * 60,
            f"  健康状态: {'健康' if diagnosis['healthy'] else '存在问题'}",
            f"  问题数:   {diagnosis['issues_count']}",
            f"  警告数:   {diagnosis['warnings_count']}",
            "",
        ]

        if diagnosis["issues"]:
            lines.append("  [问题]")
            for issue in diagnosis["issues"]:
                lines.append(
                    f"    [{issue.get('code', '')}] "
                    f"{issue.get('message', '')}"
                )
                if issue.get("suggestion"):
                    lines.append(f"      -> {issue['suggestion']}")
            lines.append("")

        if diagnosis["warnings"]:
            lines.append("  [警告]")
            for warn in diagnosis["warnings"]:
                lines.append(
                    f"    [{warn.get('code', '')}] "
                    f"{warn.get('message', '')}"
                )
                if warn.get("suggestion"):
                    lines.append(f"      -> {warn['suggestion']}")
            lines.append("")

        if diagnosis["info"]:
            lines.append("  [信息]")
            for info in diagnosis["info"]:
                lines.append(f"    {info.get('message', str(info))}")
            lines.append("")

        lines.extend([
            "-" * 60,
            "  详细指标",
            "-" * 60,
            f"  平均预测误差: {metrics.get('avg_prediction_error', 0):.4f}",
            f"  平均惊奇值:   {metrics.get('avg_surprise', 0):.4f}",
            f"  平均周期时间: {metrics.get('avg_cycle_time', 0):.4f} s",
            "",
        ])

        # 动作分布
        action_dist = metrics.get("action_distribution", {})
        if action_dist:
            lines.append("-" * 60)
            lines.append("  动作分布")
            lines.append("-" * 60)
            for action, count in action_dist.items():
                lines.append(f"  {action}: {count}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    # ================ 工具方法 ================

    @staticmethod
    def _write_file(file_path: str, content: str) -> None:
        """写入文件"""
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def _get_version() -> str:
    """获取SEVA版本号"""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


# =====================================================================
# 便捷函数
# =====================================================================

def visualize(agent: SelfEvolvingAgentCore,
              format: str = "html",
              output_file: Optional[str] = None) -> str:
    """
    便捷可视化函数

    Args:
        agent: SEVA Agent实例
        format: 导出格式 (html/json/csv/markdown/txt)
        output_file: 输出文件路径（可选）

    Returns:
        报告内容字符串

    Example:
        >>> from seva import quickstart, visualize
        >>> agent, _ = quickstart(steps=50)
        >>> visualize(agent, "html", "report.html")
    """
    viz = ResultVisualizer(agent)
    return viz.export(format, output_file)


def export_report(agent: SelfEvolvingAgentCore,
                  format: str = "html",
                  output_file: Optional[str] = None) -> str:
    """
    导出报告（visualize的别名）

    Args:
        agent: SEVA Agent实例
        format: 导出格式 (html/json/csv/markdown/txt)
        output_file: 输出文件路径（可选）

    Returns:
        报告内容字符串
    """
    return visualize(agent, format, output_file)


def generate_charts(agent: SelfEvolvingAgentCore,
                    output_dir: str = ".") -> List[str]:
    """
    生成所有SVG图表

    Args:
        agent: SEVA Agent实例
        output_dir: 输出目录

    Returns:
        生成的文件路径列表

    Example:
        >>> from seva import quickstart, generate_charts
        >>> agent, _ = quickstart(steps=50)
        >>> generate_charts(agent, "./charts/")
    """
    viz = ResultVisualizer(agent)
    return viz.generate_all_charts(output_dir)


def compare_runs(agents: List[SelfEvolvingAgentCore],
                 labels: Optional[List[str]] = None,
                 output_file: Optional[str] = None,
                 format: str = "html") -> str:
    """
    比较多次运行结果

    Args:
        agents: Agent实例列表
        labels: 每个运行的标签（可选，默认使用 Run 1, Run 2, ...）
        output_file: 输出文件路径（可选）
        format: 输出格式 (html/markdown/txt/json)

    Returns:
        比较报告字符串

    Example:
        >>> from seva import quickstart, compare_runs
        >>> agent1, _ = quickstart(steps=50, preset="easy")
        >>> agent2, _ = quickstart(steps=50, preset="hard")
        >>> compare_runs([agent1, agent2], labels=["Easy", "Hard"],
        ...              output_file="comparison.html")
    """
    if not agents:
        raise ValueError("agents列表不能为空")

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(agents))]
    elif len(labels) != len(agents):
        raise ValueError("labels数量必须与agents数量一致")

    # 收集所有运行的状态
    runs_data = []
    for agent, label in zip(agents, labels):
        status = get_status(agent)
        metrics = agent.get_metrics()
        runs_data.append({
            "label": label,
            "status": status,
            "metrics": metrics,
        })

    format = format.lower().strip()

    if format == "html":
        content = _compare_html(runs_data)
    elif format == "markdown":
        content = _compare_markdown(runs_data)
    elif format == "txt":
        content = _compare_txt(runs_data)
    elif format == "json":
        content = json.dumps(runs_data, indent=2, ensure_ascii=False,
                             default=str)
    else:
        raise ValueError(f"比较报告不支持的格式: {format}")

    if output_file:
        ResultVisualizer._write_file(output_file, content)

    return content


def _compare_html(runs_data: List[Dict]) -> str:
    """生成HTML比较报告"""
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>SEVA Agent 运行比较</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; "
        "margin: 20px; background: #f5f5f5; color: #333; }",
        ".container { max-width: 1000px; margin: 0 auto; }",
        ".card { background: white; border-radius: 8px; padding: 20px; "
        "margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".card h2 { margin-top: 0; color: #4e79a7; "
        "border-bottom: 2px solid #4e79a7; padding-bottom: 10px; }",
        "table { width: 100%; border-collapse: collapse; }",
        "th, td { padding: 8px 12px; text-align: left; "
        "border-bottom: 1px solid #eee; }",
        "th { background: #f8f8f8; }",
        ".header { text-align: center; margin-bottom: 30px; }",
        ".header h1 { color: #4e79a7; }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        '<div class="header"><h1>SEVA Agent 运行比较</h1></div>',
        '<div class="card">',
        '<h2>指标对比</h2>',
        '<table>',
        '<tr><th>指标</th>',
    ]

    for run in runs_data:
        html_parts.append(f'<th>{run["label"]}</th>')
    html_parts.append('</tr>')

    # 对比行
    compare_keys = [
        ("周期数", "cycle_count", "status"),
        ("总奖励", "total_reward", "status"),
        ("平均周期时间(ms)", "avg_cycle_time_ms", "status"),
        ("任务完成", "tasks_completed", "status"),
        ("技能总数", "skills_total", "status"),
        ("进化评估", "evolution_evaluations", "status"),
        ("平均预测误差", "avg_prediction_error", "metrics"),
        ("平均惊奇值", "avg_surprise", "metrics"),
    ]

    for label, key, source in compare_keys:
        html_parts.append(f'<tr><td><strong>{label}</strong></td>')
        for run in runs_data:
            value = run[source].get(key, "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            html_parts.append(f'<td>{value}</td>')
        html_parts.append('</tr>')

    html_parts.append('</table></div></div></body></html>')
    return "\n".join(html_parts)


def _compare_markdown(runs_data: List[Dict]) -> str:
    """生成Markdown比较报告"""
    lines = ["# SEVA Agent 运行比较", ""]

    # 表头
    header = "| 指标 |"
    separator = "|------|"
    for run in runs_data:
        header += f" {run['label']} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    compare_keys = [
        ("周期数", "cycle_count", "status"),
        ("总奖励", "total_reward", "status"),
        ("平均周期时间(ms)", "avg_cycle_time_ms", "status"),
        ("任务完成", "tasks_completed", "status"),
        ("技能总数", "skills_total", "status"),
        ("进化评估", "evolution_evaluations", "status"),
        ("平均预测误差", "avg_prediction_error", "metrics"),
        ("平均惊奇值", "avg_surprise", "metrics"),
    ]

    for label, key, source in compare_keys:
        row = f"| {label} |"
        for run in runs_data:
            value = run[source].get(key, "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            row += f" {value} |"
        lines.append(row)

    return "\n".join(lines)


def _compare_txt(runs_data: List[Dict]) -> str:
    """生成纯文本比较报告"""
    lines = [
        "=" * 70,
        "  SEVA Agent 运行比较",
        "=" * 70,
        "",
    ]

    compare_keys = [
        ("周期数", "cycle_count", "status"),
        ("总奖励", "total_reward", "status"),
        ("平均周期时间(ms)", "avg_cycle_time_ms", "status"),
        ("任务完成", "tasks_completed", "status"),
        ("技能总数", "skills_total", "status"),
        ("进化评估", "evolution_evaluations", "status"),
        ("平均预测误差", "avg_prediction_error", "metrics"),
        ("平均惊奇值", "avg_surprise", "metrics"),
    ]

    for label, key, source in compare_keys:
        lines.append(f"  {label}:")
        for run in runs_data:
            value = run[source].get(key, "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            lines.append(f"    {run['label']}: {value}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
