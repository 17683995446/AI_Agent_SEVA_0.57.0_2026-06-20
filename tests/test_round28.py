"""
阶段38测试: 结果可视化与报告导出

测试覆盖:
    1. SVGChartGenerator - SVG图表生成器
    2. ResultVisualizer - 结果可视化器
    3. 多格式导出 (HTML/JSON/CSV/Markdown/TXT)
    4. 便捷函数 (visualize/export_report/generate_charts/compare_runs)
    5. CLI集成 (export/charts/compare命令)
    6. 边界条件与错误处理
"""
import os
import sys
import json
import tempfile
import shutil

import pytest

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seva
from seva import (quickstart, ResultVisualizer, SVGChartGenerator,
                  visualize, export_report, generate_charts, compare_runs)
from seva.visualization import (_cos_deg, _sin_deg, _get_version,
                                _compare_html, _compare_markdown, _compare_txt)


# =====================================================================
# 测试夹具
# =====================================================================

@pytest.fixture
def agent():
    """创建一个运行了若干步的Agent"""
    agent, _ = quickstart(steps=20, preset="easy", seed=42)
    return agent


@pytest.fixture
def agents_for_compare():
    """创建多个Agent用于比较测试"""
    agent1, _ = quickstart(steps=10, preset="easy", seed=42)
    agent2, _ = quickstart(steps=10, preset="standard", seed=42)
    return [agent1, agent2]


@pytest.fixture
def temp_dir():
    """临时目录"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =====================================================================
# 1. SVGChartGenerator 测试
# =====================================================================

class TestSVGChartGenerator:
    """SVG图表生成器测试"""

    def test01_line_chart_basic(self):
        """测试01: 基本折线图生成"""
        data = [("0", 0.5), ("1", 0.3), ("2", 0.4), ("3", 0.2)]
        svg = SVGChartGenerator.line_chart(data, title="测试折线图")
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "polyline" in svg
        assert "测试折线图" in svg

    def test02_line_chart_empty_data(self):
        """测试02: 空数据折线图"""
        svg = SVGChartGenerator.line_chart([], title="空图")
        assert "<svg" in svg
        assert "暂无数据" in svg

    def test03_line_chart_with_labels(self):
        """测试03: 带轴标签的折线图"""
        data = [("a", 1.0), ("b", 2.0)]
        svg = SVGChartGenerator.line_chart(data, title="T",
                                           xlabel="X轴", ylabel="Y轴")
        assert "X轴" in svg
        assert "Y轴" in svg

    def test04_line_chart_custom_size(self):
        """测试04: 自定义尺寸"""
        data = [("0", 1.0)]
        svg = SVGChartGenerator.line_chart(data, width=800, height=400)
        assert 'width="800"' in svg
        assert 'height="400"' in svg

    def test05_bar_chart_basic(self):
        """测试05: 基本柱状图"""
        data = [("A", 10), ("B", 20), ("C", 15)]
        svg = SVGChartGenerator.bar_chart(data, title="柱状图")
        assert "<svg" in svg
        assert "rect" in svg
        assert "柱状图" in svg

    def test06_bar_chart_empty_data(self):
        """测试06: 空数据柱状图"""
        svg = SVGChartGenerator.bar_chart([], title="空")
        assert "暂无数据" in svg

    def test07_pie_chart_basic(self):
        """测试07: 基本饼图"""
        data = [("A", 30), ("B", 40), ("C", 30)]
        svg = SVGChartGenerator.pie_chart(data, title="饼图")
        assert "<svg" in svg
        assert "path" in svg
        assert "饼图" in svg

    def test08_pie_chart_empty_data(self):
        """测试08: 空数据饼图"""
        svg = SVGChartGenerator.pie_chart([], title="空")
        assert "暂无数据" in svg

    def test09_pie_chart_zero_total(self):
        """测试09: 总和为0的饼图"""
        data = [("A", 0), ("B", 0)]
        svg = SVGChartGenerator.pie_chart(data, title="零")
        assert "数据总和为0" in svg

    def test10_escape_xml(self):
        """测试10: XML转义"""
        result = SVGChartGenerator._escape_xml("<test>&\"'</test>")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result
        assert "&apos;" in result

    def test11_line_chart_single_point(self):
        """测试11: 单个数据点"""
        data = [("0", 0.5)]
        svg = SVGChartGenerator.line_chart(data, title="单点")
        assert "<svg" in svg
        assert "circle" in svg

    def test12_bar_chart_with_ylabel(self):
        """测试12: 带Y轴标签的柱状图"""
        data = [("A", 10)]
        svg = SVGChartGenerator.bar_chart(data, ylabel="数量")
        assert "数量" in svg

    def test13_pie_chart_percentages(self):
        """测试13: 饼图百分比显示"""
        data = [("A", 50), ("B", 50)]
        svg = SVGChartGenerator.pie_chart(data, title="50/50")
        assert "%" in svg

    def test14_colors_constant(self):
        """测试14: 颜色配置常量"""
        assert isinstance(SVGChartGenerator.COLORS, list)
        assert len(SVGChartGenerator.COLORS) >= 5
        assert all(c.startswith("#") for c in SVGChartGenerator.COLORS)

    def test15_empty_chart_method(self):
        """测试15: 空图表占位方法"""
        svg = SVGChartGenerator._empty_chart("标题", 200, 100, "消息")
        assert "<svg" in svg
        assert "标题" in svg
        assert "消息" in svg


# =====================================================================
# 2. 辅助函数测试
# =====================================================================

class TestHelpers:
    """辅助函数测试"""

    def test16_cos_deg(self):
        """测试16: 角度余弦"""
        import math
        assert abs(_cos_deg(0) - 1.0) < 1e-6
        assert abs(_cos_deg(90)) < 1e-6
        assert abs(_cos_deg(180) + 1.0) < 1e-6

    def test17_sin_deg(self):
        """测试17: 角度正弦"""
        import math
        assert abs(_sin_deg(0)) < 1e-6
        assert abs(_sin_deg(90) - 1.0) < 1e-6
        assert abs(_sin_deg(180)) < 1e-6

    def test18_get_version(self):
        """测试18: 获取版本号"""
        v = _get_version()
        assert isinstance(v, str)
        assert v != "unknown"  # 应该能获取到实际版本


# =====================================================================
# 3. ResultVisualizer 测试
# =====================================================================

class TestResultVisualizer:
    """结果可视化器测试"""

    def test19_init_without_agent(self):
        """测试19: 不带Agent初始化"""
        viz = ResultVisualizer()
        assert viz.agent is None

    def test20_init_with_agent(self, agent):
        """测试20: 带Agent初始化"""
        viz = ResultVisualizer(agent)
        assert viz.agent is not None

    def test21_set_agent(self, agent):
        """测试21: 设置Agent"""
        viz = ResultVisualizer()
        viz.set_agent(agent)
        assert viz.agent is not None

    def test22_supported_formats(self):
        """测试22: 支持的格式列表"""
        assert "html" in ResultVisualizer.SUPPORTED_FORMATS
        assert "json" in ResultVisualizer.SUPPORTED_FORMATS
        assert "csv" in ResultVisualizer.SUPPORTED_FORMATS
        assert "markdown" in ResultVisualizer.SUPPORTED_FORMATS
        assert "txt" in ResultVisualizer.SUPPORTED_FORMATS

    def test23_supported_charts(self):
        """测试23: 支持的图表类型"""
        assert "trend" in ResultVisualizer.SUPPORTED_CHARTS
        assert "distribution" in ResultVisualizer.SUPPORTED_CHARTS
        assert "timing" in ResultVisualizer.SUPPORTED_CHARTS
        assert "skills" in ResultVisualizer.SUPPORTED_CHARTS
        assert "tasks" in ResultVisualizer.SUPPORTED_CHARTS
        assert "evolution" in ResultVisualizer.SUPPORTED_CHARTS

    def test24_get_summary(self, agent):
        """测试24: 获取摘要"""
        viz = ResultVisualizer(agent)
        summary = viz.get_summary()
        assert "timestamp" in summary
        assert "version" in summary
        assert "status" in summary
        assert "metrics" in summary
        assert "diagnosis" in summary

    def test25_get_summary_no_agent(self):
        """测试25: 无Agent时获取摘要报错"""
        viz = ResultVisualizer()
        with pytest.raises(ValueError):
            viz.get_summary()

    def test26_generate_chart_trend(self, agent):
        """测试26: 生成趋势图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("trend")
        assert "<svg" in svg

    def test27_generate_chart_distribution(self, agent):
        """测试27: 生成分布图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("distribution")
        assert "<svg" in svg

    def test28_generate_chart_timing(self, agent):
        """测试28: 生成时序图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("timing")
        assert "<svg" in svg

    def test29_generate_chart_skills(self, agent):
        """测试29: 生成技能图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("skills")
        assert "<svg" in svg

    def test30_generate_chart_tasks(self, agent):
        """测试30: 生成任务图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("tasks")
        assert "<svg" in svg

    def test31_generate_chart_evolution(self, agent):
        """测试31: 生成进化图"""
        viz = ResultVisualizer(agent)
        svg = viz.generate_chart("evolution")
        assert "<svg" in svg

    def test32_generate_chart_invalid_type(self, agent):
        """测试32: 无效图表类型"""
        viz = ResultVisualizer(agent)
        with pytest.raises(ValueError):
            viz.generate_chart("invalid")

    def test33_generate_chart_no_agent(self):
        """测试33: 无Agent时生成图表报错"""
        viz = ResultVisualizer()
        with pytest.raises(ValueError):
            viz.generate_chart("trend")

    def test34_generate_chart_to_file(self, agent, temp_dir):
        """测试34: 生成图表到文件"""
        viz = ResultVisualizer(agent)
        path = os.path.join(temp_dir, "chart.svg")
        svg = viz.generate_chart("trend", path)
        assert os.path.exists(path)
        with open(path, "r") as f:
            assert f.read() == svg

    def test35_generate_all_charts(self, agent, temp_dir):
        """测试35: 生成所有图表"""
        viz = ResultVisualizer(agent)
        files = viz.generate_all_charts(temp_dir)
        assert len(files) > 0
        for f in files:
            assert os.path.exists(f)

    def test36_generate_all_charts_creates_dir(self, agent, temp_dir):
        """测试36: 自动创建目录"""
        viz = ResultVisualizer(agent)
        new_dir = os.path.join(temp_dir, "newdir")
        files = viz.generate_all_charts(new_dir)
        assert os.path.exists(new_dir)


# =====================================================================
# 4. 多格式导出测试
# =====================================================================

class TestExportFormats:
    """多格式导出测试"""

    def test37_export_html(self, agent):
        """测试37: 导出HTML"""
        viz = ResultVisualizer(agent)
        content = viz.export("html")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "SEVA" in content

    def test38_export_json(self, agent):
        """测试38: 导出JSON"""
        viz = ResultVisualizer(agent)
        content = viz.export("json")
        data = json.loads(content)
        assert "status" in data
        assert "metrics" in data
        assert "diagnosis" in data

    def test39_export_csv(self, agent):
        """测试39: 导出CSV"""
        viz = ResultVisualizer(agent)
        content = viz.export("csv")
        assert "状态摘要" in content
        assert "详细指标" in content

    def test40_export_markdown(self, agent):
        """测试40: 导出Markdown"""
        viz = ResultVisualizer(agent)
        content = viz.export("markdown")
        assert "# SEVA" in content
        assert "## 状态摘要" in content
        assert "|" in content  # 表格

    def test41_export_txt(self, agent):
        """测试41: 导出TXT"""
        viz = ResultVisualizer(agent)
        content = viz.export("txt")
        assert "SEVA" in content
        assert "状态摘要" in content
        assert "=" in content

    def test42_export_invalid_format(self, agent):
        """测试42: 无效格式"""
        viz = ResultVisualizer(agent)
        with pytest.raises(ValueError):
            viz.export("xml")

    def test43_export_to_file(self, agent, temp_dir):
        """测试43: 导出到文件"""
        viz = ResultVisualizer(agent)
        path = os.path.join(temp_dir, "report.html")
        content = viz.export("html", path)
        assert os.path.exists(path)
        with open(path, "r") as f:
            assert f.read() == content

    def test44_export_creates_parent_dir(self, agent, temp_dir):
        """测试44: 导出时创建父目录"""
        viz = ResultVisualizer(agent)
        path = os.path.join(temp_dir, "sub", "report.json")
        viz.export("json", path)
        assert os.path.exists(path)

    def test45_export_html_contains_charts(self, agent):
        """测试45: HTML报告包含SVG图表"""
        viz = ResultVisualizer(agent)
        content = viz.export("html")
        assert "<svg" in content
        assert "chart" in content.lower() or "图表" in content

    def test46_export_html_contains_diagnosis(self, agent):
        """测试46: HTML报告包含诊断"""
        viz = ResultVisualizer(agent)
        content = viz.export("html")
        assert "诊断" in content

    def test47_export_markdown_contains_tables(self, agent):
        """测试47: Markdown报告包含表格"""
        viz = ResultVisualizer(agent)
        content = viz.export("markdown")
        assert "| 指标 |" in content or "|指标|" in content

    def test48_export_case_insensitive(self, agent):
        """测试48: 格式不区分大小写"""
        viz = ResultVisualizer(agent)
        content = viz.export("HTML")
        assert "<!DOCTYPE html>" in content

    def test49_export_txt_contains_separator(self, agent):
        """测试49: TXT报告包含分隔线"""
        viz = ResultVisualizer(agent)
        content = viz.export("txt")
        assert "=" * 10 in content


# =====================================================================
# 5. 便捷函数测试
# =====================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test50_visualize_html(self, agent):
        """测试50: visualize函数HTML"""
        content = visualize(agent, "html")
        assert "<!DOCTYPE html>" in content

    def test51_visualize_json(self, agent):
        """测试51: visualize函数JSON"""
        content = visualize(agent, "json")
        json.loads(content)  # 不报错即可

    def test52_visualize_to_file(self, agent, temp_dir):
        """测试52: visualize写文件"""
        path = os.path.join(temp_dir, "viz.html")
        visualize(agent, "html", path)
        assert os.path.exists(path)

    def test53_export_report_alias(self, agent):
        """测试53: export_report是visualize别名"""
        c1 = visualize(agent, "txt")
        c2 = export_report(agent, "txt")
        assert c1 == c2

    def test54_generate_charts_function(self, agent, temp_dir):
        """测试54: generate_charts函数"""
        files = generate_charts(agent, temp_dir)
        assert len(files) > 0
        for f in files:
            assert os.path.exists(f)
            assert f.endswith(".svg")

    def test55_compare_runs_html(self, agents_for_compare):
        """测试55: 比较运行HTML"""
        content = compare_runs(agents_for_compare, labels=["A", "B"],
                               format="html")
        assert "<!DOCTYPE html>" in content
        assert "A" in content
        assert "B" in content

    def test56_compare_runs_markdown(self, agents_for_compare):
        """测试56: 比较运行Markdown"""
        content = compare_runs(agents_for_compare, labels=["A", "B"],
                               format="markdown")
        assert "| 指标 |" in content or "|指标|" in content

    def test57_compare_runs_txt(self, agents_for_compare):
        """测试57: 比较运行TXT"""
        content = compare_runs(agents_for_compare, labels=["A", "B"],
                               format="txt")
        assert "SEVA" in content
        assert "A" in content
        assert "B" in content

    def test58_compare_runs_json(self, agents_for_compare):
        """测试58: 比较运行JSON"""
        content = compare_runs(agents_for_compare, labels=["A", "B"],
                               format="json")
        data = json.loads(content)
        assert len(data) == 2

    def test59_compare_runs_default_labels(self, agents_for_compare):
        """测试59: 默认标签"""
        content = compare_runs(agents_for_compare, format="txt")
        assert "Run 1" in content
        assert "Run 2" in content

    def test60_compare_runs_to_file(self, agents_for_compare, temp_dir):
        """测试60: 比较运行写文件"""
        path = os.path.join(temp_dir, "compare.html")
        compare_runs(agents_for_compare, labels=["X", "Y"],
                     output_file=path, format="html")
        assert os.path.exists(path)

    def test61_compare_runs_empty_agents(self):
        """测试61: 空Agent列表"""
        with pytest.raises(ValueError):
            compare_runs([])

    def test62_compare_runs_mismatched_labels(self, agents_for_compare):
        """测试62: 标签数量不匹配"""
        with pytest.raises(ValueError):
            compare_runs(agents_for_compare, labels=["only_one"])

    def test63_compare_runs_invalid_format(self, agents_for_compare):
        """测试63: 无效比较格式"""
        with pytest.raises(ValueError):
            compare_runs(agents_for_compare, format="xml")


# =====================================================================
# 6. 比较报告内部函数测试
# =====================================================================

class TestCompareInternals:
    """比较报告内部函数测试"""

    def test64_compare_html_function(self, agents_for_compare):
        """测试64: _compare_html函数"""
        from seva.api import get_status
        runs = [{"label": "A", "status": get_status(agents_for_compare[0]),
                 "metrics": agents_for_compare[0].get_metrics()},
                {"label": "B", "status": get_status(agents_for_compare[1]),
                 "metrics": agents_for_compare[1].get_metrics()}]
        html = _compare_html(runs)
        assert "<!DOCTYPE html>" in html
        assert "A" in html
        assert "B" in html

    def test65_compare_markdown_function(self, agents_for_compare):
        """测试65: _compare_markdown函数"""
        from seva.api import get_status
        runs = [{"label": "A", "status": get_status(agents_for_compare[0]),
                 "metrics": agents_for_compare[0].get_metrics()}]
        md = _compare_markdown(runs)
        assert "# SEVA" in md
        assert "A" in md


# =====================================================================
# 7. 包导出与版本测试
# =====================================================================

class TestPackageExports:
    """包导出测试"""

    def test66_result_visualizer_exported(self):
        """测试66: ResultVisualizer已导出"""
        assert hasattr(seva, "ResultVisualizer")
        assert "ResultVisualizer" in seva.__all__

    def test67_svg_chart_generator_exported(self):
        """测试67: SVGChartGenerator已导出"""
        assert hasattr(seva, "SVGChartGenerator")
        assert "SVGChartGenerator" in seva.__all__

    def test68_visualize_exported(self):
        """测试68: visualize已导出"""
        assert hasattr(seva, "visualize")
        assert "visualize" in seva.__all__

    def test69_export_report_exported(self):
        """测试69: export_report已导出"""
        assert hasattr(seva, "export_report")
        assert "export_report" in seva.__all__

    def test70_generate_charts_exported(self):
        """测试70: generate_charts已导出"""
        assert hasattr(seva, "generate_charts")
        assert "generate_charts" in seva.__all__

    def test71_compare_runs_exported(self):
        """测试71: compare_runs已导出"""
        assert hasattr(seva, "compare_runs")
        assert "compare_runs" in seva.__all__

    def test72_version_updated(self):
        """测试72: 版本号已更新"""
        assert seva.__version__ >= "0.38.0"


# =====================================================================
# 8. CLI集成测试
# =====================================================================

class TestCLIIntegration:
    """CLI集成测试"""

    def test73_cli_export_command(self):
        """测试73: CLI export命令存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "json",
                                  "--steps", "0"])
        assert args.format == "json"
        assert args.steps == 0
        assert hasattr(args, "func")

    def test74_cli_charts_command(self):
        """测试74: CLI charts命令存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["charts", "--steps", "0"])
        assert args.steps == 0
        assert hasattr(args, "func")

    def test75_cli_compare_command(self):
        """测试75: CLI compare命令存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["compare", "easy", "standard",
                                  "--steps", "0"])
        assert args.presets == ["easy", "standard"]
        assert hasattr(args, "func")

    def test76_cli_report_format_option(self):
        """测试76: report命令支持format选项"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["report", "--format", "markdown",
                                  "--steps", "0"])
        assert args.format == "markdown"

    def test77_cli_export_formats(self):
        """测试77: export支持所有格式"""
        from seva.cli import build_parser
        for fmt in ["html", "json", "csv", "markdown", "txt"]:
            parser = build_parser()
            args = parser.parse_args(["export", "--format", fmt,
                                      "--steps", "0"])
            assert args.format == fmt

    def test78_cli_compare_formats(self):
        """测试78: compare支持所有格式"""
        from seva.cli import build_parser
        for fmt in ["html", "markdown", "txt", "json"]:
            parser = build_parser()
            args = parser.parse_args(["compare", "easy", "standard",
                                      "--format", fmt, "--steps", "0"])
            assert args.format == fmt


# =====================================================================
# 9. 端到端集成测试
# =====================================================================

class TestEndToEnd:
    """端到端集成测试"""

    def test79_full_workflow_html(self, agent, temp_dir):
        """测试79: 完整HTML工作流"""
        viz = ResultVisualizer(agent)
        # 生成图表
        charts = viz.generate_all_charts(temp_dir)
        assert len(charts) > 0
        # 导出报告
        report_path = os.path.join(temp_dir, "report.html")
        viz.export("html", report_path)
        assert os.path.exists(report_path)
        # 验证报告内容
        with open(report_path, "r") as f:
            content = f.read()
        assert "SEVA" in content
        assert "<svg" in content

    def test80_full_workflow_all_formats(self, agent, temp_dir):
        """测试80: 所有格式完整工作流"""
        viz = ResultVisualizer(agent)
        for fmt in ResultVisualizer.SUPPORTED_FORMATS:
            path = os.path.join(temp_dir, f"report.{fmt}")
            viz.export(fmt, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test81_compare_and_export(self, agents_for_compare, temp_dir):
        """测试81: 比较并导出"""
        for fmt in ["html", "markdown", "txt", "json"]:
            path = os.path.join(temp_dir, f"compare.{fmt}")
            compare_runs(agents_for_compare, labels=["Easy", "Standard"],
                         output_file=path, format=fmt)
            assert os.path.exists(path)

    def test82_visualizer_reuse(self, agent):
        """测试82: 可视化器复用"""
        viz = ResultVisualizer(agent)
        c1 = viz.export("txt")
        c2 = viz.export("txt")
        assert c1 == c2  # 同一Agent多次导出结果一致

    def test83_charts_are_valid_svg(self, agent, temp_dir):
        """测试83: 图表是有效SVG"""
        viz = ResultVisualizer(agent)
        files = viz.generate_all_charts(temp_dir)
        for f in files:
            with open(f, "r") as fh:
                content = fh.read()
            assert content.startswith("<svg")
            assert content.rstrip().endswith("</svg>")
