"""
阶段37测试：Web Dashboard可视化
================================

验证Web Dashboard模块的完整功能：
1. WebDashboard类
2. start_dashboard函数
3. generate_html_report函数
4. HTTP API端点
5. CLI web/report命令

对应第一性原理：易用 = 可视化操作，无需命令行
"""
import os
import sys
import time
import json
import tempfile
import subprocess
import urllib.request
import threading
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import (WebDashboard, start_dashboard, generate_html_report,
                  SEVAFactory, run)


# ============================================================
# 测试1-5：WebDashboard类基础
# ============================================================
def test1_dashboard_creation():
    """测试1: WebDashboard可创建"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18080)
    assert dashboard is not None
    assert dashboard.agent is not None
    assert dashboard.port == 18080


def test2_dashboard_default_host():
    """测试2: 默认host为localhost"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent)
    assert dashboard.host == "localhost"


def test3_dashboard_default_port():
    """测试3: 默认端口8080"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent)
    assert dashboard.port == 8080


def test4_dashboard_not_running_initially():
    """测试4: 初始状态未运行"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18081)
    assert not dashboard.is_running()


def test5_dashboard_get_url():
    """测试5: 获取URL"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, host="localhost", port=18082)
    url = dashboard.get_url()
    assert url == "http://localhost:18082"


# ============================================================
# 测试6-10：自动创建Agent
# ============================================================
def test6_dashboard_auto_create_agent():
    """测试6: 不提供agent时自动创建"""
    dashboard = WebDashboard(agent=None, port=18083, preset="easy")
    assert dashboard.agent is not None


def test7_dashboard_with_preset():
    """测试7: 指定预设创建Agent"""
    dashboard = WebDashboard(agent=None, port=18084, preset="standard")
    assert dashboard.agent is not None
    assert dashboard.preset == "standard"


def test8_dashboard_start_stop():
    """测试8: 启动和停止"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18085)
    dashboard.start(blocking=False)
    assert dashboard.is_running()
    dashboard.stop()
    assert not dashboard.is_running()


def test9_dashboard_start_non_blocking():
    """测试9: 非阻塞启动"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18086)
    dashboard.start(blocking=False)
    time.sleep(0.5)
    assert dashboard.is_running()
    dashboard.stop()


def test10_dashboard_stop_idempotent():
    """测试10: 重复停止不报错"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18087)
    dashboard.start(blocking=False)
    dashboard.stop()
    dashboard.stop()  # 再次停止不报错


# ============================================================
# 测试11-20：HTTP API端点
# ============================================================
def start_test_dashboard(port=18090):
    """启动测试用Dashboard"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)  # 运行几步产生数据
    dashboard = WebDashboard(agent=agent, port=port)
    dashboard.start(blocking=False)
    time.sleep(0.5)
    return dashboard


def fetch_url(url):
    """获取URL内容"""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        return None


def fetch_json(url):
    """获取JSON"""
    content = fetch_url(url)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


def test11_api_index_html():
    """测试11: 首页返回HTML"""
    dashboard = start_test_dashboard(18091)
    try:
        content = fetch_url("http://localhost:18091/")
        assert content is not None
        assert "<html" in content.lower()
        assert "SEVA" in content
    finally:
        dashboard.stop()


def test12_api_status():
    """测试12: /api/status端点"""
    dashboard = start_test_dashboard(18092)
    try:
        data = fetch_json("http://localhost:18092/api/status")
        assert data is not None
        assert "cycle_count" in data
        assert "mode" in data
    finally:
        dashboard.stop()


def test13_api_metrics():
    """测试13: /api/metrics端点"""
    dashboard = start_test_dashboard(18093)
    try:
        data = fetch_json("http://localhost:18093/api/metrics")
        assert data is not None
        assert "cycle_count" in data
    finally:
        dashboard.stop()


def test14_api_config():
    """测试14: /api/config端点"""
    dashboard = start_test_dashboard(18094)
    try:
        data = fetch_json("http://localhost:18094/api/config")
        assert data is not None
        assert "core" in data
        assert "perception" in data
    finally:
        dashboard.stop()


def test15_api_diagnose():
    """测试15: /api/diagnose端点"""
    dashboard = start_test_dashboard(18095)
    try:
        data = fetch_json("http://localhost:18095/api/diagnose")
        assert data is not None
        assert "healthy" in data
    finally:
        dashboard.stop()


def test16_api_scenarios():
    """测试16: /api/scenarios端点"""
    dashboard = start_test_dashboard(18096)
    try:
        data = fetch_json("http://localhost:18096/api/scenarios")
        assert data is not None
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
    finally:
        dashboard.stop()


def test17_api_run():
    """测试17: /api/run端点"""
    dashboard = start_test_dashboard(18097)
    try:
        data = fetch_json("http://localhost:18097/api/run?steps=5")
        assert data is not None
        assert "steps_completed" in data
        assert data["steps_completed"] > 0
    finally:
        dashboard.stop()


def test18_api_reset():
    """测试18: /api/reset端点"""
    dashboard = start_test_dashboard(18098)
    try:
        data = fetch_json("http://localhost:18098/api/reset")
        assert data is not None
        assert data.get("success") is True
    finally:
        dashboard.stop()


def test19_api_404():
    """测试19: 未知路径返回404"""
    dashboard = start_test_dashboard(18099)
    try:
        content = fetch_url("http://localhost:18099/nonexistent")
        # 404响应
        assert content is None or "404" in content or content == ""
    finally:
        dashboard.stop()


def test20_api_status_after_run():
    """测试20: 运行后状态更新"""
    dashboard = start_test_dashboard(18100)
    try:
        # 获取初始状态
        data1 = fetch_json("http://localhost:18100/api/status")
        initial_cycles = data1["cycle_count"]
        # 运行5步
        fetch_json("http://localhost:18100/api/run?steps=5")
        # 获取新状态
        data2 = fetch_json("http://localhost:18100/api/status")
        assert data2["cycle_count"] > initial_cycles
    finally:
        dashboard.stop()


# ============================================================
# 测试21-25：generate_html_report
# ============================================================
def test21_generate_html_report_returns_string():
    """测试21: 生成HTML报告返回字符串"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test22_generate_html_report_to_file():
    """测试22: 生成HTML报告到文件"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        generate_html_report(agent, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<html" in content.lower()


def test23_html_report_contains_status():
    """测试23: HTML报告包含状态信息"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "周期数" in html or "cycle" in html.lower()
    assert "健康" in html or "healthy" in html.lower()


def test24_html_report_contains_diagnosis():
    """测试24: HTML报告包含诊断信息"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "诊断" in html or "diagnosis" in html.lower()
    assert "问题" in html or "issue" in html.lower()


def test25_html_report_contains_metrics():
    """测试25: HTML报告包含指标"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "预测误差" in html or "prediction" in html.lower()


# ============================================================
# 测试26-30：start_dashboard便捷函数
# ============================================================
def test26_start_dashboard_returns_instance():
    """测试26: start_dashboard返回WebDashboard实例"""
    agent = SEVAFactory.create("easy")
    dashboard = start_dashboard(agent=agent, port=18101, blocking=False)
    assert isinstance(dashboard, WebDashboard)
    dashboard.stop()


def test27_start_dashboard_auto_create_agent():
    """测试27: start_dashboard自动创建Agent"""
    dashboard = start_dashboard(agent=None, port=18102, blocking=False,
                                preset="easy")
    assert dashboard.agent is not None
    dashboard.stop()


def test28_start_dashboard_with_host():
    """测试28: 指定host"""
    agent = SEVAFactory.create("easy")
    dashboard = start_dashboard(agent=agent, host="localhost",
                                port=18103, blocking=False)
    assert dashboard.host == "localhost"
    dashboard.stop()


def test29_start_dashboard_running():
    """测试29: 启动后处于运行状态"""
    agent = SEVAFactory.create("easy")
    dashboard = start_dashboard(agent=agent, port=18104, blocking=False)
    time.sleep(0.5)
    assert dashboard.is_running()
    dashboard.stop()


def test30_start_dashboard_url_accessible():
    """测试30: 启动后URL可访问"""
    agent = SEVAFactory.create("easy")
    dashboard = start_dashboard(agent=agent, port=18105, blocking=False)
    time.sleep(0.5)
    try:
        content = fetch_url("http://localhost:18105/")
        assert content is not None
        assert "SEVA" in content
    finally:
        dashboard.stop()


# ============================================================
# 测试31-35：HTML内容验证
# ============================================================
def test31_html_contains_title():
    """测试31: HTML包含标题"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "SEVA" in html


def test32_html_contains_chinese():
    """测试32: HTML包含中文"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "状态" in html or "报告" in html


def test33_html_contains_table():
    """测试33: HTML包含表格"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "<table" in html.lower()


def test34_html_contains_timestamp():
    """测试34: HTML包含时间戳"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    html = generate_html_report(agent)
    assert "2026" in html  # 当前年份


def test35_dashboard_html_has_styles():
    """测试35: Dashboard HTML有样式"""
    from seva.web_dashboard import DashboardHandler
    html_content = DashboardHandler._generate_html(DashboardHandler)
    assert "<style>" in html_content
    assert "font-family" in html_content


# ============================================================
# 测试36-45：CLI命令
# ============================================================
def run_cli(*args):
    """运行CLI命令"""
    cmd = [sys.executable, "-m", "seva"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def test36_cli_report():
    """测试36: seva report命令"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        code, out, err = run_cli("report", "--steps", "5", "--output", path)
        assert code == 0
        assert os.path.exists(path)
        assert "报告已生成" in out


def test37_cli_report_default_output():
    """测试37: seva report默认输出"""
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(cwd, "seva_report.html")
    try:
        code, out, err = run_cli("report", "--steps", "5")
        assert code == 0
        assert os.path.exists(default_path)
    finally:
        if os.path.exists(default_path):
            os.remove(default_path)


def test38_cli_report_with_preset():
    """测试38: seva report指定预设"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        code, out, err = run_cli("report", "--steps", "5",
                                 "--preset", "standard", "--output", path)
        assert code == 0
        assert os.path.exists(path)


def test39_cli_web_help():
    """测试39: seva web --help"""
    code, out, err = run_cli("web", "--help")
    assert code == 0
    assert "port" in out.lower()
    assert "host" in out.lower()


def test40_cli_report_help():
    """测试40: seva report --help"""
    code, out, err = run_cli("report", "--help")
    assert code == 0
    assert "output" in out.lower()


# ============================================================
# 测试41-50：模块导出与集成
# ============================================================
def test41_module_import():
    """测试41: 模块可导入"""
    from seva import web_dashboard
    assert hasattr(web_dashboard, "WebDashboard")
    assert hasattr(web_dashboard, "start_dashboard")
    assert hasattr(web_dashboard, "generate_html_report")


def test42_all_exports_available():
    """测试42: 所有导出可用"""
    from seva import (WebDashboard, start_dashboard, generate_html_report)
    assert WebDashboard is not None
    assert callable(start_dashboard)
    assert callable(generate_html_report)


def test43_exports_in_all():
    """测试43: 导出在__all__中"""
    import seva
    assert "WebDashboard" in seva.__all__
    assert "start_dashboard" in seva.__all__
    assert "generate_html_report" in seva.__all__


def test44_version_updated():
    """测试44: 版本号已更新"""
    import seva
    assert seva.__version__ >= "0.37.0"


def test45_dashboard_handler_has_methods():
    """测试45: DashboardHandler有do_GET方法"""
    from seva.web_dashboard import DashboardHandler
    assert hasattr(DashboardHandler, "do_GET")


def test46_dashboard_handler_methods():
    """测试46: DashboardHandler有辅助方法"""
    from seva.web_dashboard import DashboardHandler
    assert hasattr(DashboardHandler, "_serve_html")
    assert hasattr(DashboardHandler, "_serve_json")
    assert hasattr(DashboardHandler, "_get_status")
    assert hasattr(DashboardHandler, "_get_metrics")


def test47_dashboard_multiple_start_stop():
    """测试47: 多次启动停止"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18110)
    for i in range(3):
        dashboard.start(blocking=False)
        time.sleep(0.3)
        assert dashboard.is_running()
        dashboard.stop()
        assert not dashboard.is_running()


def test48_dashboard_different_ports():
    """测试48: 不同端口启动"""
    agent1 = SEVAFactory.create("easy")
    agent2 = SEVAFactory.create("easy")
    d1 = WebDashboard(agent=agent1, port=18111)
    d2 = WebDashboard(agent=agent2, port=18112)
    d1.start(blocking=False)
    d2.start(blocking=False)
    time.sleep(0.5)
    try:
        assert d1.is_running()
        assert d2.is_running()
        # 两个端口都可访问
        assert fetch_url("http://localhost:18111/") is not None
        assert fetch_url("http://localhost:18112/") is not None
    finally:
        d1.stop()
        d2.stop()


def test49_dashboard_thread_daemon():
    """测试49: Dashboard线程是daemon"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18113)
    dashboard.start(blocking=False)
    time.sleep(0.3)
    assert dashboard.thread is not None
    assert dashboard.thread.daemon is True
    dashboard.stop()


def test50_complete_web_system():
    """测试50: Web系统完整性"""
    # 1. 创建Dashboard
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    dashboard = WebDashboard(agent=agent, port=18114)

    # 2. 启动
    dashboard.start(blocking=False)
    time.sleep(0.5)

    try:
        # 3. 访问API
        status = fetch_json(f"{dashboard.get_url()}/api/status")
        assert status is not None
        assert "cycle_count" in status

        # 4. 运行Agent
        run_result = fetch_json(f"{dashboard.get_url()}/api/run?steps=5")
        assert run_result is not None
        assert run_result["steps_completed"] > 0

        # 5. 生成报告
        html_report = generate_html_report(agent)
        assert "<html" in html_report.lower()
    finally:
        dashboard.stop()


# ============================================================
# 性能测试
# ============================================================
def test51_dashboard_start_fast():
    """测试51: Dashboard启动应快速"""
    agent = SEVAFactory.create("easy")
    dashboard = WebDashboard(agent=agent, port=18115)
    start = time.time()
    dashboard.start(blocking=False)
    elapsed = time.time() - start
    assert elapsed < 2.0
    dashboard.stop()


def test52_html_report_fast():
    """测试52: HTML报告生成应快速"""
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    start = time.time()
    generate_html_report(agent)
    elapsed = time.time() - start
    assert elapsed < 1.0


def test53_api_response_fast():
    """测试53: API响应应快速"""
    dashboard = start_test_dashboard(18116)
    try:
        start = time.time()
        fetch_json("http://localhost:18116/api/status")
        elapsed = time.time() - start
        assert elapsed < 2.0
    finally:
        dashboard.stop()


# ============================================================
# 完整性测试
# ============================================================
def test54_web_dashboard_user_friendly():
    """测试54: Web Dashboard用户友好性"""
    from seva.web_dashboard import DashboardHandler
    html_content = DashboardHandler._generate_html(DashboardHandler)

    # 应包含中文
    assert "SEVA" in html_content
    # 应有控制按钮
    assert "button" in html_content.lower() or "btn" in html_content.lower()
    # 应有样式
    assert "<style>" in html_content


def test55_no_external_dependencies():
    """测试55: Web Dashboard无外部依赖"""
    # 只使用标准库http.server
    from seva.web_dashboard import HTTPServer, BaseHTTPRequestHandler
    assert HTTPServer is not None
    assert BaseHTTPRequestHandler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
