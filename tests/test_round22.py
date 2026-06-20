"""
第二十二轮测试：阶段32日志可视化与监控验证
验证StructuredLogger、MetricsCollector、Dashboard、monitor
核心目标：证明运行状态可见

第一性原理验证：
1. 易用 = 结构化日志：支持text/json两种格式
2. 易用 = 指标采集：运行时指标自动采集
3. 易用 = 仪表盘：终端可视化显示状态
4. 易用 = 监控装饰器：自动记录函数执行
"""
import sys
import os
import time
import json
import tempfile
import traceback
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

from seva.monitoring import (StructuredLogger, MetricsCollector, Dashboard,
                              monitor, setup_logging, export_metrics,
                              MetricPoint)
from seva.api import SEVAFactory, run

PASS = 0
FAIL = 0
RESULTS = []


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    RESULTS.append(f"\n=== {title} ===")


def capture_cli(*args) -> tuple:
    result = subprocess.run(
        [sys.executable, "-m", "seva"] + list(args),
        capture_output=True, text=True, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode, result.stdout, result.stderr


# ============================================================
# 测试1：StructuredLogger基础
# ============================================================
def test1_structured_logger_basic():
    section("测试1：StructuredLogger基础")
    logger = StructuredLogger("test1", format="text", level=logging.INFO)
    report("初始化成功", logger is not None)
    report("name正确", logger.name == "test1")
    report("format正确", logger.format == "text")
    report("records初始空", len(logger.get_records()) == 0)


# ============================================================
# 测试2：StructuredLogger记录日志
# ============================================================
def test2_structured_logger_log():
    section("测试2：StructuredLogger记录日志")
    logger = StructuredLogger("test2", format="text", level=logging.DEBUG)
    logger.info("测试消息")
    logger.warning("警告消息")
    logger.error("错误消息")
    records = logger.get_records()
    report("记录3条", len(records) == 3)
    report("第1条INFO", records[0]["level"] == "INFO")
    report("第2条WARNING", records[1]["level"] == "WARNING")
    report("第3条ERROR", records[2]["level"] == "ERROR")
    report("消息内容正确", records[0]["message"] == "测试消息")


# ============================================================
# 测试3：StructuredLogger extra字段
# ============================================================
def test3_structured_logger_extra():
    section("测试3：StructuredLogger extra字段")
    logger = StructuredLogger("test3", format="text", level=logging.DEBUG)
    logger.info("测试", extra={"key1": "value1", "key2": 123})
    records = logger.get_records()
    report("记录1条", len(records) == 1)
    report("含key1", records[0].get("key1") == "value1")
    report("含key2", records[0].get("key2") == 123)


# ============================================================
# 测试4：StructuredLogger JSON格式
# ============================================================
def test4_structured_logger_json():
    section("测试4：StructuredLogger JSON格式")
    logger = StructuredLogger("test4", format="json", level=logging.DEBUG)
    logger.info("JSON测试", extra={"preset": "easy"})
    records = logger.get_records()
    report("记录1条", len(records) == 1)
    report("含timestamp", "timestamp" in records[0])
    report("含level", "level" in records[0])
    report("含message", "message" in records[0])
    report("含preset", records[0].get("preset") == "easy")


# ============================================================
# 测试5：StructuredLogger级别过滤
# ============================================================
def test5_structured_logger_level_filter():
    section("测试5：StructuredLogger级别过滤")
    logger = StructuredLogger("test5", format="text", level=logging.WARNING)
    logger.debug("debug")  # 不应记录到底层logger
    logger.info("info")     # 不应记录到底层logger
    logger.warning("warning")  # 应记录
    logger.error("error")   # 应记录
    records = logger.get_records()
    # records记录所有（不受level限制），但底层logger只输出>=level
    report("records记录所有", len(records) == 4)


# ============================================================
# 测试6：StructuredLogger get_records过滤
# ============================================================
def test6_structured_logger_get_records_filter():
    section("测试6：get_records级别过滤")
    logger = StructuredLogger("test6", format="text", level=logging.DEBUG)
    logger.info("info1")
    logger.warning("warn1")
    logger.error("err1")
    logger.warning("warn2")

    all_records = logger.get_records()
    report("全部4条", len(all_records) == 4)

    warn_records = logger.get_records(level="WARNING")
    report("WARNING 2条", len(warn_records) == 2)

    err_records = logger.get_records(level="ERROR")
    report("ERROR 1条", len(err_records) == 1)


# ============================================================
# 测试7：StructuredLogger limit
# ============================================================
def test7_structured_logger_limit():
    section("测试7：get_records limit")
    logger = StructuredLogger("test7", format="text", level=logging.DEBUG)
    for i in range(50):
        logger.info(f"msg{i}")
    records = logger.get_records(limit=10)
    report("limit=10返回10条", len(records) == 10)
    report("返回最后10条", records[-1]["message"] == "msg49")


# ============================================================
# 测试8：StructuredLogger clear
# ============================================================
def test8_structured_logger_clear():
    section("测试8：StructuredLogger clear")
    logger = StructuredLogger("test8", format="text", level=logging.DEBUG)
    logger.info("msg1")
    logger.info("msg2")
    report("clear前2条", len(logger.get_records()) == 2)
    logger.clear()
    report("clear后0条", len(logger.get_records()) == 0)


# ============================================================
# 测试9：StructuredLogger输出到文件
# ============================================================
def test9_structured_logger_file():
    section("测试9：StructuredLogger输出到文件")
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = StructuredLogger("test9", format="text", level=logging.DEBUG,
                                 output_file=log_file)
        logger.info("文件测试")
        report("文件已创建", os.path.exists(log_file))
        with open(log_file) as f:
            content = f.read()
        report("文件含消息", "文件测试" in content)


# ============================================================
# 测试10：MetricPoint数据类
# ============================================================
def test10_metric_point():
    section("测试10：MetricPoint数据类")
    p = MetricPoint(timestamp=time.time(), value=1.5, tags={"env": "test"})
    report("timestamp正确", p.timestamp > 0)
    report("value正确", p.value == 1.5)
    report("tags正确", p.tags["env"] == "test")


# ============================================================
# 测试11：MetricsCollector基础
# ============================================================
def test11_metrics_collector_basic():
    section("测试11：MetricsCollector基础")
    mc = MetricsCollector()
    report("初始化成功", mc is not None)
    report("初始无指标", len(mc.list_metrics()) == 0)


# ============================================================
# 测试12：MetricsCollector记录
# ============================================================
def test12_metrics_collector_record():
    section("测试12：MetricsCollector记录")
    mc = MetricsCollector()
    mc.record("cpu", 50.0)
    mc.record("cpu", 60.0)
    mc.record("memory", 70.0)
    report("2个指标", len(mc.list_metrics()) == 2)
    report("含cpu", "cpu" in mc.list_metrics())
    report("含memory", "memory" in mc.list_metrics())

    cpu_points = mc.get_metric("cpu")
    report("cpu 2个点", len(cpu_points) == 2)
    report("第1个点50.0", cpu_points[0].value == 50.0)
    report("第2个点60.0", cpu_points[1].value == 60.0)


# ============================================================
# 测试13：MetricsCollector get_latest
# ============================================================
def test13_metrics_collector_latest():
    section("测试13：MetricsCollector get_latest")
    mc = MetricsCollector()
    mc.record("temp", 20.0)
    mc.record("temp", 25.0)
    mc.record("temp", 30.0)
    latest = mc.get_latest("temp")
    report("返回MetricPoint", isinstance(latest, MetricPoint))
    report("最新值30.0", latest.value == 30.0)

    # 不存在的指标
    none_latest = mc.get_latest("nonexistent")
    report("不存在返回None", none_latest is None)


# ============================================================
# 测试14：MetricsCollector get_stats
# ============================================================
def test14_metrics_collector_stats():
    section("测试14：MetricsCollector get_stats")
    mc = MetricsCollector()
    for v in [10, 20, 30, 40, 50]:
        mc.record("metric", float(v))
    stats = mc.get_stats("metric")
    report("返回dict", isinstance(stats, dict))
    report("count=5", stats["count"] == 5)
    report("avg=30", stats["avg"] == 30.0)
    report("min=10", stats["min"] == 10.0)
    report("max=50", stats["max"] == 50.0)
    report("last=50", stats["last"] == 50.0)


# ============================================================
# 测试15：MetricsCollector clear
# ============================================================
def test15_metrics_collector_clear():
    section("测试15：MetricsCollector clear")
    mc = MetricsCollector()
    mc.record("a", 1.0)
    mc.record("b", 2.0)
    mc.clear()
    report("clear后无指标", len(mc.list_metrics()) == 0)


# ============================================================
# 测试16：Dashboard基础
# ============================================================
def test16_dashboard_basic():
    section("测试16：Dashboard基础")
    agent = SEVAFactory.create("easy")
    dashboard = Dashboard(agent)
    report("初始化成功", dashboard is not None)
    report("含agent", dashboard.agent is not None)
    report("含collector", dashboard.collector is not None)


# ============================================================
# 测试17：Dashboard渲染
# ============================================================
def test17_dashboard_render():
    section("测试17：Dashboard渲染")
    agent = SEVAFactory.create("easy")
    run(agent, steps=10)
    dashboard = Dashboard(agent)
    output = dashboard.render()
    report("返回str", isinstance(output, str))
    report("含SEVA运行时仪表盘", "SEVA运行时仪表盘" in output)
    report("含周期数", "周期数" in output)
    report("含运行模式", "运行模式" in output)
    report("含总奖励", "总奖励" in output)


# ============================================================
# 测试18：Dashboard compact模式
# ============================================================
def test18_dashboard_compact():
    section("测试18：Dashboard compact模式")
    agent = SEVAFactory.create("easy")
    run(agent, steps=5)
    dashboard = Dashboard(agent)
    full = dashboard.render(compact=False)
    compact = dashboard.render(compact=True)
    report("full含任务系统", "任务系统" in full)
    report("compact不含任务系统", "任务系统" not in compact)
    report("compact更短", len(compact) < len(full))


# ============================================================
# 测试19：Dashboard采集指标
# ============================================================
def test19_dashboard_collect():
    section("测试19：Dashboard采集指标")
    agent = SEVAFactory.create("easy")
    run(agent, steps=10)
    dashboard = Dashboard(agent)
    dashboard.collect_from_agent()
    metrics = dashboard.collector.list_metrics()
    report("采集了指标", len(metrics) > 0)
    report("含cycle_count", "cycle_count" in metrics)
    report("含total_reward", "total_reward" in metrics)
    report("含avg_cycle_time", "avg_cycle_time" in metrics)


# ============================================================
# 测试20：Dashboard趋势
# ============================================================
def test20_dashboard_trend():
    section("测试20：Dashboard趋势")
    agent = SEVAFactory.create("easy")
    dashboard = Dashboard(agent)
    # 手动记录趋势数据
    for i in range(20):
        dashboard.collector.record("test_metric", float(i))
    trend = dashboard.get_trend("test_metric", window=20)
    report("趋势为↑", trend == "↑", f"trend={trend}")

    # 下降趋势
    dashboard2 = Dashboard(agent)
    for i in range(20):
        dashboard2.collector.record("test_metric", float(20-i))
    trend2 = dashboard2.get_trend("test_metric", window=20)
    report("趋势为↓", trend2 == "↓", f"trend={trend2}")


# ============================================================
# 测试21：monitor装饰器
# ============================================================
def test21_monitor_decorator():
    section("测试21：monitor装饰器")
    @monitor
    def test_func():
        return 42

    result = test_func()
    report("函数正常执行", result == 42)

    @monitor
    def failing_func():
        raise ValueError("测试异常")

    try:
        failing_func()
        report("异常被重新抛出", False)
    except ValueError:
        report("异常被重新抛出", True)


# ============================================================
# 测试22：setup_logging
# ============================================================
def test22_setup_logging():
    section("测试22：setup_logging")
    logger = setup_logging(level="INFO", format="text")
    report("返回StructuredLogger", isinstance(logger, StructuredLogger))
    report("name为seva", logger.name == "seva")

    logger_json = setup_logging(level="DEBUG", format="json")
    report("json格式", logger_json.format == "json")


# ============================================================
# 测试23：export_metrics
# ============================================================
def test23_export_metrics():
    section("测试23：export_metrics")
    mc = MetricsCollector()
    mc.record("cpu", 50.0)
    mc.record("cpu", 60.0)
    mc.record("memory", 70.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "metrics.json")
        export_metrics(mc, output_file, format="json")
        report("文件已创建", os.path.exists(output_file))

        with open(output_file) as f:
            data = json.load(f)
        report("含cpu", "cpu" in data)
        report("含memory", "memory" in data)
        report("cpu 2个点", len(data["cpu"]) == 2)


# ============================================================
# 测试24：export_metrics CSV格式
# ============================================================
def test24_export_metrics_csv():
    section("测试24：export_metrics CSV格式")
    mc = MetricsCollector()
    mc.record("temp", 25.0)
    mc.record("temp", 26.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "metrics.csv")
        export_metrics(mc, output_file, format="csv")
        report("文件已创建", os.path.exists(output_file))
        with open(output_file) as f:
            content = f.read()
        report("含temp", "temp" in content)


# ============================================================
# 测试25：Dashboard无Agent
# ============================================================
def test25_dashboard_no_agent():
    section("测试25：Dashboard无Agent")
    dashboard = Dashboard()
    output = dashboard.render()
    report("返回str", isinstance(output, str))
    report("含未连接Agent", "未连接Agent" in output)


# ============================================================
# 测试26：CLI monitor命令
# ============================================================
def test26_cli_monitor():
    section("测试26：CLI monitor命令")
    code, stdout, stderr = capture_cli("monitor", "--steps", "10", "--preset", "easy")
    report("退出码0", code == 0)
    report("输出含SEVA实时监控", "SEVA实时监控" in stdout)
    report("输出含仪表盘", "仪表盘" in stdout)


# ============================================================
# 测试27：CLI monitor导出指标
# ============================================================
def test27_cli_monitor_export():
    section("测试27：CLI monitor导出指标")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "metrics.json")
        code, stdout, stderr = capture_cli(
            "monitor", "--steps", "5", "--preset", "easy",
            "--output", output_path
        )
        report("退出码0", code == 0)
        report("指标文件已创建", os.path.exists(output_path))


# ============================================================
# 测试28：CLI logs show命令
# ============================================================
def test28_cli_logs_show():
    section("测试28：CLI logs show命令")
    code, stdout, stderr = capture_cli(
        "logs", "show", "--steps", "5", "--preset", "easy"
    )
    report("退出码0", code == 0)


# ============================================================
# 测试29：CLI logs export命令
# ============================================================
def test29_cli_logs_export():
    section("测试29：CLI logs export命令")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "export.log")
        code, stdout, stderr = capture_cli(
            "logs", "export", "--steps", "5", "--preset", "easy",
            "--output", output_path
        )
        report("退出码0", code == 0)
        report("日志文件已创建", os.path.exists(output_path))


# ============================================================
# 测试30：包级导入
# ============================================================
def test30_package_import():
    section("测试30：包级导入")
    import seva
    report("seva.StructuredLogger可导入", hasattr(seva, "StructuredLogger"))
    report("seva.MetricsCollector可导入", hasattr(seva, "MetricsCollector"))
    report("seva.Dashboard可导入", hasattr(seva, "Dashboard"))
    report("seva.monitor可导入", hasattr(seva, "monitor"))
    report("seva.setup_logging可导入", hasattr(seva, "setup_logging"))
    report("seva.export_metrics可导入", hasattr(seva, "export_metrics"))


# ============================================================
# 测试31：端到端监控流程
# ============================================================
def test31_e2e_monitoring():
    section("测试31：端到端监控流程")
    # 1. 创建Agent
    agent = SEVAFactory.create("easy")

    # 2. 创建仪表盘
    dashboard = Dashboard(agent)

    # 3. 运行并采集
    for i in range(20):
        if agent._env_done:
            break
        if not hasattr(agent, "_pending_input") or agent._pending_input is None:
            import numpy as np
            agent.inject_input(np.random.randn(16) * 0.3)
        agent.run_cycle()
        dashboard.collect_from_agent()

    # 4. 渲染仪表盘
    output = dashboard.render()
    report("仪表盘渲染成功", "SEVA运行时仪表盘" in output)
    report("周期数=20", "20" in output)

    # 5. 导出指标
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, "metrics.json")
        export_metrics(dashboard.collector, export_file)
        report("指标导出成功", os.path.exists(export_file))


# ============================================================
# 测试32：性能测试
# ============================================================
def test32_performance():
    section("测试32：监控性能开销")
    agent = SEVAFactory.create("easy")
    dashboard = Dashboard(agent)

    t0 = time.time()
    for _ in range(50):
        agent.run_cycle()
        dashboard.collect_from_agent()
    elapsed = time.time() - t0
    report("50步监控<5秒", elapsed < 5, f"{elapsed:.2f}s")

    # 仪表盘渲染性能
    t0 = time.time()
    for _ in range(100):
        dashboard.render(compact=True)
    render_time = (time.time() - t0) / 100 * 1000
    report("仪表盘渲染<5ms", render_time < 5, f"{render_time:.2f}ms")


# ============================================================
# 主函数
# ============================================================
def main_test():
    print("\n" + "="*60)
    print("第二十二轮测试：阶段32日志可视化与监控验证")
    print("="*60)

    tests = [
        test1_structured_logger_basic,
        test2_structured_logger_log,
        test3_structured_logger_extra,
        test4_structured_logger_json,
        test5_structured_logger_level_filter,
        test6_structured_logger_get_records_filter,
        test7_structured_logger_limit,
        test8_structured_logger_clear,
        test9_structured_logger_file,
        test10_metric_point,
        test11_metrics_collector_basic,
        test12_metrics_collector_record,
        test13_metrics_collector_latest,
        test14_metrics_collector_stats,
        test15_metrics_collector_clear,
        test16_dashboard_basic,
        test17_dashboard_render,
        test18_dashboard_compact,
        test19_dashboard_collect,
        test20_dashboard_trend,
        test21_monitor_decorator,
        test22_setup_logging,
        test23_export_metrics,
        test24_export_metrics_csv,
        test25_dashboard_no_agent,
        test26_cli_monitor,
        test27_cli_monitor_export,
        test28_cli_logs_show,
        test29_cli_logs_export,
        test30_package_import,
        test31_e2e_monitoring,
        test32_performance,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [ERROR] {test.__name__} 异常: {e}")
            traceback.print_exc()
            global FAIL
            FAIL += 1
            RESULTS.append(f"  [ERROR] {test.__name__}: {e}")

    print("\n" + "="*60)
    print(f"测试总结: {PASS}通过 / {FAIL}失败")
    print("="*60)

    print("\n".join(RESULTS))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main_test())
