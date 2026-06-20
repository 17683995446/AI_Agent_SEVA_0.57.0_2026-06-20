"""
示例3：运行时监控
================

本示例展示如何监控SEVA Agent的运行状态：
1. 使用Dashboard实时查看状态
2. 采集运行时指标
3. 导出指标数据
4. 使用结构化日志

运行方式:
    python examples/03_monitoring.py
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import SEVAFactory, run, Dashboard, MetricsCollector, export_metrics
from seva.monitoring import StructuredLogger, setup_logging


def main():
    print("="*60)
    print("SEVA运行时监控示例")
    print("="*60)

    # 1. 创建Agent和仪表盘
    print("\n1. 创建Agent和仪表盘...")
    agent = SEVAFactory.create("easy")
    dashboard = Dashboard(agent)
    print("   Agent和仪表盘已创建")

    # 2. 运行并定期查看仪表盘
    print("\n2. 运行50步并采集指标...")
    for i in range(50):
        if agent._env_done:
            break
        if not hasattr(agent, "_pending_input") or agent._pending_input is None:
            import numpy as np
            agent.inject_input(np.random.randn(16) * 0.3)
        agent.run_cycle()
        dashboard.collect_from_agent()

    # 3. 显示仪表盘
    print("\n3. 运行时仪表盘:")
    print("-"*40)
    dashboard.print(compact=False)

    # 4. 查看采集的指标
    print("\n4. 采集的指标:")
    metrics = dashboard.collector.list_metrics()
    for name in metrics:
        stats = dashboard.collector.get_stats(name)
        print(f"   {name}: avg={stats['avg']:.4f}, "
              f"last={stats['last']:.4f}")

    # 5. 获取趋势
    print("\n5. 指标趋势:")
    for name in ["total_reward", "avg_cycle_time"]:
        trend = dashboard.get_trend(name)
        print(f"   {name}: {trend}")

    # 6. 导出指标
    print("\n6. 导出指标...")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, "metrics.json")
        export_metrics(dashboard.collector, export_file)
        print(f"   指标已导出到 {export_file}")

    # 7. 结构化日志
    print("\n7. 结构化日志示例:")
    logger = StructuredLogger("example", format="text")
    logger.info("Agent运行完成", extra={"steps": 50, "preset": "easy"})
    logger.warning("这是警告示例")
    logger.error("这是错误示例")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
监控功能:
    - Dashboard: 实时仪表盘（compact/full模式）
    - MetricsCollector: 指标采集与统计
    - StructuredLogger: 结构化日志（text/json）
    - export_metrics: 指标导出（json/csv）

CLI命令:
    - seva monitor: 实时监控
    - seva logs show: 显示日志
    - seva logs export: 导出日志

下一步:
    - 查看 examples/04_diagnostics.py 学习诊断
    - 查看 examples/05_cli_usage.py 学习CLI使用
    """)


if __name__ == "__main__":
    main()
