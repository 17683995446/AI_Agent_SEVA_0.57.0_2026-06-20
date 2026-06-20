"""
示例6：高级用法
================

本示例展示SEVA的高级功能：
1. 自定义环境运行
2. 多Agent协调
3. 进化闭环监控
4. 长任务管理
5. 技能库查询
6. 自定义错误处理

运行方式:
    python examples/06_advanced.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from seva import (SEVAFactory, run, get_status, Config,
                  SelfEvolvingAgentCore, SevaError, ErrorCode,
                  Diagnoser, Dashboard, MetricsCollector)
from seva.environment import SimulationEnvironment


def main():
    print("="*60)
    print("SEVA高级用法示例")
    print("="*60)

    # 1. 自定义环境运行
    print("\n1. 自定义环境运行:")
    print("-"*40)
    print("   创建Agent并附加drift环境...")
    agent = SEVAFactory.create("standard", env_type="drift", seed=123)
    print(f"   Agent已创建，环境类型: drift")
    print(f"   input_dim: {agent.config.get('perception', 'input_dim')}")
    print(f"   grid_size: {agent.config.get('environment', 'grid_size')}")

    # 2. 运行并查看进化统计
    print("\n2. 运行并查看进化统计:")
    print("-"*40)
    stats = run(agent, steps=100)
    evo_stats = stats.get("evolution_stats", {})
    print(f"   完成步数:       {stats['steps_completed']}")
    print(f"   总奖励:         {stats['total_real_reward']:.4f}")
    print(f"   进化评估次数:   {evo_stats.get('total_evaluations', 0)}")
    print(f"   当前难度:       {evo_stats.get('current_difficulty', 'EASY')}")
    print(f"   超参数调整次数: {evo_stats.get('total_adjustments', 0)}")

    # 3. 长任务管理
    print("\n3. 长任务管理:")
    print("-"*40)
    task_manager = agent.modules["task_manager"]
    task_report = task_manager.get_progress_report()
    print(f"   总任务数:       {task_report.get('total_created', 0)}")
    print(f"   已完成:         {task_report.get('total_completed', 0)}")
    print(f"   进行中:         {task_report.get('total_in_progress', 0)}")
    print(f"   完成率:         {task_report.get('completion_rate', 0):.2%}")

    # 4. 技能库查询
    print("\n4. 技能库查询:")
    print("-"*40)
    skill_lib = agent.modules["skill_library"]
    skill_stats = skill_lib.get_stats()
    print(f"   技能总数:       {skill_stats.get('total_skills', 0)}")
    print(f"   已掌握:         {skill_stats.get('mastered_skills', 0)}")
    print(f"   学习中:         {skill_stats.get('learning_skills', 0)}")

    # 5. 多Agent协调
    print("\n5. 多Agent协调:")
    print("-"*40)
    coordinator = agent.modules["coordinator"]
    coord_status = coordinator.get_status()
    print(f"   协调器状态:     {coord_status}")

    # 6. 自定义错误处理
    print("\n6. 自定义错误处理:")
    print("-"*40)
    try:
        # 故意使用无效预设
        SEVAFactory.create("invalid_preset")
    except (ValueError, SevaError) as e:
        print(f"   捕获到预期错误: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:100]}")

    # 7. 诊断与监控结合
    print("\n7. 诊断与监控结合:")
    print("-"*40)
    diagnoser = Diagnoser(agent)
    report = diagnoser.diagnose()
    print(f"   健康:           {report['healthy']}")
    print(f"   问题数:         {report['issues_count']}")
    print(f"   警告数:         {report['warnings_count']}")
    print(f"   评分:           {report.get('score', 'N/A')}")

    # 8. 仪表盘采集
    print("\n8. 仪表盘采集:")
    print("-"*40)
    dashboard = Dashboard(agent)
    dashboard.collect_from_agent()
    compact = dashboard.render(compact=True)
    print(compact)

    # 9. 指标采集器
    print("\n9. 指标采集器:")
    print("-"*40)
    collector = MetricsCollector()
    for i in range(10):
        collector.record("custom_metric", float(i), tags={"run": "advanced"})
    stats = collector.get_stats("custom_metric")
    print(f"   自定义指标统计:")
    print(f"     count: {stats['count']}")
    print(f"     avg:   {stats['avg']:.2f}")
    print(f"     min:   {stats['min']:.2f}")
    print(f"     max:   {stats['max']:.2f}")

    # 10. 错误码体系
    print("\n10. 错误码体系:")
    print("-"*40)
    print(f"   CONFIG_INVALID: {ErrorCode.CONFIG_INVALID[0]}")
    print(f"     描述: {ErrorCode.CONFIG_INVALID[2]}")
    print(f"   ENV_NOT_ATTACHED: {ErrorCode.ENV_NOT_ATTACHED[0]}")
    print(f"     描述: {ErrorCode.ENV_NOT_ATTACHED[2]}")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
高级功能总览:
    - 自定义环境: SEVAFactory.create(env_type=...)
    - 进化闭环: agent.modules["evolution_loop"]
    - 长任务管理: agent.modules["task_manager"]
    - 技能库: agent.modules["skill_library"]
    - 多Agent协调: agent.modules["coordinator"]
    - 诊断: Diagnoser(agent).diagnose()
    - 监控: Dashboard(agent)
    - 指标: MetricsCollector()
    - 错误码: ErrorCode.*

模块访问:
    所有模块通过 agent.modules 字典访问:
    - perception, drive_engine, goal_generator
    - predictor, decision_maker, safety_ethics
    - executor, error_evaluator, task_update
    - skill_update, coordination, evolution
    - optimizer, memory, metacognition, timing_control

下一步:
    - 阅读 seva/ 目录下的源码了解实现细节
    - 运行 seva demo 查看完整演示
    - 查看 tests/ 目录了解测试用例
    """)


if __name__ == "__main__":
    main()
