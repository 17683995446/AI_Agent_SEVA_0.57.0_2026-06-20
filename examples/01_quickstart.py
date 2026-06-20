"""
示例1：快速启动（5分钟入门）
========================

本示例展示SEVA Agent的最简使用方式：
1. 一行代码启动Agent
2. 查看运行状态
3. 获取运行统计

运行方式:
    python examples/01_quickstart.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import quickstart, get_status


def main():
    print("="*60)
    print("SEVA快速启动示例")
    print("="*60)

    # 一行代码启动Agent并运行100步
    print("\n1. 启动Agent运行100步...")
    agent, stats = quickstart(steps=100, preset="easy")

    # 查看运行统计
    print("\n2. 运行统计:")
    print(f"   完成步数:     {stats['steps_completed']}")
    print(f"   总奖励:       {stats['total_real_reward']:.4f}")
    print(f"   运行模式:     {stats['mode']}")
    print(f"   平均周期时间: {stats['avg_cycle_time']*1000:.2f}ms")

    # 查看Agent状态
    print("\n3. Agent状态:")
    status = get_status(agent)
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
下一步:
    - 查看 examples/02_custom_config.py 学习自定义配置
    - 查看 examples/03_monitoring.py 学习监控
    - 运行 seva --help 查看所有CLI命令
    """)


if __name__ == "__main__":
    main()
