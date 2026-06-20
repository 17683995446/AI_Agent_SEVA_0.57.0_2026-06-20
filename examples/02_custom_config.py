"""
示例2：自定义配置
================

本示例展示如何自定义SEVA Agent的配置：
1. 使用预设模板
2. 覆盖特定配置项
3. 创建自定义配置
4. 不同预设对比

运行方式:
    python examples/02_custom_config.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import (SEVAFactory, PresetConfigs, create_config, run,
                  get_status)


def main():
    print("="*60)
    print("SEVA自定义配置示例")
    print("="*60)

    # 1. 查看可用预设
    print("\n1. 可用预设模板:")
    for preset in PresetConfigs.list_presets():
        cfg = PresetConfigs.get(preset)
        print(f"   {preset}: input_dim={cfg['perception']['input_dim']}, "
              f"grid_size={cfg['environment']['grid_size']}")
    # 预设说明:
    #   easy:       易用模式（入门推荐）
    #   standard:   标准模式（日常使用）
    #   research:   研究模式（大规模、高精度）
    #   production: 生产模式（稳定优先）

    # 2. 使用预设创建Agent
    print("\n2. 使用easy预设创建Agent（默认mixed环境）...")
    agent_easy = SEVAFactory.create("easy")  # env_type默认为"mixed"
    print(f"   input_dim: {agent_easy.config.get('perception', 'input_dim')}")
    print(f"   grid_size: {agent_easy.config.get('environment', 'grid_size')}")

    # 3. 覆盖特定配置项
    print("\n3. 覆盖特定配置项（research预设+自定义）...")
    agent_custom = SEVAFactory.create(
        "research",  # 使用research预设作为基础
        env_type="mixed",  # 显式指定mixed环境
        overrides={
            "core": {"max_cycles": 500},
            "perception": {"input_dim": 24}
        }
    )
    print(f"   max_cycles: {agent_custom.config.get('core', 'max_cycles')}")
    print(f"   input_dim: {agent_custom.config.get('perception', 'input_dim')}")

    # 4. 创建配置对象（不创建Agent）
    print("\n4. 创建配置对象...")
    config = create_config("standard", overrides={
        "memory": {"episodic_capacity": 5000}
    })
    print(f"   episodic_capacity: {config.get('memory', 'episodic_capacity')}")

    # 5. 运行对比
    print("\n5. 运行对比（各20步）:")
    for preset in ["easy", "standard"]:
        agent = SEVAFactory.create(preset)
        stats = run(agent, steps=20)
        print(f"   {preset}: reward={stats['total_real_reward']:.4f}, "
              f"time={stats['avg_cycle_time']*1000:.2f}ms")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
配置层级（优先级从高到低）:
    1. overrides参数（最高）
    2. 预设模板
    3. 默认配置（最低）

下一步:
    - 查看 examples/03_monitoring.py 学习监控
    - 查看 examples/04_diagnostics.py 学习诊断
    """)


if __name__ == "__main__":
    main()
