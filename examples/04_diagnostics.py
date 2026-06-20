"""
示例4：系统诊断
==============

本示例展示如何诊断SEVA Agent的健康状态：
1. 运行诊断
2. 解读诊断报告
3. 检测常见问题
4. 处理错误

运行方式:
    python examples/04_diagnostics.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import (SEVAFactory, run, Diagnoser, diagnose_agent,
                  ErrorCode, SevaError, format_exception,
                  check_common_issues)


def main():
    print("="*60)
    print("SEVA系统诊断示例")
    print("="*60)

    # 1. 创建并运行Agent
    print("\n1. 创建并运行Agent...")
    agent = SEVAFactory.create("easy")
    run(agent, steps=20)
    print(f"   已运行 {agent.cycle_count} 周期")

    # 2. 运行诊断
    print("\n2. 运行诊断...")
    report = diagnose_agent(agent)
    print(report)

    # 3. 获取诊断报告（结构化）
    print("\n3. 结构化诊断报告:")
    diagnoser = Diagnoser(agent)
    report_dict = diagnoser.diagnose()
    print(f"   健康: {report_dict['healthy']}")
    print(f"   问题数: {report_dict['issues_count']}")
    print(f"   警告数: {report_dict['warnings_count']}")

    # 4. 检查常见问题
    print("\n4. 常见问题检查:")
    issues = check_common_issues(agent)
    if issues:
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   无常见问题")

    # 5. 错误码体系
    print("\n5. 错误码体系示例:")
    print(f"   CONFIG_INVALID: {ErrorCode.CONFIG_INVALID[0]}")
    print(f"     描述: {ErrorCode.CONFIG_INVALID[2]}")
    print(f"     修复: {ErrorCode.CONFIG_INVALID[3]}")

    # 6. 友好错误处理
    print("\n6. 友好错误处理示例:")
    try:
        raise SevaError(ErrorCode.API_INVALID_PRESET, "使用了无效预设",
                        details={"given": "unknown"})
    except SevaError as e:
        print(f"   捕获SevaError:")
        print(f"   {e.format()}")

    # 7. 格式化Python异常
    print("\n7. 格式化Python异常:")
    try:
        raise ValueError("测试值错误")
    except Exception as e:
        formatted = format_exception(e)
        print(f"   {formatted}")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
诊断功能:
    - Diagnoser: 5维诊断（配置/环境/模块/运行时/性能）
    - diagnose_agent(): 便捷诊断函数
    - check_common_issues(): 常见问题检测
    - ErrorCode: 错误码体系（7分类25+错误码）
    - SevaError: 友好异常（含修复建议）
    - format_exception(): Python异常转友好信息

CLI命令:
    - seva diagnose: 诊断系统
    - seva errors: 列出所有错误码

下一步:
    - 查看 examples/05_cli_usage.py 学习CLI使用
    - 查看 examples/06_advanced.py 学习高级用法
    """)


if __name__ == "__main__":
    main()
