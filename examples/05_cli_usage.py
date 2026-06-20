"""
示例5：CLI命令行使用
====================

本示例展示如何使用SEVA的命令行工具：
1. 通过subprocess调用CLI
2. 演示各常用命令
3. 解析命令输出

运行方式:
    python examples/05_cli_usage.py
"""
import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 通过subprocess调用 seva 包的CLI入口（python -m seva）
# 这里导入seva仅用于验证包可用性
import seva  # noqa: F401


def run_cli(*args, cwd=None):
    """运行CLI命令并返回输出"""
    cmd = [sys.executable, "-m", "seva"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def main():
    print("="*60)
    print("SEVA CLI命令行使用示例")
    print("="*60)

    # 1. version命令
    print("\n1. seva version")
    print("-"*40)
    code, out, err = run_cli("version")
    print(out.strip())
    assert code == 0, f"version命令失败: {err}"

    # 2. status命令
    print("\n2. seva status")
    print("-"*40)
    code, out, err = run_cli("status")
    print(out.strip())
    assert code == 0, f"status命令失败: {err}"

    # 3. presets命令
    print("\n3. seva presets")
    print("-"*40)
    code, out, err = run_cli("presets")
    print(out.strip())
    assert code == 0, f"presets命令失败: {err}"

    # 4. config show命令
    print("\n4. seva config show --preset easy --section core")
    print("-"*40)
    code, out, err = run_cli("config", "show", "--preset", "easy",
                             "--section", "core")
    print(out.strip())
    assert code == 0, f"config show命令失败: {err}"

    # 5. config validate命令
    print("\n5. seva config validate --preset standard")
    print("-"*40)
    code, out, err = run_cli("config", "validate", "--preset", "standard")
    print(out.strip())
    assert code == 0, f"config validate命令失败: {err}"

    # 6. run命令（短运行）
    print("\n6. seva run --steps 20 --preset easy")
    print("-"*40)
    code, out, err = run_cli("run", "--steps", "20", "--preset", "easy")
    print(out.strip())
    assert code == 0, f"run命令失败: {err}"

    # 7. errors命令
    print("\n7. seva errors")
    print("-"*40)
    code, out, err = run_cli("errors")
    # errors输出较长，只显示前20行
    lines = out.strip().split("\n")
    print("\n".join(lines[:20]))
    print(f"   ... 共{len(lines)}行")
    assert code == 0, f"errors命令失败: {err}"

    # 8. diagnose命令
    print("\n8. seva diagnose --steps 5 --preset easy")
    print("-"*40)
    code, out, err = run_cli("diagnose", "--steps", "5", "--preset", "easy")
    print(out.strip())
    # diagnose可能返回0或1（取决于健康状态），都是正常完成
    assert "诊断" in out or "健康" in out, f"diagnose命令异常: {err}"

    # 9. config init命令（生成配置文件）
    print("\n9. seva config init --preset easy --output /tmp/seva_cfg.json")
    print("-"*40)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "seva_cfg.json")
        code, out, err = run_cli("config", "init", "--preset", "easy",
                                 "--output", cfg_path)
        print(out.strip())
        assert code == 0, f"config init命令失败: {err}"
        assert os.path.exists(cfg_path), "配置文件未生成"
        print(f"   配置文件已生成: {cfg_path}")

    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("""
CLI命令总览:
    seva run          启动Agent运行
    seva config init  生成配置文件
    seva config show  显示当前配置
    seva config validate  校验配置
    seva status       查看系统状态
    seva version      版本信息
    seva presets      列出预设模板
    seva demo         运行演示
    seva diagnose     诊断系统
    seva errors       列出所有错误码
    seva monitor      实时监控
    seva logs show    显示日志
    seva logs export  导出日志

CLI使用技巧:
    - 使用 --help 查看每个命令的参数
    - 使用 --verbose 查看详细输出
    - 使用 --output 保存结果到文件
    - 使用 --preset 选择预设模板

下一步:
    - 查看 examples/06_advanced.py 学习高级用法
    - 运行 seva --help 查看完整帮助
    """)


if __name__ == "__main__":
    main()
