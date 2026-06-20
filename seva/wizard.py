"""
阶段41: 首次运行向导与引导系统

为新用户提供交互式首次运行引导，降低上手门槛。

核心能力:
    1. 首次运行检测: 自动检测是否首次使用
    2. 交互式向导: 引导用户选择预设、环境、运行步数
    3. 自动配置: 根据向导结果生成配置并运行
    4. 引导提示: 运行后展示下一步建议
    5. 状态标记: 记录已完成向导，避免重复触发

快速启动:
    from seva import FirstRunWizard, run_wizard

    # 启动首次运行向导
    wizard = FirstRunWizard()
    wizard.run()

    # 便捷函数
    run_wizard()
"""
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .core import SelfEvolvingAgentCore
from .api import SEVAFactory, PresetConfigs, quickstart, run, get_status


# =====================================================================
# 状态管理
# =====================================================================

class WizardState:
    """向导状态管理 - 跟踪首次运行完成状态"""

    STATE_DIR = os.path.expanduser("~/.seva")
    STATE_FILE = "wizard_state.json"

    @classmethod
    def get_state_path(cls) -> str:
        """获取状态文件路径"""
        return os.path.join(cls.STATE_DIR, cls.STATE_FILE)

    @classmethod
    def is_first_run(cls) -> bool:
        """检测是否首次运行"""
        return not os.path.exists(cls.get_state_path())

    @classmethod
    def is_completed(cls) -> bool:
        """检测向导是否已完成"""
        if cls.is_first_run():
            return False
        try:
            data = cls.load_state()
            return data.get("completed", False)
        except Exception:
            return False

    @classmethod
    def load_state(cls) -> Dict:
        """加载状态"""
        path = cls.get_state_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def save_state(cls, state: Dict) -> None:
        """保存状态"""
        path = cls.get_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def mark_completed(cls, preset: str = "",
                       env_type: str = "",
                       steps: int = 0) -> None:
        """标记向导已完成"""
        state = cls.load_state()
        state.update({
            "completed": True,
            "completed_at": time.time(),
            "preset": preset,
            "env_type": env_type,
            "steps": steps,
            "version": _get_version(),
        })
        cls.save_state(state)

    @classmethod
    def reset(cls) -> None:
        """重置状态（重新触发向导）"""
        path = cls.get_state_path()
        if os.path.exists(path):
            os.remove(path)


# =====================================================================
# 首次运行向导
# =====================================================================

class FirstRunWizard:
    """
    首次运行向导

    引导新用户完成首次配置和运行。

    Example:
        >>> wizard = FirstRunWizard()
        >>> wizard.run()  # 交互式运行
    """

    # 预设选项
    PRESET_OPTIONS = [
        ("easy", "简单模式 - 适合初学者，参数宽松，快速见效"),
        ("standard", "标准模式 - 平衡配置，适合大多数场景"),
        ("research", "研究模式 - 详细日志，适合深度分析"),
        ("production", "生产模式 - 高性能配置，适合长期运行"),
    ]

    # 环境选项
    ENV_OPTIONS = [
        ("mixed", "混合环境 - 多种任务类型，推荐新手"),
        ("blank", "空白环境 - 无外部刺激，纯自驱"),
        ("drift", "漂移环境 - 环境会变化，测试适应性"),
        ("adversarial", "对抗环境 - 高难度，挑战极限"),
    ]

    # 步数选项
    STEPS_OPTIONS = [
        (10, "快速体验 (10步) - 约5秒"),
        (50, "标准运行 (50步) - 约15秒"),
        (100, "完整运行 (100步) - 约30秒"),
        (500, "深度运行 (500步) - 约2分钟"),
    ]

    def __init__(self, interactive: bool = True):
        """
        初始化向导

        Args:
            interactive: 是否交互式（False时使用默认值）
        """
        self.interactive = interactive
        self.choices: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """
        运行向导

        Returns:
            向导结果字典，包含:
            - preset: 选择的预设
            - env_type: 选择的环境类型
            - steps: 选择的步数
            - agent: 创建的Agent实例
            - stats: 运行统计
        """
        self._print_banner()

        if not self.interactive:
            # 非交互模式使用默认值
            self.choices = {
                "preset": "easy",
                "env_type": "mixed",
                "steps": 10,
            }
        else:
            # 交互式选择
            self.choices["preset"] = self._choose(
                "选择运行模式", self.PRESET_OPTIONS, default=0)
            self.choices["env_type"] = self._choose(
                "选择环境类型", self.ENV_OPTIONS, default=0)
            self.choices["steps"] = self._choose(
                "选择运行步数", self.STEPS_OPTIONS, default=0)

        self._print_separator()
        print(f"\n配置确认:")
        print(f"  模式: {self.choices['preset']}")
        print(f"  环境: {self.choices['env_type']}")
        print(f"  步数: {self.choices['steps']}")

        if self.interactive:
            confirm = self._ask_yes_no("\n确认开始运行?", default=True)
            if not confirm:
                print("\n已取消。可稍后运行 'seva wizard' 重新开始。")
                return {"cancelled": True, **self.choices}

        # 创建并运行Agent
        print(f"\n正在创建Agent...")
        agent = SEVAFactory.create(
            preset=self.choices["preset"],
            env_type=self.choices["env_type"],
            seed=42,
        )

        print(f"运行 {self.choices['steps']} 步...")
        stats = run(agent, steps=self.choices["steps"])

        # 标记完成
        WizardState.mark_completed(
            preset=self.choices["preset"],
            env_type=self.choices["env_type"],
            steps=self.choices["steps"],
        )

        # 展示结果
        self._show_results(agent, stats)

        # 展示下一步建议
        self._show_next_steps()

        return {
            "preset": self.choices["preset"],
            "env_type": self.choices["env_type"],
            "steps": self.choices["steps"],
            "agent": agent,
            "stats": stats,
            "cancelled": False,
        }

    def run_silent(self, preset: str = "easy",
                   env_type: str = "mixed",
                   steps: int = 10) -> Dict[str, Any]:
        """
        静默运行向导（使用指定参数，不交互）

        Args:
            preset: 预设模式
            env_type: 环境类型
            steps: 运行步数

        Returns:
            向导结果字典
        """
        self.interactive = False
        self.choices = {"preset": preset, "env_type": env_type,
                        "steps": steps}

        agent = SEVAFactory.create(preset=preset, env_type=env_type,
                                   seed=42)
        stats = run(agent, steps=steps)

        WizardState.mark_completed(preset=preset, env_type=env_type,
                                   steps=steps)

        return {
            "preset": preset,
            "env_type": env_type,
            "steps": steps,
            "agent": agent,
            "stats": stats,
            "cancelled": False,
        }

    # ================ 内部方法 ================

    def _print_banner(self) -> None:
        """打印欢迎横幅"""
        print()
        print("=" * 60)
        print("  欢迎使用 SEVA Agent 首次运行向导")
        print("  自驱永续进化预测修正Agent系统")
        print("=" * 60)
        print()
        print("  本向导将引导您完成首次配置和运行。")
        print("  您可以随时按 Ctrl+C 取消。")
        print()

    def _print_separator(self) -> None:
        """打印分隔线"""
        print("-" * 60)

    def _choose(self, title: str, options: List[Tuple[str, str]],
                default: int = 0) -> str:
        """
        交互式选择

        Args:
            title: 选择标题
            options: [(value, description), ...]
            default: 默认选项索引

        Returns:
            选择的值
        """
        print(f"\n{title}:")
        for i, (value, desc) in enumerate(options):
            marker = " (默认)" if i == default else ""
            print(f"  [{i + 1}] {desc}{marker}")

        while True:
            try:
                choice = input(f"\n请选择 [1-{len(options)}] "
                               f"(默认{default + 1}): ").strip()
                if not choice:
                    idx = default
                else:
                    idx = int(choice) - 1
                    if idx < 0 or idx >= len(options):
                        print("无效选择，请重试")
                        continue
                return options[idx][0]
            except ValueError:
                print("请输入数字")
            except (EOFError, KeyboardInterrupt):
                # 非交互环境，使用默认值
                return options[default][0]

    def _ask_yes_no(self, prompt: str, default: bool = True) -> bool:
        """是/否问题"""
        default_str = "Y/n" if default else "y/N"
        while True:
            try:
                choice = input(f"{prompt} [{default_str}]: ").strip().lower()
                if not choice:
                    return default
                return choice in ("y", "yes", "是")
            except (EOFError, KeyboardInterrupt):
                return default

    def _show_results(self, agent: SelfEvolvingAgentCore,
                      stats: Any) -> None:
        """展示运行结果"""
        self._print_separator()
        print("\n运行完成! 结果摘要:")
        print()

        status = get_status(agent)
        print(f"  周期数:       {status['cycle_count']}")
        print(f"  运行模式:     {status['mode']}")
        print(f"  健康状态:     {'✓ 健康' if status['healthy'] else '✗ 异常'}")
        print(f"  总奖励:       {status['total_reward']:.2f}")
        print(f"  平均周期时间: {status['avg_cycle_time_ms']:.2f} ms")
        print(f"  任务完成:     {status['tasks_completed']}")
        print(f"  技能总数:     {status['skills_total']}")
        print(f"  进化评估:     {status['evolution_evaluations']}")
        print(f"  当前难度:     {status['current_difficulty']}")

    def _show_next_steps(self) -> None:
        """展示下一步建议"""
        self._print_separator()
        print("\n下一步建议:")
        print()
        print("  1. 查看状态:     seva status")
        print("  2. 运行更多步:   seva run --steps 100")
        print("  3. 查看诊断:     seva diagnose")
        print("  4. 生成报告:     seva report --format html")
        print("  5. 启动Dashboard: seva web")
        print("  6. 交互式Shell:  seva shell")
        print("  7. 查看帮助:     seva --help")
        print()
        print("  完整文档: https://github.com/17683995446/seva-agent")
        print()
        print("=" * 60)


# =====================================================================
# 引导提示系统
# =====================================================================

class GuidedTips:
    """
    引导提示系统

    根据用户当前状态提供上下文相关的提示。
    """

    # 提示库
    TIPS = {
        "first_run": [
            "提示: 使用 'seva wizard' 运行首次向导",
            "提示: 使用 'seva shell' 进入交互式Shell",
            "提示: 使用 'seva --help' 查看所有命令",
        ],
        "low_cycles": [
            "提示: 尝试运行更多步数以看到进化效果: seva run --steps 100",
            "提示: Agent需要足够周期才能展现自驱进化能力",
        ],
        "high_error": [
            "提示: 预测误差较高，运行 'seva diagnose' 检查问题",
            "提示: 尝试更换环境类型: seva run --env-type mixed",
        ],
        "no_skills": [
            "提示: 尚未学习到技能，继续运行以积累经验",
            "提示: 技能学习需要足够周期，建议运行100步以上",
        ],
        "healthy": [
            "提示: 系统运行正常! 尝试 'seva report' 生成报告",
            "提示: 使用 'seva web' 启动可视化Dashboard",
        ],
        "completed": [
            "提示: 您已完成首次运行向导，可以开始探索更多功能",
            "提示: 使用 'seva tutorial' 开始交互式教程",
        ],
    }

    @classmethod
    def get_tips(cls, agent: Optional[SelfEvolvingAgentCore] = None,
                 context: str = "") -> List[str]:
        """
        获取上下文相关提示

        Args:
            agent: Agent实例（可选）
            context: 上下文标识（可选）

        Returns:
            提示列表
        """
        tips = []

        if context and context in cls.TIPS:
            tips.extend(cls.TIPS[context])

        if agent is not None:
            try:
                status = get_status(agent)

                # 根据状态生成提示
                if status["cycle_count"] < 10:
                    tips.extend(cls.TIPS["low_cycles"])

                if status["avg_prediction_error"] > 0.5:
                    tips.extend(cls.TIPS["high_error"])

                if status["skills_total"] == 0:
                    tips.extend(cls.TIPS["no_skills"])

                if status["healthy"] and status["cycle_count"] >= 50:
                    tips.extend(cls.TIPS["healthy"])

            except Exception:
                pass

        if not tips:
            # 默认提示
            tips = cls.TIPS["first_run"]

        # 去重并限制数量
        seen = set()
        unique_tips = []
        for tip in tips:
            if tip not in seen:
                seen.add(tip)
                unique_tips.append(tip)

        return unique_tips[:3]  # 最多3条提示

    @classmethod
    def show_tips(cls, agent: Optional[SelfEvolvingAgentCore] = None,
                  context: str = "") -> None:
        """显示提示"""
        tips = cls.get_tips(agent, context)
        if tips:
            print()
            for tip in tips:
                print(f"  {tip}")
            print()


# =====================================================================
# 便捷函数
# =====================================================================

def run_wizard(interactive: bool = True) -> Dict[str, Any]:
    """
    运行首次运行向导

    Args:
        interactive: 是否交互式

    Returns:
        向导结果字典

    Example:
        >>> from seva import run_wizard
        >>> result = run_wizard()
    """
    wizard = FirstRunWizard(interactive=interactive)
    return wizard.run()


def check_first_run() -> bool:
    """
    检查是否首次运行

    Returns:
        是否首次运行

    Example:
        >>> from seva import check_first_run
        >>> if check_first_run():
        ...     run_wizard()
    """
    return WizardState.is_first_run()


def show_tips(agent: Optional[SelfEvolvingAgentCore] = None,
              context: str = "") -> None:
    """
    显示引导提示

    Args:
        agent: Agent实例（可选）
        context: 上下文标识
    """
    GuidedTips.show_tips(agent, context)


def reset_wizard() -> None:
    """
    重置向导状态（重新触发首次运行）

    Example:
        >>> from seva import reset_wizard
        >>> reset_wizard()  # 下次将重新触发向导
    """
    WizardState.reset()


def _get_version() -> str:
    """获取SEVA版本号"""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
