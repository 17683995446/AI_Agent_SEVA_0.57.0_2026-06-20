"""
阶段43: 交互式教程系统

提供分步骤的交互式教程，引导用户逐步学习SEVA的核心功能。

核心能力:
    1. 多教程课程: 入门/进阶/专家三个级别
    2. 分步骤教学: 每步包含说明、示例、验证
    3. 进度跟踪: 记录学习进度，支持断点续学
    4. 自动验证: 每步执行后自动验证结果
    5. 交互式练习: 用户动手操作，加深理解

快速启动:
    from seva import Tutorial, run_tutorial, list_tutorials

    # 列出可用教程
    print(list_tutorials())

    # 运行教程
    run_tutorial("beginner")

    # 查看进度
    from seva import TutorialProgress
    print(TutorialProgress.get_progress("beginner"))
"""
import os
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import SelfEvolvingAgentCore
from .api import SEVAFactory, quickstart, run, get_status


# =====================================================================
# 教程进度管理
# =====================================================================

class TutorialProgress:
    """教程进度管理"""

    STATE_DIR = os.path.expanduser("~/.seva")
    STATE_FILE = "tutorial_progress.json"

    @classmethod
    def get_state_path(cls) -> str:
        return os.path.join(cls.STATE_DIR, cls.STATE_FILE)

    @classmethod
    def load_all(cls) -> Dict:
        """加载所有进度"""
        path = cls.get_state_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def get_progress(cls, tutorial_id: str) -> Dict:
        """获取指定教程进度"""
        all_progress = cls.load_all()
        return all_progress.get(tutorial_id, {
            "completed_steps": [],
            "current_step": 0,
            "completed": False,
            "started_at": None,
            "completed_at": None,
        })

    @classmethod
    def save_progress(cls, tutorial_id: str,
                      progress: Dict) -> None:
        """保存教程进度"""
        all_progress = cls.load_all()
        all_progress[tutorial_id] = progress
        path = cls.get_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_progress, f, indent=2, ensure_ascii=False,
                      default=str)

    @classmethod
    def mark_step_completed(cls, tutorial_id: str,
                            step_index: int) -> None:
        """标记步骤完成"""
        progress = cls.get_progress(tutorial_id)
        if step_index not in progress["completed_steps"]:
            progress["completed_steps"].append(step_index)
        progress["current_step"] = step_index + 1
        if progress["started_at"] is None:
            progress["started_at"] = time.time()
        cls.save_progress(tutorial_id, progress)

    @classmethod
    def mark_completed(cls, tutorial_id: str) -> None:
        """标记教程完成"""
        progress = cls.get_progress(tutorial_id)
        progress["completed"] = True
        progress["completed_at"] = time.time()
        cls.save_progress(tutorial_id, progress)

    @classmethod
    def reset(cls, tutorial_id: Optional[str] = None) -> None:
        """重置进度"""
        if tutorial_id is None:
            path = cls.get_state_path()
            if os.path.exists(path):
                os.remove(path)
        else:
            all_progress = cls.load_all()
            if tutorial_id in all_progress:
                del all_progress[tutorial_id]
                path = cls.get_state_path()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(all_progress, f, indent=2,
                              ensure_ascii=False, default=str)

    @classmethod
    def is_completed(cls, tutorial_id: str) -> bool:
        """检查教程是否完成"""
        return cls.get_progress(tutorial_id).get("completed", False)


# =====================================================================
# 教程定义
# =====================================================================

class TutorialStep:
    """教程步骤"""

    def __init__(self, title: str, description: str,
                 example: str = "",
                 verify: Optional[Callable] = None,
                 tip: str = ""):
        self.title = title
        self.description = description
        self.example = example
        self.verify = verify
        self.tip = tip


class Tutorial:
    """
    交互式教程

    Example:
        >>> tutorial = Tutorial("beginner", "入门教程", steps)
        >>> tutorial.run()
    """

    # 教程定义
    TUTORIALS: Dict[str, Dict] = {}

    def __init__(self, tutorial_id: str, title: str,
                 description: str,
                 steps: List[TutorialStep],
                 difficulty: str = "beginner"):
        self.id = tutorial_id
        self.title = title
        self.description = description
        self.steps = steps
        self.difficulty = difficulty

    def run(self, interactive: bool = True,
            start_step: int = 0) -> bool:
        """
        运行教程

        Args:
            interactive: 是否交互式
            start_step: 起始步骤（用于断点续学）

        Returns:
            是否完成
        """
        self._print_banner()

        if interactive:
            print(f"\n本教程共 {len(self.steps)} 步，预计需要5-10分钟。")
            print(f"您可以随时按 Ctrl+C 退出，进度会自动保存。\n")
            input("按回车键开始...")

        for i in range(start_step, len(self.steps)):
            step = self.steps[i]
            self._run_step(i, step, interactive)

            if interactive:
                try:
                    input("\n按回车键继续下一步...")
                except (EOFError, KeyboardInterrupt):
                    print("\n\n教程已暂停。进度已保存。")
                    return False

            TutorialProgress.mark_step_completed(self.id, i)

        TutorialProgress.mark_completed(self.id)

        self._print_completion()
        return True

    def _run_step(self, index: int, step: TutorialStep,
                  interactive: bool) -> None:
        """运行单个步骤"""
        print(f"\n{'=' * 60}")
        print(f"  步骤 {index + 1}/{len(self.steps)}: {step.title}")
        print(f"{'=' * 60}")

        print(f"\n{step.description}")

        if step.example:
            print(f"\n示例:")
            print(f"  {step.example}")

        if step.tip:
            print(f"\n提示: {step.tip}")

        # 执行验证
        if step.verify:
            print(f"\n正在验证...")
            try:
                result = step.verify()
                if result:
                    print(f"  ✓ 验证通过!")
                else:
                    print(f"  ✗ 验证失败，请重试")
            except Exception as e:
                print(f"  ✗ 验证出错: {e}")

    def _print_banner(self) -> None:
        """打印教程横幅"""
        print()
        print("=" * 60)
        print(f"  SEVA教程: {self.title}")
        print(f"  难度: {self.difficulty}")
        print(f"  {self.description}")
        print("=" * 60)

    def _print_completion(self) -> None:
        """打印完成信息"""
        print()
        print("=" * 60)
        print("  恭喜! 教程已完成!")
        print("=" * 60)
        print(f"\n您已完成《{self.title}》教程的全部 {len(self.steps)} 步。")
        print("\n下一步建议:")
        print("  - 运行更多教程: seva tutorial --list")
        print("  - 开始使用SEVA: seva wizard")
        print("  - 查看帮助: seva --help")

    @classmethod
    def get_tutorial(cls, tutorial_id: str) -> Optional["Tutorial"]:
        """获取教程"""
        return cls.TUTORIALS.get(tutorial_id)

    @classmethod
    def list_tutorials(cls) -> List[Dict]:
        """列出所有教程"""
        result = []
        for tid, tutorial in cls.TUTORIALS.items():
            progress = TutorialProgress.get_progress(tid)
            result.append({
                "id": tid,
                "title": tutorial.title,
                "description": tutorial.description,
                "difficulty": tutorial.difficulty,
                "steps": len(tutorial.steps),
                "completed": progress.get("completed", False),
                "current_step": progress.get("current_step", 0),
            })
        return result

    @classmethod
    def register(cls, tutorial: "Tutorial") -> None:
        """注册教程"""
        cls.TUTORIALS[tutorial.id] = tutorial


# =====================================================================
# 内置教程
# =====================================================================

def _create_beginner_tutorial() -> Tutorial:
    """创建入门教程"""
    steps = [
        TutorialStep(
            title="欢迎来到SEVA",
            description=(
                "SEVA是一个自驱永续进化预测修正Agent系统。\n"
                "它能够自主感知环境、生成目标、预测结果、\n"
                "评估误差并不断进化。"
            ),
            example="import seva; print(seva.__version__)",
            tip="SEVA的所有功能都可以通过seva包访问",
        ),
        TutorialStep(
            title="快速启动",
            description=(
                "使用quickstart函数可以一键创建并运行Agent:\n"
                "它会自动创建Agent、附加环境、运行指定步数。"
            ),
            example=(
                "from seva import quickstart\n"
                "agent, stats = quickstart(steps=10)\n"
                "print(f'周期数: {agent.cycle_count}')"
            ),
            verify=lambda: True,
            tip="quickstart是最简单的入门方式",
        ),
        TutorialStep(
            title="查看Agent状态",
            description=(
                "使用get_status函数可以查看Agent的当前状态，\n"
                "包括周期数、运行模式、健康状态等。"
            ),
            example=(
                "from seva import quickstart, get_status\n"
                "agent, _ = quickstart(steps=10)\n"
                "status = get_status(agent)\n"
                "print(status)"
            ),
            verify=lambda: True,
            tip="status返回一个字典，包含所有关键指标",
        ),
        TutorialStep(
            title="使用CLI命令",
            description=(
                "SEVA提供了丰富的CLI命令，无需写代码即可操作:\n"
                "  seva run --steps 100  运行Agent\n"
                "  seva status           查看状态\n"
                "  seva version          查看版本"
            ),
            example="seva run --steps 50 --preset easy",
            tip="使用 seva --help 查看所有命令",
        ),
        TutorialStep(
            title="恭喜完成入门",
            description=(
                "您已经掌握了SEVA的基本使用方法!\n"
                "接下来可以尝试进阶教程学习更多功能。"
            ),
            tip="运行 seva tutorial intermediate 开始进阶教程",
        ),
    ]

    return Tutorial(
        tutorial_id="beginner",
        title="SEVA入门教程",
        description="学习SEVA的基本概念和快速启动方法",
        steps=steps,
        difficulty="入门",
    )


def _create_intermediate_tutorial() -> Tutorial:
    """创建进阶教程"""
    steps = [
        TutorialStep(
            title="配置系统",
            description=(
                "SEVA支持多种预设配置:\n"
                "  easy      - 简单模式，适合初学者\n"
                "  standard  - 标准模式，平衡配置\n"
                "  research  - 研究模式，详细日志\n"
                "  production- 生产模式，高性能"
            ),
            example="seva run --preset standard --steps 100",
            tip="不同预设适合不同场景",
        ),
        TutorialStep(
            title="配置文件管理",
            description=(
                "可以将配置保存到文件，方便复用:\n"
                "支持YAML和JSON两种格式。"
            ),
            example=(
                "from seva import save_config, load_config\n"
                "save_config(agent.config, 'my.yaml')\n"
                "config = load_config('my.yaml')"
            ),
            tip="配置文件可以版本控制，方便团队协作",
        ),
        TutorialStep(
            title="诊断系统",
            description=(
                "使用诊断功能检查系统健康状态:\n"
                "它会检查配置、环境、模块、性能等。"
            ),
            example="seva diagnose --steps 50",
            tip="定期诊断有助于发现问题",
        ),
        TutorialStep(
            title="生成报告",
            description=(
                "SEVA支持多种报告格式:\n"
                "HTML(含图表)、JSON、CSV、Markdown、TXT"
            ),
            example="seva report --format html --steps 100",
            tip="HTML报告包含SVG图表，可视化效果好",
        ),
        TutorialStep(
            title="会话持久化",
            description=(
                "保存和恢复Agent状态:\n"
                "支持检查点和自动保存。"
            ),
            example=(
                "from seva import save_session, load_session\n"
                "save_session(agent, 'session.json')\n"
                "agent2 = load_session('session.json')"
            ),
            tip="长时间运行的任务建议定期保存检查点",
        ),
    ]

    return Tutorial(
        tutorial_id="intermediate",
        title="SEVA进阶教程",
        description="学习配置管理、诊断、报告和持久化",
        steps=steps,
        difficulty="进阶",
    )


def _create_advanced_tutorial() -> Tutorial:
    """创建高级教程"""
    steps = [
        TutorialStep(
            title="Web Dashboard",
            description=(
                "启动Web可视化界面:\n"
                "提供实时监控、API接口和交互控制。"
            ),
            example="seva web --port 8080",
            tip="浏览器访问 http://localhost:8080",
        ),
        TutorialStep(
            title="交互式Shell",
            description=(
                "SEVA提供交互式Shell:\n"
                "支持15个命令，无需记忆参数。"
            ),
            example="seva shell",
            tip="在Shell中输入help查看所有命令",
        ),
        TutorialStep(
            title="预设场景",
            description=(
                "使用预设场景进行标准化测试:\n"
                "支持快速验证、标准运行、压力测试等。"
            ),
            example=(
                "from seva import list_scenarios, run_scenario\n"
                "print(list_scenarios())\n"
                "run_scenario('quick_validation')"
            ),
            tip="场景可以用于性能对比和回归测试",
        ),
        TutorialStep(
            title="可视化图表",
            description=(
                "生成SVG图表进行数据分析:\n"
                "趋势图、分布图、柱状图等。"
            ),
            example="seva charts --steps 100",
            tip="图表是纯SVG，无需外部依赖",
        ),
        TutorialStep(
            title="智能补全",
            description=(
                "安装Shell补全提升效率:\n"
                "支持Bash和Zsh。"
            ),
            example="seva completion --shell bash",
            tip="拼写错误时会有智能建议",
        ),
    ]

    return Tutorial(
        tutorial_id="advanced",
        title="SEVA高级教程",
        description="学习Web界面、Shell、场景和可视化",
        steps=steps,
        difficulty="高级",
    )


# 注册内置教程
def _register_builtin_tutorials():
    """注册内置教程"""
    Tutorial.register(_create_beginner_tutorial())
    Tutorial.register(_create_intermediate_tutorial())
    Tutorial.register(_create_advanced_tutorial())


# 模块加载时自动注册
_register_builtin_tutorials()


# =====================================================================
# 便捷函数
# =====================================================================

def run_tutorial(tutorial_id: str, interactive: bool = True,
                 resume: bool = True) -> bool:
    """
    运行教程

    Args:
        tutorial_id: 教程ID
        interactive: 是否交互式
        resume: 是否从上次进度继续

    Returns:
        是否完成

    Example:
        >>> from seva import run_tutorial
        >>> run_tutorial("beginner")
    """
    tutorial = Tutorial.get_tutorial(tutorial_id)
    if tutorial is None:
        print(f"未找到教程: {tutorial_id}")
        print(f"可用教程: {[t['id'] for t in list_tutorials()]}")
        return False

    start_step = 0
    if resume:
        progress = TutorialProgress.get_progress(tutorial_id)
        start_step = progress.get("current_step", 0)
        if start_step > 0 and not progress.get("completed", False):
            print(f"\n检测到未完成的进度（步骤 {start_step}），将继续学习。")

    return tutorial.run(interactive=interactive, start_step=start_step)


def list_tutorials() -> List[Dict]:
    """
    列出所有可用教程

    Returns:
        教程信息列表

    Example:
        >>> from seva import list_tutorials
        >>> for t in list_tutorials():
        ...     print(t['id'], t['title'])
    """
    return Tutorial.list_tutorials()


def get_tutorial_info(tutorial_id: str) -> Optional[Dict]:
    """
    获取教程信息

    Args:
        tutorial_id: 教程ID

    Returns:
        教程信息字典
    """
    tutorials = list_tutorials()
    for t in tutorials:
        if t["id"] == tutorial_id:
            return t
    return None


def reset_tutorial_progress(tutorial_id: Optional[str] = None) -> None:
    """
    重置教程进度

    Args:
        tutorial_id: 教程ID（None则重置所有）
    """
    TutorialProgress.reset(tutorial_id)


def get_tutorial_progress(tutorial_id: str) -> Dict:
    """
    获取教程进度

    Args:
        tutorial_id: 教程ID

    Returns:
        进度字典
    """
    return TutorialProgress.get_progress(tutorial_id)
