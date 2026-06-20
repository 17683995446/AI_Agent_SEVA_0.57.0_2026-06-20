"""
预设场景与开箱即用模块（阶段34）
对应第一性原理：易用 = 开箱即用，无需配置即可运行常见场景

提供：
- Scenario: 场景数据类（封装完整运行配置）
- PresetScenarios: 预设场景集合（8个开箱即用场景）
- ScenarioRunner: 场景运行器（支持单/多/全量运行+对比）
- run_scenario(): 一键运行场景
- list_scenarios(): 列出所有场景
- compare_scenarios(): 对比多个场景
- quick_demo(): 快速演示
- run_all_scenarios(): 运行所有场景

使用示例:
    from seva import run_scenario, list_scenarios
    print(list_scenarios())  # 查看可用场景
    result = run_scenario("quick_validation")  # 一键运行
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .api import SEVAFactory, run, quickstart, PresetConfigs
from .diagnostics import Diagnoser


@dataclass
class Scenario:
    """
    场景定义：封装一个完整的运行配置

    属性:
        name: 场景名称（唯一标识）
        description: 场景描述（用户可读）
        preset: 使用的预设模板
        env_type: 环境类型
        steps: 运行步数
        seed: 随机种子
        overrides: 配置覆盖
        tags: 标签（用于分类）
        expected_min_reward: 预期最小奖励（用于验证通过）
    """
    name: str
    description: str
    preset: str = "easy"
    env_type: str = "mixed"
    steps: int = 100
    seed: int = 42
    overrides: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    expected_min_reward: float = -1000.0

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "preset": self.preset,
            "env_type": self.env_type,
            "steps": self.steps,
            "seed": self.seed,
            "overrides": dict(self.overrides),
            "tags": list(self.tags),
            "expected_min_reward": self.expected_min_reward,
        }


class PresetScenarios:
    """
    预设场景集合：8个开箱即用的常见场景

    场景分类:
        - 验证类: quick_validation
        - 日常类: standard_run
        - 基准类: performance_benchmark
        - 测试类: drift_adaptation, adversarial_robustness
        - 研究类: research_long_run
        - 生产类: production_stability
        - 演示类: evolution_demo
    """

    # 快速验证：10步运行，验证系统可用性（冒烟测试）
    QUICK_VALIDATION = Scenario(
        name="quick_validation",
        description="快速验证：10步运行，验证系统可用性（冒烟测试）",
        preset="easy",
        env_type="mixed",
        steps=10,
        seed=42,
        tags=["quick", "validation", "smoke"],
        expected_min_reward=-100.0,
    )

    # 标准运行：100步运行，日常使用推荐
    STANDARD_RUN = Scenario(
        name="standard_run",
        description="标准运行：100步运行，日常使用推荐",
        preset="standard",
        env_type="mixed",
        steps=100,
        seed=42,
        tags=["standard", "daily"],
        expected_min_reward=-100.0,
    )

    # 性能基准：200步运行，测量性能指标
    PERFORMANCE_BENCHMARK = Scenario(
        name="performance_benchmark",
        description="性能基准：200步运行，测量性能指标",
        preset="standard",
        env_type="mixed",
        steps=200,
        seed=42,
        tags=["benchmark", "performance"],
        expected_min_reward=-100.0,
    )

    # 漂移适应：测试Agent对环境漂移的适应能力
    DRIFT_ADAPTATION = Scenario(
        name="drift_adaptation",
        description="漂移适应：测试Agent对环境漂移的适应能力",
        preset="standard",
        env_type="drift",
        steps=150,
        seed=42,
        tags=["adaptation", "drift", "test"],
        expected_min_reward=-100.0,
    )

    # 对抗鲁棒性：测试Agent在对抗环境下的表现
    ADVERSARIAL_ROBUSTNESS = Scenario(
        name="adversarial_robustness",
        description="对抗鲁棒性：测试Agent在对抗环境下的表现",
        preset="standard",
        env_type="adversarial",
        steps=100,
        seed=42,
        tags=["robustness", "adversarial", "test"],
        expected_min_reward=-100.0,
    )

    # 研究长运行：500步运行，研究模式
    RESEARCH_LONG_RUN = Scenario(
        name="research_long_run",
        description="研究长运行：300步运行，研究模式",
        preset="research",
        env_type="mixed",
        steps=300,
        seed=42,
        tags=["research", "long"],
        expected_min_reward=-100.0,
    )

    # 生产稳定性：生产模式运行，验证稳定性
    PRODUCTION_STABILITY = Scenario(
        name="production_stability",
        description="生产稳定性：生产模式运行，验证稳定性",
        preset="production",
        env_type="mixed",
        steps=100,
        seed=42,
        tags=["production", "stability"],
        expected_min_reward=-100.0,
    )

    # 进化演示：长运行展示自主进化闭环
    EVOLUTION_DEMO = Scenario(
        name="evolution_demo",
        description="进化演示：长运行展示自主进化闭环",
        preset="easy",
        env_type="mixed",
        steps=250,
        seed=42,
        overrides={"evolution": {"evaluation_interval": 50}},
        tags=["evolution", "demo"],
        expected_min_reward=-100.0,
    )

    @classmethod
    def get(cls, name: str) -> Scenario:
        """获取场景"""
        scenarios = {
            "quick_validation": cls.QUICK_VALIDATION,
            "standard_run": cls.STANDARD_RUN,
            "performance_benchmark": cls.PERFORMANCE_BENCHMARK,
            "drift_adaptation": cls.DRIFT_ADAPTATION,
            "adversarial_robustness": cls.ADVERSARIAL_ROBUSTNESS,
            "research_long_run": cls.RESEARCH_LONG_RUN,
            "production_stability": cls.PRODUCTION_STABILITY,
            "evolution_demo": cls.EVOLUTION_DEMO,
        }
        if name not in scenarios:
            raise ValueError(
                f"未知场景: {name}。可用场景: {list(scenarios.keys())}"
            )
        return scenarios[name]

    @classmethod
    def list_scenarios(cls) -> List[str]:
        """列出所有场景名称"""
        return [
            "quick_validation", "standard_run", "performance_benchmark",
            "drift_adaptation", "adversarial_robustness", "research_long_run",
            "production_stability", "evolution_demo",
        ]

    @classmethod
    def list_with_descriptions(cls) -> List[Dict[str, Any]]:
        """列出所有场景（含描述）"""
        result = []
        for name in cls.list_scenarios():
            s = cls.get(name)
            result.append({
                "name": s.name,
                "description": s.description,
                "preset": s.preset,
                "env_type": s.env_type,
                "steps": s.steps,
                "tags": list(s.tags),
            })
        return result

    @classmethod
    def list_by_tag(cls, tag: str) -> List[str]:
        """按标签筛选场景"""
        result = []
        for name in cls.list_scenarios():
            s = cls.get(name)
            if tag in s.tags:
                result.append(name)
        return result


class ScenarioRunner:
    """
    场景运行器：支持单/多/全量运行+对比

    使用示例:
        runner = ScenarioRunner(verbose=True)
        result = runner.run(PresetScenarios.get("quick_validation"))
        report = runner.get_report()
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}

    def run(self, scenario: Scenario) -> Dict[str, Any]:
        """运行单个场景"""
        if self.verbose:
            print(f"\n[ScenarioRunner] 运行场景: {scenario.name}")
            print(f"  描述: {scenario.description}")
            print(f"  预设: {scenario.preset}, 环境: {scenario.env_type}")
            print(f"  步数: {scenario.steps}")

        start_time = time.time()

        # 创建Agent
        overrides = scenario.overrides if scenario.overrides else None
        agent = SEVAFactory.create(
            preset=scenario.preset,
            env_type=scenario.env_type,
            seed=scenario.seed,
            overrides=overrides,
        )

        # 运行
        stats = run(agent, steps=scenario.steps, verbose=False)

        elapsed = time.time() - start_time

        # 收集结果
        result = {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "preset": scenario.preset,
            "env_type": scenario.env_type,
            "steps": scenario.steps,
            "seed": scenario.seed,
            "tags": list(scenario.tags),
            "steps_completed": stats["steps_completed"],
            "total_reward": stats["total_real_reward"],
            "avg_cycle_time": stats["avg_cycle_time"],
            "avg_prediction_error": stats["avg_prediction_error"],
            "env_done": stats["env_done"],
            "mode": stats["mode"],
            "elapsed_time": elapsed,
            "tasks_completed": stats.get("task_stats", {}).get(
                "total_completed", 0),
            "skills_total": stats.get("skill_stats", {}).get(
                "total_skills", 0),
            "evolution_evaluations": stats.get("evolution_stats", {}).get(
                "total_evaluations", 0),
            "current_difficulty": stats.get("evolution_stats", {}).get(
                "current_difficulty", "EASY"),
            "passed": stats["total_real_reward"] >= scenario.expected_min_reward,
        }

        self.results[scenario.name] = result

        if self.verbose:
            print(f"  结果: reward={result['total_reward']:.4f}, "
                  f"time={elapsed:.2f}s, passed={result['passed']}")

        return result

    def run_multiple(self, scenario_names: List[str]) -> Dict[str, Dict]:
        """运行多个场景"""
        for name in scenario_names:
            scenario = PresetScenarios.get(name)
            self.run(scenario)
        return self.results

    def run_all(self) -> Dict[str, Dict]:
        """运行所有预设场景"""
        return self.run_multiple(PresetScenarios.list_scenarios())

    def compare(self, scenario_names: List[str]) -> Dict[str, Any]:
        """对比多个场景"""
        results = {}
        for name in scenario_names:
            scenario = PresetScenarios.get(name)
            results[name] = self.run(scenario)

        report = {
            "scenarios": list(results.keys()),
            "results": results,
            "summary": self._generate_summary(results),
        }
        return report

    def _generate_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """生成对比摘要"""
        if not results:
            return {}

        rewards = [r["total_reward"] for r in results.values()]
        times = [r["elapsed_time"] for r in results.values()]

        return {
            "total_scenarios": len(results),
            "passed": sum(1 for r in results.values() if r["passed"]),
            "failed": sum(1 for r in results.values() if not r["passed"]),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "fastest_scenario": min(results.items(),
                                    key=lambda x: x[1]["elapsed_time"])[0],
            "best_scenario": max(results.items(),
                                 key=lambda x: x[1]["total_reward"])[0],
        }

    def get_report(self) -> str:
        """生成文本报告"""
        if not self.results:
            return "无运行结果"

        lines = ["=" * 60, "场景运行报告", "=" * 60, ""]

        for name, result in self.results.items():
            lines.append(f"[{name}]")
            lines.append(f"  描述:       {result['description']}")
            lines.append(f"  预设:       {result['preset']}")
            lines.append(f"  环境:       {result['env_type']}")
            lines.append(f"  步数:       "
                         f"{result['steps_completed']}/{result['steps']}")
            lines.append(f"  总奖励:     {result['total_reward']:.4f}")
            lines.append(f"  平均周期:   "
                         f"{result['avg_cycle_time'] * 1000:.2f}ms")
            lines.append(f"  预测误差:   {result['avg_prediction_error']:.4f}")
            lines.append(f"  运行时间:   {result['elapsed_time']:.2f}s")
            lines.append(f"  任务完成:   {result['tasks_completed']}")
            lines.append(f"  技能总数:   {result['skills_total']}")
            lines.append(f"  进化评估:   {result['evolution_evaluations']}")
            lines.append(f"  当前难度:   {result['current_difficulty']}")
            lines.append(f"  通过:       "
                         f"{'是' if result['passed'] else '否'}")
            lines.append("")

        # 摘要
        summary = self._generate_summary(self.results)
        if summary:
            lines.append("-" * 40)
            lines.append("摘要:")
            lines.append(f"  总场景数:   {summary['total_scenarios']}")
            lines.append(f"  通过:       {summary['passed']}")
            lines.append(f"  失败:       {summary['failed']}")
            lines.append(f"  最佳奖励:   {summary['best_reward']:.4f}")
            lines.append(f"  最差奖励:   {summary['worst_reward']:.4f}")
            lines.append(f"  平均奖励:   {summary['avg_reward']:.4f}")
            lines.append(f"  总时间:     {summary['total_time']:.2f}s")
            lines.append(f"  平均时间:   {summary['avg_time']:.2f}s")
            lines.append(f"  最快场景:   {summary['fastest_scenario']}")
            lines.append(f"  最佳场景:   {summary['best_scenario']}")

        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def run_scenario(name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    一键运行预设场景

    参数:
        name: 场景名称（见 list_scenarios()）
        verbose: 是否打印详细输出

    返回:
        运行结果字典

    示例:
        from seva import run_scenario
        result = run_scenario("quick_validation")
        print(result["total_reward"])
    """
    scenario = PresetScenarios.get(name)
    runner = ScenarioRunner(verbose=verbose)
    return runner.run(scenario)


def list_scenarios() -> List[str]:
    """列出所有预设场景名称"""
    return PresetScenarios.list_scenarios()


def list_scenarios_detailed() -> List[Dict[str, Any]]:
    """列出所有预设场景（含详细信息）"""
    return PresetScenarios.list_with_descriptions()


def compare_scenarios(scenario_names: List[str],
                      verbose: bool = False) -> Dict[str, Any]:
    """
    对比多个场景

    参数:
        scenario_names: 场景名称列表
        verbose: 是否打印详细输出

    返回:
        对比报告字典（含结果和摘要）

    示例:
        from seva import compare_scenarios
        report = compare_scenarios(["quick_validation", "standard_run"])
        print(report["summary"])
    """
    runner = ScenarioRunner(verbose=verbose)
    return runner.compare(scenario_names)


def quick_demo(steps: int = 30, verbose: bool = True) -> Dict[str, Any]:
    """
    快速演示：一键运行简短演示
    适合首次使用SEVA的用户

    参数:
        steps: 演示步数（默认30）
        verbose: 是否打印输出

    返回:
        运行统计字典
    """
    if verbose:
        print("=" * 60)
        print("SEVA快速演示")
        print("=" * 60)
        print(f"\n运行 {steps} 步演示...")

    agent, stats = quickstart(steps=steps, preset="easy", verbose=False)

    if verbose:
        print(f"\n演示结果:")
        print(f"  完成步数:     {stats['steps_completed']}")
        print(f"  总奖励:       {stats['total_real_reward']:.4f}")
        print(f"  平均周期:     {stats['avg_cycle_time'] * 1000:.2f}ms")
        print(f"  运行模式:     {stats['mode']}")
        print(f"  任务完成:     "
              f"{stats.get('task_stats', {}).get('total_completed', 0)}")
        print(f"  技能总数:     "
              f"{stats.get('skill_stats', {}).get('total_skills', 0)}")
        print(f"  进化评估:     "
              f"{stats.get('evolution_stats', {}).get('total_evaluations', 0)}")
        print(f"\n演示完成！")

    return stats


def run_all_scenarios(verbose: bool = False) -> Dict[str, Dict]:
    """
    运行所有预设场景

    参数:
        verbose: 是否打印详细输出

    返回:
        所有场景的运行结果字典
    """
    runner = ScenarioRunner(verbose=verbose)
    return runner.run_all()


def get_scenario_report(results: Dict[str, Dict]) -> str:
    """
    从运行结果生成文本报告

    参数:
        results: run_all_scenarios() 或 ScenarioRunner.results 的输出

    返回:
        格式化的文本报告
    """
    runner = ScenarioRunner()
    runner.results = results
    return runner.get_report()
