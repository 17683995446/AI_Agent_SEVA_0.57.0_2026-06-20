"""
阶段63测试：智能决策与自主规划

覆盖：
    - 枚举: TaskStatus/PlanStatus/PlanStrategy/RiskLevel/ActionType
    - 数据类: Goal/Task/Plan/TaskResult/PlanResult/ReflectionResult/Issue/Action
    - Planner (目标分解/计划创建/回退分解/难度评估)
    - PlanExecutor (计划执行/任务执行/失败处理/重规划)
    - Reflector (反思/问题识别/回退反思/记忆存储)
    - DecisionMaker (决策/风险评估)
    - AutonomousAgent (完整流程/重规划/历史)
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import json
import time
from unittest.mock import patch, MagicMock

from seva.planner import (
    Goal, Task, TaskStatus, Plan, PlanStatus, PlanStrategy,
    TaskResult, PlanResult, ReflectionResult, Issue, RiskLevel,
    ActionType, Action,
    Planner, PlanExecutor, Reflector, DecisionMaker, AutonomousAgent,
    get_planner, get_plan_executor, get_reflector, get_decision_maker,
    get_autonomous_agent, reset_planner, reset_reflector,
    reset_decision_maker, reset_autonomous_agent,
)
from seva.llm import LLMError, reset_llm_client
from seva.tools import reset_tools


# ============================================================================
# 枚举测试
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_task_status(self):
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")

    def test02_plan_status(self):
        self.assertEqual(PlanStatus.DRAFT.value, "draft")
        self.assertEqual(PlanStatus.ACTIVE.value, "active")
        self.assertEqual(PlanStatus.COMPLETED.value, "completed")

    def test03_plan_strategy(self):
        self.assertEqual(PlanStrategy.SEQUENTIAL.value, "sequential")
        self.assertEqual(PlanStrategy.PARALLEL.value, "parallel")
        self.assertEqual(PlanStrategy.ADAPTIVE.value, "adaptive")

    def test04_risk_level(self):
        self.assertEqual(RiskLevel.LOW.value, "low")
        self.assertEqual(RiskLevel.HIGH.value, "high")
        self.assertEqual(RiskLevel.CRITICAL.value, "critical")

    def test05_action_type(self):
        self.assertEqual(ActionType.EXECUTE_TASK.value, "execute_task")
        self.assertEqual(ActionType.REPLAN.value, "replan")
        self.assertEqual(ActionType.COMPLETE_PLAN.value, "complete_plan")


# ============================================================================
# Goal测试
# ============================================================================

class TestGoal(unittest.TestCase):
    """测试Goal"""

    def test01_creation(self):
        g = Goal(description="完成报告", success_criteria="报告已提交")
        self.assertEqual(g.description, "完成报告")
        self.assertEqual(g.success_criteria, "报告已提交")
        self.assertEqual(g.priority, 5)
        self.assertIsNotNone(g.id)

    def test02_to_dict(self):
        g = Goal(description="test", priority=3)
        d = g.to_dict()
        self.assertEqual(d["description"], "test")
        self.assertEqual(d["priority"], 3)

    def test03_from_dict(self):
        d = {"description": "from dict", "priority": 8}
        g = Goal.from_dict(d)
        self.assertEqual(g.description, "from dict")
        self.assertEqual(g.priority, 8)

    def test04_roundtrip(self):
        g1 = Goal(description="roundtrip", success_criteria="ok")
        d = g1.to_dict()
        g2 = Goal.from_dict(d)
        self.assertEqual(g2.description, "roundtrip")
        self.assertEqual(g2.success_criteria, "ok")


# ============================================================================
# Task测试
# ============================================================================

class TestTask(unittest.TestCase):
    """测试Task"""

    def test01_creation(self):
        t = Task(description="计算", goal_id="g1")
        self.assertEqual(t.description, "计算")
        self.assertEqual(t.goal_id, "g1")
        self.assertEqual(t.status, TaskStatus.PENDING)
        self.assertEqual(t.max_retries, 2)

    def test02_to_dict(self):
        t = Task(description="test", difficulty=0.8)
        d = t.to_dict()
        self.assertEqual(d["description"], "test")
        self.assertEqual(d["difficulty"], 0.8)
        self.assertEqual(d["status"], "pending")

    def test03_from_dict(self):
        d = {"description": "from dict", "status": "completed"}
        t = Task.from_dict(d)
        self.assertEqual(t.description, "from dict")
        self.assertEqual(t.status, TaskStatus.COMPLETED)

    def test04_is_terminal(self):
        t = Task(status=TaskStatus.PENDING)
        self.assertFalse(t.is_terminal)
        t.status = TaskStatus.COMPLETED
        self.assertTrue(t.is_terminal)
        t.status = TaskStatus.FAILED
        self.assertTrue(t.is_terminal)
        t.status = TaskStatus.SKIPPED
        self.assertTrue(t.is_terminal)

    def test05_can_retry(self):
        t = Task(status=TaskStatus.FAILED, max_retries=2, retry_count=0)
        self.assertTrue(t.can_retry)
        t.retry_count = 2
        self.assertFalse(t.can_retry)
        t.status = TaskStatus.COMPLETED
        self.assertFalse(t.can_retry)


# ============================================================================
# Plan测试
# ============================================================================

class TestPlan(unittest.TestCase):
    """测试Plan"""

    def setUp(self):
        self.goal = Goal(description="测试目标")
        self.task1 = Task(id="t1", description="任务1", goal_id=self.goal.id)
        self.task2 = Task(id="t2", description="任务2", goal_id=self.goal.id,
                         dependencies=["t1"])
        self.plan = Plan(goal=self.goal, tasks=[self.task1, self.task2])

    def test01_creation(self):
        self.assertEqual(self.plan.goal.description, "测试目标")
        self.assertEqual(len(self.plan.tasks), 2)
        self.assertEqual(self.plan.status, PlanStatus.DRAFT)
        self.assertEqual(self.plan.version, 1)

    def test02_to_dict(self):
        d = self.plan.to_dict()
        self.assertEqual(d["goal"]["description"], "测试目标")
        self.assertEqual(len(d["tasks"]), 2)
        self.assertEqual(d["status"], "draft")

    def test03_progress(self):
        self.assertEqual(self.plan.progress, 0.0)
        self.task1.status = TaskStatus.COMPLETED
        self.assertEqual(self.plan.progress, 0.5)
        self.task2.status = TaskStatus.COMPLETED
        self.assertEqual(self.plan.progress, 1.0)

    def test04_get_task(self):
        t = self.plan.get_task("t1")
        self.assertIsNotNone(t)
        self.assertEqual(t.description, "任务1")
        self.assertIsNone(self.plan.get_task("nonexistent"))

    def test05_get_ready_tasks(self):
        # 初始: t1无依赖,可执行; t2依赖t1,不可执行
        ready = self.plan.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "t1")

        # t1完成后, t2可执行
        self.task1.status = TaskStatus.COMPLETED
        ready = self.plan.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "t2")

    def test06_get_ready_tasks_with_failed_dep(self):
        """依赖失败时任务应被阻塞"""
        self.task1.status = TaskStatus.FAILED
        ready = self.plan.get_ready_tasks()
        # t2应该被标记为blocked
        self.assertEqual(self.task2.status, TaskStatus.BLOCKED)

    def test07_all_tasks_terminal(self):
        self.assertFalse(self.plan.all_tasks_terminal())
        self.task1.status = TaskStatus.COMPLETED
        self.task2.status = TaskStatus.COMPLETED
        self.assertTrue(self.plan.all_tasks_terminal())

    def test08_is_terminal(self):
        self.assertFalse(self.plan.is_terminal)
        self.plan.status = PlanStatus.COMPLETED
        self.assertTrue(self.plan.is_terminal)
        self.plan.status = PlanStatus.FAILED
        self.assertTrue(self.plan.is_terminal)

    def test09_get_failed_tasks(self):
        self.task1.status = TaskStatus.FAILED
        failed = self.plan.get_failed_tasks()
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].id, "t1")


# ============================================================================
# TaskResult/PlanResult测试
# ============================================================================

class TestResults(unittest.TestCase):
    """测试结果类"""

    def test01_task_result(self):
        r = TaskResult(task_id="t1", status=TaskStatus.COMPLETED, output="done")
        self.assertTrue(r.success)
        self.assertEqual(r.output, "done")

    def test02_task_result_failed(self):
        r = TaskResult(task_id="t1", status=TaskStatus.FAILED, error="bad")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "bad")

    def test03_task_result_to_dict(self):
        r = TaskResult(task_id="t1", status=TaskStatus.COMPLETED)
        d = r.to_dict()
        self.assertEqual(d["task_id"], "t1")
        self.assertEqual(d["status"], "completed")

    def test04_plan_result(self):
        r = PlanResult(plan_id="p1", status=PlanStatus.COMPLETED)
        self.assertTrue(r.success)

    def test05_plan_result_to_dict(self):
        r = PlanResult(plan_id="p1", status=PlanStatus.FAILED,
                      final_output="failed")
        d = r.to_dict()
        self.assertEqual(d["plan_id"], "p1")
        self.assertEqual(d["status"], "failed")
        self.assertEqual(d["final_output"], "failed")


# ============================================================================
# ReflectionResult/Issue/Action测试
# ============================================================================

class TestReflectionAndAction(unittest.TestCase):
    """测试反思和动作"""

    def test01_issue(self):
        i = Issue(severity=RiskLevel.HIGH, description="问题",
                 suggestion="建议")
        self.assertEqual(i.severity, RiskLevel.HIGH)
        self.assertEqual(i.description, "问题")

    def test02_reflection_result(self):
        r = ReflectionResult(success=True, summary="成功")
        self.assertTrue(r.success)
        self.assertEqual(r.summary, "成功")

    def test03_reflection_to_dict(self):
        r = ReflectionResult(
            success=False,
            summary="失败",
            issues=[Issue(severity=RiskLevel.MEDIUM, description="问题1")],
            should_replan=True,
        )
        d = r.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(len(d["issues"]), 1)
        self.assertTrue(d["should_replan"])

    def test04_action(self):
        a = Action(action_type=ActionType.EXECUTE_TASK,
                  target_task_id="t1", reason="执行")
        self.assertEqual(a.action_type, ActionType.EXECUTE_TASK)
        self.assertEqual(a.target_task_id, "t1")

    def test05_action_to_dict(self):
        a = Action(action_type=ActionType.REPLAN, reason="重规划",
                  risk=RiskLevel.HIGH)
        d = a.to_dict()
        self.assertEqual(d["action_type"], "replan")
        self.assertEqual(d["risk"], "high")


# ============================================================================
# Planner测试
# ============================================================================

class TestPlanner(unittest.TestCase):
    """测试Planner"""

    def setUp(self):
        reset_tools()

    def tearDown(self):
        reset_tools()

    def test01_creation(self):
        p = Planner()
        self.assertIsNotNone(p)
        self.assertEqual(p.stats["plans_created"], 0)

    def test02_fallback_decompose(self):
        """测试回退分解"""
        p = Planner()
        goal = Goal(description="简单目标")
        tasks = p._fallback_decompose(goal)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].description, "简单目标")
        self.assertEqual(tasks[0].goal_id, goal.id)

    def test03_decompose_with_mock_llm(self):
        """Mock LLM分解目标"""
        p = Planner()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "tasks": [
        {"description": "步骤1", "expected_output": "输出1", "difficulty": 0.3, "dependencies": []},
        {"description": "步骤2", "expected_output": "输出2", "difficulty": 0.5, "dependencies": [0]}
    ]
}
```'''
        mock_client.chat.return_value = response
        p._client = mock_client

        goal = Goal(description="复杂目标", success_criteria="完成")
        tasks = p.decompose_goal(goal)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].description, "步骤1")
        self.assertEqual(tasks[1].description, "步骤2")
        # 依赖转换: 序号0 -> tasks[0].id
        self.assertEqual(tasks[1].dependencies, [tasks[0].id])

    def test04_decompose_llm_error(self):
        """LLM错误时回退"""
        p = Planner()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("mock error")
        p._client = mock_client

        goal = Goal(description="目标")
        tasks = p.decompose_goal(goal)
        # 应该回退到单任务
        self.assertEqual(len(tasks), 1)
        self.assertEqual(p.stats["decomposition_failures"], 1)

    def test05_decompose_invalid_json(self):
        """LLM返回无效JSON时回退"""
        p = Planner()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = "这不是JSON"
        mock_client.chat.return_value = response
        p._client = mock_client

        goal = Goal(description="目标")
        tasks = p.decompose_goal(goal)
        self.assertEqual(len(tasks), 1)

    def test06_create_plan(self):
        """测试创建计划"""
        p = Planner()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''{"tasks": [{"description": "任务1", "difficulty": 0.4}]}'''
        mock_client.chat.return_value = response
        p._client = mock_client

        goal = Goal(description="目标")
        plan = p.create_plan(goal)

        self.assertEqual(plan.status, PlanStatus.ACTIVE)
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(p.stats["plans_created"], 1)

    def test07_estimate_difficulty(self):
        """测试难度评估"""
        p = Planner()
        t = Task(description="短", tools_needed=[])
        d1 = p.estimate_difficulty(t)
        self.assertGreater(d1, 0)

        t2 = Task(description="这是一个很长的任务描述" * 20,
                  tools_needed=["tool1", "tool2", "tool3", "tool4"])
        d2 = p.estimate_difficulty(t2)
        self.assertGreater(d2, d1)

    def test08_get_stats(self):
        p = Planner()
        stats = p.get_stats()
        self.assertIn("plans_created", stats)
        self.assertIn("tasks_decomposed", stats)


# ============================================================================
# PlanExecutor测试
# ============================================================================

class TestPlanExecutor(unittest.TestCase):
    """测试PlanExecutor"""

    def setUp(self):
        reset_tools()
        self.goal = Goal(description="测试目标")
        self.task = Task(id="t1", description="计算2+2",
                        goal_id=self.goal.id, tools_needed=["calculator"])
        self.plan = Plan(goal=self.goal, tasks=[self.task],
                        status=PlanStatus.ACTIVE)

    def tearDown(self):
        reset_tools()

    def test01_creation(self):
        executor = PlanExecutor()
        self.assertIsNotNone(executor)
        self.assertEqual(executor.stats["plans_executed"], 0)

    def test02_execute_task_success(self):
        """测试任务执行成功"""
        executor = PlanExecutor()

        # Mock tool_loop
        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "2+2=4",
            "tool_calls": [{"tool_name": "calculator", "status": "success"}],
            "iterations": 1,
            "total_tokens": 50,
        }
        executor._tool_loop = mock_loop

        result = executor.execute_task(self.task, self.plan)
        self.assertTrue(result.success)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertEqual(self.task.actual_output, "2+2=4")
        self.assertEqual(result.tokens_used, 50)

    def test03_execute_task_failure(self):
        """测试任务执行失败"""
        executor = PlanExecutor()

        mock_loop = MagicMock()
        mock_loop.run.side_effect = Exception("执行错误")
        executor._tool_loop = mock_loop

        result = executor.execute_task(self.task, self.plan)
        self.assertFalse(result.success)
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("执行错误", result.error)

    def test04_execute_plan_success(self):
        """测试计划执行成功"""
        executor = PlanExecutor()

        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "结果",
            "tool_calls": [],
            "iterations": 1,
            "total_tokens": 30,
        }
        executor._tool_loop = mock_loop

        result = executor.execute_plan(self.plan, max_iterations=3)
        self.assertTrue(result.success)
        self.assertEqual(self.plan.status, PlanStatus.COMPLETED)
        self.assertEqual(len(result.task_results), 1)

    def test05_execute_plan_with_retry(self):
        """测试计划执行带重试"""
        # 创建一个会失败一次然后成功的任务
        task = Task(id="t1", description="任务", goal_id=self.goal.id,
                   max_retries=2)
        plan = Plan(goal=self.goal, tasks=[task], status=PlanStatus.ACTIVE)

        executor = PlanExecutor()

        mock_loop = MagicMock()
        # 第一次失败,第二次成功
        mock_loop.run.side_effect = [
            {"answer": "", "tool_calls": [], "iterations": 0, "total_tokens": 10},
            {"answer": "成功", "tool_calls": [], "iterations": 1, "total_tokens": 20},
        ]
        executor._tool_loop = mock_loop

        result = executor.execute_plan(plan, max_iterations=5)
        self.assertTrue(result.success)
        self.assertEqual(task.retry_count, 1)

    def test06_should_replan(self):
        """测试重规划判断"""
        executor = PlanExecutor()
        # 创建多个任务,大部分失败
        goal = Goal(description="目标")
        tasks = [
            Task(id="t1", description="任务1", goal_id=goal.id),
            Task(id="t2", description="任务2", goal_id=goal.id),
            Task(id="t3", description="任务3", goal_id=goal.id, dependencies=["t1"]),
        ]
        plan = Plan(goal=goal, tasks=tasks, status=PlanStatus.ACTIVE)

        # t1失败, t3依赖t1
        t1 = plan.get_task("t1")
        self.assertTrue(executor._should_replan(plan, t1))

    def test07_replan(self):
        """测试重规划"""
        planner = Planner()
        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''{"tasks": [{"description": "新任务"}]}'''
        mock_client.chat.return_value = response
        planner._client = mock_client

        executor = PlanExecutor(planner=planner)

        goal = Goal(description="目标")
        task = Task(id="t1", description="失败任务", goal_id=goal.id,
                   status=TaskStatus.FAILED)
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.ACTIVE)

        result = executor.replan(plan, task)
        self.assertTrue(result)
        self.assertEqual(task.status, TaskStatus.SKIPPED)
        self.assertEqual(plan.version, 2)
        # 应该新增了任务
        self.assertGreater(len(plan.tasks), 1)

    def test08_get_stats(self):
        executor = PlanExecutor()
        stats = executor.get_stats()
        self.assertIn("plans_executed", stats)
        self.assertIn("tasks_executed", stats)


# ============================================================================
# Reflector测试
# ============================================================================

class TestReflector(unittest.TestCase):
    """测试Reflector"""

    def setUp(self):
        reset_tools()
        reset_planner()

    def tearDown(self):
        reset_tools()
        reset_planner()

    def test01_creation(self):
        r = Reflector()
        self.assertIsNotNone(r)
        self.assertEqual(r.stats["reflections_made"], 0)

    def test02_reflect_success(self):
        """测试成功反思"""
        r = Reflector()
        goal = Goal(description="目标")
        task = Task(id="t1", description="任务", goal_id=goal.id,
                   status=TaskStatus.COMPLETED, actual_output="完成")
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.COMPLETED)
        result = PlanResult(plan_id=plan.id, status=PlanStatus.COMPLETED)

        reflection = r.reflect(plan, result)
        self.assertTrue(reflection.success)
        self.assertEqual(r.stats["reflections_made"], 1)

    def test03_reflect_with_llm(self):
        """测试LLM反思"""
        r = Reflector()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "success": false,
    "summary": "部分失败",
    "issues": [{"severity": "high", "description": "任务失败", "suggestion": "重试"}],
    "improvements": ["增加重试"],
    "lessons_learned": ["需要验证"],
    "should_replan": true,
    "replan_reason": "关键任务失败"
}
```'''
        mock_client.chat.return_value = response
        r._client = mock_client

        goal = Goal(description="目标", success_criteria="完成")
        task = Task(id="t1", description="任务", goal_id=goal.id,
                   status=TaskStatus.FAILED, error="失败")
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.FAILED)
        result = PlanResult(plan_id=plan.id, status=PlanStatus.FAILED)

        reflection = r.reflect(plan, result)
        self.assertFalse(reflection.success)
        self.assertEqual(len(reflection.issues), 1)
        self.assertEqual(reflection.issues[0].severity, RiskLevel.HIGH)
        self.assertTrue(reflection.should_replan)

    def test04_reflect_llm_error(self):
        """LLM错误时回退反思"""
        r = Reflector()

        mock_client = MagicMock()
        mock_client.chat.side_effect = LLMError("mock error")
        r._client = mock_client

        goal = Goal(description="目标")
        task = Task(id="t1", description="任务", goal_id=goal.id,
                   status=TaskStatus.FAILED, error="失败")
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.FAILED)
        result = PlanResult(plan_id=plan.id, status=PlanStatus.FAILED)

        reflection = r.reflect(plan, result)
        # 应该回退到规则反思
        self.assertFalse(reflection.success)
        self.assertTrue(len(reflection.issues) > 0)

    def test05_get_stats(self):
        r = Reflector()
        stats = r.get_stats()
        self.assertIn("reflections_made", stats)


# ============================================================================
# DecisionMaker测试
# ============================================================================

class TestDecisionMaker(unittest.TestCase):
    """测试DecisionMaker"""

    def test01_creation(self):
        dm = DecisionMaker()
        self.assertIsNotNone(dm)
        self.assertEqual(dm.stats["decisions_made"], 0)

    def test02_decide_execute(self):
        """决策执行任务"""
        dm = DecisionMaker()
        goal = Goal(description="目标")
        task = Task(id="t1", description="任务", goal_id=goal.id, difficulty=0.3)
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.ACTIVE)

        action = dm.decide_next_action(plan)
        self.assertEqual(action.action_type, ActionType.EXECUTE_TASK)
        self.assertEqual(action.target_task_id, "t1")
        self.assertEqual(action.risk, RiskLevel.LOW)

    def test03_decide_retry(self):
        """决策重试"""
        dm = DecisionMaker()
        goal = Goal(description="目标")
        task = Task(id="t1", description="任务", goal_id=goal.id,
                   status=TaskStatus.FAILED, max_retries=2, retry_count=0)
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.ACTIVE)
        last_result = TaskResult(task_id="t1", status=TaskStatus.FAILED)

        action = dm.decide_next_action(plan, last_result)
        self.assertEqual(action.action_type, ActionType.RETRY_TASK)

    def test04_decide_complete(self):
        """决策完成"""
        dm = DecisionMaker()
        goal = Goal(description="目标")
        task = Task(id="t1", description="任务", goal_id=goal.id,
                   status=TaskStatus.COMPLETED)
        plan = Plan(goal=goal, tasks=[task], status=PlanStatus.ACTIVE)

        action = dm.decide_next_action(plan)
        self.assertEqual(action.action_type, ActionType.COMPLETE_PLAN)

    def test05_decide_replan_many_failures(self):
        """失败过多决策重规划"""
        dm = DecisionMaker()
        goal = Goal(description="目标")
        tasks = [
            Task(id="t1", description="任务1", goal_id=goal.id, status=TaskStatus.FAILED),
            Task(id="t2", description="任务2", goal_id=goal.id, status=TaskStatus.FAILED),
            Task(id="t3", description="任务3", goal_id=goal.id, status=TaskStatus.FAILED),
        ]
        plan = Plan(goal=goal, tasks=tasks, status=PlanStatus.ACTIVE)
        last_result = TaskResult(task_id="t3", status=TaskStatus.FAILED)

        action = dm.decide_next_action(plan, last_result)
        self.assertEqual(action.action_type, ActionType.REPLAN)

    def test06_assess_risk(self):
        """测试风险评估"""
        dm = DecisionMaker()
        low_task = Task(difficulty=0.2)
        self.assertEqual(dm._assess_risk(low_task), RiskLevel.LOW)
        med_task = Task(difficulty=0.6)
        self.assertEqual(dm._assess_risk(med_task), RiskLevel.MEDIUM)
        high_task = Task(difficulty=0.9)
        self.assertEqual(dm._assess_risk(high_task), RiskLevel.HIGH)

    def test07_get_stats(self):
        dm = DecisionMaker()
        stats = dm.get_stats()
        self.assertIn("decisions_made", stats)


# ============================================================================
# AutonomousAgent测试
# ============================================================================

class TestAutonomousAgent(unittest.TestCase):
    """测试AutonomousAgent"""

    def setUp(self):
        reset_tools()
        reset_planner()
        reset_reflector()
        reset_decision_maker()

    def tearDown(self):
        reset_tools()
        reset_planner()
        reset_reflector()
        reset_decision_maker()

    def test01_creation(self):
        agent = AutonomousAgent()
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.planner)
        self.assertIsNotNone(agent.executor)
        self.assertIsNotNone(agent.reflector)
        self.assertIsNotNone(agent.decision_maker)

    def test02_run_with_string_goal(self):
        """测试字符串目标"""
        agent = AutonomousAgent()

        # Mock planner
        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''{"tasks": [{"description": "任务1", "difficulty": 0.3}]}'''
        mock_client.chat.return_value = response
        agent.planner._client = mock_client

        # Mock executor's tool_loop
        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "完成",
            "tool_calls": [],
            "iterations": 1,
            "total_tokens": 20,
        }
        agent.executor._tool_loop = mock_loop

        result = agent.run("简单目标", max_replans=0, max_iterations=3)
        self.assertTrue(result.success)
        self.assertEqual(len(agent.get_history()), 1)

    def test03_run_with_goal_object(self):
        """测试Goal对象"""
        agent = AutonomousAgent()

        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''{"tasks": [{"description": "任务1"}]}'''
        mock_client.chat.return_value = response
        agent.planner._client = mock_client

        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "完成",
            "tool_calls": [],
            "iterations": 1,
            "total_tokens": 20,
        }
        agent.executor._tool_loop = mock_loop

        goal = Goal(description="目标对象", success_criteria="完成")
        result = agent.run(goal, max_replans=0, max_iterations=3)
        self.assertTrue(result.success)

    def test04_get_report(self):
        """测试报告"""
        agent = AutonomousAgent()
        report = agent.get_report()
        self.assertIn("planner_stats", report)
        self.assertIn("executor_stats", report)
        self.assertIn("reflector_stats", report)
        self.assertIn("decision_stats", report)
        self.assertEqual(report["history_count"], 0)

    def test05_get_history(self):
        """测试历史"""
        agent = AutonomousAgent()
        history = agent.get_history()
        self.assertEqual(len(history), 0)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingletons(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_planner()
        reset_reflector()
        reset_decision_maker()

    def tearDown(self):
        reset_planner()
        reset_reflector()
        reset_decision_maker()

    def test01_planner_singleton(self):
        p1 = get_planner()
        p2 = get_planner()
        self.assertIs(p1, p2)

    def test02_executor_singleton(self):
        e1 = get_plan_executor()
        e2 = get_plan_executor()
        self.assertIs(e1, e2)

    def test03_reflector_singleton(self):
        r1 = get_reflector()
        r2 = get_reflector()
        self.assertIs(r1, r2)

    def test04_decision_maker_singleton(self):
        d1 = get_decision_maker()
        d2 = get_decision_maker()
        self.assertIs(d1, d2)

    def test05_agent_singleton(self):
        a1 = get_autonomous_agent()
        a2 = get_autonomous_agent()
        self.assertIs(a1, a2)

    def test06_reset_planner(self):
        p1 = get_planner()
        reset_planner()
        p2 = get_planner()
        self.assertIsNot(p1, p2)

    def test07_reset_agent(self):
        a1 = get_autonomous_agent()
        reset_autonomous_agent()
        a2 = get_autonomous_agent()
        self.assertIsNot(a1, a2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_planner()
        reset_reflector()
        reset_decision_maker()
        reset_tools()
        reset_llm_client()

    def tearDown(self):
        reset_planner()
        reset_reflector()
        reset_decision_maker()
        reset_tools()
        reset_llm_client()

    def test01_full_workflow_mock(self):
        """完整工作流(Mock)"""
        agent = AutonomousAgent()

        # Mock planner LLM
        mock_client = MagicMock()
        response = MagicMock()
        response.content = '''```json
{
    "tasks": [
        {"description": "计算步骤", "expected_output": "数字结果", "difficulty": 0.3, "tools_needed": ["calculator"], "dependencies": []}
    ]
}
```'''
        mock_client.chat.return_value = response
        agent.planner._client = mock_client
        agent.reflector._client = mock_client

        # Mock executor
        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "answer": "计算结果是42",
            "tool_calls": [{"tool_name": "calculator", "status": "success"}],
            "iterations": 1,
            "total_tokens": 100,
        }
        agent.executor._tool_loop = mock_loop

        result = agent.run("计算宇宙的答案", max_replans=0, max_iterations=3)

        self.assertTrue(result.success)
        self.assertEqual(result.status, PlanStatus.COMPLETED)
        self.assertGreater(result.total_tokens, 0)
        self.assertEqual(len(agent.get_history()), 1)

    def test02_real_llm_planning(self):
        """真实LLM规划(需要API key)"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        agent = get_autonomous_agent()

        # 使用真实LLM
        result = agent.run(
            "计算 sqrt(256) 的值并解释结果",
            max_replans=0,
            max_iterations=5,
        )

        # 验证有执行结果
        self.assertIsNotNone(result)
        self.assertIn(result.status, [PlanStatus.COMPLETED, PlanStatus.FAILED])
        # 应该有任务结果
        self.assertGreater(len(result.task_results), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
