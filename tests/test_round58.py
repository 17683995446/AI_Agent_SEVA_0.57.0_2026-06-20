"""
阶段68测试：工作流Agent

覆盖：
    - 枚举: StepStatus/WorkflowStatus
    - 数据类: WorkflowStep/Workflow
    - WorkflowExecutor: 执行器（默认处理器/自定义处理器/重试/阻塞/统计）
    - WorkflowBuilder: 流式构建器
    - WorkflowAgentConfig: 配置
    - WorkflowAgent: 工作流Agent
    - 全局单例
    - 集成测试 (真实API)
"""
import unittest
import os
import time
from unittest.mock import patch, MagicMock

from seva.agents import (
    StepStatus, WorkflowStatus,
    WorkflowStep, Workflow, WorkflowExecutor, WorkflowBuilder,
    WorkflowAgentConfig, WorkflowAgent,
    get_workflow_agent, reset_workflow_agent,
)
from seva.agents import AgentRunResult, AgentState
from seva.llm import reset_llm_client


# ============================================================================
# 枚举测试
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_step_status(self):
        self.assertEqual(StepStatus.PENDING.value, "pending")
        self.assertEqual(StepStatus.READY.value, "ready")
        self.assertEqual(StepStatus.RUNNING.value, "running")
        self.assertEqual(StepStatus.COMPLETED.value, "completed")
        self.assertEqual(StepStatus.FAILED.value, "failed")
        self.assertEqual(StepStatus.SKIPPED.value, "skipped")
        self.assertEqual(StepStatus.BLOCKED.value, "blocked")

    def test02_workflow_status(self):
        self.assertEqual(WorkflowStatus.DRAFT.value, "draft")
        self.assertEqual(WorkflowStatus.READY.value, "ready")
        self.assertEqual(WorkflowStatus.RUNNING.value, "running")
        self.assertEqual(WorkflowStatus.PAUSED.value, "paused")
        self.assertEqual(WorkflowStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkflowStatus.FAILED.value, "failed")
        self.assertEqual(WorkflowStatus.CANCELLED.value, "cancelled")

    def test03_step_status_count(self):
        self.assertEqual(len(StepStatus), 7)

    def test04_workflow_status_count(self):
        self.assertEqual(len(WorkflowStatus), 7)


# ============================================================================
# 数据类测试
# ============================================================================

class TestWorkflowStep(unittest.TestCase):
    """测试WorkflowStep"""

    def test01_create_default(self):
        step = WorkflowStep()
        self.assertTrue(step.id.startswith("step_"))
        self.assertEqual(step.status, StepStatus.PENDING)
        self.assertEqual(step.action, None)
        self.assertEqual(step.max_retries, 2)
        self.assertEqual(step.retry_count, 0)
        self.assertEqual(step.timeout, 60.0)

    def test02_create_with_params(self):
        step = WorkflowStep(
            name="load_data",
            description="加载数据",
            action="set_var",
            parameters={"var_name": "x", "var_value": 42},
            input_key="input",
            output_key="output",
            max_retries=5,
            timeout=30.0,
        )
        self.assertEqual(step.name, "load_data")
        self.assertEqual(step.action, "set_var")
        self.assertEqual(step.parameters["var_name"], "x")
        self.assertEqual(step.max_retries, 5)
        self.assertEqual(step.timeout, 30.0)

    def test03_to_dict(self):
        step = WorkflowStep(name="test", action="noop")
        d = step.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["action"], "noop")
        self.assertEqual(d["status"], "pending")
        self.assertIn("id", d)
        self.assertIn("dependencies", d)

    def test04_from_dict(self):
        data = {
            "id": "step_abc123",
            "name": "test",
            "action": "echo",
            "status": "completed",
            "max_retries": 3,
            "timeout": 120.0,
        }
        step = WorkflowStep.from_dict(data)
        self.assertEqual(step.id, "step_abc123")
        self.assertEqual(step.name, "test")
        self.assertEqual(step.action, "echo")
        self.assertEqual(step.status, StepStatus.COMPLETED)
        self.assertEqual(step.max_retries, 3)

    def test05_is_terminal(self):
        self.assertTrue(WorkflowStep(status=StepStatus.COMPLETED).is_terminal)
        self.assertTrue(WorkflowStep(status=StepStatus.FAILED).is_terminal)
        self.assertTrue(WorkflowStep(status=StepStatus.SKIPPED).is_terminal)
        self.assertFalse(WorkflowStep(status=StepStatus.PENDING).is_terminal)
        self.assertFalse(WorkflowStep(status=StepStatus.RUNNING).is_terminal)

    def test06_can_retry(self):
        # FAILED状态且retry_count < max_retries时可重试
        s = WorkflowStep(status=StepStatus.FAILED, max_retries=2, retry_count=0)
        self.assertTrue(s.can_retry)
        s = WorkflowStep(status=StepStatus.FAILED, max_retries=2, retry_count=1)
        self.assertTrue(s.can_retry)
        # retry_count >= max_retries时不可重试
        s = WorkflowStep(status=StepStatus.FAILED, max_retries=2, retry_count=2)
        self.assertFalse(s.can_retry)
        # 非FAILED状态不可重试
        s = WorkflowStep(status=StepStatus.PENDING, max_retries=2)
        self.assertFalse(s.can_retry)

    def test07_latency_ms(self):
        # 未执行: 0
        s = WorkflowStep()
        self.assertEqual(s.latency_ms, 0.0)
        # 已执行
        s.started_at = 1000.0
        s.completed_at = 1000.5
        self.assertAlmostEqual(s.latency_ms, 500.0, places=1)


class TestWorkflow(unittest.TestCase):
    """测试Workflow"""

    def test01_create_default(self):
        wf = Workflow()
        self.assertTrue(wf.id.startswith("wf_"))
        self.assertEqual(wf.status, WorkflowStatus.DRAFT)
        self.assertEqual(wf.steps, [])
        self.assertEqual(wf.context, {})

    def test02_create_with_steps(self):
        s1 = WorkflowStep(name="s1", action="noop")
        s2 = WorkflowStep(name="s2", action="noop", dependencies=[s1.id])
        wf = Workflow(name="test", steps=[s1, s2])
        self.assertEqual(wf.name, "test")
        self.assertEqual(len(wf.steps), 2)

    def test03_to_dict(self):
        wf = Workflow(name="test", description="desc")
        wf.context["key"] = "value"
        d = wf.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["description"], "desc")
        self.assertEqual(d["status"], "draft")
        self.assertEqual(d["context"]["key"], "value")
        self.assertIn("steps", d)

    def test04_progress_empty(self):
        wf = Workflow()
        self.assertEqual(wf.progress, 0.0)

    def test05_progress_partial(self):
        s1 = WorkflowStep(name="s1", status=StepStatus.COMPLETED)
        s2 = WorkflowStep(name="s2", status=StepStatus.PENDING)
        wf = Workflow(steps=[s1, s2])
        self.assertEqual(wf.progress, 0.5)

    def test06_progress_full(self):
        s1 = WorkflowStep(name="s1", status=StepStatus.COMPLETED)
        s2 = WorkflowStep(name="s2", status=StepStatus.COMPLETED)
        wf = Workflow(steps=[s1, s2])
        self.assertEqual(wf.progress, 1.0)

    def test07_is_terminal(self):
        self.assertTrue(Workflow(status=WorkflowStatus.COMPLETED).is_terminal)
        self.assertTrue(Workflow(status=WorkflowStatus.FAILED).is_terminal)
        self.assertTrue(Workflow(status=WorkflowStatus.CANCELLED).is_terminal)
        self.assertFalse(Workflow(status=WorkflowStatus.RUNNING).is_terminal)
        self.assertFalse(Workflow(status=WorkflowStatus.DRAFT).is_terminal)

    def test08_get_step(self):
        s1 = WorkflowStep(id="step_001", name="s1")
        wf = Workflow(steps=[s1])
        self.assertEqual(wf.get_step("step_001"), s1)
        self.assertIsNone(wf.get_step("nonexistent"))

    def test09_get_ready_steps_no_deps(self):
        s1 = WorkflowStep(name="s1", action="noop")
        s2 = WorkflowStep(name="s2", action="noop")
        wf = Workflow(steps=[s1, s2])
        ready = wf.get_ready_steps()
        self.assertEqual(len(ready), 2)

    def test10_get_ready_steps_with_deps(self):
        s1 = WorkflowStep(id="step_001", name="s1", action="noop")
        s2 = WorkflowStep(id="step_002", name="s2", action="noop",
                         dependencies=["step_001"])
        wf = Workflow(steps=[s1, s2])
        # s1完成前,只有s1就绪
        ready = wf.get_ready_steps()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "step_001")
        # s1完成后,s2就绪
        s1.status = StepStatus.COMPLETED
        ready = wf.get_ready_steps()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "step_002")

    def test11_get_ready_steps_blocked(self):
        s1 = WorkflowStep(id="step_001", name="s1", action="noop",
                         status=StepStatus.FAILED)
        s2 = WorkflowStep(id="step_002", name="s2", action="noop",
                         dependencies=["step_001"])
        wf = Workflow(steps=[s1, s2])
        ready = wf.get_ready_steps()
        self.assertEqual(len(ready), 0)
        self.assertEqual(s2.status, StepStatus.BLOCKED)

    def test12_all_steps_terminal(self):
        s1 = WorkflowStep(status=StepStatus.COMPLETED)
        s2 = WorkflowStep(status=StepStatus.COMPLETED)
        wf = Workflow(steps=[s1, s2])
        self.assertTrue(wf.all_steps_terminal())
        s2.status = StepStatus.PENDING
        self.assertFalse(wf.all_steps_terminal())

    def test13_get_failed_steps(self):
        s1 = WorkflowStep(status=StepStatus.COMPLETED)
        s2 = WorkflowStep(status=StepStatus.FAILED)
        wf = Workflow(steps=[s1, s2])
        failed = wf.get_failed_steps()
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0], s2)

    def test14_get_blocked_steps(self):
        s1 = WorkflowStep(status=StepStatus.BLOCKED)
        s2 = WorkflowStep(status=StepStatus.COMPLETED)
        wf = Workflow(steps=[s1, s2])
        blocked = wf.get_blocked_steps()
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0], s1)


# ============================================================================
# 工作流执行器测试
# ============================================================================

class TestWorkflowExecutor(unittest.TestCase):
    """测试WorkflowExecutor"""

    def test01_default_handlers_registered(self):
        executor = WorkflowExecutor()
        self.assertIn("noop", executor._action_handlers)
        self.assertIn("set_var", executor._action_handlers)
        self.assertIn("echo", executor._action_handlers)

    def test02_register_custom_handler(self):
        executor = WorkflowExecutor()

        def custom_handler(step, wf, input_value):
            return "custom_result"

        executor.register_handler("custom", custom_handler)
        self.assertIn("custom", executor._action_handlers)

    def test03_handle_noop(self):
        executor = WorkflowExecutor()
        step = WorkflowStep(name="s1", action="noop")
        wf = Workflow(steps=[step])
        result = executor._handle_noop(step, wf, None)
        self.assertIsNone(result)

    def test04_handle_set_var(self):
        executor = WorkflowExecutor()
        step = WorkflowStep(name="s1", action="set_var",
                           parameters={"var_name": "x", "var_value": 42})
        wf = Workflow(steps=[step])
        result = executor._handle_set_var(step, wf, None)
        self.assertEqual(result, 42)
        self.assertEqual(wf.context["x"], 42)

    def test05_handle_echo(self):
        executor = WorkflowExecutor()
        step = WorkflowStep(name="s1", action="echo")
        wf = Workflow(steps=[step])
        # 回显输入
        result = executor._handle_echo(step, wf, "hello")
        self.assertEqual(result, "hello")
        # 回显参数message
        step2 = WorkflowStep(name="s2", action="echo",
                            parameters={"message": "default_msg"})
        result2 = executor._handle_echo(step2, wf, None)
        self.assertEqual(result2, "default_msg")

    def test06_execute_simple(self):
        """执行简单工作流(单步骤)"""
        executor = WorkflowExecutor()
        step = WorkflowStep(name="s1", action="set_var",
                           parameters={"var_name": "result", "var_value": "done"},
                           output_key="result")
        wf = Workflow(name="simple", steps=[step])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["progress"], 1.0)
        self.assertEqual(result["steps_completed"], 1)
        self.assertEqual(result["steps_failed"], 0)
        self.assertEqual(wf.context["result"], "done")
        self.assertEqual(step.status, StepStatus.COMPLETED)

    def test07_execute_chain(self):
        """执行链式工作流(多步骤依赖)"""
        executor = WorkflowExecutor()
        s1 = WorkflowStep(id="s1", name="step1", action="set_var",
                         parameters={"var_name": "x", "var_value": 10},
                         output_key="x")
        s2 = WorkflowStep(id="s2", name="step2", action="echo",
                         dependencies=["s1"], input_key="x", output_key="y")
        s3 = WorkflowStep(id="s3", name="step3", action="noop",
                         dependencies=["s2"])
        wf = Workflow(name="chain", steps=[s1, s2, s3])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_completed"], 3)
        self.assertEqual(wf.context["x"], 10)
        self.assertEqual(wf.context["y"], 10)
        self.assertEqual(s1.status, StepStatus.COMPLETED)
        self.assertEqual(s2.status, StepStatus.COMPLETED)
        self.assertEqual(s3.status, StepStatus.COMPLETED)

    def test08_execute_parallel(self):
        """执行并行工作流(无依赖的步骤并行执行)"""
        executor = WorkflowExecutor()
        s1 = WorkflowStep(name="s1", action="set_var",
                         parameters={"var_name": "a", "var_value": 1},
                         output_key="a")
        s2 = WorkflowStep(name="s2", action="set_var",
                         parameters={"var_name": "b", "var_value": 2},
                         output_key="b")
        s3 = WorkflowStep(name="s3", action="noop",
                         dependencies=[s1.id, s2.id])
        wf = Workflow(name="parallel", steps=[s1, s2, s3])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_completed"], 3)
        self.assertEqual(wf.context["a"], 1)
        self.assertEqual(wf.context["b"], 2)

    def test09_execute_failure(self):
        """执行失败的工作流"""
        executor = WorkflowExecutor()

        def failing_handler(step, wf, input_value):
            raise RuntimeError("故意失败")

        executor.register_handler("fail", failing_handler)
        step = WorkflowStep(name="s1", action="fail", max_retries=0)
        wf = Workflow(name="fail_wf", steps=[step])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps_failed"], 1)
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertIsNotNone(step.error)
        self.assertIn("故意失败", step.error)

    def test10_execute_retry_success(self):
        """执行重试后成功的工作流"""
        executor = WorkflowExecutor()

        call_count = {"n": 0}

        def flaky_handler(step, wf, input_value):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("第一次失败")
            return "success"

        executor.register_handler("flaky", flaky_handler)
        step = WorkflowStep(name="s1", action="flaky", max_retries=2)
        wf = Workflow(name="retry_wf", steps=[step])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(step.status, StepStatus.COMPLETED)
        self.assertEqual(step.retry_count, 1)
        self.assertEqual(call_count["n"], 2)

    def test11_execute_retry_exhausted(self):
        """执行重试耗尽后失败的工作流"""
        executor = WorkflowExecutor()

        call_count = {"n": 0}

        def always_fail(step, wf, input_value):
            call_count["n"] += 1
            raise RuntimeError("总是失败")

        executor.register_handler("always_fail", always_fail)
        step = WorkflowStep(name="s1", action="always_fail", max_retries=2)
        wf = Workflow(name="retry_exhausted", steps=[step])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertEqual(step.retry_count, 2)
        # 1次初始 + 2次重试 = 3次
        self.assertEqual(call_count["n"], 3)

    def test12_execute_blocked_steps(self):
        """执行有阻塞步骤的工作流"""
        executor = WorkflowExecutor()

        def failing_handler(step, wf, input_value):
            raise RuntimeError("失败")

        executor.register_handler("fail", failing_handler)
        s1 = WorkflowStep(id="s1", name="s1", action="fail", max_retries=0)
        s2 = WorkflowStep(id="s2", name="s2", action="noop",
                         dependencies=["s1"])
        wf = Workflow(name="blocked_wf", steps=[s1, s2])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(s1.status, StepStatus.FAILED)
        self.assertEqual(s2.status, StepStatus.BLOCKED)

    def test13_execute_empty_workflow(self):
        """执行空工作流"""
        executor = WorkflowExecutor()
        wf = Workflow(name="empty")
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_total"], 0)
        self.assertEqual(result["steps_completed"], 0)

    def test14_execute_unknown_action(self):
        """执行未知动作"""
        executor = WorkflowExecutor()
        step = WorkflowStep(name="s1", action="unknown_action", max_retries=0)
        wf = Workflow(name="unknown", steps=[step])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertIn("未知动作", step.error)

    def test15_get_stats(self):
        executor = WorkflowExecutor()
        stats = executor.get_stats()
        self.assertEqual(stats["workflows_executed"], 0)
        self.assertEqual(stats["steps_executed"], 0)
        self.assertEqual(stats["steps_succeeded"], 0)
        self.assertEqual(stats["steps_failed"], 0)

    def test16_stats_updated_after_execution(self):
        executor = WorkflowExecutor()
        s1 = WorkflowStep(name="s1", action="noop")
        wf = Workflow(steps=[s1])
        wf.status = WorkflowStatus.READY
        executor.execute(wf)

        stats = executor.get_stats()
        self.assertEqual(stats["workflows_executed"], 1)
        self.assertEqual(stats["steps_executed"], 1)
        self.assertEqual(stats["steps_succeeded"], 1)
        self.assertEqual(stats["steps_failed"], 0)

    def test17_execute_diamond_dependency(self):
        """钻石型依赖(菱形依赖)"""
        executor = WorkflowExecutor()
        s1 = WorkflowStep(id="s1", name="root", action="set_var",
                         parameters={"var_name": "x", "var_value": 1},
                         output_key="x")
        s2 = WorkflowStep(id="s2", name="left", action="echo",
                         dependencies=["s1"], input_key="x", output_key="y")
        s3 = WorkflowStep(id="s3", name="right", action="echo",
                         dependencies=["s1"], input_key="x", output_key="z")
        s4 = WorkflowStep(id="s4", name="merge", action="noop",
                         dependencies=["s2", "s3"])
        wf = Workflow(name="diamond", steps=[s1, s2, s3, s4])
        wf.status = WorkflowStatus.READY

        result = executor.execute(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_completed"], 4)
        self.assertEqual(wf.context["x"], 1)
        self.assertEqual(wf.context["y"], 1)
        self.assertEqual(wf.context["z"], 1)

    def test18_workflow_status_after_execution(self):
        """执行后工作流状态正确"""
        executor = WorkflowExecutor()
        s1 = WorkflowStep(name="s1", action="noop")
        wf = Workflow(steps=[s1])
        wf.status = WorkflowStatus.READY

        executor.execute(wf)

        self.assertEqual(wf.status, WorkflowStatus.COMPLETED)
        self.assertIsNotNone(wf.started_at)
        self.assertIsNotNone(wf.completed_at)

    def test19_metadata_updated(self):
        """执行后元数据更新"""
        executor = WorkflowExecutor()
        s1 = WorkflowStep(name="s1", action="noop")
        wf = Workflow(steps=[s1])
        wf.status = WorkflowStatus.READY

        executor.execute(wf)

        self.assertIn("iterations", wf.metadata)
        self.assertIn("total_latency_ms", wf.metadata)
        self.assertGreater(wf.metadata["total_latency_ms"], 0)


# ============================================================================
# 工作流构建器测试
# ============================================================================

class TestWorkflowBuilder(unittest.TestCase):
    """测试WorkflowBuilder"""

    def test01_create_empty(self):
        builder = WorkflowBuilder("my_wf", "description")
        wf = builder.build()
        self.assertEqual(wf.name, "my_wf")
        self.assertEqual(wf.description, "description")
        self.assertEqual(wf.status, WorkflowStatus.READY)
        self.assertEqual(wf.steps, [])

    def test02_add_step(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="noop")
        wf = builder.build()
        self.assertEqual(len(wf.steps), 1)
        self.assertEqual(wf.steps[0].name, "s1")
        self.assertEqual(wf.steps[0].action, "noop")

    def test03_add_multiple_steps(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="noop")
        builder.step("s2", action="noop")
        builder.step("s3", action="noop")
        wf = builder.build()
        self.assertEqual(len(wf.steps), 3)

    def test04_step_with_dependencies(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="noop")
        builder.step("s2", action="noop", dependencies=["s1"])
        wf = builder.build()
        # 依赖名应被转换为ID
        self.assertEqual(len(wf.steps[1].dependencies), 1)
        self.assertEqual(wf.steps[1].dependencies[0], wf.steps[0].id)

    def test05_step_with_parameters(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="set_var",
                    parameters={"var_name": "x", "var_value": 42})
        wf = builder.build()
        self.assertEqual(wf.steps[0].parameters["var_name"], "x")
        self.assertEqual(wf.steps[0].parameters["var_value"], 42)

    def test06_step_with_input_output_keys(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="echo", input_key="in", output_key="out")
        wf = builder.build()
        self.assertEqual(wf.steps[0].input_key, "in")
        self.assertEqual(wf.steps[0].output_key, "out")

    def test07_context(self):
        builder = WorkflowBuilder("test")
        builder.context("key1", "value1")
        builder.context("key2", 42)
        wf = builder.build()
        self.assertEqual(wf.context["key1"], "value1")
        self.assertEqual(wf.context["key2"], 42)

    def test08_chaining(self):
        """测试流式API链式调用"""
        wf = (WorkflowBuilder("chain_test")
              .step("s1", action="noop")
              .step("s2", action="noop", dependencies=["s1"])
              .context("x", 1)
              .build())
        self.assertEqual(wf.name, "chain_test")
        self.assertEqual(len(wf.steps), 2)
        self.assertEqual(wf.context["x"], 1)

    def test09_dependency_chain(self):
        """测试依赖链"""
        builder = WorkflowBuilder("dep_chain")
        builder.step("a", action="noop")
        builder.step("b", action="noop", dependencies=["a"])
        builder.step("c", action="noop", dependencies=["b"])
        builder.step("d", action="noop", dependencies=["c"])
        wf = builder.build()

        self.assertEqual(len(wf.steps), 4)
        self.assertEqual(wf.steps[1].dependencies[0], wf.steps[0].id)
        self.assertEqual(wf.steps[2].dependencies[0], wf.steps[1].id)
        self.assertEqual(wf.steps[3].dependencies[0], wf.steps[2].id)

    def test10_step_with_timeout(self):
        builder = WorkflowBuilder("test")
        builder.step("s1", action="noop", timeout=30.0)
        wf = builder.build()
        self.assertEqual(wf.steps[0].timeout, 30.0)


# ============================================================================
# 工作流Agent配置测试
# ============================================================================

class TestWorkflowAgentConfig(unittest.TestCase):
    """测试WorkflowAgentConfig"""

    def test01_default_config(self):
        config = WorkflowAgentConfig()
        self.assertEqual(config.name, "workflow_agent")
        self.assertEqual(config.max_iterations, 100)
        self.assertTrue(config.enable_retry)
        self.assertIsNotNone(config.system_prompt)

    def test02_custom_config(self):
        config = WorkflowAgentConfig(
            name="custom_wf",
            max_iterations=50,
            enable_retry=False,
        )
        self.assertEqual(config.name, "custom_wf")
        self.assertEqual(config.max_iterations, 50)
        self.assertFalse(config.enable_retry)


# ============================================================================
# 工作流Agent测试
# ============================================================================

class TestWorkflowAgent(unittest.TestCase):
    """测试WorkflowAgent"""

    def setUp(self):
        reset_workflow_agent()

    def tearDown(self):
        reset_workflow_agent()

    def test01_create_agent(self):
        agent = WorkflowAgent()
        self.assertEqual(agent.state, AgentState.CREATED)
        self.assertIsNotNone(agent.executor)
        self.assertEqual(agent.list_workflows(), [])

    def test02_create_workflow_builder(self):
        agent = WorkflowAgent()
        builder = agent.create_workflow("test", "desc")
        self.assertIsInstance(builder, WorkflowBuilder)
        wf = builder.build()
        self.assertEqual(wf.name, "test")

    def test03_execute_workflow(self):
        agent = WorkflowAgent()
        wf = (WorkflowBuilder("test_wf")
              .step("s1", action="set_var",
                   parameters={"var_name": "x", "var_value": 42},
                   output_key="x")
              .build())
        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_completed"], 1)
        self.assertEqual(wf.context["x"], 42)

    def test04_get_workflow(self):
        agent = WorkflowAgent()
        wf = WorkflowBuilder("test").step("s1").build()
        agent.execute_workflow(wf)
        retrieved = agent.get_workflow(wf.id)
        self.assertEqual(retrieved, wf)

    def test05_get_workflow_nonexistent(self):
        agent = WorkflowAgent()
        self.assertIsNone(agent.get_workflow("nonexistent"))

    def test06_list_workflows(self):
        agent = WorkflowAgent()
        wf1 = WorkflowBuilder("wf1").step("s1").build()
        wf2 = WorkflowBuilder("wf2").step("s1").build()
        agent.execute_workflow(wf1)
        agent.execute_workflow(wf2)
        workflows = agent.list_workflows()
        self.assertEqual(len(workflows), 2)

    def test07_register_action(self):
        agent = WorkflowAgent()

        def custom_handler(step, wf, input_value):
            return "custom"

        agent.register_action("custom", custom_handler)
        self.assertIn("custom", agent.executor._action_handlers)

    def test08_execute_with_custom_action(self):
        agent = WorkflowAgent()

        def multiply_handler(step, wf, input_value):
            return input_value * 2

        agent.register_action("multiply", multiply_handler)
        wf = (WorkflowBuilder("custom_wf")
              .context("x", 5)
              .step("s1", action="multiply", input_key="x", output_key="y")
              .build())
        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(wf.context["y"], 10)

    def test09_run_with_workflow_object(self):
        """通过run方法执行Workflow对象"""
        agent = WorkflowAgent()
        agent.start()
        wf = (WorkflowBuilder("run_test")
              .step("s1", action="noop")
              .build())
        result = agent.run(wf)

        self.assertTrue(result.success)
        self.assertIn("completed", result.output)
        self.assertEqual(result.metadata["workflow_id"], wf.id)

    def test10_run_with_dict(self):
        """通过run方法执行包含workflow键的字典"""
        agent = WorkflowAgent()
        agent.start()
        wf = WorkflowBuilder("dict_test").step("s1", action="noop").build()
        result = agent.run({"workflow": wf})

        self.assertTrue(result.success)

    def test11_run_with_invalid_input(self):
        """无效输入"""
        agent = WorkflowAgent()
        agent.start()
        result = agent.run("invalid string")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test12_run_failed_workflow(self):
        """执行失败的工作流"""
        agent = WorkflowAgent()

        def fail_handler(step, wf, input_value):
            raise RuntimeError("失败")

        agent.register_action("fail", fail_handler)
        agent.start()
        wf = (WorkflowBuilder("fail_test")
              .step("s1", action="fail", max_retries=0)
              .build())
        result = agent.run(wf)

        self.assertFalse(result.success)
        self.assertIn("failed", result.output.lower())

    def test13_get_stats(self):
        agent = WorkflowAgent()
        wf = WorkflowBuilder("test").step("s1", action="noop").build()
        agent.execute_workflow(wf)

        stats = agent.get_stats()
        self.assertIn("executor", stats)
        self.assertEqual(stats["workflows_count"], 1)
        self.assertEqual(stats["executor"]["workflows_executed"], 1)

    def test14_wf_config_property(self):
        agent = WorkflowAgent()
        self.assertIsInstance(agent.wf_config, WorkflowAgentConfig)

    def test15_complex_workflow(self):
        """复杂工作流: 多步骤+依赖+上下文传递"""
        agent = WorkflowAgent()

        def add_handler(step, wf, input_value):
            return input_value + step.parameters.get("delta", 1)

        agent.register_action("add", add_handler)

        wf = (WorkflowBuilder("complex")
              .context("base", 10)
              .step("s1", action="add", input_key="base",
                   parameters={"delta": 5}, output_key="step1_result")
              .step("s2", action="add", input_key="step1_result",
                   parameters={"delta": 10}, output_key="step2_result",
                   dependencies=["s1"])
              .step("s3", action="echo", input_key="step2_result",
                   output_key="final", dependencies=["s2"])
              .build())

        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(wf.context["step1_result"], 15)
        self.assertEqual(wf.context["step2_result"], 25)
        self.assertEqual(wf.context["final"], 25)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingleton(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_workflow_agent()

    def tearDown(self):
        reset_workflow_agent()

    def test01_get_workflow_agent(self):
        agent = get_workflow_agent()
        self.assertIsInstance(agent, WorkflowAgent)

    def test02_singleton(self):
        a1 = get_workflow_agent()
        a2 = get_workflow_agent()
        self.assertIs(a1, a2)

    def test03_reset(self):
        a1 = get_workflow_agent()
        reset_workflow_agent()
        a2 = get_workflow_agent()
        self.assertIsNot(a1, a2)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_workflow_agent()
        reset_llm_client()

    def tearDown(self):
        reset_workflow_agent()
        reset_llm_client()

    def test01_full_workflow_with_custom_actions(self):
        """完整工作流: 自定义动作+依赖+上下文"""
        agent = WorkflowAgent()

        # 模拟数据处理流水线
        def load_handler(step, wf, input_value):
            return [1, 2, 3, 4, 5]

        def transform_handler(step, wf, input_value):
            return [x * 2 for x in input_value]

        def aggregate_handler(step, wf, input_value):
            return sum(input_value)

        agent.register_action("load", load_handler)
        agent.register_action("transform", transform_handler)
        agent.register_action("aggregate", aggregate_handler)

        wf = (WorkflowBuilder("data_pipeline",
                              description="数据处理流水线")
              .step("load", action="load", output_key="raw_data")
              .step("transform", action="transform",
                   dependencies=["load"],
                   input_key="raw_data", output_key="transformed")
              .step("aggregate", action="aggregate",
                   dependencies=["transform"],
                   input_key="transformed", output_key="result")
              .build())

        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_completed"], 3)
        self.assertEqual(wf.context["raw_data"], [1, 2, 3, 4, 5])
        self.assertEqual(wf.context["transformed"], [2, 4, 6, 8, 10])
        self.assertEqual(wf.context["result"], 30)

    def test02_workflow_with_retry(self):
        """工作流重试测试"""
        agent = WorkflowAgent()

        call_count = {"n": 0}

        def unstable_handler(step, wf, input_value):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("网络不稳定")
            return "recovered"

        agent.register_action("unstable", unstable_handler)

        wf = (WorkflowBuilder("retry_test")
              .step("s1", action="unstable", max_retries=3,
                   output_key="result")
              .build())

        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(wf.context["result"], "recovered")
        self.assertEqual(call_count["n"], 3)

    def test03_real_llm_workflow(self):
        """真实LLM工作流测试"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        from seva.llm import get_llm_client, Message, Role

        agent = WorkflowAgent()

        def llm_generate_handler(step, wf, input_value):
            """调用LLM生成文本"""
            client = get_llm_client()
            prompt = step.parameters.get("prompt", "你好")
            messages = [
                Message(role=Role.SYSTEM, content="你是一个简洁的助手。"),
                Message(role=Role.USER, content=prompt),
            ]
            response = client.chat(messages)
            return response.content

        agent.register_action("llm_generate", llm_generate_handler)

        wf = (WorkflowBuilder("llm_workflow",
                              description="LLM工作流")
              .step("generate", action="llm_generate",
                   parameters={"prompt": "用一句话介绍Python编程语言。"},
                   output_key="answer")
              .build())

        result = agent.execute_workflow(wf)

        self.assertEqual(result["status"], "completed")
        self.assertIn("answer", wf.context)
        self.assertGreater(len(wf.context["answer"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
