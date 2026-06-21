"""
阶段55测试：多Agent协作与编排
"""
import unittest
import time
import threading

from seva.orchestration import (
    TaskNode, TaskStatus, ExecutionMode, WorkflowStatus,
    Workflow, WorkflowResult, Orchestrator,
    get_orchestrator, create_workflow, run_workflow,
    register_workflow, run_registered_workflow,
    list_workflows, get_workflow_result,
    get_orchestration_report, get_orchestration_status,
    reset_orchestration,
)


# ============================================================================
# 测试枚举
# ============================================================================

class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test01_task_status(self):
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.SUCCESS.value, "success")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.SKIPPED.value, "skipped")
        self.assertEqual(TaskStatus.CANCELLED.value, "cancelled")

    def test02_execution_mode(self):
        self.assertEqual(ExecutionMode.SEQUENTIAL.value, "sequential")
        self.assertEqual(ExecutionMode.PARALLEL.value, "parallel")
        self.assertEqual(ExecutionMode.PIPELINE.value, "pipeline")

    def test03_workflow_status(self):
        self.assertEqual(WorkflowStatus.CREATED.value, "created")
        self.assertEqual(WorkflowStatus.RUNNING.value, "running")
        self.assertEqual(WorkflowStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkflowStatus.FAILED.value, "failed")
        self.assertEqual(WorkflowStatus.CANCELLED.value, "cancelled")


# ============================================================================
# 测试TaskNode
# ============================================================================

class TestTaskNode(unittest.TestCase):
    """测试任务节点"""

    def test01_creation(self):
        task = TaskNode("test", lambda: "result")
        self.assertEqual(task.name, "test")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.result)

    def test02_execute_success(self):
        task = TaskNode("test", lambda x: x * 2, args=(5,))
        result = task.execute()
        self.assertEqual(result, 10)
        self.assertEqual(task.status, TaskStatus.SUCCESS)
        self.assertEqual(task.result, 10)
        self.assertIsNone(task.error)

    def test03_execute_failure(self):
        def fail():
            raise ValueError("失败")

        task = TaskNode("test", fail)
        with self.assertRaises(ValueError):
            task.execute()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.error, "失败")

    def test04_execute_with_context(self):
        def func(context=None):
            return context.get("key", "default")

        task = TaskNode("test", func)
        result = task.execute(context={"key": "value"})
        self.assertEqual(result, "value")

    def test05_condition_skip(self):
        task = TaskNode("test", lambda: "result", condition=lambda: False)
        result = task.execute()
        self.assertIsNone(result)
        self.assertEqual(task.status, TaskStatus.SKIPPED)

    def test06_condition_pass(self):
        task = TaskNode("test", lambda: "result", condition=lambda: True)
        result = task.execute()
        self.assertEqual(result, "result")
        self.assertEqual(task.status, TaskStatus.SUCCESS)

    def test07_retry(self):
        attempts = [0]

        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("重试")
            return "成功"

        task = TaskNode("test", flaky, retries=2)
        result = task.execute()
        self.assertEqual(result, "成功")
        self.assertEqual(task.attempts, 3)

    def test08_retry_exhausted(self):
        def always_fail():
            raise ValueError("总是失败")

        task = TaskNode("test", always_fail, retries=2)
        with self.assertRaises(ValueError):
            task.execute()
        self.assertEqual(task.attempts, 3)
        self.assertEqual(task.status, TaskStatus.FAILED)

    def test09_is_completed(self):
        task = TaskNode("test", lambda: "ok")
        self.assertFalse(task.is_completed)
        task.execute()
        self.assertTrue(task.is_completed)
        self.assertTrue(task.is_successful)

    def test10_duration(self):
        task = TaskNode("test", lambda: time.sleep(0.05))
        task.execute()
        self.assertGreater(task.duration, 0.04)

    def test11_reset(self):
        task = TaskNode("test", lambda: "ok")
        task.execute()
        task.reset()
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.result)

    def test12_to_dict(self):
        task = TaskNode("test", lambda: "ok")
        task.execute()
        d = task.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["status"], "success")

    def test13_repr(self):
        task = TaskNode("test", lambda: "ok")
        r = repr(task)
        self.assertIn("TaskNode", r)
        self.assertIn("test", r)
        self.assertIn("pending", r)


# ============================================================================
# 测试Workflow
# ============================================================================

class TestWorkflow(unittest.TestCase):
    """测试工作流"""

    def test01_creation(self):
        wf = Workflow("test")
        self.assertEqual(wf.name, "test")
        self.assertEqual(wf.status, WorkflowStatus.CREATED)
        self.assertEqual(wf.task_count, 0)

    def test02_add_task(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("task1", lambda: "ok"))
        self.assertEqual(wf.task_count, 1)
        self.assertIn("task1", wf.list_tasks())

    def test03_add_task_with_dependency(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("task1", lambda: "ok"))
        wf.add_task(TaskNode("task2", lambda: "ok"), depends_on=["task1"])
        self.assertEqual(wf.get_dependencies("task2"), {"task1"})
        self.assertEqual(wf.get_dependents("task1"), {"task2"})

    def test04_add_duplicate_task(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("task1", lambda: "ok"))
        with self.assertRaises(ValueError):
            wf.add_task(TaskNode("task1", lambda: "ok"))

    def test05_add_task_nonexistent_dependency(self):
        wf = Workflow("test")
        with self.assertRaises(ValueError):
            wf.add_task(TaskNode("task1", lambda: "ok"), depends_on=["nonexistent"])

    def test06_remove_task(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("task1", lambda: "ok"))
        self.assertTrue(wf.remove_task("task1"))
        self.assertEqual(wf.task_count, 0)

    def test07_remove_task_nonexistent(self):
        wf = Workflow("test")
        self.assertFalse(wf.remove_task("nonexistent"))

    def test08_topological_sort(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok"))
        wf.add_task(TaskNode("b", lambda: "ok"), depends_on=["a"])
        wf.add_task(TaskNode("c", lambda: "ok"), depends_on=["a"])
        wf.add_task(TaskNode("d", lambda: "ok"), depends_on=["b", "c"])

        order = wf.topological_sort()
        self.assertEqual(order[0], "a")
        self.assertEqual(order[-1], "d")
        # b和c在a之后d之前
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("a"), order.index("c"))
        self.assertLess(order.index("b"), order.index("d"))
        self.assertLess(order.index("c"), order.index("d"))

    def test09_cycle_detection(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok"))
        wf.add_task(TaskNode("b", lambda: "ok"), depends_on=["a"])
        # 手动添加循环依赖
        wf._dependencies["a"].add("b")
        wf._dependents["b"].add("a")
        with self.assertRaises(ValueError):
            wf.topological_sort()

    def test10_validate(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok"))
        wf.add_task(TaskNode("b", lambda: "ok"), depends_on=["a"])
        self.assertTrue(wf.validate())

    def test11_execute_sequential(self):
        wf = Workflow("test", mode=ExecutionMode.SEQUENTIAL)
        wf.add_task(TaskNode("a", lambda context=None: 1))
        wf.add_task(TaskNode("b", lambda context=None: context.get("a", 0) + 1),
                    depends_on=["a"])
        wf.add_task(TaskNode("c", lambda context=None: context.get("b", 0) + 1),
                    depends_on=["b"])

        result = wf.execute()
        self.assertTrue(result.is_success)
        self.assertEqual(result.get_result("a"), 1)
        self.assertEqual(result.get_result("b"), 2)
        self.assertEqual(result.get_result("c"), 3)

    def test12_execute_parallel(self):
        wf = Workflow("test", mode=ExecutionMode.PARALLEL)
        wf.add_task(TaskNode("a", lambda context=None: 1))
        wf.add_task(TaskNode("b", lambda context=None: 2))
        wf.add_task(TaskNode("c", lambda context=None: 3))

        result = wf.execute()
        self.assertTrue(result.is_success)
        self.assertEqual(result.success_count, 3)

    def test13_execute_with_failure(self):
        wf = Workflow("test", mode=ExecutionMode.SEQUENTIAL)
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        wf.add_task(TaskNode("b", lambda context=None: (_ for _ in ()).throw(ValueError("失败"))),
                    depends_on=["a"])
        wf.add_task(TaskNode("c", lambda context=None: "ok"), depends_on=["b"])

        result = wf.execute()
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertEqual(result.tasks["a"].status, TaskStatus.SUCCESS)
        self.assertEqual(result.tasks["b"].status, TaskStatus.FAILED)
        self.assertEqual(result.tasks["c"].status, TaskStatus.SKIPPED)

    def test14_execute_empty(self):
        wf = Workflow("test")
        result = wf.execute()
        self.assertTrue(result.is_success)
        self.assertEqual(result.success_count, 0)

    def test15_chain_add_task(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok")).add_task(TaskNode("b", lambda: "ok"))
        self.assertEqual(wf.task_count, 2)

    def test16_get_info(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok"))
        info = wf.get_info()
        self.assertEqual(info["name"], "test")
        self.assertEqual(info["task_count"], 1)
        self.assertEqual(info["status"], "created")

    def test17_reset(self):
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda: "ok"))
        wf.execute()
        wf.reset()
        self.assertEqual(wf.status, WorkflowStatus.CREATED)

    def test18_repr(self):
        wf = Workflow("test")
        r = repr(wf)
        self.assertIn("Workflow", r)
        self.assertIn("test", r)

    def test19_pipeline_mode(self):
        wf = Workflow("test", mode=ExecutionMode.PIPELINE)
        wf.add_task(TaskNode("a", lambda context=None: [1, 2, 3]))
        wf.add_task(TaskNode("b", lambda x, context=None: [i * 2 for i in x]),
                    depends_on=["a"])

        result = wf.execute()
        self.assertTrue(result.is_success)

    def test20_fail_fast(self):
        wf = Workflow("test", mode=ExecutionMode.SEQUENTIAL, fail_fast=True)
        wf.add_task(TaskNode("a", lambda context=None: (_ for _ in ()).throw(ValueError("失败"))))
        wf.add_task(TaskNode("b", lambda context=None: "ok"), depends_on=["a"])

        result = wf.execute()
        self.assertEqual(result.status, WorkflowStatus.FAILED)


# ============================================================================
# 测试WorkflowResult
# ============================================================================

class TestWorkflowResult(unittest.TestCase):
    """测试工作流结果"""

    def test01_creation(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        self.assertEqual(result.workflow_id, "wf1")
        self.assertEqual(result.workflow_name, "test")
        self.assertTrue(result.is_success)

    def test02_counts(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        result.tasks["a"] = TaskNode("a", lambda: "ok")
        result.tasks["a"].status = TaskStatus.SUCCESS
        result.tasks["b"] = TaskNode("b", lambda: "ok")
        result.tasks["b"].status = TaskStatus.FAILED
        result.tasks["c"] = TaskNode("c", lambda: "ok")
        result.tasks["c"].status = TaskStatus.SKIPPED

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.skipped_count, 1)

    def test03_get_result(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        task = TaskNode("a", lambda: 42)
        task.result = 42
        task.status = TaskStatus.SUCCESS
        result.tasks["a"] = task
        self.assertEqual(result.get_result("a"), 42)

    def test04_get_results(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        task1 = TaskNode("a", lambda: 1)
        task1.result = 1
        task1.status = TaskStatus.SUCCESS
        task2 = TaskNode("b", lambda: 2)
        task2.status = TaskStatus.FAILED
        result.tasks["a"] = task1
        result.tasks["b"] = task2

        results = result.get_results()
        self.assertEqual(results, {"a": 1})

    def test05_to_dict(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        d = result.to_dict()
        self.assertEqual(d["workflow_id"], "wf1")
        self.assertEqual(d["status"], "completed")

    def test06_summary(self):
        result = WorkflowResult("wf1", "test", WorkflowStatus.COMPLETED)
        s = result.summary()
        self.assertIn("test", s)
        self.assertIn("completed", s)


# ============================================================================
# 测试Orchestrator
# ============================================================================

class TestOrchestrator(unittest.TestCase):
    """测试编排器"""

    def setUp(self):
        Orchestrator.reset_instance()

    def tearDown(self):
        Orchestrator.reset_instance()

    def test01_singleton(self):
        o1 = Orchestrator()
        o2 = Orchestrator()
        self.assertIs(o1, o2)

    def test02_register(self):
        orch = Orchestrator()
        wf = Workflow("test")
        self.assertTrue(orch.register(wf))
        self.assertIn("test", orch.list_workflows())

    def test03_register_duplicate(self):
        orch = Orchestrator()
        orch.register(Workflow("test"))
        self.assertFalse(orch.register(Workflow("test")))

    def test04_unregister(self):
        orch = Orchestrator()
        orch.register(Workflow("test"))
        self.assertTrue(orch.unregister("test"))
        self.assertNotIn("test", orch.list_workflows())

    def test05_unregister_nonexistent(self):
        orch = Orchestrator()
        self.assertFalse(orch.unregister("nonexistent"))

    def test06_get_workflow(self):
        orch = Orchestrator()
        wf = Workflow("test")
        orch.register(wf)
        self.assertIs(orch.get_workflow("test"), wf)

    def test07_run(self):
        orch = Orchestrator()
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        orch.register(wf)
        result = orch.run("test")
        self.assertIsNotNone(result)
        self.assertTrue(result.is_success)

    def test08_run_nonexistent(self):
        orch = Orchestrator()
        self.assertIsNone(orch.run("nonexistent"))

    def test09_run_all(self):
        orch = Orchestrator()
        wf1 = Workflow("wf1")
        wf1.add_task(TaskNode("a", lambda context=None: "ok"))
        wf2 = Workflow("wf2")
        wf2.add_task(TaskNode("b", lambda context=None: "ok"))
        orch.register(wf1)
        orch.register(wf2)
        results = orch.run_all()
        self.assertEqual(len(results), 2)
        self.assertTrue(results["wf1"].is_success)
        self.assertTrue(results["wf2"].is_success)

    def test10_get_result(self):
        orch = Orchestrator()
        wf = Workflow("test")
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        orch.register(wf)
        orch.run("test")
        result = orch.get_result("test")
        self.assertIsNotNone(result)
        self.assertTrue(result.is_success)

    def test11_get_report(self):
        orch = Orchestrator()
        orch.register(Workflow("test"))
        report = orch.get_report()
        self.assertEqual(report["total_workflows"], 1)
        self.assertIn("test", report["workflow_names"])

    def test12_format_report(self):
        orch = Orchestrator()
        orch.register(Workflow("test"))
        report = orch.format_report()
        self.assertIsInstance(report, str)
        self.assertIn("工作流编排报告", report)
        self.assertIn("test", report)

    def test13_reset(self):
        orch = Orchestrator()
        orch.register(Workflow("test"))
        orch.reset()
        self.assertEqual(len(orch.list_workflows()), 0)


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        Orchestrator.reset_instance()

    def tearDown(self):
        Orchestrator.reset_instance()

    def test01_get_orchestrator(self):
        orch = get_orchestrator()
        self.assertIsInstance(orch, Orchestrator)

    def test02_create_workflow(self):
        wf = create_workflow("test")
        self.assertIsInstance(wf, Workflow)
        self.assertEqual(wf.name, "test")

    def test03_run_workflow(self):
        wf = create_workflow("test")
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        result = run_workflow(wf)
        self.assertTrue(result.is_success)

    def test04_register_workflow(self):
        wf = create_workflow("test")
        self.assertTrue(register_workflow(wf))
        self.assertIn("test", list_workflows())

    def test05_run_registered_workflow(self):
        wf = create_workflow("test")
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        register_workflow(wf)
        result = run_registered_workflow("test")
        self.assertIsNotNone(result)
        self.assertTrue(result.is_success)

    def test06_get_workflow_result(self):
        wf = create_workflow("test")
        wf.add_task(TaskNode("a", lambda context=None: "ok"))
        register_workflow(wf)
        run_registered_workflow("test")
        result = get_workflow_result("test")
        self.assertIsNotNone(result)

    def test07_get_orchestration_report(self):
        report = get_orchestration_report()
        self.assertIsInstance(report, str)
        self.assertIn("工作流编排报告", report)

    def test08_get_orchestration_status(self):
        status = get_orchestration_status()
        self.assertIn("total_workflows", status)

    def test09_reset_orchestration(self):
        reset_orchestration()  # 不应抛异常


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        Orchestrator.reset_instance()

    def tearDown(self):
        Orchestrator.reset_instance()

    def test01_complex_dag(self):
        """复杂DAG工作流"""
        wf = Workflow("complex", mode=ExecutionMode.PARALLEL)

        # 构建DAG:
        #     a
        #    / \
        #   b   c
        #    \ /
        #     d
        wf.add_task(TaskNode("a", lambda context=None: 1))
        wf.add_task(TaskNode("b", lambda context=None: context.get("a", 0) + 1),
                    depends_on=["a"])
        wf.add_task(TaskNode("c", lambda context=None: context.get("a", 0) + 2),
                    depends_on=["a"])
        wf.add_task(TaskNode("d", lambda context=None:
                             context.get("b", 0) + context.get("c", 0)),
                    depends_on=["b", "c"])

        result = wf.execute()
        self.assertTrue(result.is_success)
        self.assertEqual(result.get_result("a"), 1)
        self.assertEqual(result.get_result("b"), 2)
        self.assertEqual(result.get_result("c"), 3)
        self.assertEqual(result.get_result("d"), 5)

    def test02_parallel_speedup(self):
        """并行执行加速"""
        def slow_task(context=None):
            time.sleep(0.1)
            return "done"

        # 顺序执行
        wf_seq = Workflow("seq", mode=ExecutionMode.SEQUENTIAL)
        for i in range(3):
            wf_seq.add_task(TaskNode(f"task_{i}", slow_task))

        start = time.time()
        wf_seq.execute()
        seq_time = time.time() - start

        # 并行执行
        wf_par = Workflow("par", mode=ExecutionMode.PARALLEL, max_workers=3)
        for i in range(3):
            wf_par.add_task(TaskNode(f"task_{i}", slow_task))

        start = time.time()
        wf_par.execute()
        par_time = time.time() - start

        # 并行应该更快
        self.assertLess(par_time, seq_time)

    def test03_error_handling(self):
        """错误处理"""
        wf = Workflow("error_test", mode=ExecutionMode.SEQUENTIAL)
        wf.add_task(TaskNode("ok", lambda context=None: "ok"))
        wf.add_task(TaskNode("fail", lambda context=None:
                             (_ for _ in ()).throw(ValueError("失败"))),
                    depends_on=["ok"])
        wf.add_task(TaskNode("skip", lambda context=None: "ok"), depends_on=["fail"])

        result = wf.execute()
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertEqual(result.tasks["ok"].status, TaskStatus.SUCCESS)
        self.assertEqual(result.tasks["fail"].status, TaskStatus.FAILED)
        self.assertEqual(result.tasks["skip"].status, TaskStatus.SKIPPED)

    def test04_orchestrator_multiple_workflows(self):
        """编排器管理多个工作流"""
        orch = Orchestrator()

        wf1 = Workflow("wf1")
        wf1.add_task(TaskNode("a", lambda context=None: 1))

        wf2 = Workflow("wf2")
        wf2.add_task(TaskNode("b", lambda context=None: 2))

        orch.register(wf1)
        orch.register(wf2)

        results = orch.run_all()
        self.assertTrue(results["wf1"].is_success)
        self.assertTrue(results["wf2"].is_success)

    def test05_thread_safety(self):
        """线程安全测试"""
        orch = Orchestrator()
        errors = []

        def worker(i):
            try:
                wf = Workflow(f"wf_{i}")
                wf.add_task(TaskNode("task", lambda context=None: i))
                orch.register(wf)
                orch.run(f"wf_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(orch.list_workflows()), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
