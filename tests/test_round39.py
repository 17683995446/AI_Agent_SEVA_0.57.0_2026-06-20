"""
阶段49测试：进度反馈与状态指示
验证 ProgressBar / Spinner / StatusIndicator / TaskProgress
及便捷函数和CLI集成
"""
import os
import sys
import io
import time
import unittest
from unittest.mock import patch


import seva
from seva.progress import (ProgressBar, Spinner, StatusIndicator, TaskProgress,
                           SPINNER_CHARS, SPINNER_CHARS_ASCII,
                           STATUS_ICONS, STATUS_ICONS_ASCII,
                           progress_bar, spinner, status,
                           track_progress, track_progress_async,
                           format_status, get_progress_report)


class TestProgressBar(unittest.TestCase):
    """测试ProgressBar"""

    def test01_init_basic(self):
        """基本初始化"""
        bar = ProgressBar("测试", total=100)
        self.assertEqual(bar.description, "测试")
        self.assertEqual(bar.total, 100)
        self.assertEqual(bar.current, 0)
        self.assertFalse(bar.is_complete)

    def test02_init_defaults(self):
        """默认值"""
        bar = ProgressBar()
        self.assertEqual(bar.description, "")
        self.assertEqual(bar.total, 100)
        self.assertEqual(bar.width, 30)

    def test03_total_minimum(self):
        """total最小为1"""
        bar = ProgressBar(total=0)
        self.assertEqual(bar.total, 1)

    def test04_percentage(self):
        """百分比"""
        bar = ProgressBar(total=100)
        bar._current = 50
        self.assertEqual(bar.percentage, 50.0)

    def test05_is_complete(self):
        """完成判断"""
        bar = ProgressBar(total=10)
        self.assertFalse(bar.is_complete)
        bar._current = 10
        self.assertTrue(bar.is_complete)
        bar._current = 15
        self.assertTrue(bar.is_complete)

    def test06_update(self):
        """update方法"""
        bar = ProgressBar(total=10, output=io.StringIO())
        bar.start()
        bar.update(1)
        self.assertEqual(bar.current, 1)
        bar.update(3)
        self.assertEqual(bar.current, 4)

    def test07_update_overflow(self):
        """update不会超过total"""
        bar = ProgressBar(total=5, output=io.StringIO())
        bar.start()
        bar.update(10)
        self.assertEqual(bar.current, 5)

    def test08_set_progress(self):
        """set_progress"""
        bar = ProgressBar(total=100, output=io.StringIO())
        bar.start()
        bar.set_progress(30)
        self.assertEqual(bar.current, 30)
        bar.set_progress(150)
        self.assertEqual(bar.current, 100)
        bar.set_progress(-10)
        self.assertEqual(bar.current, 0)

    def test09_start(self):
        """start设置开始时间"""
        bar = ProgressBar(total=10, output=io.StringIO())
        self.assertIsNone(bar._start_time)
        bar.start()
        self.assertIsNotNone(bar._start_time)

    def test10_finish(self):
        """finish完成"""
        bar = ProgressBar(total=10, output=io.StringIO())
        bar.start()
        bar.finish()
        self.assertTrue(bar.is_complete)
        self.assertIsNotNone(bar._end_time)
        self.assertFalse(bar._running)

    def test11_elapsed_time(self):
        """elapsed_time"""
        bar = ProgressBar(total=10, output=io.StringIO())
        self.assertEqual(bar.elapsed_time, 0.0)
        bar.start()
        time.sleep(0.01)
        self.assertGreater(bar.elapsed_time, 0)

    def test12_eta(self):
        """ETA预估"""
        bar = ProgressBar(total=100, output=io.StringIO())
        # 未开始时ETA为None
        self.assertIsNone(bar.eta)
        bar.start()
        bar._current = 50
        eta = bar.eta
        self.assertIsNotNone(eta)

    def test13_eta_zero_rate(self):
        """ETA在速率为0时为None"""
        bar = ProgressBar(total=100, output=io.StringIO())
        bar.start()
        # current为0时eta为None
        self.assertIsNone(bar.eta)

    def test14_rate(self):
        """速率"""
        bar = ProgressBar(total=100, output=io.StringIO())
        self.assertEqual(bar.rate, 0.0)
        bar.start()
        bar._current = 10
        time.sleep(0.01)
        self.assertGreater(bar.rate, 0)

    def test15_context_manager(self):
        """上下文管理器"""
        output = io.StringIO()
        with ProgressBar("测试", total=5, output=output) as bar:
            for _ in range(5):
                bar.update(1)
        self.assertTrue(bar.is_complete)

    def test16_context_manager_with_exception(self):
        """上下文管理器异常处理"""
        output = io.StringIO()
        bar = ProgressBar("测试", total=10, output=output)
        with self.assertRaises(ValueError):
            with bar:
                raise ValueError("test")
        self.assertFalse(bar._running)

    def test17_unicode_mode(self):
        """Unicode模式"""
        bar = ProgressBar(total=10, output=io.StringIO(), use_unicode=True)
        bar.start()
        bar.update(1)
        # 不报错即可
        self.assertTrue(True)

    def test18_ascii_mode(self):
        """ASCII模式"""
        bar = ProgressBar(total=10, output=io.StringIO(), use_unicode=False)
        bar.start()
        bar.update(1)
        self.assertTrue(True)

    def test19_format_time(self):
        """时间格式化"""
        self.assertEqual(ProgressBar._format_time(30), "30s")
        self.assertEqual(ProgressBar._format_time(90), "1.5m")
        self.assertEqual(ProgressBar._format_time(3700), "1.0h")

    def test20_render_output(self):
        """渲染输出"""
        output = io.StringIO()
        bar = ProgressBar("test", total=10, output=output)
        bar.start()
        bar.update(5)
        content = output.getvalue()
        self.assertIn("test", content)
        self.assertIn("50.0", content)


class TestSpinner(unittest.TestCase):
    """测试Spinner"""

    def test01_init_basic(self):
        """基本初始化"""
        sp = Spinner("加载中")
        self.assertEqual(sp.message, "加载中")
        self.assertFalse(sp._running)

    def test02_init_defaults(self):
        """默认值"""
        sp = Spinner()
        self.assertEqual(sp.message, "")
        self.assertEqual(sp.interval, 0.1)

    def test03_unicode_chars(self):
        """Unicode字符集"""
        sp = Spinner(use_unicode=True)
        self.assertEqual(sp._chars, SPINNER_CHARS)

    def test04_ascii_chars(self):
        """ASCII字符集"""
        sp = Spinner(use_unicode=False)
        self.assertEqual(sp._chars, SPINNER_CHARS_ASCII)

    def test05_start_stop(self):
        """start/stop"""
        sp = Spinner("test", output=io.StringIO(), interval=0.01)
        sp.start()
        self.assertTrue(sp._running)
        time.sleep(0.05)
        sp.stop()
        self.assertFalse(sp._running)

    def test06_succeed(self):
        """succeed"""
        output = io.StringIO()
        sp = Spinner("test", output=output, interval=0.01)
        sp.start()
        time.sleep(0.02)
        sp.succeed("完成")
        content = output.getvalue()
        self.assertIn("完成", content)

    def test07_fail(self):
        """fail"""
        output = io.StringIO()
        sp = Spinner("test", output=output, interval=0.01)
        sp.start()
        time.sleep(0.02)
        sp.fail("失败")
        content = output.getvalue()
        self.assertIn("失败", content)

    def test08_warn(self):
        """warn"""
        output = io.StringIO()
        sp = Spinner("test", output=output, interval=0.01)
        sp.start()
        time.sleep(0.02)
        sp.warn("警告")
        content = output.getvalue()
        self.assertIn("警告", content)

    def test09_context_manager_success(self):
        """上下文管理器成功"""
        output = io.StringIO()
        with Spinner("test", output=output, interval=0.01):
            time.sleep(0.02)
        self.assertTrue(True)

    def test10_context_manager_failure(self):
        """上下文管理器失败"""
        output = io.StringIO()
        with self.assertRaises(ValueError):
            with Spinner("test", output=output, interval=0.01):
                raise ValueError("err")

    def test11_succeed_default_message(self):
        """succeed默认消息"""
        output = io.StringIO()
        sp = Spinner("默认", output=output, interval=0.01)
        sp.start()
        time.sleep(0.02)
        sp.succeed()
        content = output.getvalue()
        self.assertIn("默认", content)


class TestStatusIndicator(unittest.TestCase):
    """测试StatusIndicator"""

    def test01_init_basic(self):
        """基本初始化"""
        si = StatusIndicator("任务1")
        self.assertEqual(si.name, "任务1")
        self.assertEqual(si.status, "idle")

    def test02_status_constants(self):
        """状态常量"""
        self.assertEqual(StatusIndicator.RUNNING, "running")
        self.assertEqual(StatusIndicator.PAUSED, "paused")
        self.assertEqual(StatusIndicator.COMPLETED, "completed")
        self.assertEqual(StatusIndicator.ERROR, "error")
        self.assertEqual(StatusIndicator.WARNING, "warning")
        self.assertEqual(StatusIndicator.IDLE, "idle")
        self.assertEqual(StatusIndicator.PENDING, "pending")

    def test03_set_running(self):
        """set_running"""
        si = StatusIndicator("test")
        si.set_running("处理中")
        self.assertTrue(si.is_running)
        self.assertEqual(si._message, "处理中")

    def test04_set_paused(self):
        """set_paused"""
        si = StatusIndicator("test")
        si.set_paused()
        self.assertEqual(si.status, "paused")

    def test05_set_completed(self):
        """set_completed"""
        si = StatusIndicator("test")
        si.set_completed()
        self.assertTrue(si.is_completed)

    def test06_set_error(self):
        """set_error"""
        si = StatusIndicator("test")
        si.set_error("出错了")
        self.assertTrue(si.is_error)
        self.assertEqual(si._message, "出错了")

    def test07_set_warning(self):
        """set_warning"""
        si = StatusIndicator("test")
        si.set_warning()
        self.assertEqual(si.status, "warning")

    def test08_set_idle(self):
        """set_idle"""
        si = StatusIndicator("test")
        si.set_idle()
        self.assertEqual(si.status, "idle")

    def test09_set_pending(self):
        """set_pending"""
        si = StatusIndicator("test")
        si.set_pending()
        self.assertEqual(si.status, "pending")

    def test10_set_progress(self):
        """set_progress"""
        si = StatusIndicator("test")
        si.set_progress(50)
        self.assertEqual(si._progress, 50)
        si.set_progress(150)
        self.assertEqual(si._progress, 100)
        si.set_progress(-10)
        self.assertEqual(si._progress, 0)

    def test11_get_icon_unicode(self):
        """Unicode图标"""
        si = StatusIndicator("test", use_unicode=True)
        si.set_running()
        self.assertEqual(si.get_icon(), STATUS_ICONS["running"])
        si.set_completed()
        self.assertEqual(si.get_icon(), STATUS_ICONS["completed"])

    def test12_get_icon_ascii(self):
        """ASCII图标"""
        si = StatusIndicator("test", use_unicode=False)
        si.set_running()
        self.assertEqual(si.get_icon(), STATUS_ICONS_ASCII["running"])

    def test13_format(self):
        """format"""
        si = StatusIndicator("任务", use_unicode=True)
        si.set_running("处理中")
        si.set_progress(50)
        text = si.format()
        self.assertIn("任务", text)
        self.assertIn("处理中", text)
        self.assertIn("50.0", text)

    def test14_elapsed_time(self):
        """elapsed_time"""
        si = StatusIndicator("test")
        self.assertEqual(si.elapsed_time, 0.0)
        si.set_running()
        time.sleep(0.01)
        self.assertGreater(si.elapsed_time, 0)

    def test15_to_dict(self):
        """to_dict"""
        si = StatusIndicator("任务", initial_status="running")
        si.set_progress(30)
        d = si.to_dict()
        self.assertEqual(d["name"], "任务")
        self.assertEqual(d["status"], "running")
        self.assertEqual(d["progress"], 30)

    def test16_all_status_icons(self):
        """所有状态都有图标"""
        for status_name in ["running", "paused", "completed", "error",
                            "warning", "idle", "pending"]:
            self.assertIn(status_name, STATUS_ICONS)
            self.assertIn(status_name, STATUS_ICONS_ASCII)


class TestTaskProgress(unittest.TestCase):
    """测试TaskProgress"""

    def test01_init_basic(self):
        """基本初始化"""
        tp = TaskProgress()
        self.assertEqual(tp.total_tasks, 0)
        self.assertEqual(tp.completed_tasks, 0)
        self.assertEqual(tp.overall_progress, 0.0)

    def test02_add_task(self):
        """add_task"""
        tp = TaskProgress()
        tid = tp.add_task("任务1", total=100)
        self.assertEqual(tid, 0)
        self.assertEqual(tp.total_tasks, 1)
        tid2 = tp.add_task("任务2", total=50)
        self.assertEqual(tid2, 1)

    def test03_start_task(self):
        """start_task"""
        tp = TaskProgress()
        tid = tp.add_task("任务1")
        tp.start_task(tid)
        task = tp.get_task(tid)
        self.assertEqual(task.status, StatusIndicator.RUNNING)

    def test04_update_task(self):
        """update_task"""
        tp = TaskProgress()
        tid = tp.add_task("任务1", total=10)
        tp.update_task(tid, 5)
        self.assertEqual(tp._task_currents[tid], 5)

    def test05_update_task_auto_start(self):
        """update自动开始任务"""
        tp = TaskProgress()
        tid = tp.add_task("任务1", total=10)
        tp.update_task(tid, 1)
        task = tp.get_task(tid)
        self.assertEqual(task.status, StatusIndicator.RUNNING)

    def test06_update_task_overflow(self):
        """update不会超过total"""
        tp = TaskProgress()
        tid = tp.add_task("任务1", total=5)
        tp.update_task(tid, 10)
        self.assertEqual(tp._task_currents[tid], 5)

    def test07_complete_task(self):
        """complete_task"""
        tp = TaskProgress()
        tid = tp.add_task("任务1", total=10)
        tp.complete_task(tid)
        task = tp.get_task(tid)
        self.assertTrue(task.is_completed)
        self.assertEqual(tp.completed_tasks, 1)

    def test08_fail_task(self):
        """fail_task"""
        tp = TaskProgress()
        tid = tp.add_task("任务1")
        tp.fail_task(tid, "出错了")
        task = tp.get_task(tid)
        self.assertTrue(task.is_error)

    def test09_get_task_invalid(self):
        """获取不存在的任务"""
        tp = TaskProgress()
        self.assertIsNone(tp.get_task(999))

    def test10_get_all_tasks(self):
        """get_all_tasks"""
        tp = TaskProgress()
        tp.add_task("任务1")
        tp.add_task("任务2")
        all_tasks = tp.get_all_tasks()
        self.assertEqual(len(all_tasks), 2)

    def test11_overall_progress(self):
        """overall_progress"""
        tp = TaskProgress()
        t1 = tp.add_task("任务1", total=100)
        t2 = tp.add_task("任务2", total=100)
        tp.update_task(t1, 50)
        tp.update_task(t2, 50)
        # 平均进度50%
        self.assertAlmostEqual(tp.overall_progress, 50.0, places=1)

    def test12_overall_progress_empty(self):
        """空任务列表的overall_progress"""
        tp = TaskProgress()
        self.assertEqual(tp.overall_progress, 0.0)

    def test13_completed_tasks_count(self):
        """completed_tasks计数"""
        tp = TaskProgress()
        t1 = tp.add_task("任务1")
        t2 = tp.add_task("任务2")
        t3 = tp.add_task("任务3")
        tp.complete_task(t1)
        tp.complete_task(t2)
        self.assertEqual(tp.completed_tasks, 2)

    def test14_format(self):
        """format"""
        tp = TaskProgress()
        t1 = tp.add_task("任务1", total=10)
        tp.start_task(t1)
        tp.update_task(t1, 5)
        text = tp.format()
        self.assertIn("任务1", text)
        self.assertIn("0/1", text)  # 0完成，1总数（任务还在running）

    def test15_to_dict(self):
        """to_dict"""
        tp = TaskProgress()
        t1 = tp.add_task("任务1", total=10)
        tp.update_task(t1, 5)
        d = tp.to_dict()
        self.assertEqual(d["total_tasks"], 1)
        self.assertIn("tasks", d)
        self.assertIn("0", d["tasks"])

    def test16_thread_safety(self):
        """线程安全（多任务并发添加）"""
        import threading
        tp = TaskProgress()

        def add_tasks():
            for i in range(10):
                tp.add_task(f"任务{i}")

        threads = [threading.Thread(target=add_tasks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(tp.total_tasks, 30)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def test01_progress_bar(self):
        """progress_bar函数"""
        bar = progress_bar("测试", 100)
        self.assertIsInstance(bar, ProgressBar)
        self.assertEqual(bar.description, "测试")
        self.assertEqual(bar.total, 100)

    def test02_progress_bar_unicode(self):
        """progress_bar Unicode选项"""
        bar = progress_bar("测试", 100, use_unicode=False)
        self.assertFalse(bar.use_unicode)

    def test03_spinner(self):
        """spinner函数"""
        sp = spinner("测试")
        self.assertIsInstance(sp, Spinner)
        self.assertEqual(sp.message, "测试")

    def test04_status(self):
        """status函数"""
        si = status("任务", initial_status="running")
        self.assertIsInstance(si, StatusIndicator)
        self.assertEqual(si.name, "任务")
        self.assertEqual(si.status, "running")

    def test05_track_progress(self):
        """track_progress"""
        items = list(range(10))
        result = []
        for item in track_progress(items, "处理", use_unicode=False):
            result.append(item)
        self.assertEqual(result, items)

    def test06_track_progress_with_total(self):
        """track_progress指定total"""
        items = list(range(5))
        result = list(track_progress(items, "处理", total=5,
                                     use_unicode=False))
        self.assertEqual(len(result), 5)

    def test07_track_progress_async(self):
        """track_progress_async"""
        items = list(range(5))
        result = list(track_progress_async(items, "处理"))
        self.assertEqual(len(result), 5)

    def test08_format_status(self):
        """format_status"""
        icon = format_status("running")
        self.assertEqual(icon, STATUS_ICONS["running"])
        icon = format_status("completed", use_unicode=False)
        self.assertEqual(icon, STATUS_ICONS_ASCII["completed"])

    def test09_format_status_unknown(self):
        """format_status未知状态"""
        icon = format_status("unknown")
        self.assertEqual(icon, "?")

    def test10_get_progress_report(self):
        """get_progress_report"""
        tp = TaskProgress()
        t1 = tp.add_task("任务1", total=10)
        tp.update_task(t1, 5)
        report = get_progress_report(tp)
        self.assertIn("任务1", report)


class TestModuleExports(unittest.TestCase):
    """测试模块导出"""

    def test01_all_exports(self):
        """__all__包含所有导出"""
        from seva.progress import __all__
        expected = [
            "ProgressBar", "Spinner", "StatusIndicator", "TaskProgress",
            "SPINNER_CHARS", "SPINNER_CHARS_ASCII",
            "STATUS_ICONS", "STATUS_ICONS_ASCII",
            "progress_bar", "spinner", "status",
            "track_progress", "track_progress_async",
            "format_status", "get_progress_report",
        ]
        for name in expected:
            self.assertIn(name, __all__)

    def test02_seva_exports(self):
        """seva包导出"""
        self.assertTrue(hasattr(seva, "ProgressBar"))
        self.assertTrue(hasattr(seva, "Spinner"))
        self.assertTrue(hasattr(seva, "StatusIndicator"))
        self.assertTrue(hasattr(seva, "TaskProgress"))
        self.assertTrue(hasattr(seva, "progress_bar"))
        self.assertTrue(hasattr(seva, "track_progress"))
        self.assertTrue(hasattr(seva, "format_status"))

    def test03_version(self):
        """版本号"""
        self.assertGreaterEqual(seva.__version__, "0.49.0")

    def test04_constants(self):
        """常量"""
        self.assertEqual(len(SPINNER_CHARS), 10)
        self.assertEqual(len(SPINNER_CHARS_ASCII), 4)
        self.assertEqual(len(STATUS_ICONS), 7)
        self.assertEqual(len(STATUS_ICONS_ASCII), 7)

    def test05_spinner_chars_unique(self):
        """Spinner字符唯一"""
        self.assertEqual(len(set(SPINNER_CHARS)), len(SPINNER_CHARS))

    def test06_status_icons_keys(self):
        """状态图标键完整"""
        expected_keys = {"running", "paused", "completed", "error",
                         "warning", "idle", "pending"}
        self.assertEqual(set(STATUS_ICONS.keys()), expected_keys)
        self.assertEqual(set(STATUS_ICONS_ASCII.keys()), expected_keys)


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test01_cmd_progress_default(self):
        """CLI progress默认"""
        from seva.cli import main
        try:
            ret = main(["progress"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test02_cmd_progress_status_list(self):
        """CLI progress --status-list"""
        from seva.cli import main
        try:
            ret = main(["progress", "--status-list"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test03_cmd_progress_status_list_ascii(self):
        """CLI progress --status-list --ascii"""
        from seva.cli import main
        try:
            ret = main(["progress", "--status-list", "--ascii"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test04_cmd_progress_track(self):
        """CLI progress --track"""
        from seva.cli import main
        try:
            ret = main(["progress", "--track", "--total", "5", "--ascii"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test05_cmd_progress_tasks(self):
        """CLI progress --tasks"""
        from seva.cli import main
        try:
            ret = main(["progress", "--tasks", "--total", "3", "--ascii"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test06_cmd_progress_demo(self):
        """CLI progress --demo"""
        from seva.cli import main
        try:
            ret = main(["progress", "--demo", "--ascii"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test07_progress_parser_exists(self):
        """progress子解析器存在"""
        from seva.cli import build_parser
        parser = build_parser()
        # 解析progress命令
        args = parser.parse_args(["progress", "--demo"])
        self.assertEqual(args.command, "progress")
        self.assertTrue(args.demo)
        self.assertTrue(hasattr(args, "func"))


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test01_progress_bar_negative_total(self):
        """负数total"""
        bar = ProgressBar(total=-10)
        self.assertEqual(bar.total, 1)

    def test02_progress_bar_zero_width(self):
        """宽度0"""
        bar = ProgressBar(total=10, width=0, output=io.StringIO())
        bar.start()
        bar.update(1)
        # 不报错即可
        self.assertTrue(True)

    def test03_status_indicator_unknown_status_icon(self):
        """未知状态图标"""
        si = StatusIndicator("test")
        si._status = "unknown_status"
        icon = si.get_icon()
        self.assertEqual(icon, "?")

    def test04_task_progress_update_invalid_id(self):
        """更新无效任务ID"""
        tp = TaskProgress()
        # 不应抛出异常
        tp.update_task(999, 5)
        tp.complete_task(999)
        tp.fail_task(999)
        tp.start_task(999)
        self.assertTrue(True)

    def test05_progress_bar_render_no_output(self):
        """渲染到关闭的输出"""
        output = io.StringIO()
        output.close()
        bar = ProgressBar(total=10, output=output)
        bar.start()
        bar.update(1)
        # 不应抛出异常
        self.assertTrue(True)

    def test06_track_progress_empty(self):
        """空迭代"""
        result = list(track_progress([], "处理", use_unicode=False))
        self.assertEqual(result, [])

    def test07_spinner_stop_without_start(self):
        """未启动就stop"""
        sp = Spinner("test", output=io.StringIO())
        sp.stop("done")
        self.assertTrue(True)

    def test08_task_progress_zero_total(self):
        """任务total为0"""
        tp = TaskProgress()
        tid = tp.add_task("任务", total=0)
        tp.update_task(tid, 1)
        # 不应抛出异常
        self.assertTrue(True)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test01_full_workflow(self):
        """完整工作流"""
        output = io.StringIO()
        # 创建进度条
        bar = ProgressBar("处理", total=10, output=output)
        bar.start()
        for i in range(10):
            bar.update(1)
        bar.finish()
        self.assertTrue(bar.is_complete)
        self.assertEqual(bar.current, 10)

    def test02_multi_task_workflow(self):
        """多任务工作流"""
        tp = TaskProgress(use_unicode=False)
        # 添加3个任务
        t1 = tp.add_task("下载", total=10)
        t2 = tp.add_task("解析", total=10)
        t3 = tp.add_task("保存", total=10)

        # 完成第一个
        tp.start_task(t1)
        for _ in range(10):
            tp.update_task(t1)
        tp.complete_task(t1)
        self.assertEqual(tp.completed_tasks, 1)

        # 第二个失败
        tp.start_task(t2)
        tp.update_task(t2, 5)
        tp.fail_task(t2, "解析错误")
        task2 = tp.get_task(t2)
        self.assertTrue(task2.is_error)

        # 第三个完成
        tp.start_task(t3)
        for _ in range(10):
            tp.update_task(t3)
        tp.complete_task(t3)
        self.assertEqual(tp.completed_tasks, 2)

    def test03_status_indicator_lifecycle(self):
        """状态指示器生命周期"""
        si = StatusIndicator("任务", use_unicode=True)
        # pending -> running -> completed
        si.set_pending("等待开始")
        self.assertEqual(si.status, "pending")
        si.set_running("处理中")
        self.assertTrue(si.is_running)
        si.set_completed("完成")
        self.assertTrue(si.is_completed)
        self.assertGreater(si.elapsed_time, 0)

    def test04_progress_bar_with_spinner(self):
        """进度条与Spinner组合"""
        output = io.StringIO()
        # 先用Spinner等待
        with Spinner("初始化...", output=output, interval=0.01):
            time.sleep(0.02)
        # 再用ProgressBar处理
        bar = ProgressBar("处理", total=5, output=output)
        bar.start()
        for _ in range(5):
            bar.update(1)
        bar.finish()
        self.assertTrue(bar.is_complete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
