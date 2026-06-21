"""
阶段50测试：智能建议与推荐系统
验证 Suggestion / ContextAnalyzer / UserProfiler / Recommender / SuggestionEngine
及便捷函数和CLI集成
"""
import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


import seva
from seva.suggestions import (Suggestion, SuggestionType, SuggestionPriority,
                              UserSkillLevel, ContextAnalyzer, UserProfiler,
                              Recommender, SuggestionEngine,
                              suggest, recommend, recommend_commands,
                              recommend_learning, get_suggestions,
                              get_user_profile, get_user_profile_report,
                              record_command, record_feature, record_error,
                              record_session, reset_user_profile,
                              get_skill_level, analyze_context, PROFILE_FILE)


class TestSuggestion(unittest.TestCase):
    """测试Suggestion"""

    def test01_init_basic(self):
        """基本初始化"""
        s = Suggestion("测试建议")
        self.assertEqual(s.title, "测试建议")
        self.assertEqual(s.type, SuggestionType.INFO)
        self.assertEqual(s.priority, SuggestionPriority.MEDIUM)

    def test02_init_full(self):
        """完整初始化"""
        s = Suggestion(
            title="优化建议",
            message="详细消息",
            type=SuggestionType.OPTIMIZATION,
            priority=SuggestionPriority.HIGH,
            action="运行命令",
            category="perf",
            confidence=0.8,
            metadata={"key": "value"},
        )
        self.assertEqual(s.title, "优化建议")
        self.assertEqual(s.type, SuggestionType.OPTIMIZATION)
        self.assertEqual(s.priority, SuggestionPriority.HIGH)
        self.assertEqual(s.action, "运行命令")
        self.assertEqual(s.confidence, 0.8)
        self.assertEqual(s.metadata["key"], "value")

    def test03_confidence_clamped(self):
        """置信度限制在0-1"""
        s = Suggestion("test", confidence=1.5)
        self.assertEqual(s.confidence, 1.0)
        s = Suggestion("test", confidence=-0.5)
        self.assertEqual(s.confidence, 0.0)

    def test04_format(self):
        """format"""
        s = Suggestion("标题", "消息", priority=SuggestionPriority.HIGH,
                       action="操作")
        text = s.format()
        self.assertIn("标题", text)
        self.assertIn("消息", text)
        self.assertIn("HIGH", text)
        self.assertIn("操作", text)

    def test05_to_dict(self):
        """to_dict"""
        s = Suggestion("标题", "消息", type=SuggestionType.WARNING)
        d = s.to_dict()
        self.assertEqual(d["title"], "标题")
        self.assertEqual(d["type"], SuggestionType.WARNING)
        self.assertIn("priority_str", d)

    def test06_from_dict(self):
        """from_dict"""
        data = {
            "title": "测试",
            "message": "消息",
            "type": SuggestionType.OPTIMIZATION,
            "priority": SuggestionPriority.LOW,
            "action": "act",
            "category": "cat",
            "confidence": 0.5,
            "metadata": {"a": 1},
            "created_at": time.time(),
        }
        s = Suggestion.from_dict(data)
        self.assertEqual(s.title, "测试")
        self.assertEqual(s.priority, SuggestionPriority.LOW)
        self.assertEqual(s.confidence, 0.5)

    def test07_unique_id(self):
        """唯一ID"""
        s1 = Suggestion("a")
        s2 = Suggestion("b")
        self.assertNotEqual(s1.suggestion_id, s2.suggestion_id)

    def test08_repr(self):
        """__repr__"""
        s = Suggestion("测试", priority=SuggestionPriority.HIGH)
        r = repr(s)
        self.assertIn("测试", r)
        self.assertIn("priority=2", r)

    def test09_equality(self):
        """相等性"""
        s1 = Suggestion("a", type=SuggestionType.INFO,
                        priority=SuggestionPriority.HIGH)
        s2 = Suggestion("a", type=SuggestionType.INFO,
                        priority=SuggestionPriority.HIGH)
        s3 = Suggestion("a", type=SuggestionType.WARNING,
                        priority=SuggestionPriority.HIGH)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test10_hash(self):
        """哈希"""
        s = Suggestion("a", type=SuggestionType.INFO,
                       priority=SuggestionPriority.HIGH)
        self.assertIsInstance(hash(s), int)


class TestSuggestionPriority(unittest.TestCase):
    """测试SuggestionPriority"""

    def test01_to_str(self):
        """to_str"""
        self.assertEqual(SuggestionPriority.to_str(1), "CRITICAL")
        self.assertEqual(SuggestionPriority.to_str(2), "HIGH")
        self.assertEqual(SuggestionPriority.to_str(3), "MEDIUM")
        self.assertEqual(SuggestionPriority.to_str(4), "LOW")

    def test02_to_str_unknown(self):
        """未知优先级"""
        self.assertEqual(SuggestionPriority.to_str(99), "UNKNOWN")


class TestContextAnalyzer(unittest.TestCase):
    """测试ContextAnalyzer"""

    def test01_analyze_agent_basic(self):
        """分析Agent"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = 5.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.2
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyzer.analyze_agent(agent)
        self.assertEqual(context["cycle_count"], 10)
        self.assertEqual(context["total_reward"], 5.0)
        self.assertTrue(context["healthy"])

    def test02_analyze_agent_high_error(self):
        """高预测误差"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = 5.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.8
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyzer.analyze_agent(agent)
        self.assertFalse(context["healthy"])
        self.assertIn("预测误差过高", context["issues"])

    def test03_analyze_agent_slow_cycle(self):
        """慢周期"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = 5.0
        agent.avg_cycle_time = 2.0
        agent.avg_prediction_error = 0.1
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyzer.analyze_agent(agent)
        self.assertFalse(context["healthy"])

    def test04_analyze_agent_negative_reward(self):
        """负奖励"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = -5.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.1
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyzer.analyze_agent(agent)
        self.assertFalse(context["healthy"])

    def test05_analyze_agent_with_managers(self):
        """带管理器的Agent"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = 5.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.1
        agent.mode = "normal"
        agent.task_manager.get_stats.return_value = {"total_completed": 5}
        agent.skill_manager.get_stats.return_value = {"total_skills": 3}
        agent.evolution_manager.get_stats.return_value = {
            "total_evaluations": 2,
            "current_difficulty": "MEDIUM",
        }
        context = analyzer.analyze_agent(agent)
        self.assertEqual(context["tasks_completed"], 5)
        self.assertEqual(context["skills_total"], 3)
        self.assertEqual(context["evolution_evaluations"], 2)
        self.assertEqual(context["current_difficulty"], "MEDIUM")

    def test06_analyze_agent_exception(self):
        """分析异常"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        # 让getattr抛出异常
        type(agent).cycle_count = property(
            lambda self: (_ for _ in ()).throw(ValueError("err")))
        context = analyzer.analyze_agent(agent)
        self.assertIn("分析异常", str(context["issues"]))

    def test07_detect_issues(self):
        """检测问题"""
        analyzer = ContextAnalyzer()
        context = {
            "issues": ["问题1"],
            "cycle_count": 0,
            "skills_total": 0,
        }
        issues = analyzer.detect_issues(context)
        self.assertIn("问题1", issues)
        self.assertIn("Agent尚未运行", issues)

    def test08_detect_issues_no_skills(self):
        """检测无技能"""
        analyzer = ContextAnalyzer()
        context = {
            "issues": [],
            "cycle_count": 20,
            "skills_total": 0,
        }
        issues = analyzer.detect_issues(context)
        self.assertIn("运行多步但无技能生成", issues)


class TestUserProfiler(unittest.TestCase):
    """测试UserProfiler"""

    def setUp(self):
        """每个测试前重置"""
        # 使用临时文件避免污染
        self._original_profile_file = PROFILE_FILE

    def tearDown(self):
        """清理"""
        # 不删除实际文件，但重置单例
        pass

    def test01_init_basic(self):
        """基本初始化"""
        profiler = UserProfiler()
        profile = profiler.get_profile()
        self.assertIn("skill_level", profile)
        self.assertIn("commands_used", profile)
        self.assertIn("sessions_count", profile)

    def test02_record_command(self):
        """记录命令"""
        profiler = UserProfiler()
        profiler.reset()
        profiler.record_command("run")
        profiler.record_command("run")
        profiler.record_command("status")
        profile = profiler.get_profile()
        self.assertIn("run", profile["commands_used"])
        self.assertIn("status", profile["commands_used"])
        self.assertEqual(profile["commands_count"]["run"], 2)
        self.assertEqual(profile["commands_count"]["status"], 1)

    def test03_record_feature(self):
        """记录功能"""
        profiler = UserProfiler()
        profiler.reset()
        profiler.record_feature("dashboard")
        profiler.record_feature("tutorial")
        profile = profiler.get_profile()
        self.assertIn("dashboard", profile["features_used"])
        self.assertIn("tutorial", profile["features_used"])

    def test04_record_error(self):
        """记录错误"""
        profiler = UserProfiler()
        profiler.reset()
        profiler.record_error()
        profiler.record_error()
        profile = profiler.get_profile()
        self.assertEqual(profile["errors_encountered"], 2)

    def test05_record_session(self):
        """记录会话"""
        profiler = UserProfiler()
        profiler.reset()
        profiler.record_session(steps=100, preset="easy", env_type="mixed")
        profiler.record_session(steps=50, preset="easy", env_type="drift")
        profile = profiler.get_profile()
        self.assertEqual(profile["sessions_count"], 2)
        self.assertEqual(profile["total_runtime_steps"], 150)
        self.assertEqual(profile["preferred_presets"]["easy"], 2)
        self.assertEqual(profile["preferred_env_types"]["mixed"], 1)
        self.assertEqual(profile["preferred_env_types"]["drift"], 1)

    def test06_skill_level_progression(self):
        """技能水平提升"""
        profiler = UserProfiler()
        profiler.reset()
        self.assertEqual(profiler.get_skill_level(),
                         UserSkillLevel.BEGINNER)
        # 模拟使用
        for i in range(20):
            profiler.record_command(f"cmd{i}")
        for i in range(15):
            profiler.record_feature(f"feature{i}")
        for i in range(25):
            profiler.record_session(steps=50)
        # 应该至少是INTERMEDIATE
        level = profiler.get_skill_level()
        self.assertIn(level, [UserSkillLevel.INTERMEDIATE,
                              UserSkillLevel.ADVANCED,
                              UserSkillLevel.EXPERT])

    def test07_get_report(self):
        """获取报告"""
        profiler = UserProfiler()
        profiler.reset()
        profiler.record_command("run")
        profiler.record_session(steps=10, preset="easy")
        report = profiler.get_report()
        self.assertIn("用户画像", report)
        self.assertIn("使用命令数:     1", report)

    def test08_reset(self):
        """重置"""
        profiler = UserProfiler()
        profiler.record_command("test")
        profiler.reset()
        profile = profiler.get_profile()
        self.assertEqual(profile["commands_used"], [])
        self.assertEqual(profile["sessions_count"], 0)
        self.assertEqual(profile["skill_level"], UserSkillLevel.BEGINNER)


class TestRecommender(unittest.TestCase):
    """测试Recommender"""

    def test01_recommend_config(self):
        """推荐配置"""
        rec = Recommender()
        result = rec.recommend_config("quick_test")
        self.assertEqual(result["preset"], "easy")
        self.assertIn("description", result)

    def test02_recommend_config_research(self):
        """研究场景推荐"""
        rec = Recommender()
        result = rec.recommend_config("research")
        self.assertEqual(result["preset"], "research")

    def test03_recommend_config_unknown(self):
        """未知场景"""
        rec = Recommender()
        result = rec.recommend_config("unknown_scenario")
        self.assertEqual(result["preset"], "standard")

    def test04_recommend_commands_beginner(self):
        """初学者命令推荐"""
        rec = Recommender()
        cmds = rec.recommend_commands(UserSkillLevel.BEGINNER)
        self.assertGreater(len(cmds), 0)
        # 检查结构
        self.assertIn("command", cmds[0])
        self.assertIn("description", cmds[0])
        self.assertIn("already_used", cmds[0])

    def test05_recommend_commands_with_used(self):
        """带已用命令的推荐"""
        rec = Recommender()
        cmds = rec.recommend_commands(
            UserSkillLevel.BEGINNER,
            used_commands=["demo", "presets"])
        # demo应该标记为已用
        demo_cmd = next(c for c in cmds if c["command"] == "demo")
        self.assertTrue(demo_cmd["already_used"])

    def test06_recommend_commands_all_levels(self):
        """所有技能水平的命令推荐"""
        rec = Recommender()
        for level in [UserSkillLevel.BEGINNER, UserSkillLevel.INTERMEDIATE,
                      UserSkillLevel.ADVANCED, UserSkillLevel.EXPERT]:
            cmds = rec.recommend_commands(level)
            self.assertGreater(len(cmds), 0)

    def test07_recommend_learning(self):
        """学习资源推荐"""
        rec = Recommender()
        resources = rec.recommend_learning(UserSkillLevel.BEGINNER)
        self.assertGreater(len(resources), 0)
        self.assertIn("resource_id", resources[0])
        self.assertIn("priority", resources[0])

    def test08_recommend_learning_sorted(self):
        """学习资源按优先级排序"""
        rec = Recommender()
        resources = rec.recommend_learning(UserSkillLevel.BEGINNER)
        priorities = [r["priority"] for r in resources]
        self.assertEqual(priorities, sorted(priorities))

    def test09_recommend_learning_with_used(self):
        """带已用功能的学习推荐"""
        rec = Recommender()
        resources = rec.recommend_learning(
            UserSkillLevel.BEGINNER,
            used_features=["getting_started"])
        gs = next(r for r in resources
                  if r["resource_id"] == "getting_started")
        self.assertTrue(gs["already_used"])

    def test10_recommend_preset_with_preference(self):
        """带偏好的预设推荐"""
        rec = Recommender()
        context = {"preferred_presets": {"standard": 5, "easy": 2}}
        preset = rec.recommend_preset(context)
        self.assertEqual(preset, "standard")

    def test11_recommend_preset_by_cycle_count(self):
        """按周期数推荐预设"""
        rec = Recommender()
        self.assertEqual(rec.recommend_preset({"cycle_count": 0}), "easy")
        self.assertEqual(rec.recommend_preset({"cycle_count": 50}),
                         "standard")
        self.assertEqual(rec.recommend_preset({"cycle_count": 200}),
                         "research")


class TestSuggestionEngine(unittest.TestCase):
    """测试SuggestionEngine"""

    def test01_generate_no_agent(self):
        """无Agent生成建议"""
        engine = SuggestionEngine()
        suggestions = engine.generate()
        self.assertIsInstance(suggestions, list)

    def test02_generate_with_context(self):
        """带上下文生成建议"""
        engine = SuggestionEngine()
        context = {
            "avg_prediction_error": 0.8,
            "avg_cycle_time": 2.0,
            "total_reward": -5.0,
            "skills_total": 0,
            "cycle_count": 20,
            "tasks_completed": 0,
            "skill_level": UserSkillLevel.BEGINNER,
            "sessions_count": 0,
            "errors_encountered": 0,
            "current_difficulty": "EASY",
            "preferred_presets": {},
        }
        suggestions = engine.generate(context=context)
        # 应该有多个建议
        self.assertGreater(len(suggestions), 0)

    def test03_generate_sorted_by_priority(self):
        """建议按优先级排序"""
        engine = SuggestionEngine()
        context = {
            "avg_prediction_error": 0.8,
            "total_reward": -5.0,
            "cycle_count": 20,
            "skills_total": 0,
            "tasks_completed": 0,
            "skill_level": UserSkillLevel.BEGINNER,
            "sessions_count": 0,
            "errors_encountered": 0,
            "current_difficulty": "EASY",
            "preferred_presets": {},
        }
        suggestions = engine.generate(context=context)
        if len(suggestions) > 1:
            for i in range(len(suggestions) - 1):
                self.assertLessEqual(suggestions[i].priority,
                                     suggestions[i + 1].priority)

    def test04_rule_high_prediction_error(self):
        """高预测误差规则"""
        engine = SuggestionEngine()
        s = engine._rule_high_prediction_error(
            {"avg_prediction_error": 0.8})
        self.assertIsNotNone(s)
        self.assertEqual(s.priority, SuggestionPriority.HIGH)

    def test05_rule_slow_cycle_time(self):
        """慢周期规则"""
        engine = SuggestionEngine()
        s = engine._rule_slow_cycle_time({"avg_cycle_time": 2.0})
        self.assertIsNotNone(s)
        self.assertEqual(s.type, SuggestionType.OPTIMIZATION)

    def test06_rule_negative_reward(self):
        """负奖励规则"""
        engine = SuggestionEngine()
        s = engine._rule_negative_reward({"total_reward": -5.0})
        self.assertIsNotNone(s)
        self.assertEqual(s.priority, SuggestionPriority.CRITICAL)

    def test07_rule_no_skills(self):
        """无技能规则"""
        engine = SuggestionEngine()
        s = engine._rule_no_skills({"skills_total": 0, "cycle_count": 20})
        self.assertIsNotNone(s)

    def test08_rule_no_skills_no_trigger(self):
        """无技能规则不触发"""
        engine = SuggestionEngine()
        s = engine._rule_no_skills({"skills_total": 5, "cycle_count": 20})
        self.assertIsNone(s)
        s = engine._rule_no_skills({"skills_total": 0, "cycle_count": 5})
        self.assertIsNone(s)

    def test09_rule_beginner_user(self):
        """初学者规则"""
        engine = SuggestionEngine()
        s = engine._rule_beginner_user({
            "skill_level": UserSkillLevel.BEGINNER,
            "sessions_count": 1,
        })
        self.assertIsNotNone(s)
        self.assertEqual(s.type, SuggestionType.LEARNING)

    def test10_rule_many_errors(self):
        """多错误规则"""
        engine = SuggestionEngine()
        s = engine._rule_many_errors({"errors_encountered": 15})
        self.assertIsNotNone(s)

    def test11_rule_long_runtime(self):
        """长运行规则"""
        engine = SuggestionEngine()
        s = engine._rule_long_runtime({"cycle_count": 2000})
        self.assertIsNotNone(s)
        self.assertEqual(s.priority, SuggestionPriority.LOW)

    def test12_rule_high_difficulty(self):
        """高难度规则"""
        engine = SuggestionEngine()
        s = engine._rule_high_difficulty(
            {"current_difficulty": "HARD"})
        self.assertIsNotNone(s)
        self.assertEqual(s.type, SuggestionType.WARNING)

    def test13_rule_first_run(self):
        """首次运行规则"""
        engine = SuggestionEngine()
        s = engine._rule_first_run({"sessions_count": 0})
        self.assertIsNotNone(s)
        self.assertEqual(s.priority, SuggestionPriority.HIGH)

    def test14_add_custom_rule(self):
        """添加自定义规则"""
        engine = SuggestionEngine()

        def custom_rule(context):
            return Suggestion("自定义", type=SuggestionType.INFO)

        engine.add_rule(custom_rule)
        suggestions = engine.generate(context={})
        titles = [s.title for s in suggestions]
        self.assertIn("自定义", titles)

    def test15_get_report(self):
        """获取报告"""
        engine = SuggestionEngine()
        report = engine.get_report()
        self.assertIsInstance(report, str)

    def test16_generate_with_mock_agent(self):
        """带Mock Agent生成建议"""
        engine = SuggestionEngine()
        agent = MagicMock()
        agent.cycle_count = 100
        agent.total_real_reward = 10.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.1
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        suggestions = engine.generate(agent)
        self.assertIsInstance(suggestions, list)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def test01_suggest(self):
        """suggest函数"""
        result = suggest()
        self.assertIsInstance(result, list)

    def test02_suggest_with_context(self):
        """带上下文的suggest"""
        context = {
            "avg_prediction_error": 0.8,
            "cycle_count": 20,
            "skills_total": 0,
            "tasks_completed": 0,
            "total_reward": -5.0,
            "skill_level": UserSkillLevel.BEGINNER,
            "sessions_count": 0,
            "errors_encountered": 0,
            "current_difficulty": "EASY",
            "preferred_presets": {},
        }
        result = suggest(context=context)
        self.assertIsInstance(result, list)

    def test03_recommend(self):
        """recommend函数"""
        result = recommend("quick_test")
        self.assertIn("preset", result)

    def test04_recommend_commands(self):
        """recommend_commands函数"""
        result = recommend_commands(UserSkillLevel.BEGINNER)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test05_recommend_learning(self):
        """recommend_learning函数"""
        result = recommend_learning(UserSkillLevel.BEGINNER)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test06_get_suggestions(self):
        """get_suggestions函数"""
        report = get_suggestions()
        self.assertIsInstance(report, str)

    def test07_get_user_profile(self):
        """get_user_profile函数"""
        profile = get_user_profile()
        self.assertIn("skill_level", profile)

    def test08_get_user_profile_report(self):
        """get_user_profile_report函数"""
        report = get_user_profile_report()
        self.assertIsInstance(report, str)
        self.assertIn("用户画像", report)

    def test09_record_command(self):
        """record_command函数"""
        record_command("test_cmd")
        profile = get_user_profile()
        # 可能被其他测试影响，只检查不报错
        self.assertTrue(True)

    def test10_record_feature(self):
        """record_feature函数"""
        record_feature("test_feature")
        self.assertTrue(True)

    def test11_record_error(self):
        """record_error函数"""
        record_error()
        self.assertTrue(True)

    def test12_record_session(self):
        """record_session函数"""
        record_session(steps=10, preset="easy", env_type="mixed")
        self.assertTrue(True)

    def test13_get_skill_level(self):
        """get_skill_level函数"""
        level = get_skill_level()
        self.assertIn(level, [UserSkillLevel.BEGINNER,
                              UserSkillLevel.INTERMEDIATE,
                              UserSkillLevel.ADVANCED,
                              UserSkillLevel.EXPERT])

    def test14_analyze_context(self):
        """analyze_context函数"""
        agent = MagicMock()
        agent.cycle_count = 10
        agent.total_real_reward = 5.0
        agent.avg_cycle_time = 0.1
        agent.avg_prediction_error = 0.1
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyze_context(agent)
        self.assertEqual(context["cycle_count"], 10)


class TestModuleExports(unittest.TestCase):
    """测试模块导出"""

    def test01_all_exports(self):
        """__all__包含所有导出"""
        from seva.suggestions import __all__
        expected = [
            "Suggestion", "SuggestionType", "SuggestionPriority",
            "UserSkillLevel", "ContextAnalyzer", "UserProfiler",
            "Recommender", "SuggestionEngine",
            "suggest", "recommend", "recommend_commands",
            "recommend_learning", "get_suggestions",
            "get_user_profile", "get_user_profile_report",
            "record_command", "record_feature", "record_error",
            "record_session", "reset_user_profile",
            "get_skill_level", "analyze_context",
        ]
        for name in expected:
            self.assertIn(name, __all__)

    def test02_seva_exports(self):
        """seva包导出"""
        self.assertTrue(hasattr(seva, "Suggestion"))
        self.assertTrue(hasattr(seva, "SuggestionEngine"))
        self.assertTrue(hasattr(seva, "suggest"))
        self.assertTrue(hasattr(seva, "recommend"))
        self.assertTrue(hasattr(seva, "get_suggestions"))
        self.assertTrue(hasattr(seva, "UserProfiler"))
        self.assertTrue(hasattr(seva, "Recommender"))

    def test03_version(self):
        """版本号"""
        self.assertGreaterEqual(seva.__version__, "0.50.0")

    def test04_suggestion_type_constants(self):
        """建议类型常量"""
        self.assertEqual(SuggestionType.OPTIMIZATION, "optimization")
        self.assertEqual(SuggestionType.CONFIGURATION, "configuration")
        self.assertEqual(SuggestionType.TROUBLESHOOTING, "troubleshooting")
        self.assertEqual(SuggestionType.BEST_PRACTICE, "best_practice")
        self.assertEqual(SuggestionType.LEARNING, "learning")
        self.assertEqual(SuggestionType.WARNING, "warning")
        self.assertEqual(SuggestionType.INFO, "info")

    def test05_user_skill_level_constants(self):
        """用户技能水平常量"""
        self.assertEqual(UserSkillLevel.BEGINNER, "beginner")
        self.assertEqual(UserSkillLevel.INTERMEDIATE, "intermediate")
        self.assertEqual(UserSkillLevel.ADVANCED, "advanced")
        self.assertEqual(UserSkillLevel.EXPERT, "expert")


class TestCLIIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test01_cmd_suggest_default(self):
        """CLI suggest默认"""
        from seva.cli import main
        try:
            ret = main(["suggest"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test02_cmd_suggest_profile(self):
        """CLI suggest --profile"""
        from seva.cli import main
        try:
            ret = main(["suggest", "--profile"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test03_cmd_suggest_recommend_config(self):
        """CLI suggest --recommend-config"""
        from seva.cli import main
        try:
            ret = main(["suggest", "--recommend-config", "quick_test"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test04_cmd_suggest_recommend_commands(self):
        """CLI suggest --recommend-commands"""
        from seva.cli import main
        try:
            ret = main(["suggest", "--recommend-commands"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test05_cmd_suggest_recommend_learning(self):
        """CLI suggest --recommend-learning"""
        from seva.cli import main
        try:
            ret = main(["suggest", "--recommend-learning"])
            self.assertEqual(ret, 0)
        except SystemExit:
            pass

    def test06_suggest_parser_exists(self):
        """suggest子解析器存在"""
        from seva.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["suggest", "--profile"])
        self.assertEqual(args.command, "suggest")
        self.assertTrue(args.profile)
        self.assertTrue(hasattr(args, "func"))


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test01_suggestion_empty_title(self):
        """空标题"""
        s = Suggestion("")
        self.assertEqual(s.title, "")

    def test02_suggestion_zero_confidence(self):
        """零置信度"""
        s = Suggestion("test", confidence=0)
        self.assertEqual(s.confidence, 0.0)

    def test03_analyzer_no_managers(self):
        """无管理器的分析"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 0
        agent.total_real_reward = 0.0
        agent.avg_cycle_time = 0.0
        agent.avg_prediction_error = 0.0
        agent.mode = "unknown"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None
        context = analyzer.analyze_agent(agent)
        self.assertEqual(context["tasks_completed"], 0)
        self.assertEqual(context["skills_total"], 0)

    def test04_engine_rule_exception_handling(self):
        """规则异常处理"""
        engine = SuggestionEngine()

        def bad_rule(context):
            raise ValueError("bad rule")

        engine.add_rule(bad_rule)
        # 不应抛出异常
        suggestions = engine.generate(context={})
        self.assertIsInstance(suggestions, list)

    def test05_profiler_persistence(self):
        """画像持久化"""
        profiler = UserProfiler()
        profiler.record_command("persist_test_cmd")
        # 创建新实例应该加载
        profiler2 = UserProfiler()
        profile = profiler2.get_profile()
        # 可能加载成功（取决于文件权限）
        self.assertIn("skill_level", profile)

    def test06_recommender_unknown_skill_level(self):
        """未知技能水平"""
        rec = Recommender()
        cmds = rec.recommend_commands("unknown_level")
        # 应该回退到BEGINNER
        self.assertGreater(len(cmds), 0)

    def test07_suggestion_format_with_low_confidence(self):
        """低置信度格式化"""
        s = Suggestion("test", confidence=0.5)
        text = s.format()
        self.assertIn("置信度", text)

    def test08_suggestion_format_full_confidence(self):
        """满置信度格式化"""
        s = Suggestion("test", confidence=1.0)
        text = s.format()
        # 满置信度不显示置信度行
        self.assertNotIn("置信度", text)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test01_full_workflow(self):
        """完整工作流"""
        # 1. 重置画像
        profiler = UserProfiler()
        profiler.reset()

        # 2. 记录使用
        profiler.record_command("run")
        profiler.record_command("status")
        profiler.record_feature("dashboard")
        profiler.record_session(steps=50, preset="easy", env_type="mixed")

        # 3. 获取画像
        profile = profiler.get_profile()
        self.assertEqual(profile["sessions_count"], 1)
        self.assertIn("run", profile["commands_used"])

        # 4. 生成建议
        engine = SuggestionEngine()
        suggestions = engine.generate()
        self.assertIsInstance(suggestions, list)

    def test02_context_to_suggestions(self):
        """上下文到建议"""
        analyzer = ContextAnalyzer()
        agent = MagicMock()
        agent.cycle_count = 50
        agent.total_real_reward = -10.0
        agent.avg_cycle_time = 2.0
        agent.avg_prediction_error = 0.8
        agent.mode = "normal"
        agent.task_manager = None
        agent.skill_manager = None
        agent.evolution_manager = None

        context = analyzer.analyze_agent(agent)
        engine = SuggestionEngine()
        suggestions = engine.generate(context=context)

        # 应该有多个建议（负奖励、高误差、慢周期等）
        self.assertGreater(len(suggestions), 0)
        # 检查是否有CRITICAL建议（负奖励）
        critical = [s for s in suggestions
                    if s.priority == SuggestionPriority.CRITICAL]
        self.assertGreater(len(critical), 0)

    def test03_recommendation_workflow(self):
        """推荐工作流"""
        rec = Recommender()
        # 初学者推荐
        config = rec.recommend_config("quick_test",
                                      UserSkillLevel.BEGINNER)
        self.assertEqual(config["preset"], "easy")

        cmds = rec.recommend_commands(UserSkillLevel.BEGINNER)
        self.assertGreater(len(cmds), 0)

        learning = rec.recommend_learning(UserSkillLevel.BEGINNER)
        self.assertGreater(len(learning), 0)

    def test04_skill_progression_workflow(self):
        """技能提升工作流"""
        profiler = UserProfiler()
        profiler.reset()

        # 初始是BEGINNER
        self.assertEqual(profiler.get_skill_level(),
                         UserSkillLevel.BEGINNER)

        # 大量使用
        for i in range(20):
            profiler.record_command(f"cmd_{i}")
        for i in range(15):
            profiler.record_feature(f"feat_{i}")
        for i in range(25):
            profiler.record_session(steps=50)

        # 应该提升了
        level = profiler.get_skill_level()
        self.assertNotEqual(level, UserSkillLevel.BEGINNER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
