"""
阶段69测试：Agent注册中心与工厂模式

覆盖：
    - AgentTypeInfo: 类型信息数据类
    - AgentRegistry: 注册中心(注册/注销/查询/别名/分类)
    - AgentFactory: 工厂(创建/默认创建/字典创建/验证)
    - AgentLoader: 加载器(字典/文件/批量/保存)
    - 全局单例
    - 便捷函数
    - 集成测试 (真实API)
"""
import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from seva.agents import (
    AgentTypeInfo, AgentRegistry, AgentFactory, AgentLoader,
    get_agent_registry, reset_agent_registry,
    get_agent_factory, reset_agent_factory,
    get_agent_loader, reset_agent_loader,
    reset_all_registry,
    create_agent, list_agent_types, register_agent_type,
    BaseAgent, AgentConfig, AgentRunResult, AgentState,
    ConversationAgent, ConversationConfig,
    CodeAgent, CodeAgentConfig,
    ResearchAgent, ResearchAgentConfig,
    WorkflowAgent, WorkflowAgentConfig,
)
from seva.llm import reset_llm_client


# ============================================================================
# AgentTypeInfo测试
# ============================================================================

class TestAgentTypeInfo(unittest.TestCase):
    """测试AgentTypeInfo"""

    def test01_create(self):
        info = AgentTypeInfo(
            type_name="test",
            agent_class=ConversationAgent,
            config_class=ConversationConfig,
            description="测试类型",
        )
        self.assertEqual(info.type_name, "test")
        self.assertEqual(info.agent_class, ConversationAgent)
        self.assertEqual(info.config_class, ConversationConfig)
        self.assertEqual(info.description, "测试类型")
        self.assertEqual(info.version, "1.0.0")
        self.assertEqual(info.category, "custom")
        self.assertEqual(info.aliases, [])

    def test02_create_with_all_params(self):
        info = AgentTypeInfo(
            type_name="test",
            agent_class=CodeAgent,
            config_class=CodeAgentConfig,
            description="测试",
            version="2.0.0",
            category="builtin",
            aliases=["t", "test_alias"],
            metadata={"author": "test"},
        )
        self.assertEqual(info.version, "2.0.0")
        self.assertEqual(info.category, "builtin")
        self.assertEqual(info.aliases, ["t", "test_alias"])
        self.assertEqual(info.metadata["author"], "test")

    def test03_to_dict(self):
        info = AgentTypeInfo(
            type_name="test",
            agent_class=ConversationAgent,
            config_class=ConversationConfig,
            description="测试",
            aliases=["t"],
        )
        d = info.to_dict()
        self.assertEqual(d["type_name"], "test")
        self.assertEqual(d["agent_class"], "ConversationAgent")
        self.assertEqual(d["config_class"], "ConversationConfig")
        self.assertEqual(d["description"], "测试")
        self.assertEqual(d["aliases"], ["t"])
        # 不应包含类引用
        self.assertNotIn("agent_class", [k for k, v in d.items()
                                         if not isinstance(v, str)])


# ============================================================================
# AgentRegistry测试
# ============================================================================

class TestAgentRegistry(unittest.TestCase):
    """测试AgentRegistry"""

    def setUp(self):
        self.registry = AgentRegistry()

    def test01_builtin_types_registered(self):
        """内置类型已注册"""
        self.assertTrue(self.registry.is_registered("base"))
        self.assertTrue(self.registry.is_registered("conversation"))
        self.assertTrue(self.registry.is_registered("code"))
        self.assertTrue(self.registry.is_registered("research"))
        self.assertTrue(self.registry.is_registered("workflow"))

    def test02_count(self):
        self.assertEqual(self.registry.count(), 5)

    def test03_list_types(self):
        types = self.registry.list_types()
        self.assertIn("base", types)
        self.assertIn("conversation", types)
        self.assertIn("code", types)
        self.assertIn("research", types)
        self.assertIn("workflow", types)

    def test04_get_info(self):
        info = self.registry.get("conversation")
        self.assertIsNotNone(info)
        self.assertEqual(info.type_name, "conversation")
        self.assertEqual(info.agent_class, ConversationAgent)

    def test05_get_class(self):
        cls = self.registry.get_class("code")
        self.assertEqual(cls, CodeAgent)

    def test06_get_config_class(self):
        cls = self.registry.get_config_class("research")
        self.assertEqual(cls, ResearchAgentConfig)

    def test07_get_nonexistent(self):
        self.assertIsNone(self.registry.get("nonexistent"))
        self.assertIsNone(self.registry.get_class("nonexistent"))
        self.assertIsNone(self.registry.get_config_class("nonexistent"))

    def test08_is_registered(self):
        self.assertTrue(self.registry.is_registered("conversation"))
        self.assertFalse(self.registry.is_registered("nonexistent"))

    def test09_register_custom(self):
        """注册自定义类型"""

        class CustomAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True, output="custom",
                )

        class CustomConfig(AgentConfig):
            pass

        result = self.registry.register(
            "custom", CustomAgent, CustomConfig,
            description="自定义Agent",
            aliases=["my_custom"],
        )
        self.assertTrue(result)
        self.assertTrue(self.registry.is_registered("custom"))
        self.assertEqual(self.registry.get_class("custom"), CustomAgent)

    def test10_register_with_aliases(self):
        class CustomAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                )

        class CustomConfig(AgentConfig):
            pass

        self.registry.register(
            "aliased", CustomAgent, CustomConfig,
            aliases=["a1", "a2", "a3"],
        )
        # 通过别名查询
        self.assertTrue(self.registry.is_registered("a1"))
        self.assertTrue(self.registry.is_registered("a2"))
        self.assertTrue(self.registry.is_registered("a3"))
        # 解析别名
        self.assertEqual(self.registry.resolve_alias("a1"), "aliased")

    def test11_register_invalid_class(self):
        """注册非BaseAgent子类应失败"""
        with self.assertRaises(TypeError):
            self.registry.register("invalid", str, AgentConfig)

    def test12_register_invalid_config(self):
        """注册非AgentConfig子类应失败"""
        with self.assertRaises(TypeError):
            self.registry.register("invalid", BaseAgent, str)

    def test13_register_empty_name(self):
        """注册空名应失败"""
        with self.assertRaises(ValueError):
            self.registry.register("", BaseAgent, AgentConfig)

    def test14_unregister(self):
        """注销类型"""
        class CustomAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                )

        class CustomConfig(AgentConfig):
            pass

        self.registry.register("temp", CustomAgent, CustomConfig,
                              aliases=["t"])
        self.assertTrue(self.registry.is_registered("temp"))
        self.assertTrue(self.registry.is_registered("t"))

        result = self.registry.unregister("temp")
        self.assertTrue(result)
        self.assertFalse(self.registry.is_registered("temp"))
        # 别名也应被移除
        self.assertFalse(self.registry.is_registered("t"))

    def test15_unregister_nonexistent(self):
        """注销不存在的类型"""
        result = self.registry.unregister("nonexistent")
        self.assertFalse(result)

    def test16_resolve_alias(self):
        """解析别名"""
        self.assertEqual(self.registry.resolve_alias("chat"), "conversation")
        self.assertEqual(self.registry.resolve_alias("coder"), "code")
        self.assertEqual(self.registry.resolve_alias("conversation"), "conversation")
        # 未知别名返回原值
        self.assertEqual(self.registry.resolve_alias("unknown"), "unknown")

    def test17_list_aliases(self):
        aliases = self.registry.list_aliases()
        self.assertIn("chat", aliases)
        self.assertIn("coder", aliases)
        self.assertEqual(aliases["chat"], "conversation")

    def test18_list_detailed(self):
        infos = self.registry.list_detailed()
        self.assertEqual(len(infos), 5)
        type_names = [i.type_name for i in infos]
        self.assertIn("base", type_names)
        self.assertIn("conversation", type_names)

    def test19_list_by_category(self):
        builtins = self.registry.list_by_category("builtin")
        self.assertEqual(len(builtins), 5)
        customs = self.registry.list_by_category("custom")
        self.assertEqual(len(customs), 0)

    def test20_clear(self):
        self.registry.clear()
        self.assertEqual(self.registry.count(), 0)
        self.assertEqual(len(self.registry.list_types()), 0)

    def test21_reset(self):
        """重置后内置类型恢复"""
        self.registry.clear()
        self.assertEqual(self.registry.count(), 0)
        self.registry.reset()
        self.assertEqual(self.registry.count(), 5)
        self.assertTrue(self.registry.is_registered("conversation"))

    def test22_get_report(self):
        report = self.registry.get_report()
        self.assertEqual(report["total_types"], 5)
        self.assertIn("categories", report)
        self.assertIn("types", report)
        self.assertEqual(report["categories"]["builtin"], 5)

    def test23_reregister_overwrites(self):
        """重复注册会覆盖"""
        class CustomAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                )

        class CustomConfig(AgentConfig):
            pass

        self.registry.register("conversation", CustomAgent, CustomConfig)
        # 应该被覆盖
        self.assertEqual(self.registry.get_class("conversation"), CustomAgent)
        # 恢复
        self.registry.reset()


# ============================================================================
# AgentFactory测试
# ============================================================================

class TestAgentFactory(unittest.TestCase):
    """测试AgentFactory"""

    def setUp(self):
        self.factory = AgentFactory(AgentRegistry())

    def test01_create_conversation(self):
        agent = self.factory.create("conversation")
        self.assertIsInstance(agent, ConversationAgent)
        self.assertEqual(agent.state, AgentState.CREATED)

    def test02_create_code(self):
        agent = self.factory.create("code")
        self.assertIsInstance(agent, CodeAgent)

    def test03_create_research(self):
        agent = self.factory.create("research")
        self.assertIsInstance(agent, ResearchAgent)

    def test04_create_workflow(self):
        agent = self.factory.create("workflow")
        self.assertIsInstance(agent, WorkflowAgent)

    def test05_create_with_config(self):
        config = ConversationConfig(name="my_conv")
        agent = self.factory.create("conversation", config=config)
        self.assertIsInstance(agent, ConversationAgent)
        self.assertEqual(agent.config.name, "my_conv")

    def test06_create_with_agent_id(self):
        agent = self.factory.create("conversation", agent_id="my_id")
        self.assertEqual(agent.agent_id, "my_id")

    def test07_create_default(self):
        agent = self.factory.create_default("conversation")
        self.assertIsInstance(agent, ConversationAgent)

    def test08_create_with_alias(self):
        """通过别名创建"""
        agent = self.factory.create("chat")
        self.assertIsInstance(agent, ConversationAgent)

        agent2 = self.factory.create("coder")
        self.assertIsInstance(agent2, CodeAgent)

    def test09_create_unknown_type(self):
        """创建未知类型应失败"""
        with self.assertRaises(ValueError):
            self.factory.create("unknown_type")

    def test10_create_with_wrong_config(self):
        """配置类型不匹配应失败"""
        # 用CodeAgentConfig创建ConversationAgent
        wrong_config = CodeAgentConfig()
        with self.assertRaises(TypeError):
            self.factory.create("conversation", config=wrong_config)

    def test11_create_from_dict(self):
        """从字典创建"""
        data = {
            "type": "conversation",
            "config": {"name": "dict_agent"},
        }
        agent = self.factory.create_from_dict(data)
        self.assertIsInstance(agent, ConversationAgent)
        self.assertEqual(agent.config.name, "dict_agent")

    def test12_create_from_dict_with_agent_id(self):
        data = {
            "type": "code",
            "agent_id": "my_agent",
            "config": {"name": "dict_code"},
        }
        agent = self.factory.create_from_dict(data)
        self.assertIsInstance(agent, CodeAgent)
        self.assertEqual(agent.agent_id, "my_agent")
        self.assertEqual(agent.config.name, "dict_code")

    def test13_create_from_dict_no_type(self):
        """字典缺少type字段"""
        with self.assertRaises(ValueError):
            self.factory.create_from_dict({"config": {}})

    def test14_create_from_dict_unknown_type(self):
        with self.assertRaises(ValueError):
            self.factory.create_from_dict({"type": "unknown"})

    def test15_create_from_dict_bad_config(self):
        """字典配置参数错误"""
        data = {
            "type": "conversation",
            "config": {"invalid_param": "value"},
        }
        with self.assertRaises(ValueError):
            self.factory.create_from_dict(data)

    def test16_validate_config(self):
        """验证配置"""
        conv_config = ConversationConfig()
        code_config = CodeAgentConfig()

        self.assertTrue(self.factory.validate_config("conversation", conv_config))
        self.assertTrue(self.factory.validate_config("code", code_config))
        # 不匹配
        self.assertFalse(self.factory.validate_config("conversation", code_config))
        self.assertFalse(self.factory.validate_config("code", conv_config))

    def test17_validate_config_unknown_type(self):
        self.assertFalse(self.factory.validate_config("unknown", AgentConfig()))

    def test18_get_supported_types(self):
        types = self.factory.get_supported_types()
        self.assertIn("conversation", types)
        self.assertIn("code", types)

    def test19_stats_updated(self):
        """统计更新"""
        self.factory.create("conversation")
        self.factory.create("code")
        stats = self.factory.get_stats()
        self.assertEqual(stats["agents_created"], 2)
        self.assertEqual(stats["creation_errors"], 0)

    def test20_stats_errors(self):
        """错误统计"""
        try:
            self.factory.create("unknown")
        except ValueError:
            pass
        stats = self.factory.get_stats()
        self.assertEqual(stats["creation_errors"], 1)

    def test21_registry_property(self):
        self.assertIsInstance(self.factory.registry, AgentRegistry)


# ============================================================================
# AgentLoader测试
# ============================================================================

class TestAgentLoader(unittest.TestCase):
    """测试AgentLoader"""

    def setUp(self):
        self.loader = AgentLoader(AgentFactory(AgentRegistry()))

    def test01_load_from_dict(self):
        data = {"type": "conversation"}
        agent = self.loader.load_from_dict(data)
        self.assertIsInstance(agent, ConversationAgent)

    def test02_load_from_dict_with_config(self):
        data = {
            "type": "code",
            "config": {"name": "loaded_code"},
        }
        agent = self.loader.load_from_dict(data)
        self.assertIsInstance(agent, CodeAgent)
        self.assertEqual(agent.config.name, "loaded_code")

    def test03_load_from_file(self):
        """从文件加载"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump({"type": "conversation",
                      "config": {"name": "file_agent"}}, f)
            f.flush()
            path = f.name

        try:
            agent = self.loader.load_from_file(path)
            self.assertIsInstance(agent, ConversationAgent)
            self.assertEqual(agent.config.name, "file_agent")
        finally:
            os.unlink(path)

    def test04_load_from_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_from_file("/nonexistent/path.json")

    def test05_load_batch(self):
        """批量加载"""
        configs = [
            {"type": "conversation"},
            {"type": "code"},
            {"type": "research"},
        ]
        agents = self.loader.load_batch(configs)
        self.assertEqual(len(agents), 3)
        self.assertIsInstance(agents[0], ConversationAgent)
        self.assertIsInstance(agents[1], CodeAgent)
        self.assertIsInstance(agents[2], ResearchAgent)

    def test06_load_batch_with_error(self):
        """批量加载含错误"""
        configs = [
            {"type": "conversation"},
            {"type": "unknown_type"},  # 错误
            {"type": "code"},
        ]
        agents = self.loader.load_batch(configs)
        self.assertEqual(len(agents), 3)
        self.assertIsInstance(agents[0], ConversationAgent)
        self.assertIsNone(agents[1])  # 错误的为None
        self.assertIsInstance(agents[2], CodeAgent)

    def test07_load_batch_from_file(self):
        """从文件批量加载"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump([
                {"type": "conversation"},
                {"type": "code"},
            ], f)
            f.flush()
            path = f.name

        try:
            agents = self.loader.load_batch_from_file(path)
            self.assertEqual(len(agents), 2)
        finally:
            os.unlink(path)

    def test08_load_batch_from_file_not_array(self):
        """批量加载文件不是数组"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump({"type": "conversation"}, f)
            f.flush()
            path = f.name

        try:
            with self.assertRaises(ValueError):
                self.loader.load_batch_from_file(path)
        finally:
            os.unlink(path)

    def test09_save_to_dict(self):
        """保存为字典"""
        agent = ConversationAgent(config=ConversationConfig(name="saved"))
        data = self.loader.save_to_dict(agent)
        self.assertEqual(data["type"], "conversation")
        self.assertEqual(data["agent_id"], agent.agent_id)
        self.assertIn("config", data)
        self.assertEqual(data["config"]["name"], "saved")

    def test10_save_to_file(self):
        """保存到文件"""
        agent = CodeAgent(config=CodeAgentConfig(name="file_saved"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            path = f.name

        try:
            result = self.loader.save_to_file(agent, path)
            self.assertTrue(result)
            # 验证文件内容
            with open(path, "r") as f:
                data = json.load(f)
            self.assertEqual(data["type"], "code")
            self.assertEqual(data["config"]["name"], "file_saved")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test11_save_and_load_roundtrip(self):
        """保存后加载的往返测试"""
        agent = ConversationAgent(config=ConversationConfig(name="roundtrip"))
        data = self.loader.save_to_dict(agent)
        agent2 = self.loader.load_from_dict(data)
        self.assertIsInstance(agent2, ConversationAgent)
        self.assertEqual(agent2.config.name, "roundtrip")

    def test12_save_unknown_agent_type(self):
        """保存未注册类型的Agent"""

        class UnknownAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                )

        agent = UnknownAgent()
        with self.assertRaises(ValueError):
            self.loader.save_to_dict(agent)

    def test13_factory_property(self):
        self.assertIsInstance(self.loader.factory, AgentFactory)


# ============================================================================
# 全局单例测试
# ============================================================================

class TestGlobalSingleton(unittest.TestCase):
    """测试全局单例"""

    def setUp(self):
        reset_all_registry()

    def tearDown(self):
        reset_all_registry()

    def test01_get_agent_registry(self):
        reg = get_agent_registry()
        self.assertIsInstance(reg, AgentRegistry)

    def test02_registry_singleton(self):
        r1 = get_agent_registry()
        r2 = get_agent_registry()
        self.assertIs(r1, r2)

    def test03_reset_agent_registry(self):
        r1 = get_agent_registry()
        reset_agent_registry()
        r2 = get_agent_registry()
        self.assertIsNot(r1, r2)

    def test04_get_agent_factory(self):
        factory = get_agent_factory()
        self.assertIsInstance(factory, AgentFactory)

    def test05_factory_singleton(self):
        f1 = get_agent_factory()
        f2 = get_agent_factory()
        self.assertIs(f1, f2)

    def test06_reset_agent_factory(self):
        f1 = get_agent_factory()
        reset_agent_factory()
        f2 = get_agent_factory()
        self.assertIsNot(f1, f2)

    def test07_get_agent_loader(self):
        loader = get_agent_loader()
        self.assertIsInstance(loader, AgentLoader)

    def test08_loader_singleton(self):
        l1 = get_agent_loader()
        l2 = get_agent_loader()
        self.assertIs(l1, l2)

    def test09_reset_agent_loader(self):
        l1 = get_agent_loader()
        reset_agent_loader()
        l2 = get_agent_loader()
        self.assertIsNot(l1, l2)

    def test10_reset_all(self):
        r1 = get_agent_registry()
        f1 = get_agent_factory()
        l1 = get_agent_loader()
        reset_all_registry()
        r2 = get_agent_registry()
        f2 = get_agent_factory()
        l2 = get_agent_loader()
        self.assertIsNot(r1, r2)
        self.assertIsNot(f1, f2)
        self.assertIsNot(l1, l2)

    def test11_factory_uses_global_registry(self):
        """全局工厂使用全局注册中心"""
        reg = get_agent_registry()
        factory = get_agent_factory()
        self.assertIs(factory.registry, reg)


# ============================================================================
# 便捷函数测试
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        reset_all_registry()

    def tearDown(self):
        reset_all_registry()

    def test01_create_agent(self):
        agent = create_agent("conversation")
        self.assertIsInstance(agent, ConversationAgent)

    def test02_create_agent_with_config(self):
        config = ConversationConfig(name="conv")
        agent = create_agent("conversation", config=config)
        self.assertEqual(agent.config.name, "conv")

    def test03_list_agent_types(self):
        types = list_agent_types()
        self.assertIn("conversation", types)
        self.assertIn("code", types)
        self.assertIn("research", types)
        self.assertIn("workflow", types)

    def test04_register_agent_type(self):
        class CustomAgent(BaseAgent):
            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                )

        class CustomConfig(AgentConfig):
            pass

        result = register_agent_type("custom", CustomAgent, CustomConfig,
                                     description="自定义")
        self.assertTrue(result)
        self.assertIn("custom", list_agent_types())
        agent = create_agent("custom")
        self.assertIsInstance(agent, CustomAgent)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        reset_all_registry()
        reset_llm_client()

    def tearDown(self):
        reset_all_registry()
        reset_llm_client()

    def test01_full_lifecycle_via_factory(self):
        """通过工厂创建Agent的完整生命周期"""
        # 1. 创建Agent
        agent = create_agent("conversation")
        self.assertEqual(agent.state, AgentState.CREATED)

        # 2. 配置并启动
        agent.configure()
        agent.start()
        self.assertEqual(agent.state, AgentState.RUNNING)

        # 3. 停止
        agent.stop()
        self.assertEqual(agent.state, AgentState.STOPPED)

    def test02_multi_agent_management(self):
        """多Agent管理"""
        agents = []
        for agent_type in ["conversation", "code", "research", "workflow"]:
            agent = create_agent(agent_type)
            agents.append(agent)

        # 验证所有Agent都是不同实例
        ids = [a.agent_id for a in agents]
        self.assertEqual(len(ids), len(set(ids)))

        # 验证类型
        self.assertIsInstance(agents[0], ConversationAgent)
        self.assertIsInstance(agents[1], CodeAgent)
        self.assertIsInstance(agents[2], ResearchAgent)
        self.assertIsInstance(agents[3], WorkflowAgent)

    def test03_save_load_and_run(self):
        """保存-加载-运行往返测试"""
        # 1. 创建并配置Agent
        agent = create_agent("conversation",
                            config=ConversationConfig(name="persisted"))
        agent.start()

        # 2. 保存
        loader = get_agent_loader()
        data = loader.save_to_dict(agent)
        self.assertEqual(data["type"], "conversation")

        # 3. 加载
        agent2 = loader.load_from_dict(data)
        self.assertIsInstance(agent2, ConversationAgent)
        self.assertEqual(agent2.config.name, "persisted")

    def test04_custom_agent_registration_and_creation(self):
        """自定义Agent注册与创建"""

        class DataProcessorAgent(BaseAgent):
            """数据处理Agent"""

            def _run_impl(self, input, **kwargs):
                return AgentRunResult(
                    agent_id=self.agent_id,
                    session_id=self.context.session_id,
                    success=True,
                    output=f"处理: {input}",
                )

        class DataProcessorConfig(AgentConfig):
            name: str = "data_processor"

        # 注册
        register_agent_type("data_processor", DataProcessorAgent,
                           DataProcessorConfig,
                           description="数据处理Agent",
                           aliases=["dp"])

        # 通过别名创建
        agent = create_agent("dp")
        self.assertIsInstance(agent, DataProcessorAgent)

        # 运行
        agent.start()
        result = agent.run("测试数据")
        self.assertTrue(result.success)
        self.assertIn("测试数据", result.output)

    def test05_real_llm_via_factory(self):
        """通过工厂创建Agent并调用真实LLM"""
        api_key = os.environ.get("SEVA_LLM_API_KEY", "")
        if not api_key:
            self.skipTest("未设置 SEVA_LLM_API_KEY, 跳过真实API测试")

        # 通过工厂创建对话Agent
        agent = create_agent("conversation")
        agent.start()

        # 运行对话
        result = agent.run("你好,请用一句话介绍自己。")

        self.assertTrue(result.success)
        self.assertGreater(len(result.output), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
