"""
阶段59测试：Prompt工程与模板管理

覆盖:
    - SectionType枚举
    - PromptVariable数据类
    - FewShotExample数据类
    - PromptSection数据类 + 命名冲突修复（section_examples字段）
    - PromptFormat枚举
    - PromptTemplate数据类（to_dict/from_dict/get_required_variables/validate_inputs）
    - RenderedPrompt数据类（hash/to_messages）
    - PromptFormatter（变量插值、Few-shot、Message组装）
    - PromptChain（add_step/step_count/clear/get_steps）
    - PromptRegistry（register/get/list/unregister/find_by_tag/count/clear/persistence）
    - 便捷函数（register_prompt/get_prompt/list_prompts/render_prompt等）
    - 集成测试（多变量插值+Few-shot+序列化往返）
"""
import unittest
import os
import json
import time
import shutil
import tempfile
import threading
from pathlib import Path

from seva.prompts import (
    SectionType, PromptVariable, FewShotExample, PromptSection,
    PromptFormat, PromptTemplate, RenderedPrompt,
    PromptFormatter, PromptChain, PromptRegistry,
    get_prompt_registry, register_prompt, get_prompt,
    list_prompts, unregister_prompt, render_prompt,
    delete_prompt, reset_prompts,
)


# ============================================================================
# SectionType枚举
# ============================================================================

class TestSectionType(unittest.TestCase):
    """测试SectionType枚举"""

    def test01_values(self):
        self.assertEqual(SectionType.ROLE.value, "role")
        self.assertEqual(SectionType.CONTEXT.value, "context")
        self.assertEqual(SectionType.TASK.value, "task")
        self.assertEqual(SectionType.EXAMPLES.value, "examples")
        self.assertEqual(SectionType.INPUT.value, "input")
        self.assertEqual(SectionType.CONSTRAINTS.value, "constraints")
        self.assertEqual(SectionType.OUTPUT.value, "output")
        self.assertEqual(SectionType.CUSTOM.value, "custom")

    def test02_count(self):
        self.assertEqual(len(list(SectionType)), 8)


# ============================================================================
# PromptVariable数据类
# ============================================================================

class TestPromptVariable(unittest.TestCase):
    """测试PromptVariable"""

    def test01_defaults(self):
        v = PromptVariable(name="x")
        self.assertEqual(v.name, "x")
        self.assertEqual(v.description, "")
        self.assertTrue(v.required)
        self.assertIsNone(v.default)
        self.assertEqual(v.var_type, "string")

    def test02_full(self):
        v = PromptVariable(name="count", description="数量", required=False,
                          default=10, var_type="number")
        self.assertEqual(v.name, "count")
        self.assertFalse(v.required)
        self.assertEqual(v.default, 10)
        self.assertEqual(v.var_type, "number")

    def test03_to_dict(self):
        v = PromptVariable(name="user", description="用户", required=True)
        d = v.to_dict()
        self.assertEqual(d["name"], "user")
        self.assertEqual(d["description"], "用户")
        self.assertTrue(d["required"])
        self.assertEqual(d["var_type"], "string")


# ============================================================================
# FewShotExample数据类
# ============================================================================

class TestFewShotExample(unittest.TestCase):
    """测试FewShotExample"""

    def test01_creation(self):
        ex = FewShotExample(user_input="hi", assistant_output="hello")
        self.assertEqual(ex.user_input, "hi")
        self.assertEqual(ex.assistant_output, "hello")
        self.assertEqual(ex.description, "")

    def test02_to_dict(self):
        ex = FewShotExample(user_input="u", assistant_output="a",
                            description="desc")
        d = ex.to_dict()
        self.assertEqual(d["user"], "u")
        self.assertEqual(d["assistant"], "a")
        self.assertEqual(d["description"], "desc")

    def test03_repr(self):
        ex = FewShotExample(user_input="very long input that exceeds thirty chars",
                            assistant_output="a")
        r = repr(ex)
        self.assertIn("FewShotExample", r)
        self.assertIn("user=", r)


# ============================================================================
# PromptSection数据类
# ============================================================================

class TestPromptSection(unittest.TestCase):
    """测试PromptSection（关键：验证命名冲突已修复）"""

    def setUp(self):
        self.examples = [
            FewShotExample(user_input="1+1", assistant_output="2"),
            FewShotExample(user_input="2+2", assistant_output="4"),
        ]

    def test01_class_methods(self):
        # 验证所有classmethod都正确工作
        role_sec = PromptSection.role("你是助手")
        self.assertEqual(role_sec.section_type, SectionType.ROLE)
        self.assertEqual(role_sec.content, "你是助手")
        self.assertEqual(role_sec.section_examples, [])

        ctx_sec = PromptSection.context("背景信息")
        self.assertEqual(ctx_sec.section_type, SectionType.CONTEXT)

        task_sec = PromptSection.task("完成任务")
        self.assertEqual(task_sec.section_type, SectionType.TASK)

        ex_sec = PromptSection.examples(self.examples)
        self.assertEqual(ex_sec.section_type, SectionType.EXAMPLES)
        self.assertEqual(len(ex_sec.section_examples), 2)

        in_sec = PromptSection.input_data("用户输入: {{q}}")
        self.assertEqual(in_sec.section_type, SectionType.INPUT)

        c_sec = PromptSection.constraints("只能回答是/否")
        self.assertEqual(c_sec.section_type, SectionType.CONSTRAINTS)

        o_sec = PromptSection.output_format("JSON")
        self.assertEqual(o_sec.section_type, SectionType.OUTPUT)

        c2_sec = PromptSection.custom("自定义")
        self.assertEqual(c2_sec.section_type, SectionType.CUSTOM)

    def test02_section_examples_is_list_not_method(self):
        """关键测试：修复了 examples 字段名和 classmethod 冲突"""
        sec = PromptSection.examples(self.examples)
        # 必须是 list，不能是 method
        self.assertIsInstance(sec.section_examples, list)
        self.assertEqual(len(sec.section_examples), 2)
        self.assertEqual(sec.section_examples[0].user_input, "1+1")

    def test03_default_examples_is_empty_list(self):
        sec = PromptSection.task("test")
        self.assertEqual(sec.section_examples, [])
        self.assertIsInstance(sec.section_examples, list)

    def test04_to_dict(self):
        sec = PromptSection.examples(self.examples)
        d = sec.to_dict()
        self.assertEqual(d["type"], "examples")
        self.assertEqual(d["content"], "以下是一些示例：")
        self.assertEqual(len(d["examples"]), 2)
        self.assertEqual(d["examples"][0]["user"], "1+1")
        self.assertEqual(d["examples"][0]["assistant"], "2")
        self.assertTrue(d["enabled"])

    def test05_disabled(self):
        sec = PromptSection.task("test", enabled=False)
        self.assertFalse(sec.enabled)
        d = sec.to_dict()
        self.assertFalse(d["enabled"])


# ============================================================================
# PromptFormat枚举
# ============================================================================

class TestPromptFormat(unittest.TestCase):
    """测试PromptFormat枚举"""

    def test01_values(self):
        self.assertEqual(PromptFormat.RAW.value, "raw")
        self.assertEqual(PromptFormat.CHATML.value, "chatml")
        self.assertEqual(PromptFormat.ALPACA.value, "alpaca")
        self.assertEqual(PromptFormat.MESSAGES.value, "messages")


# ============================================================================
# PromptTemplate数据类
# ============================================================================

class TestPromptTemplate(unittest.TestCase):
    """测试PromptTemplate"""

    def test01_minimal(self):
        t = PromptTemplate(name="simple")
        self.assertEqual(t.name, "simple")
        self.assertEqual(t.version, "1.0.0")
        self.assertEqual(t.sections, [])
        self.assertEqual(t.variables, [])
        self.assertEqual(t.output_format, PromptFormat.MESSAGES)

    def test02_full(self):
        t = PromptTemplate(
            name="complex",
            description="复杂模板",
            version="2.0.0",
            sections=[PromptSection.role("助手")],
            variables=[PromptVariable(name="q")],
            tags=["test", "demo"],
            output_format=PromptFormat.RAW,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        )
        self.assertEqual(t.description, "复杂模板")
        self.assertEqual(t.version, "2.0.0")
        self.assertEqual(t.tags, ["test", "demo"])
        self.assertEqual(t.model, "gpt-4")
        self.assertEqual(t.temperature, 0.7)
        self.assertEqual(t.max_tokens, 100)

    def test03_to_dict(self):
        t = PromptTemplate(
            name="x",
            sections=[PromptSection.task("处理{{q}}")],
            variables=[PromptVariable(name="q")],
        )
        d = t.to_dict()
        self.assertEqual(d["name"], "x")
        self.assertEqual(d["version"], "1.0.0")
        self.assertEqual(d["output_format"], "messages")
        self.assertEqual(len(d["sections"]), 1)
        self.assertEqual(len(d["variables"]), 1)

    def test04_from_dict_round_trip(self):
        """测试 to_dict -> from_dict 往返一致性"""
        t1 = PromptTemplate(
            name="rt_test",
            description="round trip",
            version="3.1.0",
            sections=[
                PromptSection.role("助手"),
                PromptSection.task("处理{{q}}"),
                PromptSection.examples([
                    FewShotExample(user_input="1", assistant_output="one"),
                ]),
            ],
            variables=[
                PromptVariable(name="q", description="query", required=True,
                              var_type="string"),
            ],
            tags=["test"],
        )
        d = t1.to_dict()
        t2 = PromptTemplate.from_dict(d)
        self.assertEqual(t1.name, t2.name)
        self.assertEqual(t1.description, t2.description)
        self.assertEqual(t1.version, t2.version)
        self.assertEqual(len(t1.sections), len(t2.sections))
        self.assertEqual(len(t1.variables), len(t2.variables))
        # 验证 examples 字段正确恢复
        ex_sec = [s for s in t2.sections if s.section_type == SectionType.EXAMPLES][0]
        self.assertEqual(len(ex_sec.section_examples), 1)
        self.assertEqual(ex_sec.section_examples[0].user_input, "1")

    def test05_get_required_variables(self):
        t = PromptTemplate(
            name="x",
            variables=[
                PromptVariable(name="a", required=True),
                PromptVariable(name="b", required=False, default="B"),
                PromptVariable(name="c", required=True),
            ],
        )
        req = t.get_required_variables()
        names = [v.name for v in req]
        self.assertIn("a", names)
        self.assertIn("c", names)
        self.assertNotIn("b", names)

    def test06_validate_inputs(self):
        t = PromptTemplate(
            name="x",
            variables=[
                PromptVariable(name="a", required=True),
                PromptVariable(name="b", required=False, default="B"),
            ],
        )
        # 缺少 a
        self.assertIn("a", t.validate_inputs({}))
        # 提供 a
        self.assertEqual(t.validate_inputs({"a": "x"}), [])
        # 有默认值的 b 可不传
        self.assertEqual(t.validate_inputs({"a": "x"}), [])

    def test07_repr(self):
        t = PromptTemplate(name="repr_test", version="1.2.3")
        r = repr(t)
        self.assertIn("repr_test", r)
        self.assertIn("1.2.3", r)


# ============================================================================
# RenderedPrompt数据类
# ============================================================================

class TestRenderedPrompt(unittest.TestCase):
    """测试RenderedPrompt"""

    def test01_creation(self):
        r = RenderedPrompt(
            template_name="t",
            template_version="1.0.0",
            raw_content="hello",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(r.template_name, "t")
        self.assertEqual(r.raw_content, "hello")
        self.assertEqual(len(r.messages), 1)
        self.assertNotEqual(r.hash, "")

    def test02_hash_auto_generated(self):
        r1 = RenderedPrompt(template_name="x", template_version="1.0",
                            raw_content="hello", messages=[])
        r2 = RenderedPrompt(template_name="x", template_version="1.0",
                            raw_content="hello", messages=[])
        # 相同内容相同版本应该 hash 一致
        self.assertEqual(r1.hash, r2.hash)
        self.assertEqual(len(r1.hash), 16)

    def test03_to_messages_from_explicit(self):
        msgs = [{"role": "user", "content": "hi"}]
        r = RenderedPrompt(template_name="t", template_version="1.0",
                          raw_content="", messages=msgs)
        self.assertEqual(r.to_messages(), msgs)

    def test04_to_messages_from_system_user(self):
        r = RenderedPrompt(
            template_name="t",
            template_version="1.0",
            raw_content="",
            messages=[],
            system_content="sys",
            user_content="user",
        )
        msgs = r.to_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test05_to_dict(self):
        r = RenderedPrompt(
            template_name="t", template_version="1.0",
            raw_content="raw", messages=[],
            variables_used={"x": 1},
        )
        d = r.to_dict()
        self.assertEqual(d["template_name"], "t")
        self.assertEqual(d["raw_content"], "raw")
        self.assertEqual(d["variables_used"], {"x": 1})


# ============================================================================
# PromptFormatter
# ============================================================================

class TestPromptFormatter(unittest.TestCase):
    """测试PromptFormatter"""

    def test01_basic_messages(self):
        t = PromptTemplate(
            name="fmt1",
            sections=[
                PromptSection.role("你是助手"),
                PromptSection.task("处理{{q}}"),
            ],
            variables=[PromptVariable(name="q")],
        )
        r = PromptFormatter.format(t, {"q": "测试"})
        msgs = r.to_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("你是助手", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("处理测试", msgs[1]["content"])

    def test02_default_values(self):
        t = PromptTemplate(
            name="fmt2",
            sections=[PromptSection.task("处理{{q}}")],
            variables=[PromptVariable(name="q", default="默认问题")],
        )
        r = PromptFormatter.format(t, {})
        self.assertIn("默认问题", r.user_content)

    def test03_section_ordering(self):
        """验证 system = role+context+constraints, user = task+examples+input+output+custom"""
        t = PromptTemplate(
            name="fmt3",
            sections=[
                PromptSection.task("任务"),
                PromptSection.role("角色"),
                PromptSection.context("上下文"),
                PromptSection.constraints("约束"),
                PromptSection.input_data("输入"),
                PromptSection.output_format("输出格式"),
            ],
        )
        r = PromptFormatter.format(t, {})
        msgs = r.to_messages()
        sys_msg = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        # role/context/constraints 应该在 system
        self.assertIn("角色", sys_msg)
        self.assertIn("上下文", sys_msg)
        self.assertIn("约束", sys_msg)
        # task/input/output 应该在 user
        self.assertIn("任务", user_msg)
        self.assertIn("输入", user_msg)
        self.assertIn("输出格式", user_msg)

    def test04_disabled_section(self):
        t = PromptTemplate(
            name="fmt4",
            sections=[
                PromptSection.role("A"),
                PromptSection.task("B", enabled=False),
            ],
        )
        r = PromptFormatter.format(t, {})
        self.assertIn("A", r.system_content)
        self.assertNotIn("B", r.user_content or "")

    def test05_few_shot_examples(self):
        t = PromptTemplate(
            name="fmt5",
            sections=[
                PromptSection.examples([
                    FewShotExample(user_input="1+1", assistant_output="2"),
                    FewShotExample(user_input="2+2", assistant_output="4"),
                ]),
            ],
        )
        r = PromptFormatter.format(t, {})
        # 验证示例被格式化
        self.assertIn("1+1", r.user_content)
        self.assertIn("2", r.user_content)
        self.assertIn("示例 1", r.user_content)
        self.assertIn("示例 2", r.user_content)

    def test06_dollar_syntax(self):
        """测试 $var 语法"""
        t = PromptTemplate(
            name="fmt6",
            sections=[PromptSection.task("值: $value")],
            variables=[PromptVariable(name="value")],
        )
        r = PromptFormatter.format(t, {"value": "X"})
        self.assertIn("值: X", r.user_content)

    def test07_missing_variable_preserved(self):
        """测试缺失变量保留占位符"""
        t = PromptTemplate(
            name="fmt7",
            sections=[PromptSection.task("处理{{missing}}")],
        )
        r = PromptFormatter.format(t, {})
        self.assertIn("{{missing}}", r.user_content)

    def test08_raw_format(self):
        t = PromptTemplate(
            name="fmt8",
            sections=[PromptSection.role("A"), PromptSection.task("B")],
        )
        r = PromptFormatter.format(t, {}, format_type=PromptFormat.RAW)
        # RAW 格式下，raw_content 应包含所有内容
        self.assertIn("A", r.raw_content)
        self.assertIn("B", r.raw_content)


# ============================================================================
# PromptChain
# ============================================================================

class TestPromptChain(unittest.TestCase):
    """测试PromptChain"""

    def test01_creation(self):
        chain = PromptChain(name="c1")
        self.assertEqual(chain.name, "c1")
        self.assertEqual(chain.step_count(), 0)
        self.assertEqual(chain.get_steps(), [])

    def test02_add_steps(self):
        chain = PromptChain(name="c2")
        t1 = PromptTemplate(name="s1", sections=[PromptSection.task("step1")])
        t2 = PromptTemplate(name="s2", sections=[PromptSection.task("step2")])

        chain.add_step(t1).add_step(t2)
        self.assertEqual(chain.step_count(), 2)
        steps = chain.get_steps()
        self.assertEqual(steps[0].name, "s1")
        self.assertEqual(steps[1].name, "s2")

    def test03_clear(self):
        chain = PromptChain(name="c3")
        chain.add_step(PromptTemplate(name="s1"))
        self.assertEqual(chain.step_count(), 1)
        chain.clear()
        self.assertEqual(chain.step_count(), 0)

    def test04_repr(self):
        chain = PromptChain(name="c4")
        chain.add_step(PromptTemplate(name="s1"))
        r = repr(chain)
        self.assertIn("c4", r)
        self.assertIn("steps=1", r)


# ============================================================================
# PromptRegistry - 使用临时目录避免污染 ~/.seva
# ============================================================================

class TestPromptRegistry(unittest.TestCase):
    """测试PromptRegistry（单例）"""

    def setUp(self):
        # 重置单例以确保干净状态
        PromptRegistry.reset_instance()
        # 创建临时目录
        self.tmpdir = tempfile.mkdtemp()
        # Monkey-patch PROMPTS_DIR
        import seva.prompts as p_module
        self._original_dir = p_module.PROMPTS_DIR
        p_module.PROMPTS_DIR = Path(self.tmpdir) / "prompts"

    def tearDown(self):
        PromptRegistry.reset_instance()
        import seva.prompts as p_module
        p_module.PROMPTS_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_singleton(self):
        r1 = PromptRegistry()
        r2 = PromptRegistry()
        self.assertIs(r1, r2)

    def test02_register_and_get(self):
        reg = PromptRegistry()
        t = PromptTemplate(name="reg1", sections=[PromptSection.task("x")])
        self.assertTrue(reg.register(t))
        self.assertIsNotNone(reg.get("reg1"))
        self.assertEqual(reg.get("reg1").name, "reg1")

    def test03_register_overwrite_false(self):
        reg = PromptRegistry()
        t = PromptTemplate(name="reg2", sections=[PromptSection.task("x")])
        reg.register(t)
        # overwrite=False 时不应替换
        self.assertFalse(reg.register(t, overwrite=False))
        # overwrite=True 时应替换
        self.assertTrue(reg.register(t, overwrite=True))

    def test04_unregister(self):
        reg = PromptRegistry()
        t = PromptTemplate(name="reg3", sections=[PromptSection.task("x")])
        reg.register(t)
        self.assertTrue(reg.unregister("reg3"))
        self.assertIsNone(reg.get("reg3"))
        # 重复 unregister 返回 False
        self.assertFalse(reg.unregister("reg3"))

    def test05_list(self):
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="l1"))
        reg.register(PromptTemplate(name="l2"))
        names = reg.list()
        self.assertIn("l1", names)
        self.assertIn("l2", names)
        self.assertEqual(len(names), 2)

    def test06_find_by_tag(self):
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="t1", tags=["a", "b"]))
        reg.register(PromptTemplate(name="t2", tags=["b", "c"]))
        reg.register(PromptTemplate(name="t3", tags=["c"]))
        # 查找 tag b
        found = reg.find_by_tag("b")
        names = [t.name for t in found]
        self.assertIn("t1", names)
        self.assertIn("t2", names)
        self.assertNotIn("t3", names)

    def test07_count(self):
        reg = PromptRegistry()
        self.assertEqual(reg.count(), 0)
        reg.register(PromptTemplate(name="c1"))
        reg.register(PromptTemplate(name="c2"))
        self.assertEqual(reg.count(), 2)

    def test08_clear(self):
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="c1"))
        reg.register(PromptTemplate(name="c2"))
        reg.clear()
        self.assertEqual(reg.count(), 0)

    def test09_persistence_save_and_load(self):
        """测试持久化：save 到磁盘，再读取"""
        reg1 = PromptRegistry()
        t = PromptTemplate(
            name="persist",
            description="持久化测试",
            sections=[PromptSection.task("处理{{q}}")],
            variables=[PromptVariable(name="q")],
        )
        reg1.register(t)
        # 验证文件已创建
        self.assertTrue((Path(self.tmpdir) / "prompts" / "registry.json").exists())

        # 重置单例以触发 load
        PromptRegistry.reset_instance()
        reg2 = PromptRegistry()
        t_loaded = reg2.get("persist")
        self.assertIsNotNone(t_loaded)
        self.assertEqual(t_loaded.description, "持久化测试")
        # 验证内容往返
        self.assertEqual(t_loaded.sections[0].content, "处理{{q}}")

    def test10_persistence_unregister(self):
        """测试持久化：unregister 后文件被更新"""
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="del_me"))
        reg.register(PromptTemplate(name="keep_me"))
        reg.unregister("del_me")

        PromptRegistry.reset_instance()
        reg2 = PromptRegistry()
        self.assertIsNone(reg2.get("del_me"))
        self.assertIsNotNone(reg2.get("keep_me"))

    def test11_thread_safety(self):
        """测试线程安全"""
        reg = PromptRegistry()
        errors = []

        def worker(i):
            try:
                for j in range(5):
                    reg.register(PromptTemplate(name=f"thread_{i}_{j}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        # 应该有 25 个模板
        self.assertEqual(reg.count(), 25)


# ============================================================================
# 便捷函数
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        PromptRegistry.reset_instance()
        import seva.prompts as p_module
        self._original_dir = p_module.PROMPTS_DIR
        self.tmpdir = tempfile.mkdtemp()
        p_module.PROMPTS_DIR = Path(self.tmpdir) / "prompts"

    def tearDown(self):
        PromptRegistry.reset_instance()
        import seva.prompts as p_module
        p_module.PROMPTS_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_get_prompt_registry(self):
        reg = get_prompt_registry()
        self.assertIsInstance(reg, PromptRegistry)

    def test02_register_and_get(self):
        t = PromptTemplate(name="cf1", sections=[PromptSection.task("x")])
        self.assertTrue(register_prompt(t))
        loaded = get_prompt("cf1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "cf1")

    def test03_list_prompts(self):
        register_prompt(PromptTemplate(name="lp1"))
        register_prompt(PromptTemplate(name="lp2"))
        names = list_prompts()
        self.assertGreaterEqual(len(names), 2)
        self.assertIn("lp1", names)

    def test04_unregister(self):
        register_prompt(PromptTemplate(name="u1"))
        self.assertTrue(unregister_prompt("u1"))
        self.assertIsNone(get_prompt("u1"))

    def test05_delete_prompt(self):
        register_prompt(PromptTemplate(name="d1"))
        self.assertTrue(delete_prompt("d1"))
        self.assertIsNone(get_prompt("d1"))

    def test06_render_prompt(self):
        t = PromptTemplate(
            name="r1",
            sections=[PromptSection.task("处理{{q}}")],
            variables=[PromptVariable(name="q")],
        )
        register_prompt(t)
        rendered = render_prompt("r1", inputs={"q": "测试"})
        self.assertIsNotNone(rendered)
        self.assertIn("处理测试", rendered.user_content)

    def test07_render_prompt_not_found(self):
        result = render_prompt("nonexistent")
        self.assertIsNone(result)

    def test08_reset_prompts(self):
        register_prompt(PromptTemplate(name="x1"))
        register_prompt(PromptTemplate(name="x2"))
        reset_prompts()
        # 重置后应该是空的（因为reset_instance清空）
        self.assertEqual(len(list_prompts()), 0)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试：端到端验证"""

    def setUp(self):
        PromptRegistry.reset_instance()
        import seva.prompts as p_module
        self._original_dir = p_module.PROMPTS_DIR
        self.tmpdir = tempfile.mkdtemp()
        p_module.PROMPTS_DIR = Path(self.tmpdir) / "prompts"

    def tearDown(self):
        PromptRegistry.reset_instance()
        import seva.prompts as p_module
        p_module.PROMPTS_DIR = self._original_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test01_full_workflow(self):
        """完整工作流：注册 → 渲染 → 注销"""
        # 1. 创建模板
        template = PromptTemplate(
            name="intent_classifier",
            description="意图分类器",
            version="1.0.0",
            sections=[
                PromptSection.role("你是意图分类器"),
                PromptSection.context("支持类别: 体育/科技/娱乐"),
                PromptSection.task("对用户输入分类: {{text}}"),
                PromptSection.examples([
                    FewShotExample(
                        user_input="今天NBA比赛",
                        assistant_output='{"intent": "体育", "confidence": 0.95}'
                    ),
                    FewShotExample(
                        user_input="iPhone 16发布",
                        assistant_output='{"intent": "科技", "confidence": 0.9}'
                    ),
                ]),
                PromptSection.output_format("返回JSON"),
            ],
            variables=[
                PromptVariable(name="text", description="用户输入", required=True),
            ],
            model="gpt-4",
            temperature=0.3,
            author="tester",
        )
        # 2. 注册
        self.assertTrue(register_prompt(template))
        # 3. 渲染
        rendered = render_prompt("intent_classifier", inputs={"text": "世界杯"})
        self.assertIsNotNone(rendered)
        # 4. 验证消息结构
        msgs = rendered.to_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("意图分类器", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")
        # 验证变量插值
        self.assertIn("世界杯", msgs[1]["content"])
        # 验证示例
        self.assertIn("NBA", msgs[1]["content"])
        # 验证 hash
        self.assertEqual(len(rendered.hash), 16)
        # 5. 验证模型参数
        self.assertEqual(rendered.model, "gpt-4")
        self.assertEqual(rendered.temperature, 0.3)
        # 6. 注销
        self.assertTrue(unregister_prompt("intent_classifier"))
        self.assertIsNone(get_prompt("intent_classifier"))

    def test02_validation_missing_required(self):
        """验证必填变量检查"""
        t = PromptTemplate(
            name="val_test",
            sections=[PromptSection.task("处理{{required}}")],
            variables=[PromptVariable(name="required", required=True)],
        )
        register_prompt(t)
        # 渲染时缺少必填变量，应该保留占位符
        rendered = render_prompt("val_test", inputs={})
        self.assertIn("{{required}}", rendered.user_content)

    def test03_json_complex_variables(self):
        """测试复杂变量（dict/list）的 JSON 序列化"""
        t = PromptTemplate(
            name="json_test",
            sections=[PromptSection.task("数据: {{data}}")],
            variables=[PromptVariable(name="data")],
        )
        register_prompt(t)
        data = {"items": [1, 2, 3], "name": "test"}
        rendered = render_prompt("json_test", inputs={"data": data})
        # dict 应该被 json 序列化
        self.assertIsNotNone(rendered)
        self.assertIn("items", rendered.user_content)
        self.assertIn("test", rendered.user_content)

    def test04_chain_composition(self):
        """测试 PromptChain 组合"""
        chain = PromptChain(name="pipeline")
        chain.add_step(PromptTemplate(name="step1",
                                      sections=[PromptSection.task("第一步")]))
        chain.add_step(PromptTemplate(name="step2",
                                      sections=[PromptSection.task("第二步")]))
        # 链中有两个步骤
        self.assertEqual(chain.step_count(), 2)
        # 渲染每个步骤
        for tmpl in chain.get_steps():
            r = PromptFormatter.format(tmpl, {})
            self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
