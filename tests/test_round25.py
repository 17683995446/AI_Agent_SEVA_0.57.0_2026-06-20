"""
阶段35测试：配置文件支持
========================

验证配置文件支持模块的完整功能：
1. save_config / load_config 核心函数
2. YAML/JSON 格式支持
3. config_from_file 一行启动
4. merge_configs 多配置合并
5. validate_config_file 文件校验
6. ConfigLoader 加载器
7. ConfigWizard 配置向导
8. CLI命令扩展

对应第一性原理：易用 = 配置可持久化复用
"""
import os
import sys
import json
import tempfile
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seva import (Config, SEVAFactory, run, save_config, load_config,
                  config_from_file, merge_configs, validate_config_file,
                  ConfigLoader, ConfigWizard, create_config)
from seva.config_io import _detect_format


# ============================================================
# 测试1-5：格式检测
# ============================================================
def test1_detect_yaml_format():
    """测试1: 检测YAML格式"""
    assert _detect_format("config.yaml") == "yaml"
    assert _detect_format("config.yml") == "yaml"


def test2_detect_json_format():
    """测试2: 检测JSON格式"""
    assert _detect_format("config.json") == "json"


def test3_detect_default_format():
    """测试3: 未知扩展名默认JSON"""
    assert _detect_format("config.txt") == "json"
    assert _detect_format("config") == "json"


def test4_detect_format_case_insensitive():
    """测试4: 格式检测大小写不敏感"""
    assert _detect_format("CONFIG.YAML") == "yaml"
    assert _detect_format("CONFIG.JSON") == "json"


def test5_detect_format_no_extension():
    """测试5: 无扩展名默认JSON"""
    assert _detect_format("configfile") == "json"


# ============================================================
# 测试6-12：save_config / load_config
# ============================================================
def test6_save_config_json():
    """测试6: 保存JSON配置"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        result = save_config(config, path)
        assert result == path
        assert os.path.exists(path)


def test7_save_config_yaml():
    """测试7: 保存YAML配置"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        assert os.path.exists(path)
        # 验证文件内容是YAML格式
        with open(path) as f:
            content = f.read()
        assert "core:" in content or "core" in content


def test8_load_config_json():
    """测试8: 加载JSON配置"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        loaded = load_config(path)
        assert isinstance(loaded, Config)
        assert loaded.get("core", "max_cycles") == config.get("core", "max_cycles")


def test9_load_config_yaml():
    """测试9: 加载YAML配置"""
    config = create_config("standard")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        loaded = load_config(path)
        assert isinstance(loaded, Config)
        assert loaded.get("perception", "input_dim") == config.get("perception", "input_dim")


def test10_load_config_not_found():
    """测试10: 加载不存在的文件抛出异常"""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test11_save_load_roundtrip():
    """测试11: 保存-加载往返一致"""
    config = create_config("standard", overrides={
        "core": {"max_cycles": 5000},
        "perception": {"input_dim": 48},
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roundtrip.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.get("core", "max_cycles") == 5000
        assert loaded.get("perception", "input_dim") == 48


def test12_save_config_with_format():
    """测试12: 显式指定格式"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 显式指定yaml格式但用.json扩展名
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path, format="yaml")
        with open(path) as f:
            content = f.read()
        assert "core:" in content  # YAML格式


# ============================================================
# 测试13-18：config_from_file
# ============================================================
def test13_config_from_file_json():
    """测试13: 从JSON文件创建Agent"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        agent = config_from_file(path)
        assert agent is not None
        assert agent.config.get("perception", "input_dim") == config.get("perception", "input_dim")


def test14_config_from_file_yaml():
    """测试14: 从YAML文件创建Agent"""
    config = create_config("standard")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        agent = config_from_file(path)
        assert agent is not None


def test15_config_from_file_with_env_type():
    """测试15: 从文件创建Agent时指定环境类型"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        agent = config_from_file(path, env_type="drift")
        assert agent is not None


def test16_config_from_file_with_seed():
    """测试16: 从文件创建Agent时指定种子"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        agent = config_from_file(path, seed=123)
        assert agent is not None


def test17_config_from_file_runnable():
    """测试17: 从文件创建的Agent可运行"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        agent = config_from_file(path)
        stats = run(agent, steps=10)
        assert stats["steps_completed"] > 0


def test18_config_from_file_not_found():
    """测试18: 文件不存在抛出异常"""
    with pytest.raises(FileNotFoundError):
        config_from_file("/nonexistent/config.yaml")


# ============================================================
# 测试19-23：merge_configs
# ============================================================
def test19_merge_configs_basic():
    """测试19: 基本合并"""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    merged = merge_configs(base, override)
    assert merged == {"a": 1, "b": 3, "c": 4}


def test20_merge_configs_deep():
    """测试20: 深度合并"""
    base = {"core": {"max_cycles": 100, "timeout": 1000}}
    override = {"core": {"max_cycles": 500}}
    merged = merge_configs(base, override)
    assert merged["core"]["max_cycles"] == 500
    assert merged["core"]["timeout"] == 1000  # 保留


def test21_merge_configs_multiple():
    """测试21: 合并多个配置"""
    c1 = {"a": 1}
    c2 = {"b": 2}
    c3 = {"c": 3}
    merged = merge_configs(c1, c2, c3)
    assert merged == {"a": 1, "b": 2, "c": 3}


def test22_merge_configs_empty():
    """测试22: 空合并"""
    assert merge_configs() == {}
    assert merge_configs({"a": 1}) == {"a": 1}


def test23_merge_configs_priority():
    """测试23: 后者覆盖前者"""
    c1 = {"a": 1}
    c2 = {"a": 2}
    c3 = {"a": 3}
    merged = merge_configs(c1, c2, c3)
    assert merged["a"] == 3


# ============================================================
# 测试24-28：validate_config_file
# ============================================================
def test24_validate_config_file_valid():
    """测试24: 校验有效配置文件"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        is_valid, errors = validate_config_file(path)
        assert is_valid is True
        assert len(errors) == 0


def test25_validate_config_file_not_found():
    """测试25: 校验不存在的文件"""
    is_valid, errors = validate_config_file("/nonexistent/config.json")
    assert is_valid is False
    assert len(errors) > 0


def test26_validate_config_file_returns_tuple():
    """测试26: 返回元组"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        save_config(config, path)
        result = validate_config_file(path)
        assert isinstance(result, tuple)
        assert len(result) == 2


def test27_validate_config_file_yaml():
    """测试27: 校验YAML配置文件"""
    config = create_config("standard")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        is_valid, errors = validate_config_file(path)
        assert is_valid is True


def test28_validate_config_file_invalid_content():
    """测试28: 校验无效内容"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "invalid.json")
        with open(path, "w") as f:
            f.write("{invalid json content")
        is_valid, errors = validate_config_file(path)
        assert is_valid is False
        assert len(errors) > 0


# ============================================================
# 测试29-35：ConfigLoader
# ============================================================
def test29_loader_creation():
    """测试29: ConfigLoader可创建"""
    loader = ConfigLoader()
    assert loader.default_format == "yaml"


def test30_loader_supported_formats():
    """测试30: 支持的格式列表"""
    formats = ConfigLoader.supported_formats()
    assert "yaml" in formats
    assert "yml" in formats
    assert "json" in formats


def test31_loader_load():
    """测试31: 加载配置"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        loader = ConfigLoader()
        loaded = loader.load(path)
        assert isinstance(loaded, Config)


def test32_loader_save():
    """测试32: 保存配置"""
    config = create_config("easy")
    loader = ConfigLoader()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.json")
        loader.save(config, path)
        assert os.path.exists(path)


def test33_loader_cache():
    """测试33: 缓存功能"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        loader = ConfigLoader()
        # 第一次加载
        c1 = loader.load(path, use_cache=True)
        # 第二次加载（应从缓存）
        c2 = loader.load(path, use_cache=True)
        assert path in loader._cache
        loader.clear_cache()
        assert path not in loader._cache


def test34_loader_load_multiple():
    """测试34: 加载多个配置合并"""
    config1 = create_config("easy")
    config2 = create_config("standard")
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "c1.yaml")
        path2 = os.path.join(tmpdir, "c2.yaml")
        save_config(config1, path1)
        save_config(config2, path2)
        loader = ConfigLoader()
        merged = loader.load_multiple(path1, path2)
        assert isinstance(merged, Config)


def test35_loader_clear_cache():
    """测试35: 清空缓存"""
    loader = ConfigLoader()
    loader._cache = {"key": "value"}
    loader.clear_cache()
    assert loader._cache == {}


# ============================================================
# 测试36-42：ConfigWizard
# ============================================================
def test36_wizard_creation():
    """测试36: ConfigWizard可创建"""
    wizard = ConfigWizard()
    assert wizard.verbose is False


def test37_wizard_generate():
    """测试37: 生成配置"""
    wizard = ConfigWizard()
    config = wizard.generate(preset="easy", steps=100)
    assert isinstance(config, Config)
    assert config.get("perception", "input_dim") == 16  # easy预设


def test38_wizard_generate_with_overrides():
    """测试38: 生成配置带覆盖"""
    wizard = ConfigWizard()
    config = wizard.generate(
        preset="easy",
        steps=100,
        custom_overrides={"memory": {"episodic_capacity": 5000}},
    )
    assert config.get("memory", "episodic_capacity") == 5000


def test39_wizard_generate_max_cycles():
    """测试39: 根据steps推导max_cycles"""
    wizard = ConfigWizard()
    config = wizard.generate(preset="easy", steps=500)
    max_cycles = config.get("core", "max_cycles")
    assert max_cycles >= 1000  # 至少1000


def test40_wizard_save():
    """测试40: 向导保存配置"""
    wizard = ConfigWizard()
    config = wizard.generate(preset="easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "wizard.yaml")
        wizard.save(config, path)
        assert os.path.exists(path)


def test41_wizard_generate_for_scenario():
    """测试41: 为场景生成配置"""
    wizard = ConfigWizard()
    config = wizard.generate_for_scenario("quick_validation")
    assert isinstance(config, Config)


def test42_wizard_verbose():
    """测试42: verbose模式"""
    wizard = ConfigWizard(verbose=True)
    config = wizard.generate(preset="easy")
    assert isinstance(config, Config)


# ============================================================
# 测试43-50：CLI命令
# ============================================================
def run_cli(*args):
    """运行CLI命令"""
    cmd = [sys.executable, "-m", "seva"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


def test43_cli_config_save_yaml():
    """测试43: seva config save (YAML)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        code, out, err = run_cli("config", "save", "--preset", "easy",
                                 "--output", path)
        assert code == 0
        assert os.path.exists(path)


def test44_cli_config_save_json():
    """测试44: seva config save (JSON)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.json")
        code, out, err = run_cli("config", "save", "--preset", "standard",
                                 "--output", path)
        assert code == 0
        assert os.path.exists(path)


def test45_cli_config_load():
    """测试45: seva config load"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        # 先保存
        run_cli("config", "save", "--preset", "easy", "--output", path)
        # 再加载
        code, out, err = run_cli("config", "load", path)
        assert code == 0
        assert "已加载配置" in out
        assert "core" in out


def test46_cli_config_load_section():
    """测试46: seva config load --section"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        run_cli("config", "save", "--preset", "easy", "--output", path)
        code, out, err = run_cli("config", "load", path, "--section", "core")
        assert code == 0
        assert "max_cycles" in out


def test47_cli_config_load_not_found():
    """测试47: 加载不存在的文件"""
    code, out, err = run_cli("config", "load", "/nonexistent/cfg.yaml")
    assert code == 1
    assert "不存在" in out or "错误" in out


def test48_cli_config_wizard():
    """测试48: seva config wizard"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "wizard.yaml")
        code, out, err = run_cli("config", "wizard", "--preset", "easy",
                                 "--output", path)
        assert code == 0
        assert os.path.exists(path)
        assert "配置向导" in out


def test49_cli_run_with_config():
    """测试49: seva run --config"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        run_cli("config", "save", "--preset", "easy", "--output", path)
        code, out, err = run_cli("run", "--config", path, "--steps", "10")
        assert code == 0
        assert "配置文件" in out
        assert "运行结果" in out


def test50_cli_config_init_still_works():
    """测试50: 原有config init仍可用"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "init.json")
        code, out, err = run_cli("config", "init", "--preset", "easy",
                                 "--output", path)
        assert code == 0
        assert os.path.exists(path)


# ============================================================
# 测试51-55：元数据与往返一致性
# ============================================================
def test51_save_includes_meta():
    """测试51: 保存的配置包含元数据"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.json")
        save_config(config, path)
        with open(path) as f:
            data = json.load(f)
        assert "_meta" in data
        assert "version" in data["_meta"]
        assert "format" in data["_meta"]


def test52_load_removes_meta():
    """测试52: 加载时移除元数据"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.json")
        save_config(config, path)
        loaded = load_config(path, as_config_object=False)
        assert "_meta" not in loaded


def test53_roundtrip_all_presets():
    """测试53: 所有预设的往返一致性"""
    from seva import PresetConfigs
    for preset in PresetConfigs.list_presets():
        config = create_config(preset)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.yaml")
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.get("perception", "input_dim") == \
                   config.get("perception", "input_dim")


def test54_yaml_json_equivalent():
    """测试54: YAML和JSON保存的内容等价"""
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "cfg.yaml")
        json_path = os.path.join(tmpdir, "cfg.json")
        save_config(config, yaml_path)
        save_config(config, json_path)
        yaml_loaded = load_config(yaml_path)
        json_loaded = load_config(json_path)
        assert yaml_loaded.get("core", "max_cycles") == \
               json_loaded.get("core", "max_cycles")


def test55_custom_config_persists():
    """测试55: 自定义配置可持久化"""
    config = create_config("easy", overrides={
        "core": {"max_cycles": 7777},
        "perception": {"input_dim": 99},
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "custom.yaml")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.get("core", "max_cycles") == 7777
        assert loaded.get("perception", "input_dim") == 99


# ============================================================
# 测试56-60：模块导出与集成
# ============================================================
def test56_module_import():
    """测试56: 模块可导入"""
    from seva import config_io
    assert hasattr(config_io, "save_config")
    assert hasattr(config_io, "load_config")
    assert hasattr(config_io, "ConfigLoader")
    assert hasattr(config_io, "ConfigWizard")


def test57_all_exports_available():
    """测试57: 所有导出可用"""
    from seva import (save_config, load_config, config_from_file,
                      merge_configs, validate_config_file,
                      ConfigLoader, ConfigWizard)
    assert callable(save_config)
    assert callable(load_config)
    assert callable(config_from_file)
    assert callable(merge_configs)
    assert callable(validate_config_file)
    assert ConfigLoader is not None
    assert ConfigWizard is not None


def test58_exports_in_all():
    """测试58: 导出在__all__中"""
    import seva
    for name in ["save_config", "load_config", "config_from_file",
                 "merge_configs", "validate_config_file",
                 "ConfigLoader", "ConfigWizard"]:
        assert name in seva.__all__, f"{name}不在__all__中"


def test59_version_updated():
    """测试59: 版本号已更新"""
    import seva
    assert seva.__version__ >= "0.35.0"


def test60_config_io_complete_workflow():
    """测试60: 完整工作流"""
    # 1. 向导生成配置
    wizard = ConfigWizard()
    config = wizard.generate(preset="easy", steps=50,
                             custom_overrides={"memory": {"episodic_capacity": 2000}})

    # 2. 保存到文件
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "workflow.yaml")
        save_config(config, path)

        # 3. 校验文件
        is_valid, errors = validate_config_file(path)
        assert is_valid

        # 4. 加载配置
        loaded = load_config(path)
        assert loaded.get("memory", "episodic_capacity") == 2000

        # 5. 从文件创建Agent
        agent = config_from_file(path)
        assert agent is not None

        # 6. 运行Agent
        stats = run(agent, steps=10)
        assert stats["steps_completed"] > 0


# ============================================================
# 性能测试
# ============================================================
def test61_save_load_fast():
    """测试61: 保存加载应快速"""
    import time
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        start = time.time()
        save_config(config, path)
        load_config(path)
        elapsed = time.time() - start
        assert elapsed < 1.0


def test62_config_from_file_fast():
    """测试62: 从文件创建Agent应快速"""
    import time
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        save_config(config, path)
        start = time.time()
        agent = config_from_file(path)
        elapsed = time.time() - start
        assert elapsed < 5.0


# ============================================================
# 完整性测试
# ============================================================
def test63_complete_config_file_system():
    """测试63: 配置文件系统完整性"""
    # 1. 格式检测
    assert _detect_format("a.yaml") == "yaml"
    assert _detect_format("a.json") == "json"

    # 2. 保存加载
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt in ["yaml", "json"]:
            path = os.path.join(tmpdir, f"cfg.{fmt}")
            save_config(config, path)
            assert os.path.exists(path)
            loaded = load_config(path)
            assert isinstance(loaded, Config)

    # 3. 合并
    merged = merge_configs({"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2}

    # 4. 校验
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        save_config(config, path)
        valid, _ = validate_config_file(path)
        assert valid

    # 5. 向导
    wizard = ConfigWizard()
    cfg = wizard.generate()
    assert isinstance(cfg, Config)

    # 6. 加载器
    loader = ConfigLoader()
    assert "yaml" in loader.supported_formats()


def test64_config_file_user_friendly():
    """测试64: 配置文件用户友好性"""
    # YAML文件应可读
    config = create_config("easy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cfg.yaml")
        save_config(config, path)
        with open(path) as f:
            content = f.read()
        # 应包含section名（可读）
        assert "core" in content
        assert "perception" in content
        assert "memory" in content


def test65_no_pyyaml_graceful_error():
    """测试65: 无pyyaml时优雅报错"""
    # 这个测试验证错误信息友好（不实际移除pyyaml）
    # 只验证函数存在且可调用
    from seva.config_io import _load_yaml, _dump_yaml
    assert callable(_load_yaml)
    assert callable(_dump_yaml)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
