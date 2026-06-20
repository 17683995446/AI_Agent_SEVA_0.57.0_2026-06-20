"""
配置文件支持模块（阶段35）
对应第一性原理：易用 = 配置可持久化复用，无需每次重写

提供：
- load_config(): 从YAML/JSON文件加载配置
- save_config(): 保存配置到YAML/JSON文件
- ConfigLoader: 配置加载器（支持多格式）
- ConfigWizard: 配置向导（引导式生成配置）
- merge_configs(): 合并多个配置
- validate_config_file(): 校验配置文件
- config_from_file(): 从文件创建Agent（一行启动）

使用示例:
    from seva import save_config, load_config, config_from_file

    # 保存当前配置
    save_config(agent.config, "my_config.yaml")

    # 加载配置创建Agent
    agent = config_from_file("my_config.yaml")
"""
import os
import json
import copy
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .api import SEVAFactory, PresetConfigs


# ============================================================
# 格式检测与转换
# ============================================================

def _detect_format(file_path: str) -> str:
    """根据文件扩展名检测格式"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".yaml", ".yml"):
        return "yaml"
    elif ext == ".json":
        return "json"
    else:
        # 默认JSON
        return "json"


def _load_yaml(file_path: str) -> Dict:
    """加载YAML文件"""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "加载YAML文件需要pyyaml包。请运行: pip install pyyaml"
        ) from e
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _dump_yaml(data: Dict, file_path: str):
    """保存为YAML文件"""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "保存YAML文件需要pyyaml包。请运行: pip install pyyaml"
        ) from e
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False,
                       allow_unicode=True, sort_keys=False)


def _load_json(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Dict, file_path: str):
    """保存为JSON文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# 核心函数
# ============================================================

def save_config(config: Config, file_path: str,
                format: Optional[str] = None) -> str:
    """
    保存配置到文件

    参数:
        config: Config对象
        file_path: 输出文件路径
        format: 格式（yaml/json），None则自动检测

    返回:
        实际保存的文件路径

    示例:
        save_config(agent.config, "my_config.yaml")
        save_config(agent.config, "my_config.json")
    """
    fmt = format or _detect_format(file_path)
    config_dict = config.to_dict()

    # 添加元数据
    config_dict["_meta"] = {
        "version": _get_version(),
        "format": fmt,
        "saved_by": "seva.save_config",
    }

    if fmt == "yaml":
        _dump_yaml(config_dict, file_path)
    else:
        _dump_json(config_dict, file_path)

    return file_path


def load_config(file_path: str,
                as_config_object: bool = True) -> Config:
    """
    从文件加载配置

    参数:
        file_path: 配置文件路径
        as_config_object: True返回Config对象，False返回dict

    返回:
        Config对象或dict

    示例:
        config = load_config("my_config.yaml")
        agent = SEVAFactory.create("standard", overrides=config.to_dict())
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    fmt = _detect_format(file_path)
    if fmt == "yaml":
        data = _load_yaml(file_path)
    else:
        data = _load_json(file_path)

    # 移除元数据
    data.pop("_meta", None)

    if as_config_object:
        return Config(instance_overrides=data)
    return data


def config_from_file(file_path: str,
                     env_type: str = "mixed",
                     seed: int = 42):
    """
    从配置文件创建Agent（一行启动）

    参数:
        file_path: 配置文件路径
        env_type: 环境类型
        seed: 随机种子

    返回:
        已初始化的Agent实例

    示例:
        agent = config_from_file("my_config.yaml")
        stats = run(agent, steps=100)
    """
    config_dict = load_config(file_path, as_config_object=False)
    # 使用加载的配置作为overrides
    agent = SEVAFactory.create(
        preset="standard",  # 基础预设
        env_type=env_type,
        seed=seed,
        overrides=config_dict,
    )
    return agent


def merge_configs(*configs: Dict) -> Dict:
    """
    合并多个配置字典（后者覆盖前者）

    参数:
        *configs: 多个配置字典

    返回:
        合并后的配置字典

    示例:
        merged = merge_configs(base_config, custom_config, runtime_config)
    """
    if not configs:
        return {}
    result = copy.deepcopy(configs[0])
    for cfg in configs[1:]:
        _deep_update(result, cfg)
    return result


def _deep_update(base: Dict, override: Dict):
    """深度更新字典"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def validate_config_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    校验配置文件

    参数:
        file_path: 配置文件路径

    返回:
        (is_valid, errors) 元组

    示例:
        valid, errors = validate_config_file("my_config.yaml")
        if not valid:
            for e in errors:
                print(f"  - {e}")
    """
    try:
        config = load_config(file_path)
        errors = config.validate()
        return len(errors) == 0, errors
    except FileNotFoundError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"加载配置失败: {e}"]


def _get_version() -> str:
    """获取版本号"""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


# ============================================================
# 配置加载器
# ============================================================

class ConfigLoader:
    """
    配置加载器：支持多格式、多源加载

    使用示例:
        loader = ConfigLoader()
        config = loader.load("my_config.yaml")
        loader.save(config, "output.json")
    """

    SUPPORTED_FORMATS = ["yaml", "yml", "json"]

    def __init__(self, default_format: str = "yaml"):
        self.default_format = default_format
        self._cache: Dict[str, Config] = {}

    def load(self, file_path: str,
             use_cache: bool = False) -> Config:
        """加载配置文件"""
        if use_cache and file_path in self._cache:
            return self._cache[file_path]

        config = load_config(file_path)

        if use_cache:
            self._cache[file_path] = config

        return config

    def save(self, config: Config, file_path: str,
             format: Optional[str] = None) -> str:
        """保存配置到文件"""
        fmt = format or _detect_format(file_path)
        if fmt not in ("yaml", "json"):
            fmt = self.default_format
        return save_config(config, file_path, format=fmt)

    def load_multiple(self, *file_paths: str) -> Config:
        """
        加载多个配置文件并合并（后者覆盖前者）

        示例:
            config = loader.load_multiple("base.yaml", "override.yaml")
        """
        if not file_paths:
            return Config()

        merged_dict = {}
        for path in file_paths:
            cfg_dict = load_config(path, as_config_object=False)
            _deep_update(merged_dict, cfg_dict)

        return Config(instance_overrides=merged_dict)

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    @classmethod
    def supported_formats(cls) -> List[str]:
        """返回支持的格式列表"""
        return list(cls.SUPPORTED_FORMATS)


# ============================================================
# 配置向导
# ============================================================

class ConfigWizard:
    """
    配置向导：引导式生成配置

    使用示例:
        wizard = ConfigWizard()
        config = wizard.generate(
            preset="standard",
            env_type="mixed",
            steps=100,
            custom_overrides={"memory": {"episodic_capacity": 5000}},
        )
        wizard.save(config, "my_config.yaml")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def generate(self,
                 preset: str = "standard",
                 env_type: str = "mixed",
                 steps: int = 100,
                 seed: int = 42,
                 custom_overrides: Optional[Dict] = None) -> Config:
        """
        生成配置

        参数:
            preset: 基础预设模板
            env_type: 环境类型
            steps: 运行步数（用于推导max_cycles）
            seed: 随机种子
            custom_overrides: 自定义覆盖

        返回:
            Config对象
        """
        if self.verbose:
            print(f"[ConfigWizard] 生成配置")
            print(f"  预设: {preset}")
            print(f"  环境: {env_type}")
            print(f"  步数: {steps}")

        # 获取预设配置
        preset_config = copy.deepcopy(PresetConfigs.get(preset))

        # 根据steps推导max_cycles（确保足够运行指定步数）
        if "core" not in preset_config:
            preset_config["core"] = {}
        preset_config["core"]["max_cycles"] = max(
            preset_config["core"].get("max_cycles", 1000),
            steps * 2,  # 留余量
        )

        # 应用自定义覆盖
        if custom_overrides:
            _deep_update(preset_config, custom_overrides)

        # 创建Config对象
        config = Config(instance_overrides=preset_config)

        if self.verbose:
            print(f"  max_cycles: {config.get('core', 'max_cycles')}")
            print(f"  input_dim: {config.get('perception', 'input_dim')}")

        return config

    def generate_for_scenario(self,
                              scenario_name: str,
                              custom_overrides: Optional[Dict] = None
                              ) -> Config:
        """
        为预设场景生成配置

        参数:
            scenario_name: 场景名称
            custom_overrides: 自定义覆盖

        返回:
            Config对象
        """
        from .scenarios import PresetScenarios
        scenario = PresetScenarios.get(scenario_name)

        # 合并场景的overrides
        overrides = copy.deepcopy(scenario.overrides)
        if custom_overrides:
            _deep_update(overrides, custom_overrides)

        return self.generate(
            preset=scenario.preset,
            env_type=scenario.env_type,
            steps=scenario.steps,
            seed=scenario.seed,
            custom_overrides=overrides,
        )

    def save(self, config: Config, file_path: str,
             format: Optional[str] = None) -> str:
        """保存配置到文件"""
        return save_config(config, file_path, format=format)

    def interactive(self) -> Config:
        """
        交互式生成配置（在终端中逐步询问）

        返回:
            Config对象

        注意: 此方法需要终端交互，在非交互环境中使用默认值
        """
        # 在非交互环境中使用默认值
        # 实际交互环境会读取stdin
        try:
            if not sys.stdin.isatty():
                # 非交互环境，使用默认值
                if self.verbose:
                    print("[ConfigWizard] 非交互环境，使用默认值")
                return self.generate()
        except Exception:
            return self.generate()

        # 交互式询问
        print("=" * 50)
        print("SEVA配置向导")
        print("=" * 50)

        preset = input("选择预设 (easy/standard/research/production) "
                       "[standard]: ").strip() or "standard"
        env_type = input("环境类型 (blank/mixed/drift/adversarial) "
                         "[mixed]: ").strip() or "mixed"
        steps_str = input("运行步数 [100]: ").strip() or "100"
        try:
            steps = int(steps_str)
        except ValueError:
            steps = 100

        seed_str = input("随机种子 [42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 42

        output_file = input("保存配置到文件（留空则不保存）: ").strip()

        config = self.generate(
            preset=preset,
            env_type=env_type,
            steps=steps,
            seed=seed,
        )

        if output_file:
            self.save(config, output_file)
            print(f"\n配置已保存到: {output_file}")

        return config


# 导入sys用于interactive方法
import sys  # noqa: E402
