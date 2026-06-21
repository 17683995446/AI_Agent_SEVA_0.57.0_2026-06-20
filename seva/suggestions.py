"""
智能建议与推荐系统（阶段50）
对应第一性原理：主动 = 系统主动给出有用建议，而非被动等待用户提问

提供能力:
    - Suggestion: 建议数据结构（类型/优先级/消息/操作）
    - ContextAnalyzer: 上下文分析器（分析Agent状态/用户操作）
    - UserProfiler: 用户画像（技能水平/使用模式）
    - Recommender: 推荐系统（配置/命令/资源推荐）
    - SuggestionEngine: 建议引擎（综合生成建议）
    - 便捷函数: suggest / recommend / get_suggestions

使用示例:
    from seva import SuggestionEngine, suggest

    # 获取建议
    suggestions = suggest(agent)
    for s in suggestions:
        print(s.format())

    # 推荐配置
    from seva import Recommender
    rec = Recommender().recommend_config(scenario="research")
    print(rec)
"""
from __future__ import annotations

import os
import time
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from pathlib import Path
from enum import Enum


# ============================================================================
# 常量
# ============================================================================

class SuggestionType:
    """建议类型"""
    OPTIMIZATION = "optimization"      # 优化建议
    CONFIGURATION = "configuration"    # 配置建议
    TROUBLESHOOTING = "troubleshooting"  # 故障排除
    BEST_PRACTICE = "best_practice"    # 最佳实践
    LEARNING = "learning"              # 学习建议
    WARNING = "warning"                # 警告建议
    INFO = "info"                      # 信息建议


class SuggestionPriority:
    """建议优先级"""
    CRITICAL = 1    # 关键（必须处理）
    HIGH = 2        # 高（强烈建议）
    MEDIUM = 3      # 中（建议考虑）
    LOW = 4         # 低（可选）

    @classmethod
    def to_str(cls, priority: int) -> str:
        names = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
        return names.get(priority, "UNKNOWN")


class UserSkillLevel:
    """用户技能水平"""
    BEGINNER = "beginner"      # 初学者
    INTERMEDIATE = "intermediate"  # 中级
    ADVANCED = "advanced"      # 高级
    EXPERT = "expert"          # 专家


# 用户画像存储路径
PROFILE_FILE = Path.home() / ".seva" / "user_profile.json"


# ============================================================================
# Suggestion - 建议数据结构
# ============================================================================

class Suggestion:
    """
    建议数据结构

    属性:
        suggestion_id: 唯一ID
        type: 建议类型
        priority: 优先级（1-4）
        title: 标题
        message: 详细消息
        action: 建议操作（如命令）
        category: 分类
        confidence: 置信度（0-1）
    """

    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, title: str, message: str = "",
                 type: str = SuggestionType.INFO,
                 priority: int = SuggestionPriority.MEDIUM,
                 action: str = "",
                 category: str = "",
                 confidence: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        with Suggestion._id_lock:
            self.suggestion_id = Suggestion._next_id
            Suggestion._next_id += 1
        self.type = type
        self.priority = priority
        self.title = title
        self.message = message
        self.action = action
        self.category = category or type
        self.confidence = max(0.0, min(1.0, confidence))
        self.metadata = metadata or {}
        self.created_at = time.time()

    def format(self) -> str:
        """格式化建议"""
        priority_str = SuggestionPriority.to_str(self.priority)
        lines = [
            f"[{priority_str}] {self.title}",
            f"  类型: {self.type}",
            f"  内容: {self.message}",
        ]
        if self.action:
            lines.append(f"  建议: {self.action}")
        if self.confidence < 1.0:
            lines.append(f"  置信度: {self.confidence:.0%}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.type,
            "priority": self.priority,
            "priority_str": SuggestionPriority.to_str(self.priority),
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "category": self.category,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Suggestion":
        s = cls(
            title=data["title"],
            message=data.get("message", ""),
            type=data.get("type", SuggestionType.INFO),
            priority=data.get("priority", SuggestionPriority.MEDIUM),
            action=data.get("action", ""),
            category=data.get("category", ""),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
        s.created_at = data.get("created_at", time.time())
        return s

    def __repr__(self) -> str:
        return f"Suggestion({self.title!r}, priority={self.priority})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Suggestion):
            return False
        return (self.title == other.title and
                self.type == other.type and
                self.priority == other.priority)

    def __hash__(self) -> int:
        return hash((self.title, self.type, self.priority))


# ============================================================================
# ContextAnalyzer - 上下文分析器
# ============================================================================

class ContextAnalyzer:
    """
    上下文分析器

    分析Agent状态、用户操作历史，识别问题和机会
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def analyze_agent(self, agent: Any) -> Dict[str, Any]:
        """分析Agent状态"""
        context = {
            "cycle_count": 0,
            "total_reward": 0.0,
            "avg_cycle_time": 0.0,
            "avg_prediction_error": 0.0,
            "tasks_completed": 0,
            "skills_total": 0,
            "evolution_evaluations": 0,
            "current_difficulty": "EASY",
            "mode": "unknown",
            "healthy": True,
            "issues": [],
        }

        try:
            context["cycle_count"] = getattr(agent, "cycle_count", 0)
            context["total_reward"] = float(
                getattr(agent, "total_real_reward", 0.0))
            context["avg_cycle_time"] = float(
                getattr(agent, "avg_cycle_time", 0.0))
            context["avg_prediction_error"] = float(
                getattr(agent, "avg_prediction_error", 0.0))
            context["mode"] = getattr(agent, "mode", "unknown")

            # 任务统计
            task_mgr = getattr(agent, "task_manager", None)
            if task_mgr:
                stats = task_mgr.get_stats() if hasattr(
                    task_mgr, "get_stats") else {}
                context["tasks_completed"] = stats.get(
                    "total_completed", 0)

            # 技能统计
            skill_mgr = getattr(agent, "skill_manager", None)
            if skill_mgr:
                stats = skill_mgr.get_stats() if hasattr(
                    skill_mgr, "get_stats") else {}
                context["skills_total"] = stats.get("total_skills", 0)

            # 进化统计
            evo_mgr = getattr(agent, "evolution_manager", None)
            if evo_mgr:
                stats = evo_mgr.get_stats() if hasattr(
                    evo_mgr, "get_stats") else {}
                context["evolution_evaluations"] = stats.get(
                    "total_evaluations", 0)
                context["current_difficulty"] = stats.get(
                    "current_difficulty", "EASY")

            # 健康检查
            context["healthy"] = self._check_health(agent, context)

        except Exception as e:
            context["issues"].append(f"分析异常: {e}")

        return context

    def _check_health(self, agent: Any, context: Dict) -> bool:
        """健康检查"""
        healthy = True
        # 预测误差过高
        if context["avg_prediction_error"] > 0.5:
            context["issues"].append("预测误差过高")
            healthy = False
        # 周期时间过长
        if context["avg_cycle_time"] > 1.0:
            context["issues"].append("周期时间过长")
            healthy = False
        # 奖励为负
        if context["total_reward"] < 0:
            context["issues"].append("总奖励为负")
            healthy = False
        return healthy

    def analyze_config(self, config: Any) -> Dict[str, Any]:
        """分析配置"""
        context = {
            "preset": "unknown",
            "input_dim": 0,
            "grid_size": 0,
            "max_cycles": 0,
            "issues": [],
        }
        try:
            if hasattr(config, "get"):
                context["input_dim"] = config.get("perception", "input_dim")
                context["grid_size"] = config.get(
                    "environment", "grid_size")
                context["max_cycles"] = config.get("core", "max_cycles")
            elif hasattr(config, "to_dict"):
                d = config.to_dict()
                if "perception" in d:
                    context["input_dim"] = d["perception"].get(
                        "input_dim", 0)
                if "environment" in d:
                    context["grid_size"] = d["environment"].get(
                        "grid_size", 0)
                if "core" in d:
                    context["max_cycles"] = d["core"].get(
                        "max_cycles", 0)
        except Exception as e:
            context["issues"].append(f"配置分析异常: {e}")
        return context

    def detect_issues(self, context: Dict[str, Any]) -> List[str]:
        """检测问题"""
        issues = list(context.get("issues", []))
        # 额外检测
        if context.get("cycle_count", 0) == 0:
            issues.append("Agent尚未运行")
        if context.get("skills_total", 0) == 0 and \
                context.get("cycle_count", 0) > 10:
            issues.append("运行多步但无技能生成")
        return issues


# ============================================================================
# UserProfiler - 用户画像
# ============================================================================

class UserProfiler:
    """
    用户画像

    跟踪用户技能水平、使用模式，用于个性化建议
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profile: Dict[str, Any] = {
            "skill_level": UserSkillLevel.BEGINNER,
            "commands_used": [],          # 使用过的命令
            "commands_count": {},         # 命令使用次数
            "features_used": set(),       # 使用过的功能
            "errors_encountered": 0,      # 遇到的错误数
            "sessions_count": 0,          # 会话数
            "total_runtime_steps": 0,     # 总运行步数
            "first_use": time.time(),
            "last_use": time.time(),
            "preferred_presets": {},      # 偏好的预设
            "preferred_env_types": {},    # 偏好的环境类型
        }
        self._load()

    def record_command(self, command: str) -> None:
        """记录命令使用"""
        with self._lock:
            if command not in self._profile["commands_count"]:
                self._profile["commands_count"][command] = 0
            self._profile["commands_count"][command] += 1
            if command not in self._profile["commands_used"]:
                self._profile["commands_used"].append(command)
            self._profile["last_use"] = time.time()
            self._update_skill_level()

    def record_feature(self, feature: str) -> None:
        """记录功能使用"""
        with self._lock:
            self._profile["features_used"].add(feature)
            self._profile["last_use"] = time.time()
            self._update_skill_level()

    def record_error(self) -> None:
        """记录错误"""
        with self._lock:
            self._profile["errors_encountered"] += 1

    def record_session(self, steps: int = 0,
                       preset: str = "",
                       env_type: str = "") -> None:
        """记录会话"""
        with self._lock:
            self._profile["sessions_count"] += 1
            self._profile["total_runtime_steps"] += steps
            if preset:
                self._profile["preferred_presets"][preset] = \
                    self._profile["preferred_presets"].get(preset, 0) + 1
            if env_type:
                self._profile["preferred_env_types"][env_type] = \
                    self._profile["preferred_env_types"].get(env_type, 0) + 1
            self._profile["last_use"] = time.time()
            self._update_skill_level()

    def _update_skill_level(self) -> None:
        """更新技能水平"""
        score = 0
        # 命令多样性
        cmd_diversity = len(self._profile["commands_used"])
        if cmd_diversity >= 15:
            score += 3
        elif cmd_diversity >= 8:
            score += 2
        elif cmd_diversity >= 3:
            score += 1

        # 功能使用
        feat_count = len(self._profile["features_used"])
        if feat_count >= 10:
            score += 3
        elif feat_count >= 5:
            score += 2
        elif feat_count >= 1:
            score += 1

        # 会话数
        sessions = self._profile["sessions_count"]
        if sessions >= 20:
            score += 2
        elif sessions >= 5:
            score += 1

        # 总运行步数
        total_steps = self._profile["total_runtime_steps"]
        if total_steps >= 1000:
            score += 2
        elif total_steps >= 100:
            score += 1

        # 根据总分判定技能水平
        if score >= 8:
            self._profile["skill_level"] = UserSkillLevel.EXPERT
        elif score >= 5:
            self._profile["skill_level"] = UserSkillLevel.ADVANCED
        elif score >= 2:
            self._profile["skill_level"] = UserSkillLevel.INTERMEDIATE
        else:
            self._profile["skill_level"] = UserSkillLevel.BEGINNER

    def get_skill_level(self) -> str:
        """获取技能水平"""
        return self._profile["skill_level"]

    def get_profile(self) -> Dict[str, Any]:
        """获取完整画像"""
        with self._lock:
            return {
                "skill_level": self._profile["skill_level"],
                "commands_used": list(self._profile["commands_used"]),
                "commands_count": dict(self._profile["commands_count"]),
                "features_used": list(self._profile["features_used"]),
                "errors_encountered": self._profile["errors_encountered"],
                "sessions_count": self._profile["sessions_count"],
                "total_runtime_steps": self._profile["total_runtime_steps"],
                "first_use": self._profile["first_use"],
                "last_use": self._profile["last_use"],
                "preferred_presets": dict(
                    self._profile["preferred_presets"]),
                "preferred_env_types": dict(
                    self._profile["preferred_env_types"]),
            }

    def get_report(self) -> str:
        """获取画像报告"""
        p = self.get_profile()
        lines = [
            "SEVA用户画像报告",
            "=" * 50,
            f"  技能水平:       {p['skill_level']}",
            f"  会话数:         {p['sessions_count']}",
            f"  总运行步数:     {p['total_runtime_steps']}",
            f"  使用命令数:     {len(p['commands_used'])}",
            f"  使用功能数:     {len(p['features_used'])}",
            f"  遇到错误数:     {p['errors_encountered']}",
        ]
        if p["preferred_presets"]:
            top_preset = max(p["preferred_presets"].items(),
                             key=lambda x: x[1])
            lines.append(f"  最常用预设:     {top_preset[0]} "
                         f"({top_preset[1]}次)")
        if p["preferred_env_types"]:
            top_env = max(p["preferred_env_types"].items(),
                          key=lambda x: x[1])
            lines.append(f"  最常用环境:     {top_env[0]} "
                         f"({top_env[1]}次)")
        first = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(p["first_use"]))
        last = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(p["last_use"]))
        lines.append(f"  首次使用:       {first}")
        lines.append(f"  最后使用:       {last}")
        return "\n".join(lines)

    def reset(self) -> None:
        """重置画像"""
        with self._lock:
            self._profile = {
                "skill_level": UserSkillLevel.BEGINNER,
                "commands_used": [],
                "commands_count": {},
                "features_used": set(),
                "errors_encountered": 0,
                "sessions_count": 0,
                "total_runtime_steps": 0,
                "first_use": time.time(),
                "last_use": time.time(),
                "preferred_presets": {},
                "preferred_env_types": {},
            }
        self._save()

    def _load(self) -> None:
        """从文件加载"""
        try:
            if PROFILE_FILE.exists():
                with open(PROFILE_FILE, "r") as f:
                    data = json.load(f)
                # 恢复（features_used从list转set）
                if "features_used" in data:
                    data["features_used"] = set(data["features_used"])
                self._profile.update(data)
        except Exception:
            pass

    def _save(self) -> None:
        """保存到文件"""
        try:
            PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = self.get_profile()
            # set转list
            data["features_used"] = list(data["features_used"])
            with open(PROFILE_FILE, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# ============================================================================
# Recommender - 推荐系统
# ============================================================================

class Recommender:
    """
    推荐系统

    基于场景、用户画像、当前状态推荐配置/命令/资源
    """

    # 场景到预设的推荐映射
    SCENARIO_PRESETS = {
        "quick_test": "easy",
        "demo": "easy",
        "daily": "standard",
        "production": "production",
        "research": "research",
        "benchmark": "research",
    }

    # 预设描述
    PRESET_DESCRIPTIONS = {
        "easy": "易用模式：小规模、快速、低资源占用（推荐入门）",
        "standard": "标准模式：平衡性能与资源（日常使用推荐）",
        "research": "研究模式：大规模、高精度、长时运行",
        "production": "生产模式：稳定优先、资源受限、监控密集",
    }

    # 命令推荐（基于技能水平）
    COMMAND_RECOMMENDATIONS = {
        UserSkillLevel.BEGINNER: [
            ("demo", "运行演示了解系统"),
            ("presets", "查看可用预设"),
            ("tutorial", "开始交互式教程"),
            ("help", "查看帮助系统"),
            ("wizard", "首次运行向导"),
        ],
        UserSkillLevel.INTERMEDIATE: [
            ("run", "启动Agent运行"),
            ("scenarios", "查看预设场景"),
            ("status", "查看系统状态"),
            ("prefs --list", "查看用户偏好"),
            ("perf --stats", "查看性能统计"),
        ],
        UserSkillLevel.ADVANCED: [
            ("scenario compare", "对比多个场景"),
            ("diagnose", "诊断系统"),
            ("monitor", "实时监控"),
            ("charts", "生成SVG图表"),
            ("session checkpoint", "创建检查点"),
        ],
        UserSkillLevel.EXPERT: [
            ("compare", "比较多个运行"),
            ("export --format json", "导出JSON报告"),
            ("perf --optimize", "优化Agent配置"),
            ("error --suggest", "错误建议"),
            ("i18n --set en_US", "切换语言"),
        ],
    }

    # 学习资源推荐
    LEARNING_RESOURCES = {
        "getting_started": "入门指南 - 快速了解SEVA",
        "tutorial": "交互式教程 - 边学边练",
        "scenarios": "预设场景 - 学习常见用例",
        "configuration": "配置指南 - 自定义Agent",
        "performance": "性能优化 - 提升运行效率",
        "troubleshooting": "故障排除 - 解决常见问题",
        "api": "API文档 - 编程接口",
        "dashboard": "Web仪表盘 - 可视化监控",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def recommend_config(self, scenario: str = "",
                         skill_level: str = UserSkillLevel.BEGINNER
                         ) -> Dict[str, Any]:
        """推荐配置"""
        preset = self.SCENARIO_PRESETS.get(scenario, "standard")
        return {
            "preset": preset,
            "description": self.PRESET_DESCRIPTIONS.get(preset, ""),
            "scenario": scenario,
            "reason": f"基于场景 '{scenario}' 和技能水平 '{skill_level}'",
        }

    def recommend_commands(self, skill_level: str = UserSkillLevel.BEGINNER,
                           used_commands: Optional[List[str]] = None
                           ) -> List[Dict[str, str]]:
        """推荐命令"""
        used_commands = used_commands or []
        recommendations = self.COMMAND_RECOMMENDATIONS.get(
            skill_level,
            self.COMMAND_RECOMMENDATIONS[UserSkillLevel.BEGINNER])
        result = []
        for cmd, desc in recommendations:
            already_used = cmd.split()[0] in [
                uc.split()[0] for uc in used_commands]
            result.append({
                "command": cmd,
                "description": desc,
                "already_used": already_used,
            })
        return result

    def recommend_learning(self, skill_level: str = UserSkillLevel.BEGINNER,
                           used_features: Optional[List[str]] = None
                           ) -> List[Dict[str, str]]:
        """推荐学习资源"""
        used_features = used_features or []
        result = []
        for resource_id, desc in self.LEARNING_RESOURCES.items():
            already_used = resource_id in used_features
            # 根据技能水平调整推荐顺序
            priority = self._resource_priority(resource_id, skill_level)
            result.append({
                "resource_id": resource_id,
                "description": desc,
                "already_used": already_used,
                "priority": priority,
            })
        # 按优先级排序
        result.sort(key=lambda x: x["priority"])
        return result

    def _resource_priority(self, resource_id: str,
                           skill_level: str) -> int:
        """资源优先级"""
        priorities = {
            UserSkillLevel.BEGINNER: {
                "getting_started": 1,
                "tutorial": 2,
                "scenarios": 3,
                "configuration": 4,
                "troubleshooting": 5,
                "dashboard": 6,
                "performance": 7,
                "api": 8,
            },
            UserSkillLevel.INTERMEDIATE: {
                "scenarios": 1,
                "configuration": 2,
                "dashboard": 3,
                "performance": 4,
                "troubleshooting": 5,
                "api": 6,
                "getting_started": 7,
                "tutorial": 8,
            },
            UserSkillLevel.ADVANCED: {
                "performance": 1,
                "api": 2,
                "configuration": 3,
                "dashboard": 4,
                "scenarios": 5,
                "troubleshooting": 6,
                "getting_started": 7,
                "tutorial": 8,
            },
            UserSkillLevel.EXPERT: {
                "api": 1,
                "performance": 2,
                "configuration": 3,
                "dashboard": 4,
                "scenarios": 5,
                "troubleshooting": 6,
                "getting_started": 7,
                "tutorial": 8,
            },
        }
        return priorities.get(skill_level, {}).get(resource_id, 9)

    def recommend_preset(self, context: Dict[str, Any]) -> str:
        """基于上下文推荐预设"""
        # 如果用户有偏好，使用偏好
        preferred = context.get("preferred_presets", {})
        if preferred:
            return max(preferred.items(), key=lambda x: x[1])[0]
        # 默认推荐
        cycle_count = context.get("cycle_count", 0)
        if cycle_count == 0:
            return "easy"
        elif cycle_count < 100:
            return "standard"
        else:
            return "research"


# ============================================================================
# SuggestionEngine - 建议引擎
# ============================================================================

class SuggestionEngine:
    """
    建议引擎

    综合分析上下文，生成智能建议
    """

    def __init__(self) -> None:
        self._analyzer = ContextAnalyzer()
        self._profiler = UserProfiler()
        self._recommender = Recommender()
        self._lock = threading.Lock()
        self._rules: List[Callable] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """注册默认规则"""
        self._rules = [
            self._rule_high_prediction_error,
            self._rule_slow_cycle_time,
            self._rule_negative_reward,
            self._rule_no_skills,
            self._rule_no_tasks,
            self._rule_beginner_user,
            self._rule_many_errors,
            self._rule_long_runtime,
            self._rule_high_difficulty,
            self._rule_first_run,
        ]

    def add_rule(self, rule: Callable) -> None:
        """添加自定义规则"""
        with self._lock:
            self._rules.append(rule)

    def generate(self, agent: Optional[Any] = None,
                 context: Optional[Dict[str, Any]] = None
                 ) -> List[Suggestion]:
        """生成建议"""
        if context is None:
            if agent is not None:
                context = self._analyzer.analyze_agent(agent)
            else:
                context = {}

        # 添加用户画像信息
        profile = self._profiler.get_profile()
        context["skill_level"] = profile["skill_level"]
        context["commands_used"] = profile["commands_used"]
        context["features_used"] = profile["features_used"]
        context["errors_encountered"] = profile["errors_encountered"]
        context["sessions_count"] = profile["sessions_count"]
        context["preferred_presets"] = profile["preferred_presets"]

        suggestions: List[Suggestion] = []
        for rule in self._rules:
            try:
                result = rule(context)
                if result:
                    if isinstance(result, list):
                        suggestions.extend(result)
                    else:
                        suggestions.append(result)
            except Exception:
                pass

        # 按优先级排序
        suggestions.sort(key=lambda s: (s.priority, -s.confidence))
        return suggestions

    # ===== 默认规则 =====

    def _rule_high_prediction_error(self, context: Dict
                                    ) -> Optional[Suggestion]:
        """高预测误差"""
        error = context.get("avg_prediction_error", 0)
        if error > 0.5:
            return Suggestion(
                title="预测误差过高",
                message=f"当前平均预测误差为 {error:.4f}，建议优化预测模型",
                type=SuggestionType.OPTIMIZATION,
                priority=SuggestionPriority.HIGH,
                action="考虑增加训练步数或调整预测器参数",
                confidence=0.9,
            )
        return None

    def _rule_slow_cycle_time(self, context: Dict
                              ) -> Optional[Suggestion]:
        """周期时间过长"""
        cycle_time = context.get("avg_cycle_time", 0)
        if cycle_time > 1.0:
            return Suggestion(
                title="周期时间过长",
                message=f"当前平均周期时间为 {cycle_time*1000:.0f}ms，"
                        f"可能影响性能",
                type=SuggestionType.OPTIMIZATION,
                priority=SuggestionPriority.HIGH,
                action="运行 'seva perf --optimize' 优化配置",
                confidence=0.85,
            )
        return None

    def _rule_negative_reward(self, context: Dict
                              ) -> Optional[Suggestion]:
        """负奖励"""
        reward = context.get("total_reward", 0)
        if reward < 0:
            return Suggestion(
                title="总奖励为负",
                message=f"当前总奖励为 {reward:.4f}，Agent表现不佳",
                type=SuggestionType.TROUBLESHOOTING,
                priority=SuggestionPriority.CRITICAL,
                action="检查环境配置或尝试更简单的预设",
                confidence=0.95,
            )
        return None

    def _rule_no_skills(self, context: Dict) -> Optional[Suggestion]:
        """无技能生成"""
        skills = context.get("skills_total", 0)
        cycles = context.get("cycle_count", 0)
        if skills == 0 and cycles > 10:
            return Suggestion(
                title="运行多步但无技能生成",
                message=f"已运行 {cycles} 步，但未生成任何技能",
                type=SuggestionType.TROUBLESHOOTING,
                priority=SuggestionPriority.MEDIUM,
                action="检查技能管理器配置或增加运行步数",
                confidence=0.7,
            )
        return None

    def _rule_no_tasks(self, context: Dict) -> Optional[Suggestion]:
        """无任务完成"""
        tasks = context.get("tasks_completed", 0)
        cycles = context.get("cycle_count", 0)
        if tasks == 0 and cycles > 20:
            return Suggestion(
                title="运行多步但无任务完成",
                message=f"已运行 {cycles} 步，但未完成任何任务",
                type=SuggestionType.TROUBLESHOOTING,
                priority=SuggestionPriority.MEDIUM,
                action="检查任务管理器或调整环境难度",
                confidence=0.7,
            )
        return None

    def _rule_beginner_user(self, context: Dict
                            ) -> Optional[Suggestion]:
        """初学者建议"""
        skill_level = context.get("skill_level", UserSkillLevel.BEGINNER)
        sessions = context.get("sessions_count", 0)
        if skill_level == UserSkillLevel.BEGINNER and sessions < 3:
            return Suggestion(
                title="建议开始教程",
                message="作为初学者，建议先完成交互式教程",
                type=SuggestionType.LEARNING,
                priority=SuggestionPriority.MEDIUM,
                action="运行 'seva tutorial beginner'",
                confidence=0.8,
            )
        return None

    def _rule_many_errors(self, context: Dict
                          ) -> Optional[Suggestion]:
        """错误过多"""
        errors = context.get("errors_encountered", 0)
        if errors > 10:
            return Suggestion(
                title="错误次数较多",
                message=f"已遇到 {errors} 次错误，建议查看错误日志",
                type=SuggestionType.TROUBLESHOOTING,
                priority=SuggestionPriority.MEDIUM,
                action="运行 'seva error --log' 查看错误日志",
                confidence=0.75,
            )
        return None

    def _rule_long_runtime(self, context: Dict
                           ) -> Optional[Suggestion]:
        """长时间运行"""
        cycles = context.get("cycle_count", 0)
        if cycles > 1000:
            return Suggestion(
                title="长时间运行建议",
                message=f"已运行 {cycles} 步，建议定期保存会话",
                type=SuggestionType.BEST_PRACTICE,
                priority=SuggestionPriority.LOW,
                action="运行 'seva session save' 保存当前会话",
                confidence=0.6,
            )
        return None

    def _rule_high_difficulty(self, context: Dict
                              ) -> Optional[Suggestion]:
        """高难度"""
        difficulty = context.get("current_difficulty", "EASY")
        if difficulty in ("HARD", "EXPERT"):
            return Suggestion(
                title="当前难度较高",
                message=f"当前难度为 {difficulty}，可能影响性能",
                type=SuggestionType.WARNING,
                priority=SuggestionPriority.MEDIUM,
                action="考虑降低难度或优化Agent配置",
                confidence=0.7,
            )
        return None

    def _rule_first_run(self, context: Dict) -> Optional[Suggestion]:
        """首次运行"""
        sessions = context.get("sessions_count", 0)
        if sessions == 0:
            return Suggestion(
                title="欢迎使用SEVA",
                message="检测到首次使用，建议先运行向导",
                type=SuggestionType.INFO,
                priority=SuggestionPriority.HIGH,
                action="运行 'seva wizard' 开始首次向导",
                confidence=1.0,
            )
        return None

    def get_report(self, agent: Optional[Any] = None) -> str:
        """获取建议报告"""
        suggestions = self.generate(agent)
        if not suggestions:
            return "暂无建议，系统运行良好"

        lines = [
            f"SEVA智能建议报告 ({len(suggestions)}条建议)",
            "=" * 60,
        ]
        for s in suggestions:
            lines.append("")
            lines.append(s.format())
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

_engine_instance: Optional[SuggestionEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> SuggestionEngine:
    """获取单例引擎"""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = SuggestionEngine()
    return _engine_instance


def suggest(agent: Optional[Any] = None,
            context: Optional[Dict[str, Any]] = None
            ) -> List[Suggestion]:
    """生成建议"""
    return _get_engine().generate(agent, context)


def recommend(scenario: str = "",
              skill_level: str = UserSkillLevel.BEGINNER
              ) -> Dict[str, Any]:
    """推荐配置"""
    return _get_engine()._recommender.recommend_config(
        scenario, skill_level)


def recommend_commands(skill_level: str = UserSkillLevel.BEGINNER,
                       used_commands: Optional[List[str]] = None
                       ) -> List[Dict[str, str]]:
    """推荐命令"""
    return _get_engine()._recommender.recommend_commands(
        skill_level, used_commands)


def recommend_learning(skill_level: str = UserSkillLevel.BEGINNER,
                       used_features: Optional[List[str]] = None
                       ) -> List[Dict[str, str]]:
    """推荐学习资源"""
    return _get_engine()._recommender.recommend_learning(
        skill_level, used_features)


def get_suggestions(agent: Optional[Any] = None) -> str:
    """获取建议报告"""
    return _get_engine().get_report(agent)


def get_user_profile() -> Dict[str, Any]:
    """获取用户画像"""
    return _get_engine()._profiler.get_profile()


def get_user_profile_report() -> str:
    """获取用户画像报告"""
    return _get_engine()._profiler.get_report()


def record_command(command: str) -> None:
    """记录命令使用"""
    _get_engine()._profiler.record_command(command)


def record_feature(feature: str) -> None:
    """记录功能使用"""
    _get_engine()._profiler.record_feature(feature)


def record_error() -> None:
    """记录错误"""
    _get_engine()._profiler.record_error()


def record_session(steps: int = 0, preset: str = "",
                   env_type: str = "") -> None:
    """记录会话"""
    _get_engine()._profiler.record_session(steps, preset, env_type)


def reset_user_profile() -> None:
    """重置用户画像"""
    _get_engine()._profiler.reset()


def get_skill_level() -> str:
    """获取技能水平"""
    return _get_engine()._profiler.get_skill_level()


def analyze_context(agent: Any) -> Dict[str, Any]:
    """分析上下文"""
    return _get_engine()._analyzer.analyze_agent(agent)


__all__ = [
    "Suggestion",
    "SuggestionType",
    "SuggestionPriority",
    "UserSkillLevel",
    "ContextAnalyzer",
    "UserProfiler",
    "Recommender",
    "SuggestionEngine",
    "suggest",
    "recommend",
    "recommend_commands",
    "recommend_learning",
    "get_suggestions",
    "get_user_profile",
    "get_user_profile_report",
    "record_command",
    "record_feature",
    "record_error",
    "record_session",
    "reset_user_profile",
    "get_skill_level",
    "analyze_context",
    "PROFILE_FILE",
]
