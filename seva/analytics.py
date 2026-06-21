"""
用户行为分析与引导优化（阶段52）
对应第一性原理：数据驱动 = 基于真实使用数据优化用户体验

提供能力:
    - BehaviorEvent: 行为事件数据结构
    - BehaviorTracker: 行为跟踪器（记录用户操作）
    - BehaviorAnalyzer: 行为分析器（频率/路径/热点/流失分析）
    - UsagePattern: 使用模式识别
    - GuidanceOptimizer: 引导优化器（个性化引导）
    - AnalyticsReport: 分析报告
    - 便捷函数: track_event / get_analytics_report / optimize_guidance

使用示例:
    from seva import track_event, get_analytics_report

    track_event("command", name="run", args={"steps": 100})
    report = get_analytics_report()
    print(report)
"""
from __future__ import annotations

import os
import json
import time
import threading
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from pathlib import Path
from enum import Enum


# ============================================================================
# 常量
# ============================================================================

class EventType:
    """事件类型"""
    COMMAND = "command"          # 命令执行
    FEATURE = "feature"          # 功能使用
    ERROR = "error"              # 错误发生
    SESSION = "session"          # 会话事件
    NAVIGATION = "navigation"    # 导航
    CONFIGURATION = "configuration"  # 配置变更
    TUTORIAL = "tutorial"        # 教程
    HELP = "help"                # 帮助


class SessionState:
    """会话状态"""
    ACTIVE = "active"
    ENDED = "ended"
    ABANDONED = "abandoned"


# 数据存储路径
ANALYTICS_DIR = Path.home() / ".seva" / "analytics"
EVENTS_FILE = ANALYTICS_DIR / "events.json"
SESSIONS_FILE = ANALYTICS_DIR / "sessions.json"


# ============================================================================
# BehaviorEvent - 行为事件
# ============================================================================

class BehaviorEvent:
    """
    行为事件数据结构

    属性:
        event_id: 事件ID
        event_type: 事件类型
        name: 事件名称
        timestamp: 时间戳
        session_id: 会话ID
        properties: 事件属性
        context: 上下文
    """

    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, event_type: str, name: str,
                 session_id: str = "",
                 properties: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None) -> None:
        with BehaviorEvent._id_lock:
            self.event_id = BehaviorEvent._next_id
            BehaviorEvent._next_id += 1
        self.event_type = event_type
        self.name = name
        self.timestamp = time.time()
        self.session_id = session_id
        self.properties = properties or {}
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "name": self.name,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "properties": self.properties,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorEvent":
        event = cls(
            event_type=data["event_type"],
            name=data["name"],
            session_id=data.get("session_id", ""),
            properties=data.get("properties", {}),
            context=data.get("context", {}),
        )
        event.event_id = data.get("event_id", event.event_id)
        event.timestamp = data.get("timestamp", time.time())
        return event

    def __repr__(self) -> str:
        return (f"BehaviorEvent(type={self.event_type!r}, "
                f"name={self.name!r})")


# ============================================================================
# BehaviorTracker - 行为跟踪器
# ============================================================================

class BehaviorTracker:
    """
    行为跟踪器

    记录用户操作事件，支持会话管理
    """

    _instance: Optional["BehaviorTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BehaviorTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._events: List[BehaviorEvent] = []
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._current_session_id: Optional[str] = None
        self._data_lock = threading.Lock()
        self._max_events = 10000  # 最大事件数
        self._load()

    def start_session(self, session_id: Optional[str] = None) -> str:
        """开始会话"""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{threading.get_ident()}"
        with self._data_lock:
            self._current_session_id = session_id
            self._sessions[session_id] = {
                "session_id": session_id,
                "start_time": time.time(),
                "end_time": None,
                "state": SessionState.ACTIVE,
                "event_count": 0,
            }
        self.track(EventType.SESSION, "session_start",
                   session_id=session_id)
        return session_id

    def end_session(self, state: str = SessionState.ENDED) -> None:
        """结束会话"""
        if self._current_session_id is None:
            return
        with self._data_lock:
            session = self._sessions.get(self._current_session_id)
            if session:
                session["end_time"] = time.time()
                session["state"] = state
        self.track(EventType.SESSION, "session_end",
                   session_id=self._current_session_id,
                   properties={"state": state})
        self._current_session_id = None
        self._save()

    def track(self, event_type: str, name: str,
              session_id: Optional[str] = None,
              properties: Optional[Dict[str, Any]] = None,
              context: Optional[Dict[str, Any]] = None) -> BehaviorEvent:
        """记录事件"""
        sid = session_id or self._current_session_id or ""
        event = BehaviorEvent(
            event_type=event_type,
            name=name,
            session_id=sid,
            properties=properties,
            context=context,
        )
        with self._data_lock:
            self._events.append(event)
            # 限制事件数量
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            # 更新会话事件计数
            if sid and sid in self._sessions:
                self._sessions[sid]["event_count"] += 1
        return event

    def get_events(self, event_type: Optional[str] = None,
                   session_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[BehaviorEvent]:
        """获取事件"""
        with self._data_lock:
            events = list(self._events)
        # 过滤
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        # 限制
        if limit:
            events = events[-limit:]
        return events

    def get_sessions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有会话"""
        with self._data_lock:
            return dict(self._sessions)

    def get_current_session(self) -> Optional[str]:
        """获取当前会话ID"""
        return self._current_session_id

    def clear(self) -> None:
        """清空所有数据"""
        with self._data_lock:
            self._events.clear()
            self._sessions.clear()
            self._current_session_id = None
        self._save()

    def get_event_count(self) -> int:
        """获取事件总数"""
        with self._data_lock:
            return len(self._events)

    def get_session_count(self) -> int:
        """获取会话总数"""
        with self._data_lock:
            return len(self._sessions)

    def _load(self) -> None:
        """从文件加载"""
        try:
            if EVENTS_FILE.exists():
                with open(EVENTS_FILE, "r") as f:
                    data = json.load(f)
                self._events = [
                    BehaviorEvent.from_dict(d) for d in data
                ]
                # 更新ID计数器
                if self._events:
                    BehaviorEvent._next_id = max(
                        e.event_id for e in self._events) + 1
            if SESSIONS_FILE.exists():
                with open(SESSIONS_FILE, "r") as f:
                    self._sessions = json.load(f)
        except Exception:
            pass

    def _save(self) -> None:
        """保存到文件"""
        try:
            ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
            with self._data_lock:
                events_data = [e.to_dict() for e in self._events]
                sessions_data = dict(self._sessions)
            with open(EVENTS_FILE, "w") as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)
            with open(SESSIONS_FILE, "w") as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# ============================================================================
# BehaviorAnalyzer - 行为分析器
# ============================================================================

class BehaviorAnalyzer:
    """
    行为分析器

    分析用户行为数据，生成洞察
    """

    def __init__(self, tracker: Optional[BehaviorTracker] = None) -> None:
        self.tracker = tracker or BehaviorTracker()

    def analyze(self) -> Dict[str, Any]:
        """综合分析"""
        return {
            "event_stats": self.event_frequency(),
            "command_stats": self.command_frequency(),
            "feature_stats": self.feature_frequency(),
            "session_stats": self.session_stats(),
            "path_analysis": self.path_analysis(),
            "hotspots": self.hotspots(),
            "dropoff_points": self.dropoff_analysis(),
            "time_analysis": self.time_analysis(),
            "summary": self.summary(),
        }

    def event_frequency(self) -> Dict[str, int]:
        """事件频率分析"""
        events = self.tracker.get_events()
        freq = Counter(e.event_type for e in events)
        return dict(freq)

    def command_frequency(self) -> Dict[str, int]:
        """命令频率分析"""
        events = self.tracker.get_events(event_type=EventType.COMMAND)
        freq = Counter(e.name for e in events)
        return dict(freq)

    def feature_frequency(self) -> Dict[str, int]:
        """功能频率分析"""
        events = self.tracker.get_events(event_type=EventType.FEATURE)
        freq = Counter(e.name for e in events)
        return dict(freq)

    def session_stats(self) -> Dict[str, Any]:
        """会话统计"""
        sessions = self.tracker.get_sessions()
        if not sessions:
            return {
                "total_sessions": 0,
                "avg_duration": 0,
                "avg_events": 0,
                "active_sessions": 0,
            }
        durations = []
        event_counts = []
        for s in sessions.values():
            if s.get("end_time") and s.get("start_time"):
                durations.append(s["end_time"] - s["start_time"])
            event_counts.append(s.get("event_count", 0))
        return {
            "total_sessions": len(sessions),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "avg_events": sum(event_counts) / len(event_counts) if event_counts else 0,
            "active_sessions": sum(1 for s in sessions.values()
                                   if s.get("state") == SessionState.ACTIVE),
        }

    def path_analysis(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """路径分析（用户操作序列）"""
        events = self.tracker.get_events(session_id=session_id)
        # 只分析命令和功能事件
        actions = [e for e in events
                   if e.event_type in (EventType.COMMAND, EventType.FEATURE)]
        if not actions:
            return {
                "total_actions": 0,
                "common_paths": [],
                "avg_path_length": 0,
            }
        # 按会话分组
        session_paths: Dict[str, List[str]] = defaultdict(list)
        for a in actions:
            session_paths[a.session_id].append(a.name)
        # 计算路径长度
        path_lengths = [len(p) for p in session_paths.values()]
        # 常见路径（前3个动作）
        common_starts = Counter()
        for path in session_paths.values():
            if len(path) >= 2:
                common_starts[tuple(path[:2])] += 1
        return {
            "total_actions": len(actions),
            "common_paths": [
                {"path": list(p), "count": c}
                for p, c in common_starts.most_common(5)
            ],
            "avg_path_length": sum(path_lengths) / len(path_lengths)
            if path_lengths else 0,
        }

    def hotspots(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """热点分析（最常用功能）"""
        cmd_freq = self.command_frequency()
        feat_freq = self.feature_frequency()
        top_commands = [
            {"name": n, "count": c}
            for n, c in sorted(cmd_freq.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]
        ]
        top_features = [
            {"name": n, "count": c}
            for n, c in sorted(feat_freq.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]
        ]
        return {
            "top_commands": top_commands,
            "top_features": top_features,
        }

    def dropoff_analysis(self) -> Dict[str, Any]:
        """流失分析（用户在哪里放弃）"""
        sessions = self.tracker.get_sessions()
        if not sessions:
            return {"dropoff_points": [], "abandonment_rate": 0}
        # 分析放弃的会话
        abandoned = [s for s in sessions.values()
                     if s.get("state") == SessionState.ABANDONED]
        # 分析每个会话的最后一个事件
        events = self.tracker.get_events()
        last_events = {}
        for e in events:
            if e.session_id:
                last_events[e.session_id] = e.name
        dropoff_counts = Counter(last_events.values())
        return {
            "dropoff_points": [
                {"name": n, "count": c}
                for n, c in dropoff_counts.most_common(5)
            ],
            "abandonment_rate": len(abandoned) / len(sessions)
            if sessions else 0,
        }

    def time_analysis(self) -> Dict[str, Any]:
        """时间分析"""
        events = self.tracker.get_events()
        if not events:
            return {"total_time": 0, "avg_interval": 0}
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        # 总时间跨度
        total_time = sorted_events[-1].timestamp - sorted_events[0].timestamp
        # 平均事件间隔
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = sorted_events[i].timestamp - sorted_events[i-1].timestamp
            if interval < 3600:  # 忽略超过1小时间隔（可能是新会话）
                intervals.append(interval)
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        return {
            "total_time": total_time,
            "avg_interval": avg_interval,
            "event_count": len(events),
        }

    def summary(self) -> Dict[str, Any]:
        """摘要"""
        events = self.tracker.get_events()
        sessions = self.tracker.get_sessions()
        return {
            "total_events": len(events),
            "total_sessions": len(sessions),
            "event_types": len(set(e.event_type for e in events)),
            "unique_commands": len(set(e.name for e in events
                                       if e.event_type == EventType.COMMAND)),
            "unique_features": len(set(e.name for e in events
                                       if e.event_type == EventType.FEATURE)),
        }


# ============================================================================
# UsagePattern - 使用模式
# ============================================================================

class UsagePattern:
    """
    使用模式识别

    识别用户的使用模式和习惯
    """

    def __init__(self, tracker: Optional[BehaviorTracker] = None) -> None:
        self.tracker = tracker or BehaviorTracker()

    def identify_patterns(self) -> List[Dict[str, Any]]:
        """识别使用模式"""
        patterns = []
        # 模式1：频繁命令组合
        patterns.extend(self._find_command_combinations())
        # 模式2：时间模式（使用时段）
        patterns.extend(self._find_time_patterns())
        # 模式3：重复行为
        patterns.extend(self._find_repeated_behaviors())
        return patterns

    def _find_command_combinations(self) -> List[Dict[str, Any]]:
        """查找命令组合"""
        events = self.tracker.get_events(event_type=EventType.COMMAND)
        session_cmds: Dict[str, List[str]] = defaultdict(list)
        for e in events:
            session_cmds[e.session_id].append(e.name)
        # 查找常见2-gram
        bigrams = Counter()
        for cmds in session_cmds.values():
            for i in range(len(cmds) - 1):
                bigrams[(cmds[i], cmds[i+1])] += 1
        patterns = []
        for (cmd1, cmd2), count in bigrams.most_common(5):
            if count >= 2:
                patterns.append({
                    "type": "command_combination",
                    "pattern": f"{cmd1} -> {cmd2}",
                    "count": count,
                    "description": f"经常连续使用 {cmd1} 和 {cmd2}",
                })
        return patterns

    def _find_time_patterns(self) -> List[Dict[str, Any]]:
        """查找时间模式"""
        events = self.tracker.get_events()
        if not events:
            return []
        # 按小时统计
        hour_counts = Counter()
        for e in events:
            hour = time.localtime(e.timestamp).tm_hour
            hour_counts[hour] += 1
        # 找出最活跃时段
        if hour_counts:
            top_hour, top_count = hour_counts.most_common(1)[0]
            return [{
                "type": "time_pattern",
                "pattern": f"hour_{top_hour}",
                "count": top_count,
                "description": f"最常在 {top_hour} 时使用",
            }]
        return []

    def _find_repeated_behaviors(self) -> List[Dict[str, Any]]:
        """查找重复行为"""
        cmd_freq = Counter()
        for e in self.tracker.get_events(event_type=EventType.COMMAND):
            cmd_freq[e.name] += 1
        patterns = []
        for cmd, count in cmd_freq.most_common(3):
            if count >= 5:
                patterns.append({
                    "type": "repeated_behavior",
                    "pattern": cmd,
                    "count": count,
                    "description": f"频繁使用命令 '{cmd}' ({count}次)",
                })
        return patterns

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常行为"""
        anomalies = []
        events = self.tracker.get_events()
        # 异常1：错误频率过高
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        if len(error_events) > 10:
            anomalies.append({
                "type": "high_error_rate",
                "description": f"错误频率过高 ({len(error_events)}次)",
                "severity": "high",
            })
        # 异常2：会话过短
        sessions = self.tracker.get_sessions()
        short_sessions = [s for s in sessions.values()
                          if s.get("end_time") and
                          s.get("start_time") and
                          s["end_time"] - s["start_time"] < 5]
        if len(short_sessions) > 3:
            anomalies.append({
                "type": "short_sessions",
                "description": f"多个过短会话 ({len(short_sessions)}个)",
                "severity": "medium",
            })
        # 异常3：重复相同命令
        cmd_freq = Counter(e.name for e in events
                           if e.event_type == EventType.COMMAND)
        for cmd, count in cmd_freq.items():
            if count > 50:
                anomalies.append({
                    "type": "command_overuse",
                    "description": f"命令 '{cmd}' 使用过多 ({count}次)",
                    "severity": "low",
                })
        return anomalies


# ============================================================================
# GuidanceOptimizer - 引导优化器
# ============================================================================

class GuidanceOptimizer:
    """
    引导优化器

    基于行为分析优化引导流程
    """

    def __init__(self, tracker: Optional[BehaviorTracker] = None) -> None:
        self.tracker = tracker or BehaviorTracker()
        self.analyzer = BehaviorAnalyzer(tracker)
        self.patterns = UsagePattern(tracker)

    def optimize(self) -> Dict[str, Any]:
        """生成优化建议"""
        analysis = self.analyzer.analyze()
        patterns = self.patterns.identify_patterns()
        anomalies = self.patterns.detect_anomalies()
        recommendations = self._generate_recommendations(
            analysis, patterns, anomalies)
        return {
            "analysis": analysis,
            "patterns": patterns,
            "anomalies": anomalies,
            "recommendations": recommendations,
        }

    def _generate_recommendations(self, analysis: Dict,
                                  patterns: List[Dict],
                                  anomalies: List[Dict]) -> List[Dict[str, Any]]:
        """生成推荐"""
        recommendations = []
        # 基于热点推荐
        hotspots = analysis.get("hotspots", {})
        top_commands = hotspots.get("top_commands", [])
        if top_commands:
            top_cmd = top_commands[0]["name"]
            recommendations.append({
                "type": "shortcut",
                "title": f"为常用命令 '{top_cmd}' 创建快捷方式",
                "description": f"检测到 '{top_cmd}' 使用频率最高，"
                               f"考虑为其创建快捷方式",
                "priority": "medium",
            })
        # 基于流失点推荐
        dropoff = analysis.get("dropoff_points", {})
        dropoff_points = dropoff.get("dropoff_points", [])
        if dropoff_points:
            top_dropoff = dropoff_points[0]["name"]
            recommendations.append({
                "type": "guidance",
                "title": f"优化 '{top_dropoff}' 后的引导",
                "description": f"用户常在 '{top_dropoff}' 后放弃，"
                               f"考虑添加后续引导",
                "priority": "high",
            })
        # 基于异常推荐
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                recommendations.append({
                    "type": "fix",
                    "title": f"解决异常: {anomaly['type']}",
                    "description": anomaly["description"],
                    "priority": "high",
                })
        # 基于模式推荐
        for pattern in patterns:
            if pattern["type"] == "command_combination":
                recommendations.append({
                    "type": "workflow",
                    "title": f"优化工作流: {pattern['pattern']}",
                    "description": pattern["description"],
                    "priority": "low",
                })
        # 基于会话统计推荐
        session_stats = analysis.get("session_stats", {})
        if session_stats.get("avg_events", 0) < 5:
            recommendations.append({
                "type": "engagement",
                "title": "提高用户参与度",
                "description": "平均会话事件数较少，考虑增加引导提示",
                "priority": "medium",
            })
        return recommendations

    def get_personalized_path(self) -> List[Dict[str, str]]:
        """获取个性化引导路径"""
        analysis = self.analyzer.analyze()
        hotspots = analysis.get("hotspots", {})
        top_commands = hotspots.get("top_commands", [])
        # 基于常用命令构建引导路径
        path = []
        seen = set()
        for cmd in top_commands:
            if cmd["name"] not in seen:
                path.append({
                    "step": f"使用 {cmd['name']}",
                    "description": f"常用命令 (使用{cmd['count']}次)",
                })
                seen.add(cmd["name"])
        # 添加推荐的新命令
        if len(path) < 5:
            path.append({
                "step": "尝试新功能",
                "description": "探索未使用的功能",
            })
        return path


# ============================================================================
# AnalyticsReport - 分析报告
# ============================================================================

class AnalyticsReport:
    """
    分析报告生成器
    """

    def __init__(self, tracker: Optional[BehaviorTracker] = None) -> None:
        self.tracker = tracker or BehaviorTracker()
        self.analyzer = BehaviorAnalyzer(tracker)
        self.optimizer = GuidanceOptimizer(tracker)

    def generate(self) -> str:
        """生成完整报告"""
        analysis = self.analyzer.analyze()
        opt = self.optimizer.optimize()
        lines = [
            "SEVA用户行为分析报告",
            "=" * 60,
            "",
            "【概览】",
            f"  总事件数: {analysis['summary']['total_events']}",
            f"  总会话数: {analysis['summary']['total_sessions']}",
            f"  事件类型数: {analysis['summary']['event_types']}",
            f"  使用命令数: {analysis['summary']['unique_commands']}",
            f"  使用功能数: {analysis['summary']['unique_features']}",
            "",
            "【会话统计】",
            f"  总会话: {analysis['session_stats']['total_sessions']}",
            f"  平均时长: {analysis['session_stats']['avg_duration']:.1f}秒",
            f"  平均事件: {analysis['session_stats']['avg_events']:.1f}",
            f"  活跃会话: {analysis['session_stats']['active_sessions']}",
            "",
            "【热点分析】",
        ]
        # 热点命令
        hotspots = analysis.get("hotspots", {})
        top_cmds = hotspots.get("top_commands", [])
        if top_cmds:
            lines.append("  常用命令:")
            for cmd in top_cmds[:5]:
                lines.append(f"    {cmd['name']}: {cmd['count']}次")
        top_feats = hotspots.get("top_features", [])
        if top_feats:
            lines.append("  常用功能:")
            for feat in top_feats[:5]:
                lines.append(f"    {feat['name']}: {feat['count']}次")
        # 流失分析
        lines.append("")
        lines.append("【流失分析】")
        dropoff = analysis.get("dropoff_points", {})
        if dropoff.get("dropoff_points"):
            lines.append(f"  放弃率: {dropoff['abandonment_rate']:.1%}")
            lines.append("  流失点:")
            for dp in dropoff["dropoff_points"][:3]:
                lines.append(f"    {dp['name']}: {dp['count']}次")
        else:
            lines.append("  无流失数据")
        # 使用模式
        lines.append("")
        lines.append("【使用模式】")
        patterns = opt.get("patterns", [])
        if patterns:
            for p in patterns[:5]:
                lines.append(f"  [{p['type']}] {p['description']}")
        else:
            lines.append("  暂无识别到的模式")
        # 异常
        anomalies = opt.get("anomalies", [])
        if anomalies:
            lines.append("")
            lines.append("【异常检测】")
            for a in anomalies:
                lines.append(f"  [{a['severity']}] {a['description']}")
        # 推荐
        recommendations = opt.get("recommendations", [])
        if recommendations:
            lines.append("")
            lines.append("【优化建议】")
            for r in recommendations:
                lines.append(f"  [{r['priority']}] {r['title']}")
                lines.append(f"    {r['description']}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要数据"""
        return self.analyzer.summary()


# ============================================================================
# 便捷函数
# ============================================================================

def track_event(event_type: str, name: str,
                properties: Optional[Dict[str, Any]] = None,
                context: Optional[Dict[str, Any]] = None) -> BehaviorEvent:
    """记录事件"""
    return BehaviorTracker().track(event_type, name,
                                   properties=properties,
                                   context=context)


def track_command(name: str, **kwargs) -> BehaviorEvent:
    """记录命令事件"""
    return track_event(EventType.COMMAND, name, properties=kwargs)


def track_feature(name: str, **kwargs) -> BehaviorEvent:
    """记录功能事件"""
    return track_event(EventType.FEATURE, name, properties=kwargs)


def track_error(name: str, **kwargs) -> BehaviorEvent:
    """记录错误事件"""
    return track_event(EventType.ERROR, name, properties=kwargs)


def start_session(session_id: Optional[str] = None) -> str:
    """开始会话"""
    return BehaviorTracker().start_session(session_id)


def end_session(state: str = SessionState.ENDED) -> None:
    """结束会话"""
    BehaviorTracker().end_session(state)


def get_analytics_report() -> str:
    """获取分析报告"""
    return AnalyticsReport().generate()


def get_analytics_summary() -> Dict[str, Any]:
    """获取分析摘要"""
    return AnalyticsReport().get_summary()


def optimize_guidance() -> Dict[str, Any]:
    """优化引导"""
    return GuidanceOptimizer().optimize()


def get_behavior_patterns() -> List[Dict[str, Any]]:
    """获取行为模式"""
    return UsagePattern().identify_patterns()


def get_behavior_anomalies() -> List[Dict[str, Any]]:
    """获取行为异常"""
    return UsagePattern().detect_anomalies()


def clear_analytics() -> None:
    """清空分析数据"""
    BehaviorTracker().clear()


def get_event_count() -> int:
    """获取事件总数"""
    return BehaviorTracker().get_event_count()


def get_session_count() -> int:
    """获取会话总数"""
    return BehaviorTracker().get_session_count()


__all__ = [
    "BehaviorEvent",
    "EventType",
    "SessionState",
    "BehaviorTracker",
    "BehaviorAnalyzer",
    "UsagePattern",
    "GuidanceOptimizer",
    "AnalyticsReport",
    "track_event",
    "track_command",
    "track_feature",
    "track_error",
    "start_session",
    "end_session",
    "get_analytics_report",
    "get_analytics_summary",
    "optimize_guidance",
    "get_behavior_patterns",
    "get_behavior_anomalies",
    "clear_analytics",
    "get_event_count",
    "get_session_count",
    "ANALYTICS_DIR",
    "EVENTS_FILE",
    "SESSIONS_FILE",
]
