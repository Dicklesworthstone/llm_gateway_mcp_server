"""Business logic and service functions for UMS API."""

import json
import math
import sqlite3
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .ums_models import (
    MemoryDetail,
    PreviewMemory,
    CriticalPathAction,
    FlameGraphNode,
)
from .ums_database import get_db_connection


# ---------- Utility Functions ----------

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def _dict_depth(d: Dict[str, Any], depth: int = 0) -> int:
    if not isinstance(d, dict) or not d:
        return depth
    return max(_dict_depth(v, depth + 1) for v in d.values())


def _count_values(d: Dict[str, Any]) -> int:
    cnt = 0
    for v in d.values():
        if isinstance(v, dict):
            cnt += _count_values(v)
        elif isinstance(v, list):
            cnt += len(v)
        else:
            cnt += 1
    return cnt


def calculate_state_complexity(state_data: Dict[str, Any]) -> float:
    if not state_data:
        return 0.0
    comp = (
        len(state_data) * 5 + _dict_depth(state_data) * 10 + _count_values(state_data) * 0.5
    )
    return round(min(100.0, comp), 2)


def compute_state_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff = {"added": {}, "removed": {}, "modified": {}, "magnitude": 0.0}
    keys = set(a) | set(b)
    changed = 0
    for k in keys:
        if k not in a:
            diff["added"][k] = b[k]
            changed += 1
        elif k not in b:
            diff["removed"][k] = a[k]
            changed += 1
        elif a[k] != b[k]:
            diff["modified"][k] = {"before": a[k], "after": b[k]}
            changed += 1
    if keys:
        diff["magnitude"] = (changed / len(keys)) * 100
    return diff


# ---------- Action Monitor Helper Functions ----------

def get_action_status_indicator(status: str, execution_time: float) -> dict:
    """Get status indicator with color and icon for action status"""
    indicators = {
        "running": {"color": "blue", "icon": "play", "label": "Running"},
        "executing": {"color": "blue", "icon": "cpu", "label": "Executing"},
        "in_progress": {"color": "orange", "icon": "clock", "label": "In Progress"},
        "completed": {"color": "green", "icon": "check", "label": "Completed"},
        "failed": {"color": "red", "icon": "x", "label": "Failed"},
        "cancelled": {"color": "gray", "icon": "stop", "label": "Cancelled"},
        "timeout": {"color": "yellow", "icon": "timer-off", "label": "Timeout"},
    }

    indicator = indicators.get(
        status, {"color": "gray", "icon": "help", "label": "Unknown"}
    )

    # Add urgency flag for long-running actions
    if (
        status in ["running", "executing", "in_progress"] and execution_time > 120
    ):  # 2 minutes
        indicator["urgency"] = "high"
    elif (
        status in ["running", "executing", "in_progress"] and execution_time > 60
    ):  # 1 minute
        indicator["urgency"] = "medium"
    else:
        indicator["urgency"] = "low"

    return indicator


def categorize_action_performance(execution_time: float, estimated_duration: float) -> str:
    """Categorize action performance based on execution time vs estimate"""
    if estimated_duration <= 0:
        return "unknown"

    ratio = execution_time / estimated_duration

    if ratio <= 0.5:
        return "excellent"
    elif ratio <= 0.8:
        return "good"
    elif ratio <= 1.2:
        return "acceptable"
    elif ratio <= 2.0:
        return "slow"
    else:
        return "very_slow"


def get_action_resource_usage(action_id: str) -> dict:
    """Get resource usage for an action (placeholder implementation)"""
    # This is a placeholder - in a real implementation, you'd fetch actual metrics
    return {"cpu_usage": 0.0, "memory_usage": 0.0, "network_io": 0.0, "disk_io": 0.0}


def estimate_wait_time(position: int, queue: list) -> float:
    """Estimate wait time based on queue position and historical data"""
    if position == 0:
        return 0.0
    # Average action time of 30 seconds (this could be calculated from historical data)
    avg_action_time = 30.0
    return position * avg_action_time


def get_priority_label(priority: int) -> str:
    """Get human-readable priority label"""
    if priority <= 1:
        return "Critical"
    elif priority <= 3:
        return "High"
    elif priority <= 5:
        return "Normal"
    elif priority <= 7:
        return "Low"
    else:
        return "Very Low"


def calculate_action_performance_score(action: dict) -> float:
    """Calculate performance score for a completed action"""
    if action["status"] != "completed":
        return 0.0

    execution_time = action.get("execution_duration", 0)
    if execution_time <= 0:
        return 100.0

    if execution_time <= 5:
        return 100.0
    elif execution_time <= 15:
        return 90.0
    elif execution_time <= 30:
        return 80.0
    elif execution_time <= 60:
        return 70.0
    elif execution_time <= 120:
        return 60.0
    else:
        return max(50.0, 100.0 - (execution_time / 10))


def calculate_efficiency_rating(execution_time: float, result_size: int) -> str:
    """Calculate efficiency rating based on time and output"""
    if execution_time <= 0:
        return "unknown"

    efficiency_score = result_size / execution_time if execution_time > 0 else 0

    if efficiency_score >= 100:
        return "excellent"
    elif efficiency_score >= 50:
        return "good"
    elif efficiency_score >= 20:
        return "fair"
    else:
        return "poor"


def calculate_performance_summary(actions: list) -> dict:
    """Calculate performance summary from action history"""
    if not actions:
        return {
            "avg_score": 0.0,
            "top_performer": None,
            "worst_performer": None,
            "efficiency_distribution": {},
        }

    scores = [a.get("performance_score", 0) for a in actions]
    avg_score = sum(scores) / len(scores)

    best_action = max(actions, key=lambda a: a.get("performance_score", 0))
    worst_action = min(actions, key=lambda a: a.get("performance_score", 0))

    efficiency_counts = Counter(a.get("efficiency_rating", "unknown") for a in actions)

    return {
        "avg_score": round(avg_score, 2),
        "top_performer": {
            "tool_name": best_action.get("tool_name", ""),
            "score": best_action.get("performance_score", 0),
        },
        "worst_performer": {
            "tool_name": worst_action.get("tool_name", ""),
            "score": worst_action.get("performance_score", 0),
        },
        "efficiency_distribution": dict(efficiency_counts),
    }


def generate_performance_insights(
    overall_stats: dict, tool_stats: list, hourly_metrics: list
) -> list:
    """Generate actionable performance insights"""
    insights = []

    success_rate = (
        overall_stats.get("successful_actions", 0) / overall_stats.get("total_actions", 1)
    ) * 100
    if success_rate < 80:
        insights.append(
            {
                "type": "warning",
                "title": "Low Success Rate",
                "message": f"Current success rate is {success_rate:.1f}%. Consider investigating failing tools.",
                "severity": "high",
            }
        )

    if tool_stats:
        slowest_tool = max(tool_stats, key=lambda t: t.get("avg_duration", 0))
        if slowest_tool.get("avg_duration", 0) > 60:
            insights.append(
                {
                    "type": "info",
                    "title": "Performance Optimization",
                    "message": f"{slowest_tool['tool_name']} is taking {slowest_tool['avg_duration']:.1f}s on average. Consider optimization.",
                    "severity": "medium",
                }
            )

    if hourly_metrics:
        peak_hour = max(hourly_metrics, key=lambda h: h.get("action_count", 0))
        insights.append(
            {
                "type": "info",
                "title": "Peak Usage",
                "message": f"Peak usage occurs at {peak_hour['hour']}:00 with {peak_hour['action_count']} actions.",
                "severity": "low",
            }
        )

    return insights


# ---------- Memory Quality Functions ----------

def find_cognitive_patterns(
    states: List[Dict[str, Any]], min_length: int, similarity_threshold: float
) -> List[Dict[str, Any]]:
    """Find recurring patterns in cognitive states"""
    patterns = []
    type_sequences = defaultdict(list)
    for state in states:
        type_sequences[state["state_type"]].append(state)
    for state_type, sequence in type_sequences.items():
        if len(sequence) >= min_length * 2:
            for length in range(min_length, len(sequence) // 2 + 1):
                for start in range(len(sequence) - length * 2 + 1):
                    subseq1 = sequence[start : start + length]
                    subseq2 = sequence[start + length : start + length * 2]
                    similarity = calculate_sequence_similarity(subseq1, subseq2)
                    if similarity >= similarity_threshold:
                        patterns.append(
                            {
                                "type": f"repeating_{state_type}",
                                "length": length,
                                "similarity": similarity,
                                "occurrences": 2,
                                "first_occurrence": subseq1[0]["timestamp"],
                                "pattern_description": f"Repeating {state_type} sequence of {length} states",
                            }
                        )
    return sorted(patterns, key=lambda p: p["similarity"], reverse=True)


def calculate_sequence_similarity(
    seq1: List[Dict[str, Any]], seq2: List[Dict[str, Any]]
) -> float:
    """Calculate similarity between two state sequences"""
    if len(seq1) != len(seq2):
        return 0.0
    total_similarity = 0.0
    for s1, s2 in zip(seq1, seq2, strict=False):
        state_sim = calculate_single_state_similarity(s1, s2)
        total_similarity += state_sim
    return total_similarity / len(seq1)


def calculate_single_state_similarity(
    state1: Dict[str, Any], state2: Dict[str, Any]
) -> float:
    """Calculate similarity between two individual states"""
    data1 = state1.get("state_data", {})
    data2 = state2.get("state_data", {})
    if not data1 and not data2:
        return 1.0
    if not data1 or not data2:
        return 0.0
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 1.0
    common_keys = keys1 & keys2
    value_similarity = 0.0
    if common_keys:
        matching_values = sum(1 for key in common_keys if data1[key] == data2[key])
        value_similarity = matching_values / len(common_keys)
    return (key_similarity + value_similarity) / 2


def analyze_state_transitions(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze transitions between cognitive states"""
    transitions = defaultdict(int)
    for i in range(len(states) - 1):
        current_type = states[i]["state_type"]
        next_type = states[i + 1]["state_type"]
        transition = f"{current_type} → {next_type}"
        transitions[transition] += 1
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "transition": transition,
            "count": count,
            "percentage": (count / (len(states) - 1)) * 100 if len(states) > 1 else 0,
        }
        for transition, count in sorted_transitions
    ]


def detect_cognitive_anomalies(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect anomalous cognitive states"""
    anomalies = []
    if len(states) < 3:
        return anomalies
    complexities = [calculate_state_complexity(s.get("state_data", {})) for s in states]
    avg_complexity = sum(complexities) / len(complexities)
    std_complexity = (
        sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)
    ) ** 0.5
    for i, state in enumerate(states):
        complexity = complexities[i]
        z_score = (
            (complexity - avg_complexity) / std_complexity if std_complexity > 0 else 0
        )
        if abs(z_score) > 2:
            anomalies.append(
                {
                    "state_id": state["state_id"],
                    "timestamp": state["timestamp"],
                    "anomaly_type": "complexity_outlier",
                    "z_score": z_score,
                    "description": f"Unusual complexity: {complexity:.1f} (avg: {avg_complexity:.1f})",
                    "severity": "high" if abs(z_score) > 3 else "medium",
                }
            )
    return anomalies


# ---------- Working Memory System ----------

class WorkingMemorySystem:
    """
    Working memory system for managing active memories with focus capabilities.
    
    This system maintains a pool of recent memories with relevance scoring
    and focus mode for filtering based on keywords or patterns.
    """
    
    def __init__(self, capacity: int = 100, focus_threshold: float = 0.7):
        self.capacity = capacity
        self.focus_threshold = focus_threshold
        self.memory_pool = deque(maxlen=capacity)
        self.focus_mode_enabled = False
        self.focus_keywords = []
        self.memory_index = {}  # memory_id -> memory mapping
        self.category_index = defaultdict(list)  # category -> [memory_ids]
        self.access_counts = defaultdict(int)  # memory_id -> access count
        self.relevance_scores = {}  # memory_id -> relevance score
        self.initialized_at = datetime.now()
        self.last_optimization = datetime.now()
        self.optimization_count = 0
        
    def add_memory(self, memory_id: str, content: str, category: str, importance: float = 5.0):
        """Add a memory to the working pool"""
        memory = {
            'memory_id': memory_id,
            'content': content,
            'category': category,
            'importance': importance,
            'added_at': datetime.now().timestamp(),
            'last_accessed': datetime.now().timestamp()
        }
        
        # Remove old memory if exists
        if memory_id in self.memory_index:
            self.remove_memory(memory_id)
        
        # Add to pool
        self.memory_pool.append(memory)
        self.memory_index[memory_id] = memory
        self.category_index[category].append(memory_id)
        
        # Calculate initial relevance
        self._calculate_relevance(memory)
        
    def remove_memory(self, memory_id: str):
        """Remove a memory from the working pool"""
        if memory_id in self.memory_index:
            memory = self.memory_index[memory_id]
            self.memory_pool.remove(memory)
            del self.memory_index[memory_id]
            self.category_index[memory['category']].remove(memory_id)
            if memory_id in self.relevance_scores:
                del self.relevance_scores[memory_id]
            if memory_id in self.access_counts:
                del self.access_counts[memory_id]
    
    def access_memory(self, memory_id: str):
        """Record memory access and update relevance"""
        if memory_id in self.memory_index:
            self.access_counts[memory_id] += 1
            self.memory_index[memory_id]['last_accessed'] = datetime.now().timestamp()
            self._calculate_relevance(self.memory_index[memory_id])
    
    def set_focus_mode(self, enabled: bool, keywords: List[str] = None):
        """Enable or disable focus mode with optional keywords"""
        self.focus_mode_enabled = enabled
        self.focus_keywords = keywords or []
        
        # Recalculate relevance for all memories
        for memory in self.memory_pool:
            self._calculate_relevance(memory)
    
    def _calculate_relevance(self, memory: dict):
        """Calculate relevance score for a memory"""
        base_score = memory['importance'] / 10.0  # Normalize to 0-1
        
        # Recency factor
        age_hours = (datetime.now().timestamp() - memory['added_at']) / 3600
        recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
        
        # Access frequency factor
        access_factor = min(1.0, self.access_counts[memory['memory_id']] / 10.0)
        
        # Focus mode factor
        focus_factor = 1.0
        if self.focus_mode_enabled and self.focus_keywords:
            content_lower = memory['content'].lower()
            keyword_matches = sum(1 for kw in self.focus_keywords if kw.lower() in content_lower)
            focus_factor = min(2.0, 1.0 + (keyword_matches * 0.5))
        
        # Calculate final score
        relevance = base_score * recency_factor * (0.5 + 0.5 * access_factor) * focus_factor
        self.relevance_scores[memory['memory_id']] = min(1.0, relevance)
    
    def get_active_memories(self, limit: int = None) -> List[dict]:
        """Get active memories sorted by relevance"""
        memories = list(self.memory_pool)
        
        # Filter by focus threshold if in focus mode
        if self.focus_mode_enabled:
            memories = [m for m in memories if self.relevance_scores.get(m['memory_id'], 0) >= self.focus_threshold]
        
        # Sort by relevance
        memories.sort(key=lambda m: self.relevance_scores.get(m['memory_id'], 0), reverse=True)
        
        if limit:
            memories = memories[:limit]
        
        return memories
    
    def get_statistics(self) -> dict:
        """Get working memory statistics"""
        active_memories = self.get_active_memories()
        
        # Category distribution
        category_dist = {}
        for category, memory_ids in self.category_index.items():
            category_dist[category] = len(memory_ids)
        
        # Calculate average relevance
        relevance_values = list(self.relevance_scores.values())
        avg_relevance = sum(relevance_values) / len(relevance_values) if relevance_values else 0
        
        return {
            'total_memories': len(self.memory_pool),
            'active_memories': len(active_memories),
            'capacity_used': len(self.memory_pool) / self.capacity * 100,
            'avg_relevance_score': avg_relevance,
            'category_distribution': category_dist,
            'total_accesses': sum(self.access_counts.values()),
            'optimization_suggestions': self._get_optimization_suggestions()
        }
    
    def _get_optimization_suggestions(self) -> int:
        """Count optimization suggestions"""
        suggestions = 0
        
        # Check for low relevance memories
        low_relevance = sum(1 for score in self.relevance_scores.values() if score < 0.3)
        if low_relevance > self.capacity * 0.2:  # More than 20% low relevance
            suggestions += 1
        
        # Check for stale memories
        now = datetime.now().timestamp()
        stale_memories = sum(1 for m in self.memory_pool if (now - m['last_accessed']) > 3600)  # 1 hour
        if stale_memories > self.capacity * 0.3:  # More than 30% stale
            suggestions += 1
        
        # Check for unbalanced categories
        if self.category_index:
            sizes = [len(ids) for ids in self.category_index.values()]
            if max(sizes) > sum(sizes) * 0.5:  # One category has more than 50%
                suggestions += 1
        
        return suggestions
    
    def optimize(self):
        """Optimize working memory by removing low-relevance memories"""
        # Remove memories below threshold
        to_remove = [
            m['memory_id'] for m in self.memory_pool 
            if self.relevance_scores.get(m['memory_id'], 0) < 0.2
        ]
        
        for memory_id in to_remove:
            self.remove_memory(memory_id)
        
        self.last_optimization = datetime.now()
        self.optimization_count += 1
        
        return len(to_remove)


# Global working memory instance
_working_memory_system = None
_working_memory_lock = Lock()


def get_working_memory_system() -> WorkingMemorySystem:
    """Get or create the global working memory system instance"""
    global _working_memory_system
    
    with _working_memory_lock:
        if _working_memory_system is None:
            _working_memory_system = WorkingMemorySystem()
        return _working_memory_system


# ---------- Timeline Analysis Functions ----------

def generate_timeline_segments(
    timeline_data: List[Dict[str, Any]], granularity: str, hours: int
) -> List[Dict[str, Any]]:
    """Generate timeline segments summarising state counts / complexity over time."""
    if not timeline_data:
        return []

    start_ts = min(item["timestamp"] for item in timeline_data)
    end_ts = max(item["timestamp"] for item in timeline_data)

    seg_seconds = 1 if granularity == "second" else 60 if granularity == "minute" else 3600
    segments: List[Dict[str, Any]] = []
    current = start_ts

    while current < end_ts:
        seg_end = current + seg_seconds
        seg_states = [it for it in timeline_data if current <= it["timestamp"] < seg_end]
        if seg_states:
            segments.append(
                {
                    "start_time": current,
                    "end_time": seg_end,
                    "state_count": len(seg_states),
                    "avg_complexity": sum(s["complexity_score"] for s in seg_states)
                    / len(seg_states),
                    "max_change_magnitude": max(s["change_magnitude"] for s in seg_states),
                    "dominant_type": Counter(
                        s["state_type"] for s in seg_states
                    ).most_common(1)[0][0],
                }
            )
        current = seg_end
    return segments


def calculate_timeline_stats(timeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate stats about timeline complexity / changes."""
    if not timeline_data:
        return {}

    complexities = [it["complexity_score"] for it in timeline_data]
    changes = [it["change_magnitude"] for it in timeline_data if it["change_magnitude"] > 0]
    stypes = Counter(it["state_type"] for it in timeline_data)
    return {
        "avg_complexity": sum(complexities) / len(complexities),
        "max_complexity": max(complexities),
        "avg_change_magnitude": (sum(changes) / len(changes)) if changes else 0,
        "max_change_magnitude": max(changes) if changes else 0,
        "most_common_type": stypes.most_common(1)[0][0] if stypes else None,
        "type_distribution": dict(stypes),
    }


# ---------- Flame Graph Functions ----------

def build_flame_graph_structure(actions: List[Dict], workflow_id: str) -> Dict:
    """Build hierarchical flame graph structure from actions"""
    total_duration = sum(action.get('duration', 0) for action in actions if action.get('duration'))
    
    flame_graph_data = {
        'name': f'Workflow {workflow_id}',
        'value': total_duration,
        'children': []
    }
    
    # Group actions by tool for flame graph hierarchy
    tool_groups = {}
    for action in actions:
        tool_name = action.get('tool_name', 'unknown')
        if tool_name not in tool_groups:
            tool_groups[tool_name] = []
        tool_groups[tool_name].append(action)
    
    # Build hierarchical structure
    for tool_name, tool_actions in tool_groups.items():
        tool_duration = sum(action.get('duration', 0) for action in tool_actions if action.get('duration'))
        
        tool_node = {
            'name': tool_name,
            'value': tool_duration,
            'children': []
        }
        
        # Add individual actions as children
        for action in tool_actions:
            if action.get('duration'):
                action_node = {
                    'name': f"Action {action['action_id']}",
                    'value': action['duration'],
                    'action_id': action['action_id'],
                    'status': action.get('status'),
                    'reasoning': action.get('reasoning', ''),
                    'started_at': action.get('started_at'),
                    'completed_at': action.get('completed_at')
                }
                tool_node['children'].append(action_node)
        
        flame_graph_data['children'].append(tool_node)
    
    return flame_graph_data


def calculate_critical_path(actions: List[Dict]) -> List[Dict]:
    """Calculate the critical path through the workflow"""
    if not actions:
        return []
    
    # Sort actions by start time
    sorted_actions = sorted(actions, key=lambda x: x.get('started_at', 0))
    
    critical_path = []
    current_time = min(action['started_at'] for action in sorted_actions if action.get('started_at'))
    workflow_end = max(action['completed_at'] for action in sorted_actions if action.get('completed_at'))
    
    while current_time < workflow_end:
        # Find action that was running at current_time and ends latest
        running_actions = [
            a for a in sorted_actions 
            if a.get('started_at', 0) <= current_time and a.get('completed_at', 0) > current_time
        ]
        
        if running_actions:
            # Find the action that ends latest (most critical)
            critical_action = max(running_actions, key=lambda x: x.get('completed_at', 0))
            if critical_action not in [cp['action_id'] for cp in critical_path]:
                critical_path.append({
                    'action_id': critical_action['action_id'],
                    'tool_name': critical_action.get('tool_name'),
                    'duration': critical_action.get('duration', 0),
                    'start_time': critical_action.get('started_at'),
                    'end_time': critical_action.get('completed_at')
                })
            current_time = critical_action.get('completed_at', current_time + 1)
        else:
            # No action running, find next action start
            future_actions = [a for a in sorted_actions if a.get('started_at', 0) > current_time]
            if future_actions:
                current_time = min(a['started_at'] for a in future_actions)
            else:
                break
    
    return critical_path


def convert_to_model(node: Dict) -> FlameGraphNode:
    """Convert flame graph dictionary to Pydantic model"""
    return FlameGraphNode(
        name=node['name'],
        value=node['value'],
        children=[convert_to_model(child) for child in node.get('children', [])],
        action_id=node.get('action_id'),
        status=node.get('status'),
        reasoning=node.get('reasoning'),
        started_at=node.get('started_at'),
        completed_at=node.get('completed_at')
    )


# ---------- Performance Recommendation Functions ----------

def calculate_tool_reliability_score(tool_stats: dict) -> float:
    """Calculate reliability score for a tool"""
    total_calls = tool_stats.get('total_calls', 0)
    successful_calls = tool_stats.get('successful_calls', 0)
    
    if total_calls == 0:
        return 0.0
    
    success_rate = successful_calls / total_calls
    volume_factor = min(1.0, total_calls / 100)  # Normalize by 100 calls
    
    return round(success_rate * volume_factor * 100, 2)


def categorize_tool_performance(avg_execution_time: float) -> str:
    """Categorize tool performance based on average execution time"""
    if avg_execution_time is None:
        return 'unknown'
    
    if avg_execution_time <= 5:
        return 'fast'
    elif avg_execution_time <= 15:
        return 'normal'
    elif avg_execution_time <= 30:
        return 'slow'
    else:
        return 'very_slow' 