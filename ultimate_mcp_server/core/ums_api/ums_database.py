"""Database utilities for UMS API."""

import sqlite3
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

# Database path configuration
def get_database_path() -> str:
    """Get the path to the unified agent memory database."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    storage_dir = project_root / "storage"
    return str(storage_dir / "unified_agent_memory.db")


def get_db_connection() -> sqlite3.Connection:
    """
    Return a SQLite connection with row factory.
    
    This function creates a connection to the unified agent memory database
    and configures it with a row factory for easier data access.
    
    Returns:
        sqlite3.Connection: Database connection with row factory configured
    """
    conn = sqlite3.connect(get_database_path())
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(query: str, params: tuple = None) -> list:
    """
    Execute a SELECT query and return results as a list of dictionaries.
    
    Args:
        query: SQL SELECT query to execute
        params: Optional parameters for the query
        
    Returns:
        List of dictionaries representing the query results
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        return results
    finally:
        conn.close()


def execute_update(query: str, params: tuple = None) -> int:
    """
    Execute an INSERT, UPDATE, or DELETE query and return the number of affected rows.
    
    Args:
        query: SQL query to execute
        params: Optional parameters for the query
        
    Returns:
        Number of rows affected by the query
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def ensure_database_exists() -> bool:
    """
    Ensure the database file exists and is accessible.
    
    Returns:
        True if the database exists and is accessible, False otherwise
    """
    try:
        db_path = get_database_path()
        return Path(db_path).exists()
    except Exception:
        return False 
# ---------- Helper Functions for Data Processing ----------
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

# ---------- Action Monitoring Helper Functions ----------
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

# ---------- File and Data Utilities ----------
def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# ---------- Performance Analysis Functions ----------
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


# ---------- Cognitive Pattern Analysis Functions ----------
def find_cognitive_patterns(
    states: List[Dict[str, Any]], min_length: int, similarity_threshold: float
) -> List[Dict[str, Any]]:
    """Find recurring patterns in cognitive states"""
    patterns = []
    from collections import defaultdict

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
    from collections import defaultdict

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

# ---------- Pattern analysis models ----------
