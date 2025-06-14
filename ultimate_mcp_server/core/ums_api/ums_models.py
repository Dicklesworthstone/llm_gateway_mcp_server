"""Pydantic models for UMS API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- Cognitive States Models ----------

class CognitiveState(BaseModel):
    state_id: str
    timestamp: float
    formatted_timestamp: str
    state_type: str
    description: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_title: Optional[str] = None
    complexity_score: float
    change_magnitude: float
    age_minutes: float
    memory_count: int
    action_count: int
    state_data: Dict[str, Any] = {}


class CognitiveStatesResponse(BaseModel):
    states: List[CognitiveState]
    total: int
    has_more: bool


class TimelineState(BaseModel):
    state_id: str
    timestamp: float
    formatted_time: str
    state_type: str
    workflow_id: Optional[str] = None
    description: Optional[str] = None
    sequence_number: int
    complexity_score: float
    change_magnitude: float


class TimelineSummaryStats(BaseModel):
    avg_complexity: float
    total_transitions: int
    max_change_magnitude: float


class CognitiveTimelineResponse(BaseModel):
    timeline_data: List[TimelineState]
    total_states: int
    time_range_hours: int
    granularity: str
    summary_stats: TimelineSummaryStats


class Memory(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    importance: float
    created_at: float


class Action(BaseModel):
    action_id: str
    action_type: str
    tool_name: str
    status: str
    started_at: float


class DetailedCognitiveState(BaseModel):
    state_id: str
    timestamp: float
    formatted_timestamp: str
    state_type: str
    description: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_title: Optional[str] = None
    workflow_goal: Optional[str] = None
    state_data: Dict[str, Any]
    complexity_score: float
    memories: List[Memory] = []
    actions: List[Action] = []


class Pattern(BaseModel):
    type: str
    length: int
    similarity: float
    occurrences: int
    first_occurrence: float
    pattern_description: str


class Transition(BaseModel):
    transition: str
    count: int
    percentage: float


class Anomaly(BaseModel):
    state_id: str
    timestamp: float
    anomaly_type: str
    z_score: float
    description: str
    severity: str


class PatternSummary(BaseModel):
    pattern_count: int
    most_common_transition: Optional[Transition] = None
    anomaly_count: int


class CognitivePatternAnalysis(BaseModel):
    total_states: int
    time_range_hours: int
    patterns: List[Pattern] = []
    transitions: List[Transition] = []
    anomalies: List[Anomaly] = []
    summary: PatternSummary


class StateComparisonInfo(BaseModel):
    state_id: str
    timestamp: float
    formatted_timestamp: str


class StateDiff(BaseModel):
    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    modified: Dict[str, Dict[str, Any]] = {}
    magnitude: float


class StateComparisonRequest(BaseModel):
    state_id_1: str = Field(
        ...,
        description="First cognitive state ID for comparison",
        example="state_abc123"
    )
    state_id_2: str = Field(
        ...,
        description="Second cognitive state ID for comparison", 
        example="state_xyz789"
    )


class StateComparisonResponse(BaseModel):
    state_1: StateComparisonInfo
    state_2: StateComparisonInfo
    time_diff_minutes: float
    diff: StateDiff


# ---------- Action Monitor Models ----------

class StatusIndicator(BaseModel):
    """Action status indicator with visual cues"""
    color: str = Field(..., description="Color for visual representation")
    icon: str = Field(..., description="Icon name for the status")
    label: str = Field(..., description="Human-readable status label")
    urgency: str = Field(..., description="Urgency level: low, medium, high")


class ResourceUsage(BaseModel):
    """Resource usage metrics for an action"""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    network_io: float = Field(..., description="Network I/O in KB/s")
    disk_io: float = Field(..., description="Disk I/O in KB/s")


class RunningAction(BaseModel):
    """Model for a currently running action"""
    action_id: str = Field(..., description="Unique action identifier")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_title: Optional[str] = Field(None, description="Workflow title")
    tool_name: str = Field(..., description="Name of the tool being executed")
    status: str = Field(..., description="Current execution status")
    started_at: float = Field(..., description="Start timestamp")
    formatted_start_time: str = Field(..., description="ISO formatted start time")
    execution_time_seconds: float = Field(
        ..., description="Current execution duration in seconds"
    )
    estimated_duration: Optional[float] = Field(
        None, description="Estimated duration in seconds"
    )
    progress_percentage: float = Field(..., description="Estimated progress percentage")
    status_indicator: StatusIndicator = Field(..., description="Visual status indicator")
    performance_category: str = Field(..., description="Performance categorization")
    resource_usage: ResourceUsage = Field(..., description="Current resource usage")
    tool_data: Dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific data"
    )


class RunningActionsResponse(BaseModel):
    """Response for currently running actions"""
    running_actions: List[RunningAction] = Field(
        ..., description="List of currently executing actions"
    )
    total_running: int = Field(..., description="Total number of running actions")
    avg_execution_time: float = Field(
        ..., description="Average execution time of running actions"
    )
    timestamp: str = Field(..., description="Response timestamp")


class QueuedAction(BaseModel):
    """Model for a queued action"""
    action_id: str = Field(..., description="Unique action identifier")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_title: Optional[str] = Field(None, description="Workflow title")
    tool_name: str = Field(..., description="Name of the tool to be executed")
    status: str = Field(..., description="Queue status")
    created_at: float = Field(..., description="Creation timestamp")
    formatted_queue_time: str = Field(..., description="ISO formatted queue time")
    queue_position: int = Field(..., description="Position in the queue (1-based)")
    queue_time_seconds: float = Field(..., description="Time spent in queue")
    estimated_wait_time: float = Field(..., description="Estimated wait time in seconds")
    priority: int = Field(..., description="Numeric priority value")
    priority_label: str = Field(..., description="Human-readable priority label")
    tool_data: Dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific data"
    )


class ActionQueueResponse(BaseModel):
    """Response for action queue status"""
    queued_actions: List[QueuedAction] = Field(..., description="List of queued actions")
    total_queued: int = Field(..., description="Total number of queued actions")
    avg_queue_time: float = Field(..., description="Average time in queue")
    next_action: Optional[QueuedAction] = Field(
        None, description="Next action to be executed"
    )
    timestamp: str = Field(..., description="Response timestamp")


class ActionHistoryItem(BaseModel):
    """Model for a single action in history"""
    action_id: str = Field(..., description="Unique action identifier")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_title: Optional[str] = Field(None, description="Associated workflow title")
    tool_name: str = Field(..., description="Name of the tool executed")
    action_type: Optional[str] = Field(None, description="Type of action")
    status: str = Field(..., description="Action completion status")
    started_at: float = Field(..., description="Unix timestamp when action started")
    completed_at: Optional[float] = Field(
        None, description="Unix timestamp when action completed"
    )
    execution_duration_seconds: float = Field(
        ..., description="Total execution time in seconds"
    )
    performance_score: float = Field(
        ..., description="Calculated performance score (0-100)"
    )
    efficiency_rating: str = Field(
        ..., description="Efficiency rating based on time and output"
    )
    success_rate_impact: int = Field(..., description="Impact on success rate (1 or 0)")
    formatted_start_time: str = Field(..., description="ISO formatted start time")
    formatted_completion_time: Optional[str] = Field(
        None, description="ISO formatted completion time"
    )
    tool_data: Dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific data"
    )
    result_data: Dict[str, Any] = Field(
        default_factory=dict, description="Action result data"
    )
    result_size: int = Field(0, description="Size of the result data")


class PerformanceSummary(BaseModel):
    """Performance summary statistics"""
    avg_score: float = Field(..., description="Average performance score")
    top_performer: Optional[Dict[str, Any]] = Field(
        None, description="Best performing tool"
    )
    worst_performer: Optional[Dict[str, Any]] = Field(
        None, description="Worst performing tool"
    )
    efficiency_distribution: Dict[str, int] = Field(
        ..., description="Distribution of efficiency ratings"
    )


class ActionHistoryResponse(BaseModel):
    """Response model for action history"""
    action_history: List[ActionHistoryItem] = Field(
        ..., description="List of completed actions"
    )
    total_actions: int = Field(
        ..., description="Total number of actions in the time period"
    )
    success_rate: float = Field(..., description="Overall success rate percentage")
    avg_execution_time: float = Field(..., description="Average execution time in seconds")
    performance_summary: PerformanceSummary = Field(
        ..., description="Performance summary statistics"
    )
    timestamp: str = Field(..., description="Response timestamp")


class OverallMetrics(BaseModel):
    """Overall action execution metrics"""
    total_actions: int = Field(..., description="Total number of actions executed")
    successful_actions: int = Field(
        ..., description="Number of successfully completed actions"
    )
    failed_actions: int = Field(..., description="Number of failed actions")
    avg_duration: Optional[float] = Field(
        None, description="Average execution duration in seconds"
    )
    success_rate_percentage: float = Field(
        ..., description="Overall success rate as percentage"
    )
    failure_rate_percentage: float = Field(
        ..., description="Overall failure rate as percentage"
    )
    avg_duration_seconds: float = Field(..., description="Average duration in seconds")


class ToolUsageStat(BaseModel):
    """Statistics for a single tool"""
    tool_name: str = Field(..., description="Name of the tool")
    usage_count: int = Field(..., description="Number of times the tool was used")
    success_count: int = Field(..., description="Number of successful executions")
    avg_duration: Optional[float] = Field(
        None, description="Average execution time in seconds"
    )


class HourlyMetric(BaseModel):
    """Hourly performance metrics"""
    hour: str = Field(..., description="Hour of the day (0-23)")
    action_count: int = Field(..., description="Number of actions in this hour")
    avg_duration: Optional[float] = Field(
        None, description="Average duration for this hour"
    )
    success_count: int = Field(..., description="Number of successful actions")


class PerformanceInsight(BaseModel):
    """Performance insight or recommendation"""
    type: str = Field(..., description="Type of insight (warning, info, etc.)")
    title: str = Field(..., description="Title of the insight")
    message: str = Field(..., description="Detailed message")
    severity: str = Field(..., description="Severity level (high, medium, low)")


class ActionMetricsResponse(BaseModel):
    """Response model for action metrics"""
    overall_metrics: OverallMetrics = Field(..., description="Overall execution metrics")
    tool_usage_stats: List[ToolUsageStat] = Field(
        ..., description="Per-tool usage statistics"
    )
    hourly_performance: List[HourlyMetric] = Field(
        ..., description="Hourly performance breakdown"
    )
    performance_insights: List[PerformanceInsight] = Field(
        ..., description="Actionable insights and recommendations"
    )
    timestamp: str = Field(..., description="Response timestamp")


# ---------- Artifacts Models ----------

class Artifact(BaseModel):
    """Model for a single artifact"""
    artifact_id: str = Field(..., description="Unique artifact identifier")
    name: str = Field(..., description="Name of the artifact")
    artifact_type: str = Field(
        ..., description="Type of artifact (document, image, code, etc.)"
    )
    description: Optional[str] = Field(None, description="Description of the artifact")
    file_path: Optional[str] = Field(None, description="File system path to the artifact")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_title: Optional[str] = Field(None, description="Title of associated workflow")
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")
    file_size: int = Field(..., description="File size in bytes")
    file_size_human: str = Field(..., description="Human-readable file size")
    importance: Optional[float] = Field(None, description="Importance score (1-10)")
    access_count: int = Field(0, description="Number of times accessed")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    relationship_count: int = Field(0, description="Number of related artifacts")
    version_count: int = Field(0, description="Number of versions")
    formatted_created_at: str = Field(..., description="ISO formatted creation date")
    formatted_updated_at: str = Field(..., description="ISO formatted update date")
    age_days: float = Field(..., description="Age of artifact in days")


class ArtifactsFilter(BaseModel):
    """Filter parameters used in the request"""
    artifact_type: Optional[str] = Field(None, description="Type filter applied")
    workflow_id: Optional[str] = Field(None, description="Workflow filter applied")
    tags: Optional[str] = Field(None, description="Tags filter applied")
    search: Optional[str] = Field(None, description="Search query applied")
    sort_by: str = Field(..., description="Sort field used")
    sort_order: str = Field(..., description="Sort order used")


class ArtifactsResponse(BaseModel):
    """Response model for artifacts listing"""
    artifacts: List[Artifact] = Field(..., description="List of artifacts")
    total: int = Field(..., description="Total number of artifacts matching query")
    has_more: bool = Field(..., description="Whether there are more artifacts available")
    filters: ArtifactsFilter = Field(..., description="Filters that were applied")


class ArtifactTypeStats(BaseModel):
    """Statistics for a specific artifact type"""
    artifact_type: str = Field(..., description="Type of artifact")
    count: int = Field(..., description="Number of artifacts of this type")
    avg_importance: Optional[float] = Field(None, description="Average importance score")
    total_size: int = Field(..., description="Total size of all artifacts of this type")
    max_access_count: int = Field(..., description="Maximum access count for this type")


class ArtifactOverallStats(BaseModel):
    """Overall artifact statistics"""
    total_artifacts: int = Field(..., description="Total number of artifacts")
    unique_types: int = Field(..., description="Number of unique artifact types")
    unique_workflows: int = Field(..., description="Number of unique workflows")
    total_size: int = Field(..., description="Total size of all artifacts in bytes")
    total_size_human: str = Field(..., description="Human-readable total size")
    avg_size: float = Field(..., description="Average artifact size in bytes")
    latest_created: Optional[float] = Field(
        None, description="Timestamp of most recent artifact"
    )
    earliest_created: Optional[float] = Field(
        None, description="Timestamp of oldest artifact"
    )


class ArtifactStatsResponse(BaseModel):
    """Response model for artifact statistics"""
    overall: ArtifactOverallStats = Field(..., description="Overall statistics")
    by_type: List[ArtifactTypeStats] = Field(
        ..., description="Statistics broken down by type"
    )


# ---------- Memory Quality Models ----------

class MemoryDetail(BaseModel):
    """Detailed information about a memory"""
    memory_id: str = Field(..., description="Unique memory identifier")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    memory_type: str = Field(..., description="Type of memory")
    importance: float = Field(..., description="Importance score")
    created_at: float = Field(..., description="Creation timestamp")


class DuplicateGroup(BaseModel):
    """Group of duplicate memories"""
    cluster_id: str = Field(..., description="Unique identifier for this duplicate cluster")
    content_preview: str = Field(..., description="Preview of the duplicated content")
    duplicate_count: int = Field(..., description="Number of duplicates in this group")
    memory_ids: List[str] = Field(..., description="List of all memory IDs in this group")
    primary_memory_id: str = Field(..., description="Suggested primary memory to keep")
    memory_details: List[MemoryDetail] = Field(..., description="Detailed info for each memory")
    first_created: float = Field(..., description="Timestamp of earliest duplicate")
    last_created: float = Field(..., description="Timestamp of latest duplicate")
    avg_importance: float = Field(..., description="Average importance across duplicates")
    recommendation: str = Field(..., description="Recommended action (merge/review)")


class DuplicatesResponse(BaseModel):
    """Response model for duplicate analysis"""
    success: bool = Field(..., description="Whether analysis completed successfully")
    clusters: List[DuplicateGroup] = Field(..., description="List of duplicate groups")
    duplicate_groups: List[DuplicateGroup] = Field(..., description="Alias for clusters (backward compatibility)")
    total_groups: int = Field(..., description="Total number of duplicate groups found")
    total_duplicates: int = Field(..., description="Total number of duplicate memories")


class OrphanedMemory(BaseModel):
    """Model for an orphaned memory"""
    memory_id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Type of memory")
    importance: float = Field(..., description="Importance score")
    created_at: float = Field(..., description="Creation timestamp")


class OrphanedMemoriesResponse(BaseModel):
    """Response model for orphaned memories"""
    success: bool = Field(..., description="Whether query completed successfully")
    orphaned_memories: List[OrphanedMemory] = Field(..., description="List of orphaned memories")
    total_orphaned: int = Field(..., description="Total count of orphaned memories")
    recommendation: str = Field(..., description="Recommended action for orphaned memories")


class BulkOperationRequest(BaseModel):
    """Request model for bulk operations"""
    operation_type: str = Field(
        ...,
        description="Type of bulk operation to perform",
        regex="^(delete|archive|merge)$"
    )
    memory_ids: List[str] = Field(
        ...,
        description="List of memory IDs to operate on",
        min_items=1
    )
    target_memory_id: Optional[str] = Field(
        None,
        description="Target memory ID for merge operations"
    )


class BulkOperationResponse(BaseModel):
    """Response model for bulk operations"""
    success: bool = Field(..., description="Whether operation completed successfully")
    operation_type: str = Field(..., description="Type of operation performed")
    memory_ids: List[str] = Field(..., description="Memory IDs that were operated on")
    success_count: int = Field(..., description="Number of successful operations")
    error_count: int = Field(..., description="Number of failed operations")
    message: str = Field(..., description="Summary message of the operation")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    merged_into: Optional[str] = Field(None, description="Target memory ID for merge operations")


class PreviewMemory(BaseModel):
    """Memory preview for bulk operations"""
    memory_id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Type of memory")
    importance: float = Field(..., description="Importance score")
    workflow_id: Optional[str] = Field(None, description="Associated workflow")


class BulkOperationPreview(BaseModel):
    """Preview of bulk operation effects"""
    operation_type: str = Field(..., description="Type of operation to be performed")
    total_affected: int = Field(..., description="Total memories that will be affected")
    preview_description: str = Field(..., description="Description of what will happen")
    affected_memories: List[PreviewMemory] = Field(..., description="Details of affected memories")
    merge_target: Optional[PreviewMemory] = Field(None, description="Target memory for merge")
    will_be_deleted: Optional[List[PreviewMemory]] = Field(None, description="Memories to be deleted in merge")


class BulkPreviewResponse(BaseModel):
    """Response model for bulk operation preview"""
    success: bool = Field(..., description="Whether preview generated successfully")
    operation: BulkOperationPreview = Field(..., description="Preview of the operation")


# ---------- Working Memory Models ----------

class FocusMode(BaseModel):
    """Focus mode configuration"""
    enabled: bool = Field(..., description="Whether focus mode is enabled")
    focus_keywords: List[str] = Field(default_factory=list, description="Keywords for focus filtering")


class PerformanceMetrics(BaseModel):
    """Working memory performance metrics"""
    avg_relevance_score: float = Field(..., description="Average relevance score across all memories")
    optimization_suggestions: int = Field(..., description="Number of optimization suggestions")


class WorkingMemoryStatus(BaseModel):
    """Complete working memory system status"""
    initialized: bool = Field(..., description="Whether the system is initialized")
    total_capacity: int = Field(..., description="Maximum memory capacity")
    current_size: int = Field(..., description="Current number of memories in pool")
    utilization_percentage: float = Field(..., description="Percentage of capacity used")
    focus_mode: FocusMode = Field(..., description="Focus mode configuration")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    category_distribution: Dict[str, int] = Field(default_factory=dict, description="Memory count by category")
    last_optimization: str = Field(..., description="ISO timestamp of last optimization")
    optimization_count: int = Field(..., description="Total number of optimizations performed")


class InitializeRequest(BaseModel):
    """Request model for initializing working memory"""
    capacity: int = Field(
        100,
        ge=10,
        le=1000,
        description="Maximum number of memories in working pool"
    )
    focus_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Relevance threshold for focus mode"
    )


class InitializeResponse(BaseModel):
    """Response model for initialization"""
    success: bool = Field(..., description="Whether initialization was successful")
    message: str = Field(..., description="Status message")
    configuration: Dict[str, Any] = Field(..., description="Applied configuration")


class MemoryItem(BaseModel):
    """Model for a memory in the working pool"""
    memory_id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="Memory content")
    category: str = Field(..., description="Memory category")
    importance: float = Field(..., description="Importance score (0-10)")
    relevance_score: float = Field(..., description="Current relevance score (0-1)")
    added_at: float = Field(..., description="Timestamp when added to working memory")
    last_accessed: float = Field(..., description="Timestamp of last access")
    access_count: int = Field(..., description="Number of times accessed")


class ActiveMemoriesResponse(BaseModel):
    """Response for active memories query"""
    memories: List[MemoryItem] = Field(..., description="List of active memories sorted by relevance")
    total_count: int = Field(..., description="Total number of memories matching criteria")
    focus_active: bool = Field(..., description="Whether focus mode filtering is active")


class SetFocusModeRequest(BaseModel):
    """Request to set focus mode"""
    enabled: bool = Field(..., description="Enable or disable focus mode")
    keywords: List[str] = Field(default_factory=list, description="Keywords for focus filtering", max_items=20)


class OptimizeResponse(BaseModel):
    """Response for optimization operation"""
    success: bool = Field(..., description="Whether optimization was successful")
    removed_count: int = Field(..., description="Number of memories removed")
    message: str = Field(..., description="Optimization result message")


# ---------- Performance Profiler Models ----------

class PerformanceOverviewStats(BaseModel):
    """Overall performance statistics"""
    total_actions: int = Field(..., description="Total number of actions executed")
    active_workflows: int = Field(..., description="Number of unique workflows")
    avg_execution_time: float = Field(..., description="Average execution time in seconds")
    min_execution_time: Optional[float] = Field(None, description="Minimum execution time")
    max_execution_time: Optional[float] = Field(None, description="Maximum execution time")
    successful_actions: int = Field(..., description="Number of successful actions")
    failed_actions: int = Field(..., description="Number of failed actions")
    tools_used: int = Field(..., description="Number of distinct tools used")
    success_rate_percentage: float = Field(..., description="Success rate as percentage")
    throughput_per_hour: float = Field(..., description="Actions processed per hour")
    error_rate_percentage: float = Field(..., description="Error rate as percentage")
    avg_workflow_size: float = Field(..., description="Average actions per workflow")


class TimelineBucket(BaseModel):
    """Performance metrics for a time bucket"""
    time_bucket: str = Field(..., description="Time bucket identifier")
    action_count: int = Field(..., description="Number of actions in this bucket")
    avg_duration: Optional[float] = Field(None, description="Average duration in seconds")
    successful_count: int = Field(..., description="Number of successful actions")
    failed_count: int = Field(..., description="Number of failed actions")
    workflow_count: int = Field(..., description="Number of unique workflows")


class ToolUtilization(BaseModel):
    """Tool utilization metrics"""
    tool_name: str = Field(..., description="Name of the tool")
    usage_count: int = Field(..., description="Number of times used")
    avg_duration: Optional[float] = Field(None, description="Average execution duration")
    success_count: int = Field(..., description="Number of successful executions")
    max_duration: Optional[float] = Field(None, description="Maximum execution duration")


class Bottleneck(BaseModel):
    """Performance bottleneck information"""
    tool_name: str = Field(..., description="Tool causing the bottleneck")
    workflow_id: Optional[str] = Field(None, description="Associated workflow")
    action_id: str = Field(..., description="Action identifier")
    started_at: float = Field(..., description="Start timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")
    duration: float = Field(..., description="Duration in seconds")
    status: str = Field(..., description="Action status")
    reasoning: Optional[str] = Field(None, description="Action reasoning")


class PerformanceOverviewResponse(BaseModel):
    """Response model for performance overview"""
    overview: PerformanceOverviewStats
    timeline: List[TimelineBucket]
    tool_utilization: List[ToolUtilization]
    bottlenecks: List[Bottleneck]
    analysis_period: Dict[str, Any] = Field(..., description="Analysis period information")
    timestamp: str = Field(..., description="Response generation timestamp")


class ToolBottleneck(BaseModel):
    """Tool performance bottleneck analysis"""
    tool_name: str = Field(..., description="Name of the tool")
    total_calls: int = Field(..., description="Total number of calls")
    avg_duration: float = Field(..., description="Average execution duration")
    max_duration: float = Field(..., description="Maximum execution duration")
    min_duration: float = Field(..., description="Minimum execution duration")
    p95_duration: float = Field(..., description="95th percentile duration")
    p99_duration: float = Field(..., description="99th percentile duration")
    failure_count: int = Field(..., description="Number of failures")
    total_time_spent: float = Field(..., description="Total time spent in seconds")


class WorkflowBottleneck(BaseModel):
    """Workflow performance bottleneck"""
    workflow_id: str = Field(..., description="Workflow identifier")
    title: Optional[str] = Field(None, description="Workflow title")
    action_count: int = Field(..., description="Number of actions")
    avg_action_duration: float = Field(..., description="Average action duration")
    max_action_duration: float = Field(..., description="Maximum action duration")
    total_workflow_time: float = Field(..., description="Total workflow execution time")
    workflow_start: float = Field(..., description="Workflow start timestamp")
    workflow_end: float = Field(..., description="Workflow end timestamp")
    total_elapsed_time: float = Field(..., description="Total elapsed wall-clock time")


class ParallelizationOpportunity(BaseModel):
    """Workflow parallelization opportunity"""
    workflow_id: str = Field(..., description="Workflow identifier")
    sequential_actions: int = Field(..., description="Number of sequential actions")
    total_sequential_time: float = Field(..., description="Total sequential execution time")
    actual_elapsed_time: float = Field(..., description="Actual elapsed time")
    potential_time_savings: float = Field(..., description="Potential time savings in seconds")
    parallelization_efficiency: float = Field(..., description="Current parallelization efficiency percentage")
    optimization_score: float = Field(..., description="Optimization potential score (0-10)")


class ResourceContention(BaseModel):
    """Resource contention analysis"""
    tool_name: str = Field(..., description="Tool name")
    concurrent_usage: int = Field(..., description="Number of concurrent usages")
    avg_duration_under_contention: float = Field(..., description="Average duration when contended")


class OptimizationRecommendation(BaseModel):
    """Performance optimization recommendation"""
    type: str = Field(..., description="Type of optimization")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    impact: str = Field(..., description="Expected impact description")
    actions: List[str] = Field(..., description="Recommended actions to take")


class BottleneckAnalysisResponse(BaseModel):
    """Response model for bottleneck analysis"""
    tool_bottlenecks: List[ToolBottleneck]
    workflow_bottlenecks: List[WorkflowBottleneck]
    parallelization_opportunities: List[ParallelizationOpportunity]
    resource_contention: List[ResourceContention]
    recommendations: List[OptimizationRecommendation]
    analysis_summary: Dict[str, Any]
    timestamp: str


class FlameGraphNode(BaseModel):
    """Model for a flame graph node"""
    name: str = Field(..., description="Name of the node (workflow, tool, or action)")
    value: float = Field(..., description="Duration in seconds")
    children: List['FlameGraphNode'] = Field(default_factory=list, description="Child nodes")
    action_id: Optional[str] = Field(None, description="Action ID if this is an action node")
    status: Optional[str] = Field(None, description="Execution status")
    reasoning: Optional[str] = Field(None, description="Reasoning for the action")
    started_at: Optional[float] = Field(None, description="Start timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")


FlameGraphNode.model_rebuild()  # Needed for recursive model


class CriticalPathAction(BaseModel):
    """Model for a critical path action"""
    action_id: str = Field(..., description="Action identifier")
    tool_name: str = Field(..., description="Tool used for the action")
    duration: float = Field(..., description="Duration in seconds")
    start_time: float = Field(..., description="Start timestamp")
    end_time: float = Field(..., description="End timestamp")


class WorkflowMetrics(BaseModel):
    """Workflow performance metrics"""
    total_actions: int = Field(..., description="Total number of actions in workflow")
    total_cpu_time: float = Field(..., description="Total CPU time (sum of all action durations)")
    wall_clock_time: float = Field(..., description="Total wall clock time from start to end")
    parallelization_efficiency: float = Field(..., description="Efficiency percentage (0-100)")
    avg_action_duration: float = Field(..., description="Average duration per action")
    workflow_start: float = Field(..., description="Workflow start timestamp")
    workflow_end: float = Field(..., description="Workflow end timestamp")


class WorkflowAnalysis(BaseModel):
    """Analysis results for workflow optimization"""
    bottleneck_tool: Optional[str] = Field(None, description="Tool causing the main bottleneck")
    parallelization_potential: float = Field(..., description="Potential time savings through parallelization")
    optimization_score: float = Field(..., description="Overall optimization score (0-10)")


class FlameGraphResponse(BaseModel):
    """Response model for flame graph generation"""
    flame_graph: FlameGraphNode = Field(..., description="Hierarchical flame graph data")
    metrics: WorkflowMetrics = Field(..., description="Workflow performance metrics")
    critical_path: List[CriticalPathAction] = Field(..., description="Critical path through the workflow")
    analysis: WorkflowAnalysis = Field(..., description="Workflow optimization analysis")
    timestamp: str = Field(..., description="Response generation timestamp")


class DailyTrend(BaseModel):
    """Model for daily performance metrics"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    action_count: int = Field(..., description="Number of actions executed")
    avg_duration: Optional[float] = Field(None, description="Average action duration in seconds")
    success_rate: float = Field(..., description="Success rate percentage (0-100)")
    throughput: float = Field(..., description="Actions per hour")
    error_rate: float = Field(..., description="Error rate percentage (0-100)")
    successful_actions: int = Field(..., description="Number of successful actions")
    failed_actions: int = Field(..., description="Number of failed actions")
    workflow_count: int = Field(..., description="Number of unique workflows")
    tool_count: int = Field(..., description="Number of unique tools used")


class ToolTrend(BaseModel):
    """Model for tool-specific performance trends"""
    tool_name: str = Field(..., description="Name of the tool")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    usage_count: int = Field(..., description="Number of times used")
    avg_duration: Optional[float] = Field(None, description="Average execution duration")
    success_count: int = Field(..., description="Number of successful executions")


class WorkflowComplexityTrend(BaseModel):
    """Model for workflow complexity trends"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    workflow_id: str = Field(..., description="Workflow identifier")
    action_count: int = Field(..., description="Number of actions in workflow")
    total_duration: Optional[float] = Field(None, description="Total workflow duration")
    elapsed_time: Optional[float] = Field(None, description="Wall clock time")


class TrendAnalysis(BaseModel):
    """Trend analysis results"""
    performance_trend: str = Field(..., description="Overall performance trend (improving/degrading/stable/insufficient_data)")
    success_trend: str = Field(..., description="Success rate trend (improving/degrading/stable/insufficient_data)")
    data_points: int = Field(..., description="Number of data points analyzed")
    analysis_period_days: int = Field(..., description="Analysis period in days")


class InsightMetrics(BaseModel):
    """Performance insight metrics"""
    best_performing_day: Optional[DailyTrend] = Field(None, description="Day with best performance")
    worst_performing_day: Optional[DailyTrend] = Field(None, description="Day with worst performance")
    peak_throughput_day: Optional[DailyTrend] = Field(None, description="Day with highest throughput")
    avg_daily_actions: float = Field(..., description="Average actions per day")


class PerformanceTrendsResponse(BaseModel):
    """Response model for performance trends analysis"""
    daily_trends: List[DailyTrend] = Field(..., description="Daily performance metrics")
    tool_trends: List[ToolTrend] = Field(..., description="Tool-specific performance trends")
    workflow_complexity: List[WorkflowComplexityTrend] = Field(..., description="Workflow complexity trends")
    trend_analysis: TrendAnalysis = Field(..., description="Overall trend analysis")
    patterns: List[PerformancePattern] = Field(..., description="Detected performance patterns")
    insights: InsightMetrics = Field(..., description="Key performance insights")
    timestamp: str = Field(..., description="Response generation timestamp")


class ImpactEstimate(BaseModel):
    """Model for recommendation impact estimates"""
    time_savings_potential: float = Field(..., description="Estimated time savings in seconds")
    affected_actions: int = Field(..., description="Number of actions that would benefit")
    cost_benefit_ratio: float = Field(..., description="Ratio of benefit to implementation cost")
    affected_workflows: Optional[int] = Field(None, description="Number of affected workflows")
    efficiency_improvement: Optional[float] = Field(None, description="Percentage efficiency improvement")
    reliability_improvement: Optional[float] = Field(None, description="Percentage reliability improvement")
    user_experience_impact: Optional[str] = Field(None, description="Impact on user experience (high/medium/low)")


class PerformanceRecommendation(BaseModel):
    """Model for a single performance recommendation"""
    id: str = Field(..., description="Unique identifier for the recommendation")
    type: str = Field(..., description="Type of recommendation (tool_optimization, parallelization, reliability_improvement)")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    title: str = Field(..., description="Brief title of the recommendation")
    description: str = Field(..., description="Detailed description of the issue and recommendation")
    impact_estimate: ImpactEstimate = Field(..., description="Estimated impact of implementing this recommendation")
    implementation_steps: List[str] = Field(..., description="Step-by-step implementation guide")
    estimated_effort: str = Field(..., description="Estimated implementation effort (low, medium, high)")
    prerequisites: List[str] = Field(..., description="Prerequisites for implementation")
    metrics_to_track: List[str] = Field(..., description="Metrics to track after implementation")


class RecommendationSummary(BaseModel):
    """Summary statistics for recommendations"""
    total_recommendations: int = Field(..., description="Total number of recommendations generated")
    high_priority: int = Field(..., description="Number of high priority recommendations")
    medium_priority: int = Field(..., description="Number of medium priority recommendations")
    low_priority: int = Field(..., description="Number of low priority recommendations")
    estimated_total_savings: float = Field(..., description="Total estimated time savings in seconds")
    analysis_period_hours: int = Field(..., description="Hours of data analyzed")


class ImplementationRoadmap(BaseModel):
    """Categorized implementation roadmap"""
    quick_wins: List[PerformanceRecommendation] = Field(..., description="Low effort, high impact recommendations")
    major_improvements: List[PerformanceRecommendation] = Field(..., description="High effort, high impact recommendations")
    maintenance_tasks: List[PerformanceRecommendation] = Field(..., description="Low priority maintenance recommendations")


class PerformanceRecommendationsResponse(BaseModel):
    """Response model for performance recommendations"""
    recommendations: List[PerformanceRecommendation] = Field(..., description="List of actionable recommendations")
    summary: RecommendationSummary = Field(..., description="Summary statistics")
    implementation_roadmap: ImplementationRoadmap = Field(..., description="Recommendations organized by implementation strategy")
    timestamp: str = Field(..., description="ISO timestamp of analysis")


# ---------- Workflow Management Models ----------

class WorkflowScheduleRequest(BaseModel):
    """Request model for scheduling a workflow"""
    scheduled_at: datetime = Field(
        ...,
        description="ISO timestamp for when to execute the workflow",
        example="2024-01-01T12:00:00Z"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Execution priority (1=highest, 10=lowest)",
        example=3
    )


class ScheduleData(BaseModel):
    """Schedule data for the workflow"""
    workflow_id: str = Field(..., description="ID of the scheduled workflow")
    scheduled_at: str = Field(..., description="Scheduled execution time")
    priority: int = Field(..., description="Execution priority")
    status: str = Field(..., description="Schedule status")
    created_at: str = Field(..., description="When the schedule was created")


class WorkflowScheduleResponse(BaseModel):
    """Response model for workflow scheduling"""
    success: bool = Field(..., description="Whether scheduling was successful")
    schedule_id: str = Field(..., description="Unique identifier for this schedule")
    message: str = Field(..., description="Success or error message")
    schedule_data: ScheduleData = Field(..., description="Details of the created schedule")


class RestoreStateRequest(BaseModel):
    """Request model for restoring a cognitive state"""
    restore_mode: str = Field(
        default="full",
        regex="^(full|partial|snapshot)$",
        description="Type of restoration to perform",
        example="full"
    )


class RestoreData(BaseModel):
    """Restoration data"""
    state_id: str = Field(..., description="ID of the state being restored")
    restore_mode: str = Field(..., description="Restoration mode used")
    restored_at: str = Field(..., description="When the restoration occurred")
    original_timestamp: Optional[float] = Field(None, description="Original state timestamp")


class RestoreStateResponse(BaseModel):
    """Response model for state restoration"""
    success: bool = Field(..., description="Whether restoration was successful")
    message: str = Field(..., description="Success or error message")
    restore_data: RestoreData = Field(..., description="Details of the restoration")


# ---------- Health Check Models ----------

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status indicator", example="ok")
    version: str = Field(..., description="Server version string", example="0.1.0")


# ---------- Performance Trends Models ----------

class PerformancePattern(BaseModel):
    """Detected performance pattern"""
    type: str = Field(..., description="Type of pattern detected")
    description: str = Field(..., description="Description of the pattern")
    impact: str = Field(..., description="Impact level (high/medium/low)")
    recommendation: str = Field(..., description="Recommended action")
    date: Optional[str] = Field(None, description="Date of occurrence for anomalies") 