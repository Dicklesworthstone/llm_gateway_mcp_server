"""Unified Memory System

This module provides a comprehensive memory, reasoning, and workflow tracking system
designed for LLM agents, merging sophisticated cognitive modeling with structured
process tracking.

Key Features:
- Multi-level memory hierarchy (working, episodic, semantic, procedural) with rich metadata.
- Structured workflow, action, artifact, and thought chain tracking.
- Associative memory graph with automatic linking capabilities.
- Vector embeddings for semantic similarity and clustering.
- Foundational tools for recording agent activity and knowledge.
- Integrated episodic memory creation linked to actions and artifacts.
- Basic cognitive state saving (structure defined, loading/saving tools ported).
- SQLite backend using aiosqlite with performance optimizations.
"""

import asyncio
import contextlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import aiosqlite
import markdown
import networkx as nx
import numpy as np
from pygments.formatters import HtmlFormatter
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import (
    Provider as LLMGatewayProvider,  # To use provider constants
)
from ultimate_mcp_server.core.providers.base import (
    get_provider,  # For consolidation/reflection LLM calls
)

# Import error handling and decorators from agent_memory concepts
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.text import count_tokens

logger = get_logger("ultimate_mcp_server.tools.unified_memory")


# --- BEGIN UMS MONKEY PATCH FOR AIOSQLITE CONNECTION METHODS ---
# This patch adds helper methods to aiosqlite.Connection if they are missing,
# to provide execute_fetchone, execute_fetchall, and execute_fetchval functionality
# without altering widespread existing code. This addresses AttributeErrors if the
# aiosqlite version in use doesn't have these convenience methods directly.


async def _ums_patched_execute_fetchone(self, sql, parameters=None):
    """Patched version of execute_fetchone."""
    async with self.execute(sql, parameters) as cursor:
        return await cursor.fetchone()


async def _ums_patched_execute_fetchall(self, sql, parameters=None):
    """Patched version of execute_fetchall."""
    async with self.execute(sql, parameters) as cursor:
        return await cursor.fetchall()


async def _ums_patched_execute_fetchval(self, sql, parameters=None):
    """Patched version of execute_fetchval (fetches first column of first row)."""
    async with self.execute(sql, parameters) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None


try:
    # Ensure aiosqlite is imported to access its Connection class
    if not hasattr(aiosqlite.Connection, "execute_fetchone"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchone' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchone = _ums_patched_execute_fetchone

    if not hasattr(aiosqlite.Connection, "execute_fetchall"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchall' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchall = _ums_patched_execute_fetchall

    if not hasattr(aiosqlite.Connection, "execute_fetchval"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchval' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchval = _ums_patched_execute_fetchval
except ImportError:
    logger.error("UMS MONKEY-PATCH: aiosqlite module not found. Cannot apply patches.")
except AttributeError:
    logger.error(
        "UMS MONKEY-PATCH: aiosqlite.Connection not found or attribute error during patching."
    )
# --- END UMS MONKEY PATCH ---


# ======================================================
# Configuration Settings
# ======================================================

# Load config once at module level for efficiency
try:
    config = get_config()
    agent_memory_config = config.agent_memory

    # ------------------------------------------------------------------
    # Ensure all tunables exist even when the YAML omits them
    # ------------------------------------------------------------------
    _defaults = {
        # --- relevance / similarity ---
        "memory_decay_rate": 0.001,  # per-hour linear decay
        "similarity_threshold": 0.85,  # cosine-similarity cutoff
        # --- serialization ---
        "max_text_length": 64_000,  # byte-cap enforced by MemoryUtils.serialize
        # --- TTLs used by store_memory (new) ---
        "ttl_working": 3_600,  # 1 hour default for WORKING memories
        "ttl_episodic": 86_400,  # 24 hours default for EPISODIC memories
        # --- search tuning used by hybrid_search_memories (new) ---
        "max_semantic_candidates": 500,  # hard ceiling on candidate pool
        # --- multi-tool support ---
        "enable_batched_operations": True,  # allow multiple tool calls per turn
        "max_tools_per_batch": 20,  # prevent abuse
    }

    for _k, _v in _defaults.items():
        if not hasattr(agent_memory_config, _k) or getattr(agent_memory_config, _k) is None:
            setattr(agent_memory_config, _k, _v)

    # Expose fast globals for hot paths; attributes remain on the config object
    MEMORY_DECAY_RATE = agent_memory_config.memory_decay_rate
    SIMILARITY_THRESHOLD = agent_memory_config.similarity_threshold
    MAX_TEXT_LENGTH = agent_memory_config.max_text_length
    TTL_WORKING_DEFAULT = agent_memory_config.ttl_working
    TTL_EPISODIC_DEFAULT = agent_memory_config.ttl_episodic
    MAX_SEMANTIC_CANDIDATES = agent_memory_config.max_semantic_candidates

except Exception as config_e:
    logger.critical(
        f"CRITICAL: Failed to load configuration for unified_memory_system: {config_e}",
        exc_info=True,
    )
    raise RuntimeError(
        f"Failed to initialize configuration for unified_memory_system: {config_e}"
    ) from config_e

# --- UMS Tool Default Constants (can be overridden by agent via fetch_limits/show_limits) ---
# These should mirror or be inspired by the agent's CONTEXT_*_FETCH_LIMIT and CONTEXT_*_SHOW_LIMIT
UMS_DEFAULT_FETCH_LIMIT_RECENT_ACTIONS = 10
UMS_DEFAULT_FETCH_LIMIT_IMPORTANT_MEMORIES = 7
UMS_DEFAULT_FETCH_LIMIT_KEY_THOUGHTS = 7
UMS_DEFAULT_FETCH_LIMIT_PROACTIVE = 5
UMS_DEFAULT_FETCH_LIMIT_PROCEDURAL = 3
UMS_DEFAULT_FETCH_LIMIT_LINKS = 5
UMS_DEFAULT_FETCH_LIMIT_GOAL_DEPTH = 3  # Max parent goals to fetch details for
UMS_DEFAULT_SHOW_LIMIT_WORKING_MEMORY = 10
UMS_DEFAULT_SHOW_LIMIT_GOAL_STACK = 5
UMS_PKG_DEFAULT_FETCH_RECENT_ACTIONS = 10
UMS_PKG_DEFAULT_FETCH_IMPORTANT_MEMORIES = 7
UMS_PKG_DEFAULT_FETCH_KEY_THOUGHTS = 7
UMS_PKG_DEFAULT_FETCH_PROACTIVE = 5
UMS_PKG_DEFAULT_FETCH_PROCEDURAL = 3
UMS_PKG_DEFAULT_FETCH_LINKS = 5
# Show limits are mainly for compression decisions within this tool, or if it were to do truncation.
UMS_PKG_DEFAULT_SHOW_LINKS_SUMMARY = 3

MIN_CONFIDENCE_SEMANTIC: float = 0.90  #  ↔ promote_memory_level default

# ======================================================
# Batch Operation Support for Multi-Tool Agent Calls
# ======================================================

# Context variable to track batch operations
_batch_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("ums_batch_context", default=None)


class UMSBatchContext:
    """Context manager for batching multiple UMS tool calls within a single agent turn."""

    def __init__(self, batch_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.tool_calls: List[Dict[str, Any]] = []
        self.shared_connections: Dict[str, Any] = {}

    async def __aenter__(self):
        batch_data = {
            "batch_id": self.batch_id,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "tool_calls": self.tool_calls,
            "shared_connections": self.shared_connections,
        }
        _batch_context.set(batch_data)
        logger.debug(f"Started UMS batch context: {self.batch_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        batch_data = _batch_context.get()
        if batch_data:
            elapsed = time.time() - batch_data["start_time"]
            logger.info(
                f"Completed UMS batch {self.batch_id}: {len(batch_data['tool_calls'])} tools in {elapsed:.3f}s",
                emoji_key="package",
            )
        _batch_context.set(None)

    def record_tool_call(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]):
        """Record a tool call within this batch."""
        self.tool_calls.append(
            {"tool_name": tool_name, "params": params, "result": result, "timestamp": time.time()}
        )


def get_current_batch() -> Optional[Dict[str, Any]]:
    """Get the current batch context if any."""
    return _batch_context.get()


def is_in_batch() -> bool:
    """Check if we're currently in a batch context."""
    return _batch_context.get() is not None


# ======================================================
# Enums (Combined & Standardized)
# ======================================================


# --- Workflow & Action Status ---
class WorkflowStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ActionStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# --- Content Types ---
class ActionType(str, Enum):
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    RESEARCH = "research"
    DECISION = "decision"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SUMMARY = "summary"
    CONSOLIDATION = "consolidation"
    MEMORY_OPERATION = "memory_operation"
    PARALLEL_TOOLS = "parallel_tools"


class ArtifactType(str, Enum):
    FILE = "file"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    CODE = "code"
    DATA = "data"
    JSON = "json"
    URL = "url"


class ThoughtType(str, Enum):
    GOAL = "goal"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    EVIDENCE = "evidence"
    CONSTRAINT = "constraint"
    PLAN = "plan"
    DECISION = "decision"
    REFLECTION = "reflection"
    CRITIQUE = "critique"
    SUMMARY = "summary"
    USER_GUIDANCE = "user_guidance"
    INSIGHT = "insight"
    REASONING = "reasoning"
    ANALYSIS = "analysis"


# --- Memory System Types ---
class MemoryLevel(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryType(str, Enum):
    """Content type classifications for memories. Combines concepts."""

    OBSERVATION = "observation"  # Raw data or sensory input (like text)
    ACTION_LOG = "action_log"  # Record of an agent action
    TOOL_OUTPUT = "tool_output"  # Result from a tool
    ARTIFACT_CREATION = "artifact_creation"  # Record of artifact generation
    REASONING_STEP = "reasoning_step"  # Corresponds to a thought
    FACT = "fact"  # Verifiable piece of information
    INSIGHT = "insight"  # Derived understanding or pattern
    PLAN = "plan"  # Future intention or strategy
    QUESTION = "question"  # Posed question or uncertainty
    SUMMARY = "summary"  # Condensed information
    REFLECTION = "reflection"  # Meta-cognitive analysis (distinct from thought type)
    SKILL = "skill"  # Learned capability (like procedural)
    PROCEDURE = "procedure"  # Step-by-step method
    PATTERN = "pattern"  # Recognized recurring structure
    CODE = "code"  # Code snippet
    JSON = "json"  # Structured JSON data
    URL = "url"  # A web URL
    USER_INPUT = "user_input"
    TEXT = "text"  # Generic text block (fallback)
    WARNING = "warning"
    CONTRADICTION_ANALYSIS = "contradiction_analysis"
    VALIDATION_FAILURE = "validation_failure"
    CORRECTION = "correction"
    TOOL_EFFECTIVENESS = "tool_effectiveness"
    CONTEXT_INITIALIZATION = "context_initialization"
    STATE_SNAPSHOT = "state_snapshot"
    CHECKPOINT = "checkpoint"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    # Retain IMAGE? Needs blob storage/linking capability. Deferred.


class LinkType(str, Enum):
    """Types of associations between memories (from cognitive_memory)."""

    RELATED = "related"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TASK = "task"
    REFERENCES = "references"
    ELABORATES = "elaborates"
    QUESTION_OF = "question_of"
    CONSEQUENCE_OF = "consequence_of"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ABANDONED = "abandoned"


# ======================================================
# Database Schema (Defined as Individual Statements)
# ======================================================
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS ums_internal_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
    );""",
    """CREATE TABLE IF NOT EXISTS workflows (
        workflow_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT, goal TEXT, status TEXT NOT NULL,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
        parent_workflow_id TEXT, metadata TEXT, last_active INTEGER,
        idempotency_key TEXT UNIQUE NULL
    );""",
    """CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, parent_action_id TEXT, action_type TEXT NOT NULL,
        title TEXT, reasoning TEXT, tool_name TEXT, tool_args TEXT, tool_result TEXT, status TEXT NOT NULL,
        started_at INTEGER NOT NULL, completed_at INTEGER, sequence_number INTEGER,
        idempotency_key TEXT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (parent_action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, artifact_type TEXT NOT NULL,
        name TEXT NOT NULL, description TEXT, path TEXT, content TEXT, metadata TEXT,
        created_at INTEGER NOT NULL, is_output BOOLEAN DEFAULT FALSE,
        idempotency_key TEXT NULL,
        file_path TEXT, file_size INTEGER DEFAULT 0, content_hash TEXT, thumbnail_path TEXT,
        version INTEGER DEFAULT 1, parent_artifact_id TEXT, tags TEXT,
        importance REAL DEFAULT 0.5, access_count INTEGER DEFAULT 0, updated_at INTEGER DEFAULT 0, last_accessed_at INTEGER,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        FOREIGN KEY (parent_artifact_id) REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS artifact_relationships (
        relationship_id TEXT PRIMARY KEY,
        source_artifact_id TEXT NOT NULL,
        target_artifact_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        strength REAL DEFAULT 1.0,
        created_at INTEGER DEFAULT (unixepoch()),
        FOREIGN KEY (source_artifact_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE,
        FOREIGN KEY (target_artifact_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE,
        UNIQUE(source_artifact_id, target_artifact_id, relationship_type)
    );""",
    """CREATE TABLE IF NOT EXISTS thought_chains (
        thought_chain_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, title TEXT NOT NULL, created_at INTEGER NOT NULL,
        idempotency_key TEXT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        id          TEXT PRIMARY KEY,
        memory_id   TEXT UNIQUE REFERENCES memories(memory_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
        model       TEXT    NOT NULL,
        embedding   BLOB    NOT NULL,
        dimension   INTEGER NOT NULL,
        created_at  INTEGER NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS thoughts (
        thought_id      TEXT PRIMARY KEY,
        thought_chain_id TEXT NOT NULL REFERENCES thought_chains(thought_chain_id) ON DELETE CASCADE,
        parent_thought_id TEXT REFERENCES thoughts(thought_id) ON DELETE SET NULL,
        thought_type    TEXT NOT NULL,
        content         TEXT NOT NULL,
        sequence_number INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        relevant_action_id   TEXT REFERENCES actions(action_id) ON DELETE SET NULL,
        relevant_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
        relevant_memory_id   TEXT REFERENCES memories(memory_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
        idempotency_key TEXT NULL,
        UNIQUE(thought_chain_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS memories (
        memory_id   TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        content     TEXT    NOT NULL,
        memory_level TEXT   NOT NULL,
        memory_type TEXT    NOT NULL,
        importance  REAL    DEFAULT 5.0,
        confidence  REAL    DEFAULT 1.0,
        description TEXT,
        reasoning   TEXT,
        source      TEXT,
        context     TEXT,
        tags        TEXT,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL,
        last_accessed INTEGER,
        access_count INTEGER DEFAULT 0,
        ttl         INTEGER DEFAULT 0,
        embedding_id TEXT REFERENCES embeddings(id) ON DELETE SET NULL,
        action_id   TEXT REFERENCES actions(action_id) ON DELETE SET NULL,
        thought_id  TEXT REFERENCES thoughts(thought_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
        artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
        idempotency_key TEXT NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS goals (
        goal_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        parent_goal_id TEXT REFERENCES goals(goal_id) ON DELETE SET NULL,
        title TEXT, description TEXT NOT NULL, status TEXT NOT NULL,
        priority INTEGER DEFAULT 3, reasoning TEXT, acceptance_criteria TEXT, metadata TEXT,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
        sequence_number INTEGER,
        idempotency_key TEXT NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS memory_links (
        link_id TEXT PRIMARY KEY, source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL,
        link_type TEXT NOT NULL, strength REAL DEFAULT 1.0, description TEXT, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        FOREIGN KEY (target_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        UNIQUE(source_memory_id, target_memory_id, link_type)
    );""",
    """CREATE TABLE IF NOT EXISTS tags (
        tag_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, description TEXT, category TEXT, created_at INTEGER NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS workflow_tags (
        workflow_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (workflow_id, tag_id),
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS action_tags (
        action_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (action_id, tag_id),
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS artifact_tags (
        artifact_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (artifact_id, tag_id),
        FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS dependencies (
        dependency_id INTEGER PRIMARY KEY AUTOINCREMENT, source_action_id TEXT NOT NULL, target_action_id TEXT NOT NULL,
        dependency_type TEXT NOT NULL, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        FOREIGN KEY (target_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        UNIQUE(source_action_id, target_action_id, dependency_type)
    );""",
    """CREATE TABLE IF NOT EXISTS cognitive_states (
        state_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        title TEXT NOT NULL,
        working_memory TEXT, focus_areas TEXT, context_actions TEXT, current_goals TEXT,
        created_at INTEGER NOT NULL, is_latest BOOLEAN NOT NULL,
        focal_memory_id TEXT REFERENCES memories(memory_id) ON DELETE SET NULL,
        last_active INTEGER,                        
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS cognitive_timeline_states (
        state_id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        state_type TEXT NOT NULL,
        state_data TEXT NOT NULL,
        workflow_id TEXT,
        description TEXT,
        created_at REAL DEFAULT (unixepoch()),
        FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS reflections (
        reflection_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
        reflection_type TEXT NOT NULL, created_at INTEGER NOT NULL, referenced_memories TEXT,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS memory_operations (
        operation_log_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, memory_id TEXT, action_id TEXT,
        operation TEXT NOT NULL, operation_data TEXT, timestamp INTEGER NOT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE SET NULL,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL
    );""",
    "CREATE INDEX IF NOT EXISTS idx_workflows_idempotency_key ON workflows(idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_actions_idempotency ON actions(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_artifacts_idempotency ON artifacts(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_memories_idempotency ON memories(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_goals_idempotency ON goals(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_thought_chains_idempotency ON thought_chains(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_thoughts_idempotency ON thoughts(thought_chain_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_parent ON workflows(parent_workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_last_active ON workflows(last_active DESC);",
    "CREATE INDEX IF NOT EXISTS idx_actions_workflow_id ON actions(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_parent ON actions(parent_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_sequence ON actions(workflow_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_workflow_id ON artifacts(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_action_id ON artifacts(action_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_tags ON artifacts(tags);",
    "CREATE INDEX IF NOT EXISTS idx_artifact_relationships_source ON artifact_relationships(source_artifact_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifact_relationships_target ON artifact_relationships(target_artifact_id);",
    "CREATE INDEX IF NOT EXISTS idx_thought_chains_workflow ON thought_chains(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_chain ON thoughts(thought_chain_id);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_sequence ON thoughts(thought_chain_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_type ON thoughts(thought_type);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_relevant_memory ON thoughts(relevant_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_workflow ON memories(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(memory_level);",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_action_id ON memories(action_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_thought_id ON memories(thought_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_artifact_id ON memories(artifact_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_type ON memory_links(link_type);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_memory_id ON embeddings(memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_dimension ON embeddings(dimension);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_workflow ON cognitive_states(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_latest ON cognitive_states(workflow_id, is_latest);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_timeline_states_timestamp ON cognitive_timeline_states(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_timeline_states_type ON cognitive_timeline_states(state_type);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_timeline_states_workflow ON cognitive_timeline_states(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_reflections_workflow ON reflections(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_workflow ON memory_operations(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_memory ON memory_operations(memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON memory_operations(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);",
    "CREATE INDEX IF NOT EXISTS idx_workflow_tags ON workflow_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_action_tags ON action_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifact_tags ON artifact_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_last_active ON cognitive_states(last_active DESC);",
    "CREATE INDEX IF NOT EXISTS idx_goals_workflow_id ON goals(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_goals_parent_goal_id ON goals(parent_goal_id);",
    "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);",
    "CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority);",
    "CREATE INDEX IF NOT EXISTS idx_goals_sequence_number ON goals(parent_goal_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_ums_internal_metadata_key ON ums_internal_metadata(key);",
    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        content, description, reasoning, tags,
        workflow_id UNINDEXED, memory_id UNINDEXED,
        content='memories', content_rowid='rowid', tokenize='porter unicode61'
    );""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_insert AFTER INSERT ON memories BEGIN
        INSERT INTO memory_fts(rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES (new.rowid, new.content, new.description, new.reasoning, new.tags, new.workflow_id, new.memory_id);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_delete AFTER DELETE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES ('delete', old.rowid, old.content, old.description, old.reasoning, old.tags, old.workflow_id, old.memory_id);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_update_sync AFTER UPDATE ON memories BEGIN
        INSERT OR REPLACE INTO memory_fts(rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES (new.rowid, new.content, new.description, new.reasoning, new.tags, new.workflow_id, new.memory_id);
    END;""",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_sequence_unique  "
    "ON actions (workflow_id,  sequence_number);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_thoughts_sequence_unique "
    "ON thoughts(thought_chain_id, sequence_number);",
    # Fixed: Replace the incorrect goals constraint with proper partial constraints
    # Root goals: unique sequence numbers within workflow where parent_goal_id IS NULL
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_goals_root_sequence_unique "
    "ON goals (workflow_id, sequence_number) WHERE parent_goal_id IS NULL;",
    # Child goals: unique sequence numbers within parent goal where parent_goal_id IS NOT NULL
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_goals_child_sequence_unique "
    "ON goals (parent_goal_id, sequence_number) WHERE parent_goal_id IS NOT NULL;",
]


def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    # Ensure slicing doesn't go out of bounds if string is shorter than length
    return s[: min(length, len(s))]


def _validate_uuid_format(uuid_str: str, param_name: str = "id") -> str:
    """
    Validate that a string is a proper UUID format.

    For backwards compatibility, also accepts truncated UUIDs (8 hex chars)
    which will be resolved to full UUIDs via database lookup.

    Args:
        uuid_str: The string to validate
        param_name: Parameter name for error messages

    Returns:
        The validated UUID string (may be truncated for later resolution)

    Raises:
        ToolInputError: If the string is not a valid UUID or truncated UUID format
    """
    if not uuid_str:
        raise ToolInputError(f"{param_name} is required.", param_name=param_name)

    # Check if it looks like a truncated ID (8 chars, hex)
    if len(uuid_str) == 8 and all(c in "0123456789abcdefABCDEF" for c in uuid_str):
        # Return as-is, will be resolved in the calling function
        return uuid_str.lower()

    try:
        # Validate full UUID format
        uuid.UUID(uuid_str)
        return uuid_str
    except ValueError as e:
        raise ToolInputError(
            f"Invalid {param_name} format '{uuid_str}'. Must be a valid UUID or 8-character hex prefix.",
            param_name=param_name,
        ) from e


async def _resolve_truncated_id(
    conn: aiosqlite.Connection,
    truncated_id: str,
    table: str,
    id_column: str,
    param_name: str = "id",
) -> str:
    """
    Resolve a truncated UUID to a full UUID by searching the database.

    Args:
        conn: Database connection
        truncated_id: 8-character hex prefix
        table: Table name to search
        id_column: Column name containing UUIDs
        param_name: Parameter name for error messages

    Returns:
        Full UUID string

    Raises:
        ToolInputError: If no matches or multiple matches found
    """
    if len(truncated_id) != 8:
        raise ToolInputError(
            f"Truncated {param_name} must be exactly 8 characters.", param_name=param_name
        )

    # Search for UUIDs starting with the truncated portion
    rows = await conn.execute_fetchall(
        f"SELECT {id_column} FROM {table} WHERE {id_column} LIKE ? LIMIT 10", (f"{truncated_id}%",)
    )

    if not rows:
        raise ToolInputError(
            f"No {table[:-1]} found with {param_name} starting with '{truncated_id}'.",
            param_name=param_name,
        )

    if len(rows) > 1:
        matches = [row[0] for row in rows]
        raise ToolInputError(
            f"Ambiguous {param_name} '{truncated_id}' matches multiple {table}: {', '.join(_fmt_id(m) for m in matches[:5])}{'...' if len(matches) > 5 else ''}. "
            f"Please use full UUID.",
            param_name=param_name,
        )

    full_uuid = rows[0][0]
    logger.debug(f"Resolved truncated {param_name} '{truncated_id}' to full UUID '{full_uuid}'")
    return full_uuid


async def _validate_and_resolve_id(
    db_conn: "DBConnection", id_value: str, table: str, id_column: str, param_name: str = "id"
) -> str:
    """
    Convenience function that validates and resolves IDs (full UUID or truncated).

    Args:
        db_conn: Database connection instance
        id_value: ID to validate and resolve
        table: Table name to search for truncated IDs
        id_column: Column name containing UUIDs
        param_name: Parameter name for error messages

    Returns:
        Full UUID string

    Raises:
        ToolInputError: If validation fails or truncated ID cannot be resolved
    """
    # Validate format first
    validated_id = _validate_uuid_format(id_value, param_name)

    # If it's a truncated ID, resolve it to a full UUID
    if len(validated_id) == 8:
        async with db_conn.transaction(readonly=True) as conn:
            validated_id = await _resolve_truncated_id(
                conn, validated_id, table, id_column, param_name
            )

    return validated_id


_MUTATION_SQL = re.compile(r"^\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER)\b", re.I)


# ======================================================
# Secure Path Validation & File Access Control
# ======================================================


def validate_and_secure_db_path(db_path: str) -> str:
    """
    Validate and secure a database path with lightweight checks.

    This function is optimized for frequent calls during database connections.
    It performs basic security validation without expensive filesystem operations.

    Args:
        db_path: Original database path from config

    Returns:
        Safe database path
    """
    import os

    try:
        path_obj = Path(db_path).resolve()

        # Quick check for obviously unsafe paths
        path_str = str(path_obj)
        unsafe_prefixes = [
            "/bin/",
            "/sbin/",
            "/usr/bin/",
            "/usr/sbin/",
            "/etc/",
            "/var/log/",
            "/var/run/",
            "/var/spool/",
            "/sys/",
            "/proc/",
            "/dev/",
            "/root/",
        ]

        for unsafe in unsafe_prefixes:
            if path_str.startswith(unsafe) or path_str == unsafe.rstrip("/"):
                logger.debug(
                    f"Database path '{db_path}' is in restricted directory, using fallback"
                )
                return _get_safe_fallback_path(db_path)

        # Check if we can likely create the path (quick check)
        parent = path_obj.parent

        # If parent exists, check if it's writable
        if parent.exists():
            if os.access(str(parent), os.W_OK):
                return str(path_obj)
            else:
                logger.debug(f"No write permission to '{parent}', using fallback")
                return _get_safe_fallback_path(db_path)

        # If parent doesn't exist, check if we can create it
        # Find the first existing parent
        check_parent = parent
        while not check_parent.exists() and check_parent != check_parent.parent:
            check_parent = check_parent.parent

        if check_parent.exists() and os.access(str(check_parent), os.W_OK):
            return str(path_obj)
        else:
            logger.debug(f"Cannot create database directory '{parent}', using fallback")
            return _get_safe_fallback_path(db_path)

    except Exception as e:
        logger.debug(f"Path validation failed for '{db_path}': {e}, using fallback")
        return _get_safe_fallback_path(db_path)


def _get_safe_fallback_path(original_path: str) -> str:
    """Get a safe fallback path for database files (lightweight)."""
    import tempfile

    db_filename = Path(original_path).name

    # Try to check filesystem allowed directories if available (but don't fail if not)
    try:
        from ultimate_mcp_server.tools.filesystem import get_allowed_directories

        allowed_dirs = get_allowed_directories()

        if allowed_dirs:
            for allowed_dir in allowed_dirs:
                try:
                    candidate = Path(allowed_dir) / "ultimate_mcp_server" / db_filename
                    # Quick feasibility check
                    if _can_create_path_quickly(candidate):
                        return str(candidate)
                except Exception:
                    continue
    except (ImportError, Exception):
        pass

    # Standard fallbacks
    try:
        user_fallback = Path.home() / ".ultimate_mcp_server" / db_filename
        if _can_create_path_quickly(user_fallback):
            return str(user_fallback)
    except Exception:
        pass

    # Ultimate fallback to temp
    temp_fallback = Path(tempfile.gettempdir()) / "ultimate_mcp_server" / db_filename
    return str(temp_fallback)


def _can_create_path_quickly(path: Path) -> bool:
    """Quick check if a path can likely be created (no expensive operations)."""
    import os

    try:
        if path.exists():
            return True

        # Find first existing parent
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent

        return parent.exists() and os.access(str(parent), os.W_OK)
    except Exception:
        return False


def safe_mkdir(path: Path) -> bool:
    """
    Safely create a directory with proper error handling.

    This is an internal utility function that should be fast and quiet.
    For user-facing directory operations, use filesystem tools directly.

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        logger.error(
            f"Permission denied creating directory: {path}. "
            f"Please check directory permissions or configure a different path.",
            emoji_key="lock",
        )
        return False
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}", emoji_key="error")
        return False


class DBConnection:
    __slots__ = ("db_path", "_managed_conn")
    _schema_lock = asyncio.Lock()
    _schema_ready: Set[str] = set()
    _write: Dict[str, asyncio.Lock] = {}  # db_path → asyncio.Lock
    _write_guard: threading.Lock = threading.Lock()
    _MAX_TX = 6
    _MAX_COMMIT = 4
    _BASE = 0.05
    _CAP = 5.0

    def __init__(self, db_path: str = agent_memory_config.db_path):
        # Validate and secure the database path
        import tempfile

        secure_db_path = validate_and_secure_db_path(db_path)
        self.db_path = str(Path(secure_db_path).resolve())

        # Safely create parent directory
        parent_dir = Path(self.db_path).parent
        if not safe_mkdir(parent_dir):
            # If we can't create the directory, fall back to a temp location
            fallback_path = (
                Path(tempfile.gettempdir()) / "ultimate_mcp_server" / Path(self.db_path).name
            )
            logger.warning(
                f"Failed to create database directory. Using fallback: {fallback_path}",
                emoji_key="warning",
            )
            safe_mkdir(fallback_path.parent)  # Create temp directory
            self.db_path = str(fallback_path)

        self._managed_conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------ #
    # low-level helpers                                                  #
    # ------------------------------------------------------------------ #
    async def _pause(self, n: int):
        await asyncio.sleep(min(self._CAP, self._BASE * 2**n) * (0.5 + random.random() / 2))

    @classmethod
    def _w(cls, path: str) -> asyncio.Lock:
        """
        Return a process-wide asyncio.Lock scoped to *path*.

        The registry update is protected by a plain threading.Lock so
        only one asyncio.Lock can ever be created for a given database
        file, even when several event loops in different threads hit
        this code concurrently.
        """
        try:  # common fast path
            return cls._write[path]
        except KeyError:  # first access for this path
            with cls._write_guard:  # threadsafe critical section
                return cls._write.setdefault(path, asyncio.Lock())

    async def _cfg(self, conn: aiosqlite.Connection, *, readonly: bool = False) -> None:
        """
        Apply connection-scoped PRAGMAs.

        Rationale
        ---------
        • **Readers** use `journal_mode=OFF` + shared locking – they never write a
          rollback journal and therefore never block writers.

        • **Writers** use `journal_mode=TRUNCATE` (single persistent journal file)
          together with `locking_mode=EXCLUSIVE`.
          Acquiring the exclusive lock up-front removes the
          *commit-time* DELETE-mode race that produced frequent
          `OperationalError: database is locked`.  Durability remains identical to
          DELETE when `synchronous=NORMAL`.
        """
        pragmas: list[str] = [
            "PRAGMA foreign_keys=ON;",
            "PRAGMA busy_timeout=60000;",  # 60 s back-off already mirrored in Python
            "PRAGMA temp_store=MEMORY;",
        ]

        if readonly:
            pragmas[:0] = [
                "PRAGMA journal_mode=OFF;",
                "PRAGMA locking_mode=SHARED;",  # the default, but make it explicit
                "PRAGMA synchronous=OFF;",  # safe because we only read
            ]
        else:
            pragmas[:0] = [
                "PRAGMA journal_mode=TRUNCATE;",  # keeps a pre-allocated journal file
                "PRAGMA locking_mode=EXCLUSIVE;",  # grab the write lock immediately
                "PRAGMA synchronous=NORMAL;",
            ]

        await conn.executescript("".join(pragmas))

        # --- UDFs (unchanged) ----------------------------------------------------
        await conn.create_function("json_contains", 2, _json_contains, deterministic=True)
        await conn.create_function("json_contains_any", 2, _json_contains_any, deterministic=True)
        await conn.create_function("json_contains_all", 2, _json_contains_all, deterministic=True)
        await conn.create_function(
            "compute_memory_relevance", 5, _compute_memory_relevance, deterministic=True
        )

    async def _bootstrap(self):
        async with self._schema_lock:
            if self.db_path in self._schema_ready:
                return
            conn: Optional[aiosqlite.Connection] = None
            try:
                conn = await aiosqlite.connect(
                    self.db_path, timeout=agent_memory_config.connection_timeout
                )
                conn.row_factory = aiosqlite.Row
                await self._cfg(conn)
                await conn.execute("BEGIN IMMEDIATE;")
                for stmt in SCHEMA_STATEMENTS:
                    await conn.execute(stmt)
                await conn.commit()
                self._schema_ready.add(self.db_path)
            except Exception as e:
                if conn and conn.in_transaction:
                    await conn.rollback()
                logger.critical("schema bootstrap failed", exc_info=True)
                raise ToolError(f"schema bootstrap failed: {e}") from e
            finally:
                if conn:
                    await conn.close()

    # ------------------------------------------------------------------ #
    # public async-context APIs                                          #
    # ------------------------------------------------------------------ #

    @contextlib.asynccontextmanager
    async def transaction(self, *, readonly: bool = False, mode: str | None = None):
        await self._bootstrap()

        uri_path = (
            f"file:{self.db_path}?mode=ro&cache=shared" if readonly else f"file:{self.db_path}"
        )
        lock_cm = contextlib.nullcontext() if readonly else self._w(self.db_path)

        for attempt in range(self._MAX_TX):
            async with lock_cm:
                conn: Optional[aiosqlite.Connection] = None
                try:
                    conn = await aiosqlite.connect(
                        uri_path,
                        uri=True,  # always a URI now
                        timeout=agent_memory_config.connection_timeout,
                        cached_statements=64,
                    )
                    conn.row_factory = aiosqlite.Row
                    await self._cfg(conn, readonly=readonly)

                    # ----- logging-only trace callback ---------------------------------
                    def _trace(sql: str) -> None:
                        if _MUTATION_SQL.match(sql):
                            logger.debug(f"DB TRACE: {sql.split(None, 1)[0]} …")

                    await conn.set_trace_callback(_trace)
                    # ------------------------------------------------------------------

                    await conn.execute(
                        "BEGIN DEFERRED;" if readonly else f"BEGIN {mode or 'IMMEDIATE'};"
                    )
                    baseline_changes = conn.total_changes  # snapshot *after* BEGIN

                    # --------------------- caller block -------------------------------
                    try:
                        yield conn
                    finally:
                        wrote = conn.total_changes != baseline_changes
                        logger.debug(
                            f"DB transaction finished. readonly={readonly} "
                            f"wrote={wrote} total_changes={conn.total_changes}"
                        )

                        if readonly:
                            # End the read snapshot; cheap and never blocks.
                            if conn.in_transaction:
                                await conn.commit()
                        else:
                            if wrote:
                                # ---------- commit with retry loop ----------
                                for c_attempt in range(self._MAX_COMMIT):
                                    try:
                                        await conn.commit()
                                        break
                                    except aiosqlite.OperationalError as e:
                                        if "database is locked" in str(e).lower():
                                            await self._pause(c_attempt)
                                            continue
                                        await conn.rollback()
                                        break
                            else:
                                # No writes → fast rollback
                                await conn.rollback()
                except aiosqlite.OperationalError as e:
                    if "database is locked" not in str(e).lower():
                        raise
                    await self._pause(attempt)
                    continue  # retry outer loop
                finally:
                    if conn:
                        # Safety net: roll back stale txn before closing.
                        if conn.in_transaction:
                            await conn.rollback()
                        await conn.set_trace_callback(None)
                        await conn.close()

                return  # successful run
        raise ToolError("Maximum SQLite transaction retries exceeded")

    async def __aenter__(self) -> aiosqlite.Connection:
        await self._bootstrap()
        for attempt in range(self._MAX_TX):
            try:
                self._managed_conn = await aiosqlite.connect(
                    self.db_path,
                    timeout=agent_memory_config.connection_timeout,
                    cached_statements=64,
                )
                self._managed_conn.row_factory = aiosqlite.Row
                await self._cfg(self._managed_conn, readonly=False)
                return self._managed_conn
            except aiosqlite.OperationalError as e:
                if "database is locked" not in str(e).lower():
                    raise
                await self._pause(attempt)
        raise ToolError("Maximum SQLite open retries exceeded")

    async def __aexit__(self, *_):
        if self._managed_conn:
            if self._managed_conn.in_transaction:
                await self._managed_conn.rollback()
            await self._managed_conn.close()
            self._managed_conn = None


# ------------------------------------------------------------------ #
# JSON UDF helpers + relevance                                       #
# ------------------------------------------------------------------ #
def _safe_json_loads(t: str | None) -> Any | None:
    if not t:
        return None
    try:
        return json.loads(t)
    except (TypeError, json.JSONDecodeError):
        return None


def _json_contains(j: str | None, v: Any) -> bool:
    d = _safe_json_loads(j)
    return isinstance(d, list) and v in d


def _json_flatten_list(val: Any) -> Optional[list]:
    """Return a *flat* list if `val` is (JSON-encoded) list-like, else None.
    • Recurses one level ([[1,2],[3]] ➜ [1,2,3]).
    • Filters out None / NaN values.
    """
    data = _safe_json_loads(val)
    if not isinstance(data, list):
        return None
    flat: list = []
    for x in data:
        if x is None or (isinstance(x, float) and x != x):  # filter None / NaN
            continue
        if isinstance(x, list):
            flat.extend(i for i in x if i is not None)
        else:
            flat.append(x)
    return flat


def _json_contains_any(j: str | None, vs: str | None) -> bool:
    """True if ANY element in `vs` appears in `j` (both JSON arrays)."""
    d, v = _json_flatten_list(j), _json_flatten_list(vs)
    if d is None or v is None:
        return False
    ds = set(d)
    return any(item in ds for item in v)


def _json_contains_all(j: str | None, vs: str | None) -> bool:
    """True if *all* elements in `vs` appear in `j` (both JSON arrays)."""
    d, v = _json_flatten_list(j), _json_flatten_list(vs)
    if d is None or v is None:
        return False
    return set(v).issubset(set(d))


def _compute_memory_relevance(
    imp: float, conf: float, created: int, cnt: int, last: int | None
) -> float:
    """Calculate memory relevance score (0-10) using consistent time units and smooth decay."""
    now = time.time()
    created = created or now
    last = last or created

    # Use consistent time units (all in seconds)
    age_seconds = now - created
    recency_seconds = now - last

    # Smooth exponential decay instead of cliff-drop
    age_hours = age_seconds / 3600
    decay_factor = np.exp(-MEMORY_DECAY_RATE * age_hours)  # Exponential decay

    # Recency boost (higher for recently accessed memories)
    recency_days = recency_seconds / 86400
    recency_factor = 1 / (1 + recency_days)

    # Usage boost with logarithmic scaling (diminishing returns)
    usage_factor = 1 + np.log(1 + cnt) / 10  # Smooth scaling

    # Combine all factors
    score = imp * decay_factor * usage_factor * conf * recency_factor

    return max(0, min(score, 10))


# ======================================================
# Utilities
# ======================================================


async def _ensure_udfs_registered(conn: aiosqlite.Connection) -> None:
    """Re-register UDFs if they're missing on this connection."""
    try:
        # Quick test if compute_memory_relevance is available
        await conn.execute_fetchone(
            "SELECT compute_memory_relevance(5.0, 1.0, ?, 0, NULL) AS test", (int(time.time()),)
        )
    except aiosqlite.OperationalError as e:
        if "no such function" in str(e).lower() or "no such column" in str(e).lower():
            # Re-register all UDFs by calling the existing _cfg method
            db_conn = DBConnection()  # Create a temporary instance
            await db_conn._cfg(conn, readonly=False)
        else:
            raise


def to_iso_z(ts: float) -> str:  # helper ⇒  ISO‑8601 with trailing “Z”
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def safe_format_timestamp(ts_value):
    """Safely formats a timestamp value (int, float, or ISO string) to ISO Z format."""
    if isinstance(ts_value, (int, float)):
        try:
            # Reasonable timestamp range validation (1900-2100)
            MIN_TIMESTAMP = -2208988800  # January 1, 1900 00:00:00 UTC
            MAX_TIMESTAMP = 4102444800  # January 1, 2100 00:00:00 UTC

            if not (MIN_TIMESTAMP <= ts_value <= MAX_TIMESTAMP):
                logger.warning(
                    f"Timestamp {ts_value} outside reasonable range (1900-2100), returning as string."
                )
                return str(ts_value)
            return to_iso_z(ts_value)
        except (OverflowError, OSError, ValueError, TypeError) as e:
            logger.warning(f"Failed to convert numeric timestamp {ts_value} to ISO: {e}")
            return str(ts_value)  # Fallback to string representation of number
    elif isinstance(ts_value, str):
        # Try to parse and reformat to ensure consistency, but return original if parsing fails
        try:
            # Attempt parsing, assuming it might already be close to ISO
            dt_obj = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            # Reformat to our standard Z format
            return to_iso_z(dt_obj.timestamp())
        except ValueError:
            # If parsing fails, return the original string but log a warning
            logger.debug(
                f"Timestamp value '{ts_value}' is a string but not valid ISO format. Returning as is."
            )
            return ts_value
    elif ts_value is None:
        return None
    else:
        logger.warning(
            f"Unexpected timestamp type {type(ts_value)}, value: {ts_value}. Returning string representation."
        )
        return str(ts_value)


@with_error_handling
async def summarize_text(
    text_to_summarize: str,
    *,
    target_tokens: int = 500,
    prompt_template: str | None = None,
    provider: str = "openai",
    model: str | None = None,
    workflow_id: str | None = None,
    record_summary: bool = False,
    context_type: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Summarise *text_to_summarize* with an LLM and (optionally) store the result as a memory.

    Returned keys
    -------------
    summary, original_length, summary_length,
    stored_memory_id, success, processing_time
    """
    t0 = time.time()

    # ───────── basic validation ─────────
    if not text_to_summarize:
        raise ToolInputError("text_to_summarize cannot be empty", param_name="text_to_summarize")
    if record_summary and not workflow_id:
        raise ToolInputError(
            "workflow_id required when record_summary=True", param_name="workflow_id"
        )

    target_tokens = max(50, min(2_000, target_tokens))

    # ───────── provider / model resolution (FIX #13 & #14) ─────────
    cfg = get_config()
    provider_key: str = provider or cfg.default_provider or LLMGatewayProvider.OPENAI.value

    default_models: dict[str, str] = {
        LLMGatewayProvider.OPENAI.value: "gpt-4.1-mini",
        LLMGatewayProvider.ANTHROPIC.value: "claude-3-5-haiku-20241022",
    }

    # first attempt to honour explicit `model` or hard-coded defaults
    model_name: str | None = model or default_models.get(provider_key)

    # initialise provider once (avoid double await + latency)
    prov = await get_provider(provider_key)
    if prov is None:
        raise ToolError(
            f"LLM provider '{provider_key}' unavailable. "
            f"Available providers: {', '.join(p.value for p in LLMGatewayProvider)}"
        )

    # fall back to provider’s own default model if we still have none
    if model_name is None:
        model_name = prov.get_default_model()

    # ───────── default prompt (rich version) ─────────
    if prompt_template is None:
        prompt_template = (
            "You are an expert technical writer and editor.\n"
            "Your task is to produce a **concise, accurate, well-structured summary** "
            "of the text provided below.\n\n"
            "**Requirements**\n"
            "1. Length ≈ {target_tokens} tokens (±10 %).\n"
            "2. Preserve all critical facts, numbers, names, and causal relationships.\n"
            "3. Omit anecdotes, filler, or rhetorical questions unless essential to meaning.\n"
            "4. Write in clear, neutral, third-person prose; bullet lists are welcome where helpful.\n"
            "5. Do **not** add opinions or external knowledge.\n\n"
            "---\n"
            "### ORIGINAL TEXT\n"
            "{text_to_summarize}\n"
            "---\n"
            "### SUMMARY\n"
        )

    # ───────── LLM invocation ─────────
    try:
        prompt = prompt_template.format(
            text_to_summarize=text_to_summarize,
            target_tokens=target_tokens,
        )
        result = await prov.generate_completion(
            prompt=prompt,
            model=model_name,
            max_tokens=target_tokens + 100,
            temperature=0.3,
        )
        summary = result.text.strip()
        if not summary:
            raise ToolError("LLM returned empty summary.")
    except Exception as exc:
        logger.error(f"LLM summarisation failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to generate summary: {exc}") from exc

    stored_memory_id: str | None = None

    # ───────── optional persistence ─────────
    if record_summary:
        db = DBConnection(db_path)
        async with db.transaction() as conn:
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            now = int(time.time())
            stored_memory_id = MemoryUtils.generate_id()
            tags = {
                "summary",
                "automated",
                "text_summary",
                MemoryLevel.SEMANTIC.value,
                MemoryType.SUMMARY.value,
            }
            if context_type:
                tags.add(context_type)

            await conn.execute(
                """
                INSERT INTO memories (
                    memory_id, workflow_id, content,
                    memory_level, memory_type,
                    importance, confidence,
                    description, source, tags,
                    created_at, updated_at, access_count
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    stored_memory_id,
                    workflow_id,
                    summary,
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.SUMMARY.value,
                    6.0,
                    0.85,
                    f"Summary ({context_type or 'ad-hoc'}) of {len(text_to_summarize)}-character text",
                    "summarize_text_tool",
                    json.dumps(sorted(tags)),
                    now,
                    now,
                    0,
                ),
            )

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create_summary_memory",
                stored_memory_id,
                None,
                {
                    "original_length": len(text_to_summarize),
                    "summary_length": len(summary),
                    "context_type": context_type,
                },
            )

    elapsed = time.time() - t0
    logger.info(
        f"Summarised {len(text_to_summarize)} chars → {len(summary)} chars "
        f"({'stored' if stored_memory_id else 'not stored'})",
        emoji_key="scissors",
        time=elapsed,
    )

    return {
        "success": True,
        "data": {
            "summary": summary,
            "original_length": len(text_to_summarize),
            "summary_length": len(summary),
            "stored_memory_id": stored_memory_id,
        },
        "processing_time": elapsed,
    }


async def delete_expired_memories(
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Purge memories whose *ttl* has elapsed and write an operation-log entry
    for each affected workflow.

    Returns
    -------
    {
        deleted_count      : int,
        workflows_affected : list[str],
        success            : True,
        processing_time    : float   # seconds
    }
    """
    t0 = time.time()
    db = DBConnection(db_path)
    now_ts = int(time.time())

    deleted_ids: list[str] = []
    wf_affected: set[str] = set()

    async with db.transaction(mode="IMMEDIATE") as conn:
        rows = await conn.execute_fetchall(
            """
            SELECT memory_id, workflow_id
            FROM   memories
            WHERE  ttl > 0
              AND  created_at + ttl < ?
            """,
            (now_ts,),
        )

        if not rows:
            return {
                "success": True,
                "data": {
                    "deleted_count": 0,
                    "workflows_affected": [],
                },
                "processing_time": time.time() - t0,
            }

        deleted_ids = [r["memory_id"] for r in rows]
        wf_affected = {r["workflow_id"] for r in rows}

        # batch delete (SQLITE_MAX_VARIABLE_NUMBER ≈ 999)
        BATCH = 500
        for i in range(0, len(deleted_ids), BATCH):
            batch = deleted_ids[i : i + BATCH]
            ph = ",".join("?" * len(batch))
            await conn.execute(f"DELETE FROM memories WHERE memory_id IN ({ph})", batch)

        # per-workflow operation log
        for wf in wf_affected:
            expired_here = sum(r["workflow_id"] == wf for r in rows)
            await MemoryUtils._log_memory_operation(
                conn,
                wf,
                "expire_batch",
                None,  # memory_id
                None,  # action_id
                {
                    "expired_count_in_workflow": expired_here,
                    "total_expired_this_run": len(deleted_ids),
                },
            )
    # ────────────── transaction committed ──────────────

    dt = time.time() - t0
    logger.success(
        f"Expired-memory sweep removed {len(deleted_ids)} rows "
        f"across {len(wf_affected)} workflows.",
        emoji_key="wastebasket",
        time=dt,
    )
    return {
        "success": True,
        "data": {
            "deleted_count": len(deleted_ids),
            "workflows_affected": sorted(wf_affected),
        },
        "processing_time": dt,
    }


async def compute_memory_statistics(
    workflow_id: Optional[str] = None,  # None → global stats
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Compute memory / link statistics.
    """
    start_time = time.time()
    stats: Dict[str, Any] = {"scope": workflow_id or "global"}
    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # --- base WHERE + params ----------------------------------------
            where_clause = "WHERE workflow_id = ?" if workflow_id else ""
            params: list[Any] = [workflow_id] if workflow_id else []

            # --- totals ------------------------------------------------------
            row = await conn.execute_fetchone(
                f"SELECT COUNT(*) AS cnt FROM memories {where_clause}", params
            )
            total_mem = row["cnt"] if row and row["cnt"] is not None else 0
            stats["total_memories"] = total_mem

            if total_mem == 0:
                logger.info(f"No memories found for statistics in scope: {stats['scope']}")
                return {
                    "success": True,
                    "data": stats,
                    "processing_time": time.time() - start_time,
                }

            # --- by level ----------------------------------------------------
            lv_rows = await conn.execute_fetchall(
                f"SELECT memory_level, COUNT(*) AS c FROM memories {where_clause} GROUP BY memory_level",
                params,
            )
            stats["by_level"] = {r["memory_level"]: r["c"] for r in lv_rows}

            # --- by type -----------------------------------------------------
            tp_rows = await conn.execute_fetchall(
                f"SELECT memory_type, COUNT(*) AS c FROM memories {where_clause} GROUP BY memory_type",
                params,
            )
            stats["by_type"] = {r["memory_type"]: r["c"] for r in tp_rows}

            # --- confidence / importance aggregates -------------------------
            agg_row = await conn.execute_fetchone(
                f"SELECT AVG(COALESCE(confidence,0)), AVG(COALESCE(importance,0)) FROM memories {where_clause}",
                params,
            )
            stats["confidence_avg"] = round(agg_row[0], 3) if agg_row and agg_row[0] else 0.0
            stats["importance_avg"] = round(agg_row[1], 2) if agg_row and agg_row[1] else 0.0

            # --- temporal ----------------------------------------------------
            tm_row = await conn.execute_fetchone(
                f"SELECT MAX(created_at), MIN(created_at) FROM memories {where_clause}", params
            )
            stats["newest_memory_unix"] = tm_row[0] if tm_row else None
            stats["oldest_memory_unix"] = tm_row[1] if tm_row else None

            # --- link stats --------------------------------------------------
            link_where = "WHERE m.workflow_id = ?" if workflow_id else ""
            link_params = params
            link_tot = await conn.execute_fetchone(
                f"""
                SELECT COUNT(*) AS cnt
                FROM memory_links ml
                JOIN memories m ON ml.source_memory_id = m.memory_id
                {link_where}
                """,
                link_params,
            )
            stats["total_links"] = link_tot["cnt"] if link_tot else 0

            l_rows = await conn.execute_fetchall(
                f"""
                SELECT ml.link_type, COUNT(*) AS c
                FROM memory_links ml
                JOIN memories m ON ml.source_memory_id = m.memory_id
                {link_where}
                GROUP BY ml.link_type
                """,
                link_params,
            )
            stats["links_by_type"] = {r["link_type"]: r["c"] for r in l_rows}

            # --- tag stats (top-5) ------------------------------------------
            tag_where = "WHERE wt.workflow_id = ?" if workflow_id else ""
            tag_rows = await conn.execute_fetchall(
                f"""
                SELECT t.name, COUNT(*) AS c
                FROM tags t
                JOIN workflow_tags wt ON t.tag_id = wt.tag_id
                {tag_where}
                GROUP BY t.tag_id
                ORDER BY c DESC
                LIMIT 5
                """,
                params,
            )
            stats["top_workflow_tags"] = {r["name"]: r["c"] for r in tag_rows}

            # --- workflow status break-down (global only) -------------------
            if not workflow_id:
                wf_rows = await conn.execute_fetchall(
                    "SELECT status, COUNT(*) AS c FROM workflows GROUP BY status"
                )
                stats["workflows_by_status"] = {r["status"]: r["c"] for r in wf_rows}

        logger.info(
            f"Computed memory statistics for scope: {stats['scope']}", emoji_key="bar_chart"
        )
        return {
            "success": True,
            "data": stats,
            "processing_time": time.time() - start_time,
        }

    except Exception as exc:
        logger.error(f"Failed to compute statistics: {exc}", exc_info=True)
        raise ToolError(f"Failed to compute statistics: {exc}") from exc


async def visualize_reasoning_chain(
    thought_chain_id: str,
    output_format: str = "mermaid",  # 'mermaid' | 'json'
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Produce a Mermaid diagram or hierarchical JSON for a single thought-chain.
    """
    # ───────────────── input validation ─────────────────
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")

    valid_formats = {"mermaid", "json"}
    output_format_lc = (output_format or "mermaid").lower()
    if output_format_lc not in valid_formats:
        raise ToolInputError(
            f"Invalid format. Use one of: {sorted(valid_formats)}",
            param_name="output_format",
        )

    t0 = time.time()

    try:
        # ─────────────── read-only data fetch ───────────────
        async with DBConnection(db_path).transaction(readonly=True):
            thought_chain_data = await get_thought_chain(thought_chain_id, db_path=db_path)

        if not thought_chain_data.get("success"):
            raise ToolError(
                f"Failed to retrieve thought chain {thought_chain_id} for visualization."
            )

        # ─────────────── generate visualisation ─────────────
        visual: str | None = None

        if output_format_lc == "mermaid":
            visual = await _generate_thought_chain_mermaid(thought_chain_data)

        else:  # json
            structured = {
                k: v for k, v in thought_chain_data.items() if k not in {"success", "thoughts"}
            }
            child_map: Dict[str | None, list[dict]] = defaultdict(list)
            for th in thought_chain_data.get("thoughts", []):
                child_map[th.get("parent_thought_id")].append(th)

            def build(nodes: list[dict]) -> list[dict]:
                tree = []
                for node in nodes:
                    n = dict(node)
                    kids = child_map.get(node["thought_id"])
                    if kids:
                        n["children"] = build(kids)
                    tree.append(n)
                return tree

            structured["thought_tree"] = build(child_map.get(None, []))
            visual = json.dumps(structured, indent=2)

        if visual is None:
            raise ToolError(
                f"Failed to generate visualization content for format '{output_format_lc}'."
            )

        result_data = {
            "thought_chain_id": thought_chain_id,
            "title": thought_chain_data.get("title", "Thought Chain"),
            "visualization": visual,
            "format": output_format_lc,
        }
        logger.info(
            f"Generated {output_format_lc} visualization for thought chain {thought_chain_id}",
            emoji_key="projector",
        )
        return {
            "success": True,
            "data": result_data,
            "processing_time": time.time() - t0,
        }

    except (ToolInputError, ToolError):
        raise
    except Exception as exc:
        logger.error(f"Error visualizing thought chain {thought_chain_id}: {exc}", exc_info=True)
        raise ToolError(f"Failed to visualize thought chain: {exc}") from exc


async def visualize_memory_network(
    workflow_id: Optional[str] = None,
    center_memory_id: Optional[str] = None,
    depth: int = 1,
    max_nodes: int = 30,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    output_format: str = "mermaid",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Generates a Mermaid diagram of the memory graph.  All original behaviour,
    parameters, and return shape are preserved exactly; the sole change is that
    the query block is now executed inside a read-only snapshot transaction
    (`DBConnection.transaction(readonly=True)`)
    """
    # ------------- validation (unchanged) -------------
    if not workflow_id and not center_memory_id:
        raise ToolInputError(
            "Either workflow_id or center_memory_id must be provided.", param_name="workflow_id"
        )
    if output_format.lower() != "mermaid":
        raise ToolInputError(
            "Currently only 'mermaid' format is supported.", param_name="output_format"
        )
    if depth < 0:
        raise ToolInputError("Depth cannot be negative.", param_name="depth")
    if max_nodes <= 0:
        raise ToolInputError("Max nodes must be positive.", param_name="max_nodes")

    start_time = time.time()
    selected_memory_ids: set[str] = set()
    memories_data: dict[str, Any] = {}
    links_data: list[dict[str, Any]] = []

    try:
        # ---------- read-only snapshot ----------
        async with DBConnection(db_path).transaction(readonly=True) as conn:
            # --- 1. Initial memory selection (unchanged) ---
            target_workflow_id = workflow_id
            if center_memory_id:
                async with conn.execute(
                    "SELECT workflow_id FROM memories WHERE memory_id = ?", (center_memory_id,)
                ) as cursor:
                    center_row = await cursor.fetchone()
                    if not center_row:
                        raise ToolInputError(
                            f"Center memory {center_memory_id} not found.",
                            param_name="center_memory_id",
                        )
                    if not target_workflow_id:
                        target_workflow_id = center_row["workflow_id"]
                    elif target_workflow_id != center_row["workflow_id"]:
                        raise ToolInputError(
                            f"Center memory {center_memory_id} does not belong to workflow {target_workflow_id}.",
                            param_name="center_memory_id",
                        )

                queue, visited = {center_memory_id}, set()
                for current_depth in range(depth + 1):
                    if len(selected_memory_ids) >= max_nodes:
                        break
                    next_queue: set[str] = set()
                    ids_to_query = list(queue - visited)
                    if not ids_to_query:
                        break
                    visited.update(ids_to_query)
                    for node_id in ids_to_query:
                        if len(selected_memory_ids) < max_nodes:
                            selected_memory_ids.add(node_id)
                        else:
                            break
                    if current_depth < depth and len(selected_memory_ids) < max_nodes:
                        placeholders = ", ".join("?" * len(ids_to_query))
                        neighbor_query = (
                            f"SELECT target_memory_id AS neighbor_id FROM memory_links "
                            f"WHERE source_memory_id IN ({placeholders}) "
                            f"UNION "
                            f"SELECT source_memory_id AS neighbor_id FROM memory_links "
                            f"WHERE target_memory_id IN ({placeholders})"
                        )
                        async with conn.execute(neighbor_query, ids_to_query * 2) as cursor:
                            async for row in cursor:
                                if row["neighbor_id"] not in visited:
                                    next_queue.add(row["neighbor_id"])
                    queue = next_queue
            else:
                if not target_workflow_id:
                    raise ToolInputError(
                        "Workflow ID is required when not specifying a center memory.",
                        param_name="workflow_id",
                    )
                filter_where = ["workflow_id = ?"]
                params: list[Any] = [target_workflow_id]
                if memory_level:
                    filter_where.append("memory_level = ?")
                    params.append(memory_level.lower())
                if memory_type:
                    filter_where.append("memory_type = ?")
                    params.append(memory_type.lower())
                now_unix = int(time.time())
                filter_where.append("(ttl = 0 OR created_at + ttl > ?)")
                params.append(now_unix)
                where_sql = " AND ".join(filter_where)
                query = (
                    "SELECT memory_id "
                    "FROM memories "
                    f"WHERE {where_sql} "
                    "ORDER BY compute_memory_relevance("
                    "    importance, confidence, created_at, access_count, last_accessed"
                    ") DESC "
                    "LIMIT ?"
                )
                params.append(max_nodes)

                # Ensure UDFs are registered before using compute_memory_relevance
                await _ensure_udfs_registered(conn)

                async with conn.execute(query, params) as cursor:
                    selected_memory_ids = {row["memory_id"] for row in await cursor.fetchall()}

            # --- early-exit branches & rest of logic (unchanged) ---
            if not selected_memory_ids:
                logger.info("No memories selected for visualization based on criteria.")
                return {
                    "workflow_id": target_workflow_id,
                    "center_memory_id": center_memory_id,
                    "visualization": "```mermaid\ngraph TD\n    NoNodes[No memories found]\n```",
                    "node_count": 0,
                    "link_count": 0,
                    "format": "mermaid",
                    "processing_time": time.time() - start_time,
                }

            fetch_ids = list(selected_memory_ids)
            placeholders = ", ".join("?" * len(fetch_ids))
            details_query = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
            async with conn.execute(details_query, fetch_ids) as cursor:
                async for row in cursor:
                    mem_data = dict(row)
                    if center_memory_id:
                        ok = True
                        if memory_level and mem_data.get("memory_level") != memory_level.lower():
                            ok = False
                        if memory_type and mem_data.get("memory_type") != memory_type.lower():
                            ok = False
                        if ok or mem_data["memory_id"] == center_memory_id:
                            memories_data[mem_data["memory_id"]] = mem_data
                    else:
                        memories_data[mem_data["memory_id"]] = mem_data
            final_selected_ids = set(memories_data)

            if center_memory_id and center_memory_id not in final_selected_ids:
                if center_memory_id in memories_data:
                    final_selected_ids.add(center_memory_id)

            if not final_selected_ids:
                logger.info("No memories remained after applying filters for visualization.")
                return {
                    "workflow_id": target_workflow_id,
                    "center_memory_id": center_memory_id,
                    "visualization": "```mermaid\ngraph TD\n    NoNodes[No memories match filters]\n```",
                    "node_count": 0,
                    "link_count": 0,
                    "format": "mermaid",
                    "processing_time": time.time() - start_time,
                }

            final_ids_list = list(final_selected_ids)
            placeholders = ", ".join("?" * len(final_ids_list))
            links_query = (
                "SELECT * FROM memory_links "
                f"WHERE source_memory_id IN ({placeholders}) "
                f"AND target_memory_id IN ({placeholders})"
            )
            async with conn.execute(links_query, final_ids_list * 2) as cursor:
                links_data = [dict(row) for row in await cursor.fetchall()]

        # --- diagram generation & return (unchanged) ---
        mermaid_string = await _generate_memory_network_mermaid(
            list(memories_data.values()), links_data, center_memory_id
        )
        processing_time = time.time() - start_time
        node_count, link_count = len(memories_data), len(links_data)
        logger.info(
            f"Generated memory network visualization ({node_count} nodes, {link_count} links) for workflow {target_workflow_id}",
            emoji_key="network",
        )
        return {
            "success": True,
            "data": {
                "workflow_id": target_workflow_id,
                "center_memory_id": center_memory_id,
                "visualization": mermaid_string,
                "node_count": node_count,
                "link_count": link_count,
                "format": "mermaid",
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error visualizing memory network: {e}", exc_info=True)
        raise ToolError(f"Failed to visualize memory network: {e}") from e


# ======================================================
# Cognitive Timeline State Recording
# ======================================================


class CognitiveStateType(str, Enum):
    """Types of cognitive state changes to record in the timeline."""

    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_STATUS_CHANGED = "workflow_status_changed"
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    MEMORY_STORED = "memory_stored"
    MEMORY_UPDATED = "memory_updated"
    GOAL_CREATED = "goal_created"
    GOAL_STATUS_CHANGED = "goal_status_changed"
    THOUGHT_RECORDED = "thought_recorded"
    ARTIFACT_CREATED = "artifact_created"
    COGNITIVE_STATE_SAVED = "cognitive_state_saved"
    WORKING_MEMORY_UPDATED = "working_memory_updated"
    FOCUS_CHANGED = "focus_changed"
    MEMORY_LINK_CREATED = "memory_link_created"
    REFLECTION_GENERATED = "reflection_generated"


async def _record_cognitive_timeline_state(
    conn: aiosqlite.Connection,
    workflow_id: Optional[str],
    state_type: str,
    state_data: Dict[str, Any],
    description: Optional[str] = None,
) -> str:
    """
    Record a cognitive state change in the timeline.

    Args:
        conn: Database connection
        workflow_id: Associated workflow ID (can be None for global states)
        state_type: Type of cognitive state (e.g., 'workflow_created', 'action_started', 'memory_stored')
        state_data: Dictionary containing the state data
        description: Optional human-readable description

    Returns:
        The state_id of the recorded state
    """
    state_id = MemoryUtils.generate_id()
    timestamp = time.time()

    # Serialize state data to JSON
    try:
        state_data_json = json.dumps(state_data, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Failed to serialize cognitive state data: {e}")
        state_data_json = json.dumps(
            {"error": "Serialization failed", "type": str(type(state_data))}
        )

    await conn.execute(
        """
        INSERT INTO cognitive_timeline_states (
            state_id, timestamp, state_type, state_data, workflow_id, description
        ) VALUES (?, ?, ?, ?, ?, ?)
    """,
        (state_id, timestamp, state_type, state_data_json, workflow_id, description),
    )

    logger.debug(f"Recorded cognitive state: {state_type} for workflow {workflow_id}")
    return state_id


class MemoryUtils:
    """Utility methods for memory operations."""

    @staticmethod
    def generate_id() -> str:
        """Generate a unique UUID V4 string for database records."""
        return str(uuid.uuid4())

    @staticmethod
    async def serialize(obj: Any) -> Optional[str]:
        """Safely serialize an arbitrary Python object to a JSON string."""
        if obj is None:
            return None

        # ------------------------------------------------------------------ #
        # constants / helpers                                                #
        # ------------------------------------------------------------------ #
        try:
            max_len = MAX_TEXT_LENGTH  # from config block
        except NameError:  # ultra-defensive fallback
            max_len = 64_000

        def _truncate_utf8(text: str, limit: int) -> str:
            """Return text whose UTF-8 byte length ≤ limit, appending '[TRUNCATED]'.

            Ensures the cut never leaves the byte-stream inside a UTF-8 sequence:
            walk backwards past any 10xxxxxx continuation bytes, then decode;
            if that still fails, step back until it succeeds (⇒ O(bytes)).
            """
            raw = text.encode("utf-8")
            if len(raw) <= limit:
                return text
            truncated_marker = "[TRUNCATED]"
            truncated_bytes = truncated_marker.encode("utf-8")

            # Reserve space for the truncated marker
            if len(truncated_bytes) >= limit:
                return truncated_marker[:limit]  # Edge case: limit too small

            available_space = limit - len(truncated_bytes)
            cut = available_space
            # first, back up over 10xxxxxx continuation bytes
            while cut > 0 and (raw[cut] & 0xC0) == 0x80:
                cut -= 1
            # now make sure we’re not in the middle of a leading byte for a
            # multibyte code-point (rare, but handle)
            while cut > 0:
                try:
                    prefix = raw[:cut].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    cut -= 1
            else:
                prefix = ""  # could not decode anything
            result = prefix + truncated_marker
            # Double-check our work
            assert len(result.encode("utf-8")) <= limit, (
                f"Truncation failed: {len(result.encode('utf-8'))} > {limit}"
            )
            return result

        # ------------------------------------------------------------------ #
        # normal JSON path                                                   #
        # ------------------------------------------------------------------ #
        try:
            json_str = json.dumps(obj, ensure_ascii=False, default=str)
        except TypeError as e:
            logger.debug(
                f"Direct JSON serialization failed for {type(obj)}: {e}. Falling back to str()."
            )
            fallback_repr = str(obj)
            # ensure fallback fits byte budget *and* is safe UTF-8
            fallback_repr = _truncate_utf8(fallback_repr, max_len)
            json_str = json.dumps(
                {
                    "error": f"Serialization failed for type {type(obj)}.",
                    "fallback_repr": fallback_repr,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Unexpected serialization error for {type(obj)}: {e}", exc_info=True)
            json_str = json.dumps(
                {
                    "error": f"Unhandled serialization error for type {type(obj)}.",
                    "details": str(e),
                },
                ensure_ascii=False,
            )

        # ------------------------------------------------------------------ #
        # post-serialization size guard                                      #
        # ------------------------------------------------------------------ #
        if len(json_str.encode("utf-8")) > max_len:
            # Calculate proper preview size: leave space for JSON structure overhead
            error_structure_overhead = 120  # Rough estimate for JSON structure
            preview_limit = max(50, max_len - error_structure_overhead)  # At least 50 chars
            preview = _truncate_utf8(json_str, preview_limit)
            json_str = json.dumps(
                {
                    "error": "Serialized content exceeded maximum length.",
                    "original_type": str(type(obj)),
                    "preview": preview,
                },
                ensure_ascii=False,
            )
        return json_str

    @staticmethod
    async def deserialize(json_str: Optional[str]) -> Any:
        """Safely deserialize a JSON string back into a Python object.

        Handles None input and potential JSON decoding errors. If decoding fails,
        it returns the original string, assuming it might not have been JSON
        in the first place (e.g., a truncated representation).
        """
        if json_str is None:
            return None
        if not json_str.strip():  # Handle empty strings
            return None
        try:
            # Attempt to load the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If it fails, log the issue and return the original string
            # This might happen if the string stored was an error message or truncated data
            logger.debug(
                f"Failed to deserialize JSON: {e}. Content was: '{json_str[:100]}...'. Returning raw string."
            )
            return json_str
        except Exception as e:
            # Catch other potential errors during deserialization
            logger.error(
                f"Unexpected error deserializing JSON: {e}. Content: '{json_str[:100]}...'",
                exc_info=True,
            )
            return json_str  # Return original string as fallback

    @staticmethod
    def _validate_sql_identifier(identifier: str, identifier_type: str = "column/table") -> str:
        """Validates a string intended for use as an SQL table or column name.

        Prevents SQL injection by ensuring the identifier only contains
        alphanumeric characters and underscores. Raises ToolInputError if invalid.

        Args:
            identifier: The string to validate.
            identifier_type: A description of what the identifier represents (for error messages).

        Returns:
            The validated identifier if it's safe.

        Raises:
            ToolInputError: If the identifier is invalid.
        """
        # Simple regex: Allows letters, numbers, and underscores. Must start with a letter or underscore.
        # Adjust regex if more complex identifiers (e.g., quoted) are needed, but keep it strict.
        if not identifier or not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            logger.error(f"Invalid SQL identifier provided: '{identifier}'")
            raise ToolInputError(
                f"Invalid {identifier_type} name provided. Must be alphanumeric/underscore.",
                param_name=identifier_type,
            )
        # Optional: Check against a known allowlist of tables/columns if possible
        # known_tables = {"actions", "thoughts", "memories", ...}
        # if identifier_type == "table" and identifier not in known_tables:
        #     raise ToolInputError(f"Unknown table name provided: {identifier}", param_name=identifier_type)
        return identifier

    @staticmethod
    async def get_next_sequence_number(
        conn: aiosqlite.Connection,
        parent_id: str,
        table: str,
        parent_col: str,
        *,
        max_retries: int = 6,
        backoff_base: float = 0.02,
    ) -> int:
        """
        Atomically allocate the next `sequence_number` within (*table*, *parent_col*).

        Strategy
        --------
        1.  In each iteration SELECT the current MAX(sequence_number) for *parent_id*
            **inside the caller’s transaction** (writers already hold BEGIN IMMEDIATE).
        2.  Return candidate = max+1 and let the eventual INSERT hit the UNIQUE index.
            If that raises `sqlite3.IntegrityError`, the caller may call this function
            again; for convenience we retry here automatically.

        The surrounding transaction plus the UNIQUE index guarantees safety; the loop
        merely hides rare contention from the caller.
        """
        validated_table = MemoryUtils._validate_sql_identifier(table, "table")
        validated_parent_col = MemoryUtils._validate_sql_identifier(parent_col, "parent_col")

        for attempt in range(max_retries):
            sql_max = (
                f"SELECT COALESCE(MAX(sequence_number), 0) + 1 "
                f"FROM {validated_table} "
                f"WHERE {validated_parent_col} = ?"
            )
            async with conn.execute(sql_max, (parent_id,)) as cur:
                row = await cur.fetchone()
                next_seq = int(row[0] if row and row[0] is not None else 1)

            # Fast pre-check: is the candidate already present?  (Cheap index lookup)
            sql_exists = (
                f"SELECT 1 FROM {validated_table} "
                f"WHERE {validated_parent_col} = ? AND sequence_number = ? LIMIT 1"
            )
            async with conn.execute(sql_exists, (parent_id, next_seq)) as cur:
                exists = await cur.fetchone()

            if not exists:
                return next_seq  # success!

            # Contention: another writer committed the same number – back-off & retry
            await asyncio.sleep(backoff_base * (2**attempt) * (0.5 + random.random() / 2))

        raise ToolError(
            f"Unable to allocate unique sequence_number for {validated_table}"
            f" (parent {parent_id}) after {max_retries} retries."
        )

    @staticmethod
    async def process_tags(
        conn: aiosqlite.Connection, entity_id: str, tags: List[str], entity_type: str
    ) -> None:
        """Ensures tags exist in the 'tags' table and associates them with a given entity
           in the appropriate junction table (e.g., 'workflow_tags').

        Args:
            conn: The database connection.
            entity_id: The ID of the entity (workflow, action, artifact).
            tags: A list of tag names (strings) to associate. Duplicates are handled.
            entity_type: The type of the entity ('workflow', 'action', 'artifact'). Must form valid SQL identifiers when combined with '_tags' or '_id'.
        """
        if not tags:
            return  # Nothing to do if no tags are provided

        # Validate entity_type first as it forms part of identifiers
        # Allow only specific expected entity types
        allowed_entity_types = {"workflow", "action", "artifact"}
        if entity_type not in allowed_entity_types:
            raise ToolInputError(
                f"Invalid entity_type for tagging: {entity_type}", param_name="entity_type"
            )

        # Define and validate dynamic identifiers
        junction_table_name = f"{entity_type}_tags"
        id_column_name = f"{entity_type}_id"
        validated_junction_table = MemoryUtils._validate_sql_identifier(
            junction_table_name, "junction_table"
        )
        validated_id_column = MemoryUtils._validate_sql_identifier(id_column_name, "id_column")
        # --- End Validation ---

        tag_ids_to_link = []
        unique_tags = list(
            set(str(tag).strip().lower() for tag in tags if str(tag).strip())
        )  # Clean, lowercase, unique tags
        now_unix = int(time.time())

        if not unique_tags:
            return  # Nothing to do if tags are empty after cleaning

        # Ensure all unique tags exist in the 'tags' table and get their IDs
        for tag_name in unique_tags:
            # Use UPSERT with retry to handle race conditions robustly
            for attempt in range(3):  # Retry up to 3 times for race conditions
                try:
                    # Try INSERT with RETURNING to get ID atomically
                    cursor = await conn.execute(
                        """
                        INSERT INTO tags (name, created_at) VALUES (?, ?)
                        ON CONFLICT(name) DO UPDATE SET name = name
                        RETURNING tag_id
                        """,
                        (tag_name, now_unix),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()

                    if row:
                        tag_ids_to_link.append(row["tag_id"])
                        break  # Success, move to next tag
                    else:
                        # Fallback: SELECT separately (rare case)
                        cursor = await conn.execute(
                            "SELECT tag_id FROM tags WHERE name = ?", (tag_name,)
                        )
                        row = await cursor.fetchone()
                        await cursor.close()

                        if row:
                            tag_ids_to_link.append(row["tag_id"])
                            break
                        elif attempt == 2:  # Last attempt
                            logger.warning(f"Could not find or create tag_id for tag: {tag_name}")
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.warning(f"Tag processing failed for '{tag_name}': {e}")
                    else:
                        await asyncio.sleep(0.01 * (attempt + 1))  # Brief retry delay

        # Link the retrieved tag IDs to the entity in the junction table
        if tag_ids_to_link:
            link_values = [(entity_id, tag_id) for tag_id in tag_ids_to_link]
            # Use INSERT OR IGNORE to handle potential race conditions or duplicate calls gracefully
            # Use validated identifiers in the f-string
            await conn.executemany(
                f"INSERT OR IGNORE INTO {validated_junction_table} ({validated_id_column}, tag_id) VALUES (?, ?)",
                link_values,
            )
            logger.debug(f"Associated {len(link_values)} tags with {entity_type} {entity_id}")

    @staticmethod
    async def _log_memory_operation(
        conn: aiosqlite.Connection,
        workflow_id: str,
        operation: str,
        memory_id: Optional[str] = None,
        action_id: Optional[str] = None,
        operation_data: Optional[Dict] = None,
    ) -> str:
        """
        Persist a memory-operation audit record.

        Returns
        -------
        str
            The generated `operation_log_id`.

        Notes
        -----
        • The caller provides an already-open connection and is responsible
          for committing or rolling back the surrounding transaction.
        • Any failure is logged **and re-raised** so the caller can roll back.
        """
        op_id = MemoryUtils.generate_id()
        ts_unix = int(time.time())

        try:
            op_data_json = (
                await MemoryUtils.serialize(operation_data) if operation_data is not None else None
            )

            await conn.execute(
                """
                INSERT INTO memory_operations (
                    operation_log_id, workflow_id, memory_id, action_id,
                    operation, operation_data, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    op_id,
                    workflow_id,
                    memory_id,
                    action_id,
                    operation,
                    op_data_json,
                    ts_unix,
                ),
            )
            return op_id

        except Exception as exc:
            logger.error(
                f"CRITICAL: failed to log memory operation '{operation}' "
                f"(wf={_fmt_id(workflow_id)}, mem={_fmt_id(memory_id)}, act={_fmt_id(action_id)}): {exc}",
                exc_info=True,
            )
            raise

    @staticmethod
    async def _update_memory_access(conn: aiosqlite.Connection, memory_id: str):
        """Updates the last_accessed timestamp and increments access_count for a memory. Internal helper."""
        now_unix = int(time.time())
        try:
            # Use COALESCE to handle the first access correctly
            await conn.execute(
                """
                UPDATE memories
                SET last_accessed = ?,
                    access_count = COALESCE(access_count, 0) + 1
                WHERE memory_id = ?
                """,
                (now_unix, memory_id),
            )
        except Exception as e:
            logger.warning(
                f"Failed to update memory access stats for {memory_id}: {e}", exc_info=True
            )


# ======================================================
# Embedding Service Integration & Semantic Search Logic
# ======================================================


async def _store_embedding(conn: aiosqlite.Connection, memory_id: str, text: str) -> Optional[str]:
    """Generates and stores an embedding for a memory using the EmbeddingService.

    Args:
        conn: Database connection.
        memory_id: ID of the memory.
        text: Text content to generate embedding for (often content + description).

    Returns:
        ID of the stored embedding record in the embeddings table, or None if failed.
    """
    try:
        from ultimate_mcp_server.services.vector.embeddings import get_embedding_service

        embedding_service = get_embedding_service()
        if not embedding_service.client:  # Check if service was initialized correctly (has client)
            logger.warning(
                "EmbeddingService client not available. Cannot generate embedding.",
                emoji_key="warning",
            )
            return None

        # Generate embedding using the service (handles caching internally)
        embedding_list = await embedding_service.create_embeddings(texts=[text])
        if not embedding_list or not embedding_list[0]:  # Extra check for empty embedding
            logger.warning(f"Failed to generate embedding for memory {memory_id}")
            return None
        embedding_array = np.array(embedding_list[0], dtype=np.float32)  # Ensure consistent dtype
        if embedding_array.size == 0:
            logger.warning(f"Generated embedding is empty for memory {memory_id}")
            return None

        # Get the embedding dimension
        embedding_dimension = embedding_array.shape[0]

        # Generate a unique ID for this embedding entry in our DB table
        embedding_db_id = MemoryUtils.generate_id()
        embedding_bytes = embedding_array.tobytes()
        model_used = embedding_service.model_name

        # Store embedding in our DB
        await conn.execute(
            """
            INSERT INTO embeddings (id, memory_id, model, embedding, dimension, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                id = excluded.id,
                model = excluded.model,
                embedding = excluded.embedding,
                dimension = excluded.dimension,
                created_at = excluded.created_at
            """,
            (
                embedding_db_id,
                memory_id,
                model_used,
                embedding_bytes,
                embedding_dimension,
                int(time.time()),
            ),
        )
        # Update the memory record to link to this *embedding table entry ID*
        # Note: The cognitive_memory schema had embedding_id as FK to embeddings.id
        # We will store embedding_db_id here.
        await conn.execute(
            "UPDATE memories SET embedding_id = ? WHERE memory_id = ?", (embedding_db_id, memory_id)
        )

        logger.debug(
            f"Stored embedding {embedding_db_id} (Dim: {embedding_dimension}) for memory {memory_id}"
        )
        return embedding_db_id  # Return the ID of the row in the embeddings table

    except Exception as e:
        logger.error(f"Failed to store embedding for memory {memory_id}: {e}", exc_info=True)
        return None


async def _find_similar_memories(
    conn: aiosqlite.Connection,
    query_text: str,
    workflow_id: Optional[str] = None,
    limit: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Find memories semantically close to *query_text* using stored embeddings.
    Returns [(memory_id, similarity_score)] sorted by similarity desc.
    """
    try:
        from ultimate_mcp_server.services.vector.embeddings import get_embedding_service

        embedding_service = get_embedding_service()
        if not embedding_service.client:
            logger.warning("EmbeddingService unavailable; semantic search disabled.")
            return []

        # 1. ─ Generate embedding for the query text ───────────────────────────
        q_emb_list = await embedding_service.create_embeddings(texts=[query_text])
        if not q_emb_list or not q_emb_list[0]:
            logger.warning("Failed to embed query text.")
            return []
        q_vec = np.asarray(q_emb_list[0], dtype=np.float32)
        if q_vec.size == 0:
            logger.warning("Query embedding empty.")
            return []
        q_dim = q_vec.shape[0]
        q_vec_2d = q_vec.reshape(1, -1)

        # 2. ─ Collect candidate embeddings from DB ────────────────────────────
        # First check if we have any embeddings with the current dimension
        dimension_check_sql = "SELECT COUNT(*) FROM embeddings WHERE dimension = ?"
        async with conn.execute(dimension_check_sql, (q_dim,)) as cur:
            count_row = await cur.fetchone()
            current_dim_count = count_row[0] if count_row else 0

        if current_dim_count == 0:
            logger.warning(
                f"No embeddings found with current model dimension {q_dim}. "
                "This may indicate an embedding model change. Consider re-generating embeddings."
            )
            return []

        sql = """
        SELECT m.memory_id, e.embedding
        FROM   memories  m
        JOIN   embeddings e ON e.id = m.embedding_id
        WHERE  e.dimension = ?
        """
        params: list[Any] = [q_dim]
        if workflow_id:
            sql += " AND m.workflow_id = ?"
            params.append(workflow_id)
        if memory_level:
            sql += " AND m.memory_level = ?"
            params.append(memory_level.lower())
        if memory_type:
            sql += " AND m.memory_type  = ?"
            params.append(memory_type.lower())

        now_unix = int(time.time())
        sql += " AND (m.ttl = 0 OR m.created_at + m.ttl > ?)"
        params.append(now_unix)

        cand_limit = max(limit * 5, 50)
        sql += """
        ORDER BY m.last_accessed IS NULL, m.last_accessed DESC
        LIMIT ?
        """
        params.append(cand_limit)

        async with conn.execute(sql, params) as cur:
            candidates = await cur.fetchall()

        if not candidates:
            return []

        # 3. ─ Compute cosine similarities ─────────────────────────────────────
        sims: list[Tuple[str, float]] = []
        for mem_id, emb_blob in candidates:
            emb_vec = np.frombuffer(emb_blob, dtype=np.float32)
            if emb_vec.size != q_dim:
                logger.warning(
                    f"Dim mismatch for memory {mem_id}: {emb_vec.size} vs {q_dim}; skipped."
                )
                continue
            sim = sk_cosine_similarity(q_vec_2d, emb_vec.reshape(1, -1))[0][0]
            if sim >= threshold:
                sims.append((mem_id, float(sim)))

        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[:limit]

    except Exception as e:
        logger.error(f"_find_similar_memories failed: {e}", exc_info=True)
        return []


# ======================================================
# Public Tool Functions
# ======================================================


@with_tool_metrics
@with_error_handling
async def initialize_memory_system(
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Prepare the Unified Memory System for use.

    Steps
    -----
    1. Ensure the SQLite file exists and the full schema is present.
    2. Open a *read-only* snapshot to prove the DB is readable.
    3. Initialise the EmbeddingService.
    4. Return a status payload.

    Returns
    -------
    {
        success: bool,
        message: str,
        db_path: str,               # absolute
        embedding_service_functional: bool,
        embedding_service_warning: Optional[str],
        processing_time: float      # seconds
    }
    """
    t0 = time.perf_counter()
    logger.info("Initializing Unified Memory System…", emoji_key="rocket")

    db = DBConnection(db_path)

    try:
        # ───────── 1. Schema bootstrap (idempotent) ─────────
        await db._bootstrap()

        # ───────── 2. Sanity check – read-only snapshot ─────
        async with db.transaction(readonly=True) as conn:
            await conn.execute_fetchone("SELECT count(*) FROM ums_internal_metadata")

        logger.success("Database schema verified.", emoji_key="database")

        # ───────── 3. Embedding service ─────────────────────
        embedding_service_warning: str | None = None
        try:
            from ultimate_mcp_server.services.vector.embeddings import get_embedding_service

            es = get_embedding_service()
            if es.client is None:
                embedding_service_warning = (
                    "EmbeddingService client not available; embeddings disabled."
                )
                logger.error(embedding_service_warning, emoji_key="warning")
                raise ToolError(embedding_service_warning)

            # Test embedding creation with a small sample
            logger.info("Testing embedding service functionality...", emoji_key="test_tube")
            test_embeddings = await es.create_embeddings(["test"])
            if test_embeddings and test_embeddings[0]:
                embedding_dim = len(test_embeddings[0])
                logger.success(
                    f"EmbeddingService fully functional - Model: {es.model_name}, Dimension: {embedding_dim}",
                    emoji_key="brain",
                )
            else:
                embedding_service_warning = "EmbeddingService test failed - no embeddings generated"
                logger.error(embedding_service_warning, emoji_key="warning")
                raise ToolError(embedding_service_warning)

            embedding_ok = True
        except Exception as exc:
            if not isinstance(exc, ToolError):
                embedding_service_warning = (
                    f"Failed to initialise EmbeddingService: {exc}; embeddings disabled."
                )
                logger.error(embedding_service_warning, emoji_key="error", exc_info=True)
                raise ToolError(embedding_service_warning) from exc
            raise  # propagate pre-built ToolError

        # ───────── 4. Done ──────────────────────────────────
        dt = time.perf_counter() - t0
        logger.success(
            "Unified Memory System ready.",
            emoji_key="white_check_mark",
            time=dt,
        )
        return {
            "success": True,
            "data": {
                "message": "Unified Memory System initialised successfully.",
                "db_path": os.path.abspath(db_path),
                "embedding_service_functional": embedding_ok,
                "embedding_service_warning": embedding_service_warning,
            },
            "processing_time": dt,
        }

    except Exception as exc:
        dt = time.perf_counter() - t0
        logger.error(
            f"Memory system initialisation failed: {exc}",
            emoji_key="x",
            exc_info=True,
            time=dt,
        )
        if isinstance(exc, ToolError):
            raise
        raise ToolError(f"Memory system initialisation failed: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def create_workflow(
    title: str,
    *,
    description: str | None = None,
    goal: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    parent_workflow_id: str | None = None,
    idempotency_key: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """Create a new workflow with optional idempotency-key support."""
    logger.info(f"UMS:create_workflow called – title='{title[:80]}', idem='{idempotency_key}'")
    if not isinstance(title, str) or not title.strip():
        raise ToolInputError("Workflow title must be a non-empty string.", param_name="title")

    now = int(time.time())
    db = DBConnection(db_path)

    def iso(ts):
        return safe_format_timestamp(ts)

    try:
        # ── 1. Idempotency check ──────────────────────────────────────────────
        if idempotency_key:
            async with db.transaction(readonly=True) as conn_chk:
                row = await conn_chk.execute_fetchone(
                    "SELECT * FROM workflows WHERE idempotency_key = ?", (idempotency_key,)
                )
            if row:
                wf = dict(row)  # ← convert once, avoid Row.get
                wf_id = wf["workflow_id"]

                async with db.transaction(readonly=True) as conn_det:
                    tag_rows = await conn_det.execute_fetchall(
                        """
                        SELECT t.name
                        FROM   tags t
                        JOIN   workflow_tags wt ON wt.tag_id = t.tag_id
                        WHERE  wt.workflow_id = ?
                        """,
                        (wf_id,),
                    )
                    tags_list = [r["name"] for r in tag_rows]
                    chain_row = await conn_det.execute_fetchone(
                        """
                        SELECT thought_chain_id
                        FROM   thought_chains
                        WHERE  workflow_id = ?
                        ORDER  BY created_at
                        LIMIT 1
                        """,
                        (wf_id,),
                    )
                    chain_id = chain_row["thought_chain_id"] if chain_row else None
                    meta = await MemoryUtils.deserialize(wf["metadata"])

                payload = {
                    "workflow_id": wf_id,
                    "title": wf["title"],
                    "description": wf["description"],
                    "goal": wf["goal"],
                    "status": wf["status"],
                    "created_at": wf["created_at"],
                    "created_at_iso": iso(wf["created_at"]),
                    "updated_at": wf["updated_at"],
                    "updated_at_iso": iso(wf["updated_at"]),
                    "completed_at": wf["completed_at"],
                    "completed_at_iso": iso(wf["completed_at"]) if wf["completed_at"] else None,
                    "parent_workflow_id": wf["parent_workflow_id"],
                    "metadata": meta,
                    "last_active": wf["last_active"],
                    "last_active_iso": iso(wf["last_active"]) if wf["last_active"] else None,
                    "tags": tags_list,
                    "primary_thought_chain_id": chain_id,
                    "idempotency_hit": True,
                }
                logger.info(f"UMS:create_workflow idem-hit → {wf_id}")
                return {"success": True, "data": payload}
        # ── 2. Normal create path ────────────────────────────────────────────
        wf_id = MemoryUtils.generate_id()
        chain_id = MemoryUtils.generate_id()

        async with db.transaction(mode="IMMEDIATE") as conn:
            if parent_workflow_id:
                ok = await conn.execute_fetchone(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (parent_workflow_id,)
                )
                if not ok:
                    raise ToolInputError(
                        f"Parent workflow '{parent_workflow_id}' not found.",
                        param_name="parent_workflow_id",
                    )

            meta_json = await MemoryUtils.serialize(metadata)
            title_clean = title.strip()

            await conn.execute(
                """
                INSERT INTO workflows (
                    workflow_id,title,description,goal,status,
                    created_at,updated_at,parent_workflow_id,metadata,last_active,
                    idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    wf_id,
                    title_clean,
                    description,
                    goal,
                    WorkflowStatus.ACTIVE.value,
                    now,
                    now,
                    parent_workflow_id,
                    meta_json,
                    now,
                    idempotency_key,
                ),
            )

            if tags:
                await MemoryUtils.process_tags(conn, wf_id, tags, "workflow")

            chain_title = f"Main reasoning for: {title_clean[:100]}"
            await conn.execute(
                """
                INSERT INTO thought_chains (
                    thought_chain_id,workflow_id,title,created_at
                ) VALUES (?,?,?,?)
                """,
                (chain_id, wf_id, chain_title, now),
            )

            if goal:
                thought_id = MemoryUtils.generate_id()
                seq = await MemoryUtils.get_next_sequence_number(
                    conn, chain_id, "thoughts", "thought_chain_id"
                )
                await conn.execute(
                    """
                    INSERT INTO thoughts (
                        thought_id,thought_chain_id,thought_type,content,sequence_number,created_at
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (
                        thought_id,
                        chain_id,
                        ThoughtType.GOAL.value,
                        goal,
                        seq,
                        now,
                    ),
                )

            # initial cognitive_state
            await conn.execute(
                """
                INSERT OR IGNORE INTO cognitive_states (
                    state_id,workflow_id,title,created_at,is_latest,last_active
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    wf_id,
                    wf_id,
                    f"Primary context for workflow: {title_clean[:100]}",
                    now,
                    True,
                    now,
                ),
            )

            # Record cognitive state change in timeline
            try:
                await _record_cognitive_timeline_state(
                    conn,
                    wf_id,
                    CognitiveStateType.WORKFLOW_CREATED,
                    {
                        "workflow_id": wf_id,
                        "title": title_clean,
                        "description": description,
                        "goal": goal,
                        "status": WorkflowStatus.ACTIVE.value,
                        "parent_workflow_id": parent_workflow_id,
                        "tags": tags or [],
                    },
                    f"Created workflow: {title_clean}",
                )
            except Exception as e:
                logger.warning(f"Failed to record cognitive state for workflow creation: {e}")

        payload = {
            "workflow_id": wf_id,
            "title": title_clean,
            "description": description,
            "goal": goal,
            "status": WorkflowStatus.ACTIVE.value,
            "created_at": now,
            "created_at_iso": iso(now),
            "updated_at": now,
            "updated_at_iso": iso(now),
            "tags": tags or [],
            "primary_thought_chain_id": chain_id,
            "idempotency_hit": False,
        }
        logger.info(f"UMS:create_workflow created → {wf_id}")
        return {"success": True, "data": payload}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"create_workflow failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to create workflow: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def update_workflow_status(
    workflow_id: str,
    status: str,
    *,
    completion_message: str | None = None,
    update_tags: list[str] | None = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Change the *status* of a workflow, optionally append a completion / reflection
    thought and/or replace its tags.
    """
    try:
        status_enum = WorkflowStatus(status.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid status '{status}'. "
            f"Must be one of: {', '.join(s.value for s in WorkflowStatus)}",
            param_name="status",
        ) from exc
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    now = int(time.time())
    db = DBConnection(db_path)  # DBConnection instance for this specific db_path

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            exists = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

            cols: dict[str, Any] = {
                "status": status_enum.value,
                "updated_at": now,
                "last_active": now,
            }
            if status_enum in (
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.ABANDONED,
            ):
                cols["completed_at"] = now

            set_sql = ", ".join(f"{k}=?" for k in cols)
            await conn.execute(
                f"UPDATE workflows SET {set_sql} WHERE workflow_id = ?",
                [*cols.values(), workflow_id],
            )

            if completion_message:
                tc_row = await conn.execute_fetchone(
                    "SELECT thought_chain_id "
                    "FROM thought_chains WHERE workflow_id = ? "
                    "ORDER BY created_at LIMIT 1",
                    (workflow_id,),
                )
                if tc_row:
                    chain_id = tc_row["thought_chain_id"]
                    seq_no = await MemoryUtils.get_next_sequence_number(
                        conn, chain_id, "thoughts", "thought_chain_id"
                    )
                    thought_id = MemoryUtils.generate_id()
                    thought_type_enum = (
                        ThoughtType.SUMMARY
                        if status_enum == WorkflowStatus.COMPLETED
                        else ThoughtType.REFLECTION
                    )
                    await conn.execute(
                        "INSERT INTO thoughts "
                        "(thought_id, thought_chain_id, thought_type, content, "
                        " sequence_number, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            thought_id,
                            chain_id,
                            thought_type_enum.value,
                            completion_message,
                            seq_no,
                            now,
                        ),
                    )
                    logger.debug(
                        f"Added {thought_type_enum.value} thought {thought_id} for workflow {workflow_id}"
                    )
                else:
                    logger.warning(
                        f"No thought chain found for workflow {workflow_id}; completion message skipped."
                    )

            if update_tags:
                await MemoryUtils.process_tags(
                    conn, workflow_id, update_tags, entity_type="workflow"
                )
        # Transaction committed successfully here

        result = {
            "workflow_id": workflow_id,
            "status": status_enum.value,
            "updated_at_iso": safe_format_timestamp(now),
        }
        if "completed_at" in cols:
            result["completed_at_iso"] = safe_format_timestamp(now)

        logger.info(
            f"Workflow {_fmt_id(workflow_id)} status ➜ '{status_enum.value}'",
            emoji_key="arrows_counterclockwise",
        )
        return {"success": True, "data": result}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"update_workflow_status({_fmt_id(workflow_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to update workflow status: {exc}") from exc


# --- 3. Action Tracking Tools ---
@with_tool_metrics
@with_error_handling
async def record_action_start(
    workflow_id: str,
    action_type: str,
    reasoning: str,
    *,
    tool_name: str | None = None,
    tool_args: Dict[str, Any] | None = None,
    title: str | None = None,
    parent_action_id: str | None = None,
    tags: list[str] | None = None,
    related_thought_id: str | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    # ... (validation for action_type, reasoning, tool_name remains the same) ...
    try:
        action_type_enum = ActionType(action_type.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid action_type '{action_type}'. Valid types: {', '.join(t.value for t in ActionType)}",
            param_name="action_type",
        ) from exc
    if not reasoning:
        raise ToolInputError("Reasoning must be a non-empty string.", param_name="reasoning")
    if action_type_enum is ActionType.TOOL_USE and not tool_name:
        raise ToolInputError("tool_name required for 'tool_use' actions.", param_name="tool_name")

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    # Helper to fetch full action details for consistent return on idempotency hit
    async def _fetch_existing_action_details(
        conn_fetch: aiosqlite.Connection, existing_action_id: str
    ) -> Dict[str, Any]:
        action_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM actions WHERE action_id = ?", (existing_action_id,)
        )
        if not action_row:
            raise ToolError(
                f"Failed to re-fetch existing action {existing_action_id} on idempotency hit."
            )

        action_data = dict(action_row)
        tag_rows = await conn_fetch.execute_fetchall(
            "SELECT t.name FROM tags t JOIN action_tags at ON at.tag_id = t.tag_id WHERE at.action_id = ?",
            (existing_action_id,),
        )
        action_data["tags"] = [r["name"] for r in tag_rows]
        action_data["tool_args"] = await MemoryUtils.deserialize(action_data.get("tool_args"))

        # Find linked memory_id if exists (the one created when action originally started)
        linked_mem_row = await conn_fetch.execute_fetchone(
            "SELECT memory_id FROM memories WHERE action_id = ? AND memory_type = ? ORDER BY created_at ASC LIMIT 1",
            (existing_action_id, MemoryType.ACTION_LOG.value),
        )
        linked_memory_id_existing = linked_mem_row["memory_id"] if linked_mem_row else None

        return {
            "success": True,
            "data": {
                "action_id": existing_action_id,
                "workflow_id": action_data["workflow_id"],
                "action_type": action_data["action_type"],
                "title": action_data["title"],
                "tool_name": action_data.get("tool_name"),
                "status": action_data[
                    "status"
                ],  # Should be IN_PROGRESS or a terminal state if it somehow got completed/failed
                "started_at_unix": action_data["started_at"],
                "started_at_iso": to_iso_z(action_data["started_at"]),
                "sequence_number": action_data["sequence_number"],
                "tags": action_data["tags"],
                "linked_memory_id": linked_memory_id_existing,
                "idempotency_hit": True,
                "processing_time": time.perf_counter() - t0_perf,
            },
        }

    try:
        if idempotency_key:
            async with db.transaction(readonly=True) as conn_check:
                existing_action_row = await conn_check.execute_fetchone(
                    "SELECT action_id FROM actions WHERE workflow_id = ? AND idempotency_key = ?",
                    (workflow_id, idempotency_key),
                )
            if existing_action_row:
                existing_action_id = existing_action_row["action_id"]
                logger.info(
                    f"Idempotency hit for record_action_start (key='{idempotency_key}'). Returning existing action {_fmt_id(existing_action_id)}."
                )
                async with db.transaction(
                    readonly=True
                ) as conn_details:  # New transaction for fetching details
                    return await _fetch_existing_action_details(conn_details, existing_action_id)

        action_id = MemoryUtils.generate_id()
        memory_id = MemoryUtils.generate_id()  # For the new episodic memory

        async with db.transaction(mode="IMMEDIATE") as conn:
            # ... (existence checks for workflow_id, parent_action_id, related_thought_id remain the same) ...
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
            ):
                raise ToolInputError("Workflow not found.", param_name="workflow_id")
            if parent_action_id and not await conn.execute_fetchone(
                "SELECT 1 FROM actions WHERE action_id=? AND workflow_id=?",
                (parent_action_id, workflow_id),
            ):
                raise ToolInputError(
                    f"Parent action '{parent_action_id}' not in workflow.",
                    param_name="parent_action_id",
                )
            if related_thought_id and not await conn.execute_fetchone(
                "SELECT 1 FROM thoughts t JOIN thought_chains c ON c.thought_chain_id = t.thought_chain_id WHERE t.thought_id=? AND c.workflow_id=?",
                (related_thought_id, workflow_id),
            ):
                raise ToolInputError(
                    "related_thought_id not found in workflow.",
                    param_name="related_thought_id",
                )

            seq_no = await MemoryUtils.get_next_sequence_number(
                conn, workflow_id, "actions", "workflow_id"
            )
            auto_title = title or (
                f"Using {tool_name}"
                if action_type_enum is ActionType.TOOL_USE and tool_name
                else (
                    reasoning.split(".", 1)[0][:50] or f"{action_type_enum.value.title()} #{seq_no}"
                )
            )

            await conn.execute(
                """
                INSERT INTO actions (
                    action_id, workflow_id, parent_action_id, action_type, title, reasoning, 
                    tool_name, tool_args, status, started_at, sequence_number, idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,  # MODIFIED: Added idempotency_key
                (
                    action_id,
                    workflow_id,
                    parent_action_id,
                    action_type_enum.value,
                    auto_title,
                    reasoning,
                    tool_name,
                    await MemoryUtils.serialize(tool_args),
                    ActionStatus.IN_PROGRESS.value,
                    now_unix,
                    seq_no,
                    idempotency_key,  # MODIFIED: Added value
                ),
            )

            if tags:
                await MemoryUtils.process_tags(conn, action_id, tags, entity_type="action")
            if related_thought_id:
                await conn.execute(
                    "UPDATE thoughts SET relevant_action_id=? WHERE thought_id=?",
                    (action_id, related_thought_id),
                )

            mem_content = (
                f"Started action [{seq_no}] '{auto_title}' ({action_type_enum.value}). Reasoning: {reasoning}"
                + (f" Tool: {tool_name}." if tool_name else "")
            )
            final_mem_tags_list = ["action_start", action_type_enum.value]
            if tags:
                final_mem_tags_list.extend(tags)
            mem_tags_json = json.dumps(list(set(final_mem_tags_list)))

            await conn.execute(
                """INSERT INTO memories (memory_id, workflow_id, action_id, content, memory_level, memory_type, importance, confidence, tags, created_at, updated_at, access_count, last_accessed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
                (
                    memory_id,
                    workflow_id,
                    action_id,
                    mem_content,
                    MemoryLevel.EPISODIC.value,
                    MemoryType.ACTION_LOG.value,
                    5.0,
                    1.0,
                    mem_tags_json,
                    now_unix,
                    now_unix,
                    0,
                ),
            )
            await MemoryUtils._log_memory_operation(
                conn, workflow_id, "create_from_action_start", memory_id, action_id
            )
            await conn.execute(
                "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
                (now_unix, now_unix, workflow_id),
            )

            # Record cognitive state change in timeline
            try:
                await _record_cognitive_timeline_state(
                    conn,
                    workflow_id,
                    CognitiveStateType.ACTION_STARTED,
                    {
                        "action_id": action_id,
                        "action_type": action_type_enum.value,
                        "title": auto_title,
                        "tool_name": tool_name,
                        "reasoning": reasoning,
                        "sequence_number": seq_no,
                    },
                    f"Started action: {auto_title}",
                )
            except Exception as e:
                logger.warning(f"Failed to record cognitive state for action start: {e}")

        return {
            "success": True,
            "data": {
                "action_id": action_id,
                "workflow_id": workflow_id,
                "action_type": action_type_enum.value,
                "title": auto_title,
                "tool_name": tool_name,
                "status": ActionStatus.IN_PROGRESS.value,
                "started_at_unix": now_unix,
                "started_at_iso": to_iso_z(now_unix),
                "sequence_number": seq_no,
                "tags": tags or [],
                "linked_memory_id": memory_id,
                "idempotency_hit": False,
                "processing_time": time.perf_counter() - t0_perf,
            },
        }
    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"record_action_start failed for workflow {workflow_id}: {exc}", exc_info=True)
        raise ToolError(f"Failed to start action: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def record_action_completion(
    action_id: str,
    *,
    status: str = "completed",
    tool_result: Optional[Any] = None,
    summary: Optional[str] = None,
    conclusion_thought: Optional[str] = None,
    conclusion_thought_type: str = "inference",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Mark an action terminal (completed / failed / skipped), persist the tool-result,
    optionally append a concluding thought, and update any linked action-log memory.
    """
    start_perf = time.perf_counter()
    try:
        status_enum = ActionStatus(status.lower())
        if status_enum not in (ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.SKIPPED):
            raise ValueError
    except ValueError as e:
        raise ToolInputError(f"Invalid status '{status}'.", param_name="status") from e
    thought_enum = None
    if conclusion_thought:
        try:
            thought_enum = ThoughtType(conclusion_thought_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid thought type '{conclusion_thought_type}'.",
                param_name="conclusion_thought_type",
            ) from e
    now = int(time.time())
    conclusion_thought_id = None
    workflow_id_for_response = None
    db = DBConnection(db_path)
    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            row = await conn.execute_fetchone(
                "SELECT workflow_id,status FROM actions WHERE action_id=?", (action_id,)
            )
            if row is None:
                raise ToolInputError(f"Action not found: {action_id}", param_name="action_id")
            workflow_id_for_response = row["workflow_id"]
            serialized_tool_result = await MemoryUtils.serialize(tool_result)
            overflow_artifact_id = None
            if (
                serialized_tool_result is not None
                and len(serialized_tool_result.encode("utf-8")) > MAX_TEXT_LENGTH
            ):
                overflow_artifact_id = MemoryUtils.generate_id()
                await conn.execute(
                    """INSERT INTO artifacts(artifact_id,workflow_id,action_id,artifact_type,name,description,content,created_at,is_output)
                       VALUES(?,?,?,?,?,?,?, ?,1)""",
                    (
                        overflow_artifact_id,
                        workflow_id_for_response,
                        action_id,
                        ArtifactType.JSON.value
                        if isinstance(tool_result, (dict, list))
                        else ArtifactType.TEXT.value,
                        f"tool_result_{_fmt_id(action_id)}",
                        "Tool result stored externally due to size limit",
                        serialized_tool_result,
                        now,
                    ),
                )
                preview = (
                    serialized_tool_result[:200] + "…"
                    if len(serialized_tool_result) > 200
                    else serialized_tool_result
                )
                serialized_tool_result = json.dumps(
                    {
                        "artifact_id": overflow_artifact_id,
                        "preview": preview,
                        "stored_externally": True,
                    },
                    ensure_ascii=False,
                )
            await conn.execute(
                """UPDATE actions SET status=?,completed_at=?,tool_result=? WHERE action_id=?""",
                (status_enum.value, now, serialized_tool_result, action_id),
            )
            await conn.execute(
                "UPDATE workflows SET updated_at=?,last_active=? WHERE workflow_id=?",
                (now, now, workflow_id_for_response),
            )
            if conclusion_thought and thought_enum:
                chain_id = await conn.execute_fetchval(
                    "SELECT thought_chain_id FROM thought_chains WHERE workflow_id=? ORDER BY created_at LIMIT 1",
                    (workflow_id_for_response,),
                )
                if chain_id:
                    seq = await MemoryUtils.get_next_sequence_number(
                        conn, chain_id, "thoughts", "thought_chain_id"
                    )
                    conclusion_thought_id = MemoryUtils.generate_id()
                    await conn.execute(
                        "INSERT INTO thoughts(thought_id,thought_chain_id,thought_type,content,sequence_number,created_at,relevant_action_id) VALUES(?,?,?,?,?,?,?)",
                        (
                            conclusion_thought_id,
                            chain_id,
                            thought_enum.value,
                            conclusion_thought,
                            seq,
                            now,
                            action_id,
                        ),
                    )
            mem_row = await conn.execute_fetchone(
                "SELECT memory_id,content FROM memories WHERE action_id=? AND memory_type=?",
                (action_id, MemoryType.ACTION_LOG.value),
            )
            if mem_row:
                mem_id = mem_row["memory_id"]
                parts = [f"Completed ({status_enum.value})."]
                if summary:
                    parts.append(f"Summary: {summary}")
                if tool_result is not None:
                    parts.append(
                        "Result: stored_externally"
                        if overflow_artifact_id
                        else f"Result: {str(tool_result)[:50]}{'…' if len(str(tool_result)) > 50 else ''}"
                    )
                new_content = f"{mem_row['content']} {' '.join(parts)}"
                if len(new_content.encode("utf-8")) > MAX_TEXT_LENGTH:
                    new_content = (
                        new_content.encode("utf-8")[: MAX_TEXT_LENGTH - 3].decode(
                            "utf-8", "replace"
                        )
                        + "…"
                    )
                new_importance = min(
                    10.0,
                    max(
                        0.0,
                        (
                            await conn.execute_fetchone(
                                "SELECT importance FROM memories WHERE memory_id=?", (mem_id,)
                            )
                        )["importance"]
                        * (
                            1.2
                            if status_enum == ActionStatus.FAILED
                            else 0.8
                            if status_enum == ActionStatus.SKIPPED
                            else 1.0
                        ),
                    ),
                )
                await conn.execute(
                    "UPDATE memories SET content=?,importance=?,updated_at=?,last_accessed=? WHERE memory_id=?",
                    (new_content, new_importance, now, now, mem_id),
                )
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id_for_response,
                    "update_from_action_completion",
                    mem_id,
                    action_id,
                    {
                        "status": status_enum.value,
                        "summary_added": bool(summary),
                        "overflow_artifact_id": overflow_artifact_id,
                    },
                )

            # Record cognitive state change in timeline
            try:
                await _record_cognitive_timeline_state(
                    conn,
                    workflow_id_for_response,
                    CognitiveStateType.ACTION_COMPLETED,
                    {
                        "action_id": action_id,
                        "status": status_enum.value,
                        "had_summary": bool(summary),
                        "had_conclusion_thought": bool(conclusion_thought),
                        "overflow_artifact_id": overflow_artifact_id,
                    },
                    f"Completed action with status: {status_enum.value}",
                )
            except Exception as e:
                logger.warning(f"Failed to record cognitive state for action completion: {e}")

        return {
            "success": True,
            "data": {
                "action_id": action_id,
                "workflow_id": workflow_id_for_response,
                "status": status_enum.value,
                "completed_at_iso": to_iso_z(now),
                "conclusion_thought_id": conclusion_thought_id,
                "overflow_artifact_id": overflow_artifact_id,
                "processing_time": time.perf_counter() - start_perf,
            },
        }
    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"record_action_completion({action_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to record action completion: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_action_details(
    *,
    action_id: str | None = None,
    action_ids: list[str] | None = None,
    include_dependencies: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Fetch one or many actions (plus optional dependency graph).

    • Raw integer timestamps are preserved; *_iso companions are added.
    • `tool_args` / `tool_result` columns are JSON-deserialised.
    • When *include_dependencies* is True a bidirectional dependency map is returned
      for each action:  `{depends_on: [...], dependent_actions: [...]}`.
    """
    if not action_id and not action_ids:
        raise ToolInputError("Provide action_id or action_ids", param_name="action_id")

    targets: list[str] = [action_id] if action_id else action_ids or []
    if not targets:
        raise ToolInputError("No valid action IDs supplied.", param_name="action_id")

    start = time.perf_counter()
    db = DBConnection(db_path)

    def _add_iso(obj: dict[str, Any], *cols: str) -> None:
        for c in cols:
            if (ts := obj.get(c)) is not None:
                obj[f"{c}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            ph = ", ".join("?" * len(targets))

            # ───────── primary query ─────────
            rows = await conn.execute_fetchall(
                f"""
                SELECT a.*,
                       GROUP_CONCAT(DISTINCT t.name) AS tags_str
                FROM   actions a
                LEFT   JOIN action_tags at ON at.action_id = a.action_id
                LEFT   JOIN tags        t  ON t.tag_id    = at.tag_id
                WHERE  a.action_id IN ({ph})
                GROUP  BY a.action_id
                """,
                targets,
            )

            if not rows:
                raise ToolInputError(
                    f"No actions found for IDs: {', '.join(targets[:5])}"
                    + ("…" if len(targets) > 5 else ""),
                    param_name="action_id",
                )

            # ───────── dependencies (single query) ─────────
            dep_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
            if include_dependencies:
                dep_rows = await conn.execute_fetchall(
                    """
                    SELECT source_action_id, target_action_id, dependency_type
                    FROM   dependencies
                    WHERE  source_action_id IN ({ph})
                       OR  target_action_id IN ({ph})
                    """.format(ph=ph),
                    targets * 2,
                )
                for r in dep_rows:
                    src, tgt, typ = r
                    dep_map.setdefault(src, {}).setdefault("depends_on", []).append(
                        {"action_id": tgt, "type": typ}
                    )
                    dep_map.setdefault(tgt, {}).setdefault("dependent_actions", []).append(
                        {"action_id": src, "type": typ}
                    )

            # ───────── row post-processing ─────────
            actions: list[dict[str, Any]] = []
            for r in rows:
                a = dict(r)
                a["tags"] = r["tags_str"].split(",") if r["tags_str"] else []
                a.pop("tags_str", None)

                # JSON columns
                for col in ("tool_args", "tool_result"):
                    if a.get(col):
                        a[col] = await MemoryUtils.deserialize(a[col])

                # Dependency attach
                if include_dependencies:
                    a["dependencies"] = dep_map.get(
                        a["action_id"],
                        {"depends_on": [], "dependent_actions": []},
                    )

                _add_iso(a, "started_at", "completed_at")
                actions.append(a)

            elapsed = time.perf_counter() - start
            logger.info(
                f"get_action_details: {len(actions)} actions returned in {elapsed:.3f}s",
                emoji_key="bolt",
            )
            return {"success": True, "data": {"actions": actions}, "processing_time": elapsed}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_action_details failed", exc_info=True)
        raise ToolError(f"Failed to get action details: {exc}") from exc


# ======================================================
# Contextual Summarization
# ======================================================


@with_tool_metrics
@with_error_handling
async def summarize_context_block(
    text_to_summarize: str,
    *,
    target_tokens: int = 500,
    context_type: str = "actions",  # "actions" | "memories" | "thoughts" | …
    workflow_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Compress an arbitrary context blob using the configured LLM provider.

    • Prompt is specialised per *context_type*.
    • Provider/model defaulting mirrors get_workflow_details logic.
    • Logs a `compress_context` entry in *memory_operations* when workflow_id supplied.
    """
    t0 = time.perf_counter()

    if not text_to_summarize:
        raise ToolInputError("Text to summarise cannot be empty.", param_name="text_to_summarize")

    # ───────────────────────── prompt selection ─────────────────────────
    tmpl: str
    if context_type == "actions":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise the following ACTION HISTORY,
retaining IDs and the most salient events.

Focus:
1. State-changing or output-producing actions
2. Failures + error reasons
3. Last 2-3 actions (even if trivial)
4. Actions that created artifacts / memories
5. Sequencing

Aim for ~{target_tokens} tokens.

ACTION HISTORY:
{text}

SUMMARY:
"""
    elif context_type == "memories":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise these MEMORY ENTRIES.

Prioritise:
1. importance > 7
2. confidence > 0.8
3. insights over observations
4. preserve memory IDs
5. linked memories / networks

~{target_tokens} tokens.

MEMORIES:
{text}

SUMMARY:
"""
    elif context_type == "thoughts":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise the following THOUGHT CHAINS.

Prioritise:
1. goals, decisions, conclusions
2. key hypotheses / reflections
3. most recent thoughts
4. preserve thought IDs

~{target_tokens} tokens.

THOUGHTS:
{text}

SUMMARY:
"""
    else:  # generic
        tmpl = """
Summarise the text below for an AI agent. Preserve:
1. recent, goal-relevant info
2. critical state details
3. unique identifiers
4. significant events/insights

Target length: ~{target_tokens} tokens.

TEXT:
{text}

SUMMARY:
"""

    # ───────────────────────── provider / model ─────────────────────────
    cfg = get_config()
    provider_name = provider or cfg.default_provider or LLMGatewayProvider.ANTHROPIC.value
    prov = await get_provider(provider_name)
    if prov is None:
        raise ToolError(f"Provider '{provider_name}' unavailable.")

    model_name = model or prov.get_default_model()
    if model_name is None:  # hard fallback
        fallbacks = {
            LLMGatewayProvider.OPENAI.value: "gpt-3.5-turbo",
            LLMGatewayProvider.ANTHROPIC.value: "claude-3-haiku-20240307",
        }
        model_name = fallbacks.get(provider_name)
        if model_name is None:
            raise ToolError(f"No model specified and no default for provider '{provider_name}'.")

    # ───────────────────────── LLM call ─────────────────────────
    prompt = tmpl.format(text=text_to_summarize, target_tokens=target_tokens).lstrip()
    out = await prov.generate_completion(
        prompt=prompt,
        model=model_name,
        max_tokens=target_tokens + 150,
        temperature=0.2,
    )

    summary = out.text.strip()
    if not summary:
        logger.warning(f"LLM returned empty summary for context_type='{context_type}'.")
        summary = ""

    comp_ratio = len(summary) / max(len(text_to_summarize), 1)

    # ───────────────────────── logging (optional) ───────────────────────
    if workflow_id:
        db = DBConnection(db_path)
        async with db.transaction(mode="IMMEDIATE") as conn:  # write txn
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "compress_context",
                memory_id=None,
                action_id=None,
                operation_data={
                    "context_type": context_type,
                    "original_length": len(text_to_summarize),
                    "summary_length": len(summary),
                    "compression_ratio": comp_ratio,
                    "provider": provider_name,
                    "model": model_name,
                },
            )

    # ───────────────────────── return ─────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Summarised {context_type} context ({len(text_to_summarize)}→{len(summary)} chars, ratio {comp_ratio:.2f}) via {provider_name}/{model_name}",
        emoji_key="compression",
        time=elapsed,  # Pass elapsed as 'time' kwarg
    )
    return {
        "success": True,
        "data": {
            "summary": summary,
            "context_type": context_type,
            "compression_ratio": comp_ratio,
            "processing_time": elapsed,
        },
    }


# ======================================================
# 3.5 Action Dependency Tools
# ======================================================


@with_tool_metrics
@with_error_handling
async def add_action_dependency(
    source_action_id: str,
    target_action_id: str,
    *,
    dependency_type: str = "requires",  # e.g. requires | informs | blocks …
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Register (or confirm) a directed edge between two actions.

    Fix 2025-05-17 (#8): The earlier race where a dependency could be deleted
    after the `INSERT OR IGNORE` but before the follow-up `SELECT` is now
    handled.  We retry the INSERT exactly once inside the same transaction; if
    it still doesn’t appear we raise `ToolError`, guaranteeing the caller never
    receives a payload that claims success without a persisted row.

    • `dependency_type` is normalised (`strip().lower()`).
    • Detects INSERT vs. reuse via `SELECT changes()`.
    • Guarantees a non-NULL `dependency_id` on success.
    """
    # ─── basic validation ───
    if not source_action_id:
        raise ToolInputError("Source action ID required.", param_name="source_action_id")
    if not target_action_id:
        raise ToolInputError("Target action ID required.", param_name="target_action_id")
    if source_action_id == target_action_id:
        raise ToolInputError("Source and target IDs must differ.", param_name="source_action_id")

    dep_type = dependency_type.strip().lower()
    if not dep_type:
        raise ToolInputError("Dependency type cannot be empty.", param_name="dependency_type")

    t0_perf = time.perf_counter()
    now_unix: int = int(time.time())
    dependency_id: Optional[int] = None

    db = DBConnection(db_path)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            # ─── FK checks ───
            src = await conn.execute_fetchone(
                "SELECT workflow_id FROM actions WHERE action_id = ?",
                (source_action_id,),
            )
            if src is None:
                raise ToolInputError(
                    f"Source action {_fmt_id(source_action_id)} not found.",
                    param_name="source_action_id",
                )

            tgt = await conn.execute_fetchone(
                "SELECT workflow_id FROM actions WHERE action_id = ?",
                (target_action_id,),
            )
            if tgt is None:
                raise ToolInputError(
                    f"Target action {_fmt_id(target_action_id)} not found.",
                    param_name="target_action_id",
                )

            if src["workflow_id"] != tgt["workflow_id"]:
                raise ToolInputError(
                    "Source and target actions belong to different workflows.",
                    param_name="target_action_id",
                )

            workflow_id: str = src["workflow_id"]

            # ─── UPSERT (INSERT OR IGNORE) ───
            async def _insert_edge() -> bool:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO dependencies
                        (source_action_id, target_action_id, dependency_type, created_at)
                    VALUES (?,?,?,?)
                    """,
                    (source_action_id, target_action_id, dep_type, now_unix),
                )
                return bool(await conn.execute_fetchval("SELECT changes()"))

            fresh_insert = await _insert_edge()

            if fresh_insert:
                dependency_id = await conn.execute_fetchval("SELECT last_insert_rowid()")
            else:
                dependency_row = await conn.execute_fetchone(
                    """
                    SELECT dependency_id
                    FROM dependencies
                    WHERE source_action_id = ?
                      AND target_action_id = ?
                      AND dependency_type  = ?
                    """,
                    (source_action_id, target_action_id, dep_type),
                )

                # ---------- race-condition patch ----------
                if dependency_row is None:
                    # Edge was deleted after our first INSERT OR IGNORE; try once more.
                    if await _insert_edge():
                        dependency_id = await conn.execute_fetchval("SELECT last_insert_rowid()")
                    else:
                        raise ToolError(
                            "Failed to create dependency edge after concurrent deletion."
                        )
                else:
                    dependency_id = dependency_row["dependency_id"]
            # ─── touch workflow & audit log ───
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            )

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "add_dependency",
                None,  # memory_id
                source_action_id,  # log against source action
                {
                    "target_action_id": target_action_id,
                    "dependency_type": dep_type,
                    "db_dependency_id": dependency_id,
                },
            )

        processing_time = time.perf_counter() - t0_perf
        logger.info(
            f"Dependency '{dep_type}' {_fmt_id(source_action_id)} → "
            f"{_fmt_id(target_action_id)} "
            f"(id={dependency_id if dependency_id is not None else 'exists'}) "
            f"created/verified.",
            emoji_key="link",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "source_action_id": source_action_id,
                "target_action_id": target_action_id,
                "dependency_type": dep_type,
                "dependency_id": dependency_id,
                "created_at_unix": now_unix,
                "created_at_iso": to_iso_z(now_unix),
                "processing_time": processing_time,
            },
        }

    except ToolInputError:
        raise
    except Exception as exc:
        processing_time = time.perf_counter() - t0_perf
        logger.error(
            f"add_action_dependency failed for "
            f"{_fmt_id(source_action_id)}→{_fmt_id(target_action_id)}: {exc}",
            exc_info=True,
            time=processing_time,
        )
        raise ToolError(f"Failed to add action dependency: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_action_dependencies(
    action_id: str,
    *,
    direction: str = "downstream",  # 'downstream' → children, 'upstream' → parents
    dependency_type: str | None = None,  # filter by dependency edge type (case-insensitive)
    include_details: bool = False,  # include extra cols + ISO timestamps
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return the set of actions directly connected to *action_id* via the *dependencies* table.

    • `dependency_type` is normalised (`strip().lower()`) so callers can use any case.
    • Sequence ordering preserved; timestamps decorated with *_iso when requested.
    """
    # ───── basic validation ─────
    if not action_id:
        raise ToolInputError("Action ID required.", param_name="action_id")
    if direction not in {"downstream", "upstream"}:
        raise ToolInputError(
            "Direction must be 'downstream' or 'upstream'.", param_name="direction"
        )

    # normalise dependency_type *exactly* the same way add_action_dependency stores it
    dep_type_norm: str | None = None
    if dependency_type is not None:
        dep_type_norm = dependency_type.strip().lower()
        if not dep_type_norm:
            raise ToolInputError("dependency_type cannot be empty.", param_name="dependency_type")

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    # helper for ISO decoration
    def _iso(obj: dict, keys: tuple[str, ...]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ─── confirm root action exists ───
            if not await conn.execute_fetchone(
                "SELECT 1 FROM actions WHERE action_id = ?", (action_id,)
            ):
                raise ToolInputError(f"Action {action_id} not found.", param_name="action_id")

            # ─── build SELECT list ───
            cols: list[str] = [
                "a.action_id",
                "a.title",
                "dep.dependency_type",
            ]
            if include_details:
                cols += [
                    "a.action_type",
                    "a.status",
                    "a.started_at",
                    "a.completed_at",
                    "a.sequence_number",
                ]

            # ─── choose join and where clauses based on direction ───
            if direction == "downstream":
                join_cond = "dep.source_action_id = a.action_id"
                where_cond = "dep.target_action_id = ?"
            else:  # upstream
                join_cond = "dep.target_action_id = a.action_id"
                where_cond = "dep.source_action_id = ?"

            sql = (
                f"SELECT {', '.join(cols)} "
                "FROM dependencies dep "
                f"JOIN actions a ON {join_cond} "
                f"WHERE {where_cond}"
            )
            params: list[Any] = [action_id]
            if dep_type_norm:
                sql += " AND dep.dependency_type = ?"
                params.append(dep_type_norm)
            sql += " ORDER BY a.sequence_number"

            # ─── fetch ───
            related: list[dict] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    rec = dict(row)
                    if include_details:
                        _iso(rec, ("started_at", "completed_at"))
                    related.append(rec)

        return {
            "success": True,
            "data": {
                "action_id": action_id,
                "direction": direction,
                "dependency_type_applied": dep_type_norm,  # echo back the normalised filter
                "related_actions": related,
                "processing_time": time.perf_counter() - t0,
            },
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_action_dependencies failed", exc_info=True)
        raise ToolError(f"Failed to get action dependencies: {exc}") from exc


# --- 4. Artifact Tracking Tools ---
@with_tool_metrics
@with_error_handling
async def record_artifact(
    workflow_id: str,
    name: str,
    artifact_type: str,
    *,
    action_id: str | None = None,
    description: str | None = None,
    path: str | None = None,
    content: str | None = None,
    metadata: Dict[str, Any] | None = None,
    is_output: bool = False,
    tags: list[str] | None = None,
    idempotency_key: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Persist an artifact row, create a linked episodic memory, and update workflow timestamps.
    """
    # ────────── validation ──────────
    if not name:
        raise ToolInputError("Artifact name required", param_name="name")
    try:
        art_type_enum = ArtifactType(artifact_type.lower())
    except ValueError as err:
        raise ToolInputError(
            f"Invalid artifact_type '{artifact_type}'. "
            f"Expected one of {[t.value for t in ArtifactType]}",
            param_name="artifact_type",
        ) from err

    t_start = time.perf_counter()
    now_unix = int(time.time())
    db = DBConnection(db_path)

    # ────────── idempotency helpers ──────────
    async def _fetch_existing_artifact_details(
        conn_fetch: aiosqlite.Connection,
        existing_artifact_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve a previously-created artifact row plus its derived metadata,
        handling legacy rows whose `metadata` column is NULL.

        Returns
        -------
        Dict[str, Any]
            Payload mirroring the structure produced by the happy-path insert,
            with `"idempotency_hit": True`.
        """
        row = await conn_fetch.execute_fetchone(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (existing_artifact_id,),
        )
        if row is None:
            raise ToolError(
                f"Failed to re-fetch existing artifact {existing_artifact_id} on idempotency hit."
            )

        art: dict = dict(row)

        # --- robust deserialisation -----------------------------------------
        meta_raw = art.get("metadata")
        meta_obj = await MemoryUtils.deserialize(meta_raw)
        if not isinstance(meta_obj, dict):
            meta_obj = {}  # legacy NULL / non-dict payloads
        art["metadata"] = meta_obj

        # --- tags -----------------------------------------------------------
        tag_rows = await conn_fetch.execute_fetchall(
            """
            SELECT t.name
            FROM   tags t
            JOIN   artifact_tags at ON at.tag_id = t.tag_id
            WHERE  at.artifact_id = ?
            """,
            (existing_artifact_id,),
        )
        art["tags"] = [r["name"] for r in tag_rows]

        # --- booleans / convenience casts -----------------------------------
        art["is_output"] = bool(art.get("is_output", False))
        content_in_db = art.get("content") is not None
        content_trunc_flag = bool(meta_obj.get("_content_truncated", False))

        # --- linked memory ---------------------------------------------------
        mem_row = await conn_fetch.execute_fetchone(
            """
            SELECT memory_id
            FROM   memories
            WHERE  artifact_id = ?
              AND  memory_type = ?
            ORDER  BY created_at ASC
            LIMIT  1
            """,
            (existing_artifact_id, MemoryType.ARTIFACT_CREATION.value),
        )
        linked_mem_id = mem_row["memory_id"] if mem_row else None

        return {
            "success": True,
            "data": {
                "artifact_id": existing_artifact_id,
                "linked_memory_id": linked_mem_id,
                "workflow_id": art["workflow_id"],
                "name": art["name"],
                "artifact_type": art["artifact_type"],
                "path": art.get("path"),
                "created_at_unix": art["created_at"],
                "created_at_iso": to_iso_z(art["created_at"]),
                "content_stored_in_db": content_in_db,
                "content_truncated_in_db": content_trunc_flag,
                "is_output": art["is_output"],
                "tags": art["tags"],
                "idempotency_hit": True,
                "processing_time": time.perf_counter() - t_start,
            },
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_chk:
            ex_row = await conn_chk.execute_fetchone(
                "SELECT artifact_id FROM artifacts WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if ex_row:
            async with db.transaction(readonly=True) as conn_det:
                return await _fetch_existing_artifact_details(conn_det, ex_row["artifact_id"])

    # ────────── content byte-cap enforcement (UTF-8 safe) ──────────
    content_truncated = False
    db_content_to_store: str | None = None
    if content is not None:
        content_bytes = content.encode("utf-8", errors="replace")
        max_len = MAX_TEXT_LENGTH
        if len(content_bytes) > max_len:
            # slice to the raw-byte ceiling, then walk back until it is
            # valid UTF-8 -- this avoids the “empty slice → cut[-1]” crash
            cut = content_bytes[:max_len]
            cut_str = ""
            while True:  # exit only on successful decode
                try:
                    cut_str = cut.decode("utf-8")  # 🌱 valid
                    break
                except UnicodeDecodeError:  # 🌱 still split code-point
                    cut = cut[:-1]  # drop the last byte and retry
                    if not cut:  # degenerate safeguard
                        cut_str = ""
                        break
            db_content_to_store = f"{cut_str}[TRUNCATED]"
            content_truncated = True
        else:
            db_content_to_store = content

    # ────────── prepare metadata ──────────
    meta: dict[str, Any] = dict(metadata or {})
    meta["_content_truncated"] = content_truncated
    metadata_json = await MemoryUtils.serialize(meta)

    # ────────── main insert ──────────
    artifact_id = MemoryUtils.generate_id()
    linked_mem_id = MemoryUtils.generate_id()

    async with db.transaction(mode="IMMEDIATE") as conn:
        # FK checks
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(
                f"Workflow {_fmt_id(workflow_id)} not found", param_name="workflow_id"
            )
        if action_id and not await conn.execute_fetchone(
            "SELECT 1 FROM actions WHERE action_id = ? AND workflow_id = ?",
            (action_id, workflow_id),
        ):
            raise ToolInputError(
                f"Action {_fmt_id(action_id)} does not belong to workflow {_fmt_id(workflow_id)}",
                param_name="action_id",
            )

        await conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, workflow_id, action_id, artifact_type, name,
                description, path, content, metadata, created_at,
                is_output, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                artifact_id,
                workflow_id,
                action_id,
                art_type_enum.value,
                name,
                description,
                path,
                db_content_to_store,
                metadata_json,
                now_unix,
                is_output,
                idempotency_key,
            ),
        )

        if tags:
            await MemoryUtils.process_tags(conn, artifact_id, tags, entity_type="artifact")

        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # linked episodic memory
        mem_parts: list[str] = [f"Artifact '{name}' ({art_type_enum.value}) created"]
        if action_id:
            mem_parts.append(f"in action '{_fmt_id(action_id)}'")
        if description:
            mem_parts.append(f"Description: {description[:100]}")
        if path:
            mem_parts.append(f"path: {path}")
        if db_content_to_store is not None:
            size_b = len(db_content_to_store.encode("utf-8"))
            extra = " (truncated)" if content_truncated else ""
            mem_parts.append(f"content stored ({size_b} bytes{extra})")
        else:
            mem_parts.append("no inline content")
        if is_output:
            mem_parts.append("marked as output")
        mem_content = ". ".join(mem_parts) + "."

        await conn.execute(
            """
            INSERT INTO memories (
                memory_id, workflow_id, action_id, artifact_id, content,
                memory_level, memory_type, importance, confidence, tags,
                created_at, updated_at, access_count, last_accessed
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
            """,
            (
                linked_mem_id,
                workflow_id,
                action_id,
                artifact_id,
                mem_content,
                MemoryLevel.EPISODIC.value,
                MemoryType.ARTIFACT_CREATION.value,
                6.0 if is_output else 5.0,
                1.0,
                json.dumps(list({*(tags or []), "artifact_creation", art_type_enum.value})),
                now_unix,
                now_unix,
                0,
            ),
        )

        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_artifact",
            None,
            action_id,
            {
                "artifact_id": artifact_id,
                "name": name,
                "type": art_type_enum.value,
                "linked_memory_id": linked_mem_id,
                "content_truncated": content_truncated,
            },
        )
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_from_artifact",
            linked_mem_id,
            action_id,
            {"artifact_id": artifact_id},
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Artifact '{name}' ({_fmt_id(artifact_id)}, {art_type_enum.value}) stored "
        f"{'with truncation' if content_truncated else 'fully'}; "
        f"linked memory {_fmt_id(linked_mem_id)}.",
        emoji_key="page_facing_up",
        time=elapsed,
    )
    return {
        "success": True,
        "data": {
            "artifact_id": artifact_id,
            "linked_memory_id": linked_mem_id,
            "workflow_id": workflow_id,
            "name": name,
            "artifact_type": art_type_enum.value,
            "path": path,
            "created_at_unix": now_unix,
            "created_at_iso": to_iso_z(now_unix),
            "content_stored_in_db": db_content_to_store is not None,
            "content_truncated_in_db": content_truncated,
            "is_output": is_output,
            "tags": tags or [],
            "idempotency_hit": False,
            "processing_time": elapsed,
        },
    }


# --- 5. Thought & Reasoning Tools ---
@with_tool_metrics
@with_error_handling
async def record_thought(
    workflow_id: str,
    content: str,
    *,
    thought_type: str = "inference",
    thought_chain_id: str | None = None,
    parent_thought_id: str | None = None,
    relevant_action_id: str | None = None,
    relevant_artifact_id: str | None = None,
    relevant_memory_id: str | None = None,
    idempotency_key: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create (or idempotently return) a thought and optionally promote it
    to semantic memory.  Guaranteed to always include a *non-None*
    `linked_memory_id` key in the response.

    Fix: prevent `None`-deref by using an empty-string sentinel when no
    memory promotion occurs.
    """
    # ─── validation with lenient content handling ───
    if not content or not isinstance(content, str):
        raise ToolInputError("Thought content must be a non-empty string", "content")

    # Sanitize complex content before validation
    original_content = content
    if content:
        # Handle JSON-like content that might be malformed plan data
        if content.strip().startswith("{") and '"tool"' in content:
            # This is likely a malformed plan JSON being recorded as thought
            logger.debug(
                f"Detected JSON-like content in thought, truncating for safety: {content[:100]}..."
            )
            content = f"MALFORMED_PLAN_JSON (truncated): {content[:200]}..."

        # Ensure content doesn't exceed reasonable length limits for thoughts
        max_thought_length = 2000  # Reasonable limit for thought content
        if len(content) > max_thought_length:
            logger.debug(
                f"Truncating long thought content from {len(content)} to {max_thought_length} chars"
            )
            content = content[: max_thought_length - 3] + "..."

        # Additional safety: if content contains complex nested structures, simplify
        if content.count("{") > 5 or content.count("[") > 5:
            # Likely complex nested data, create a summary instead
            content_preview = content[:150].replace("\n", " ").replace("\r", " ")
            content = f"COMPLEX_DATA_SUMMARY: {content_preview}... [Original length: {len(original_content)} chars]"

    try:
        thought_type_enum = ThoughtType(thought_type.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid thought_type '{thought_type}'. "
            f"Must be one of: {', '.join(t.value for t in ThoughtType)}",
            "thought_type",
        ) from exc

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db_main = DBConnection(db_path)

    # ─── helper: fetch existing for idempotency ───
    async def _existing_payload(c: aiosqlite.Connection, th_id: str, ch_id: str) -> Dict[str, Any]:
        row = await c.execute_fetchone("SELECT * FROM thoughts WHERE thought_id=?", (th_id,))
        if not row:
            raise ToolError(f"Thought {th_id} vanished during idempotency resolution.")
        mem_row = await c.execute_fetchone(
            "SELECT memory_id FROM memories "
            "WHERE thought_id=? AND memory_type=? ORDER BY created_at LIMIT 1",
            (th_id, MemoryType.REASONING_STEP.value),
        )
        return {
            "success": True,
            "data": {
                "thought_id": th_id,
                "thought_chain_id": ch_id,
                "thought_type": row["thought_type"],
                "content": row["content"],
                "sequence_number": row["sequence_number"],
                "created_at": to_iso_z(row["created_at"]),
                "linked_memory_id": mem_row["memory_id"] if mem_row else "",
                "idempotency_hit": True,
                "processing_time": time.perf_counter() - t0_perf,
            },
        }

    # ─── inner transactional worker ───
    async def _tx(db_conn: aiosqlite.Connection) -> tuple[str, str, int, str]:
        """Return (chain_id, thought_id, seq_no, linked_mem_id_or_empty)."""
        # FK checks -----------------------------------------------------------
        _PK = {
            "workflows": "workflow_id",
            "thoughts": "thought_id",
            "actions": "action_id",
            "artifacts": "artifact_id",
            "memories": "memory_id",
            "thought_chains": "thought_chain_id",
        }

        async def _exists(table: str, key: Optional[str], pname: str) -> None:
            if not key:
                return
            tbl = MemoryUtils._validate_sql_identifier(table, "table")
            pk = _PK[tbl]
            sql = (
                f"SELECT 1 FROM {tbl} WHERE {pk}=?"
                if tbl == "workflows"
                else f"SELECT 1 FROM {tbl} WHERE {pk}=? AND workflow_id=?"
            )
            params = (key,) if tbl == "workflows" else (key, workflow_id)
            if await db_conn.execute_fetchone(sql, params) is None:
                raise ToolInputError(f"{pname.replace('_', ' ').title()} not found: {key}", pname)

        await _exists("workflows", workflow_id, "workflow_id")
        await _exists("thoughts", parent_thought_id, "parent_thought_id")
        await _exists("actions", relevant_action_id, "relevant_action_id")
        await _exists("artifacts", relevant_artifact_id, "relevant_artifact_id")
        await _exists("memories", relevant_memory_id, "relevant_memory_id")
        if thought_chain_id:
            await _exists("thought_chains", thought_chain_id, "thought_chain_id")

        # chain resolve / create ---------------------------------------------
        if thought_chain_id:
            chain_id = thought_chain_id
        else:
            row = await db_conn.execute_fetchone(
                "SELECT thought_chain_id FROM thought_chains "
                "WHERE workflow_id=? ORDER BY created_at LIMIT 1",
                (workflow_id,),
            )
            if row:
                chain_id = row["thought_chain_id"]
            else:
                chain_id = MemoryUtils.generate_id()
                await db_conn.execute(
                    "INSERT INTO thought_chains "
                    "(thought_chain_id,workflow_id,title,created_at) "
                    "VALUES (?,?,?,?)",
                    (chain_id, workflow_id, "Main reasoning", now_unix),
                )

        # idempotency check ---------------------------------------------------
        if idempotency_key:
            r = await db_conn.execute_fetchone(
                "SELECT thought_id FROM thoughts WHERE thought_chain_id=? AND idempotency_key=?",
                (chain_id, idempotency_key),
            )
            if r:
                raise ToolError(f"IDEMPOTENCY_HIT:{r['thought_id']}:{chain_id}")

        # insert thought ------------------------------------------------------
        thought_id = MemoryUtils.generate_id()
        seq_no = await MemoryUtils.get_next_sequence_number(
            db_conn, chain_id, "thoughts", "thought_chain_id"
        )
        await db_conn.execute(
            "INSERT INTO thoughts (thought_id,thought_chain_id,parent_thought_id,thought_type,"
            "content,sequence_number,created_at,relevant_action_id,relevant_artifact_id,"
            "relevant_memory_id,idempotency_key) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                thought_id,
                chain_id,
                parent_thought_id,
                thought_type_enum.value,
                content,
                seq_no,
                now_unix,
                relevant_action_id,
                relevant_artifact_id,
                relevant_memory_id,
                idempotency_key,
            ),
        )
        await db_conn.execute(
            "UPDATE workflows SET updated_at=?,last_active=? WHERE workflow_id=?",
            (now_unix, now_unix, workflow_id),
        )

        # optional memory promotion ------------------------------------------
        linked_mem_id = ""
        if thought_type_enum in {
            ThoughtType.GOAL,
            ThoughtType.DECISION,
            ThoughtType.SUMMARY,
            ThoughtType.REFLECTION,
            ThoughtType.HYPOTHESIS,
            ThoughtType.INSIGHT,
            ThoughtType.REASONING,
            ThoughtType.ANALYSIS,
        }:
            linked_mem_id = MemoryUtils.generate_id()
            await db_conn.execute(
                "INSERT INTO memories (memory_id,workflow_id,thought_id,content,memory_level,"
                "memory_type,importance,confidence,tags,created_at,updated_at,access_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    linked_mem_id,
                    workflow_id,
                    thought_id,
                    f"Thought [{seq_no}] ({thought_type_enum.value.title()}): {content}",
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.REASONING_STEP.value,
                    7.5 if thought_type_enum in {ThoughtType.GOAL, ThoughtType.DECISION} else 6.5,
                    1.0,
                    json.dumps(["reasoning", thought_type_enum.value]),
                    now_unix,
                    now_unix,
                ),
            )
            await MemoryUtils._log_memory_operation(
                db_conn,
                workflow_id,
                "create_from_thought",
                linked_mem_id,
                None,
                {"thought_id": thought_id},
            )

        # Record cognitive state change in timeline
        try:
            await _record_cognitive_timeline_state(
                db_conn,
                workflow_id,
                CognitiveStateType.THOUGHT_RECORDED,
                {
                    "thought_id": thought_id,
                    "thought_type": thought_type_enum.value,
                    "thought_chain_id": chain_id,
                    "sequence_number": seq_no,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "memory_promoted": bool(linked_mem_id),
                },
                f"Recorded {thought_type_enum.value} thought",
            )
        except Exception as e:
            logger.warning(f"Failed to record cognitive state for thought creation: {e}")
        return chain_id, thought_id, seq_no, linked_mem_id

    # ─── transaction orchestration ───
    idemp_hit = False
    try:
        if conn is not None:
            if not isinstance(conn, aiosqlite.Connection):
                raise ToolError("conn must be an aiosqlite.Connection")
            chain_id, thought_id, seq_no, mem_id = await _tx(conn)
        else:
            async with db_main.transaction(mode="IMMEDIATE") as tx_conn:
                chain_id, thought_id, seq_no, mem_id = await _tx(tx_conn)
    except ToolError as exc:
        if str(exc).startswith("IDEMPOTENCY_HIT:"):
            idemp_hit = True
            _, thought_id, chain_id = str(exc).split(":")
        else:
            raise

    # ─── idempotency return path ───
    if idemp_hit:
        async with db_main.transaction(readonly=True) as c:
            return await _existing_payload(c, thought_id, chain_id)

    # ─── success payload ───
    return {
        "success": True,
        "data": {
            "thought_id": thought_id,
            "thought_chain_id": chain_id,
            "thought_type": thought_type_enum.value,
            "content": content,
            "sequence_number": seq_no,
            "created_at": to_iso_z(now_unix),
            "linked_memory_id": mem_id,  # always non-None (may be "")
            "idempotency_hit": False,
            "processing_time": time.perf_counter() - t0_perf,
        },
    }


# --- 6. Core Memory Tools ---
@with_tool_metrics
@with_error_handling
async def store_memory(
    workflow_id: str,
    content: str,
    memory_type: str,
    *,
    memory_level: str = MemoryLevel.EPISODIC.value,
    importance: float = 5.0,
    confidence: float = 1.0,
    description: str | None = None,
    reasoning: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    ttl: int | None = None,
    context_data: dict[str, Any] | None = None,
    generate_embedding: bool = True,
    suggest_links: bool = True,
    link_suggestion_threshold: float = agent_memory_config.similarity_threshold,
    max_suggested_links: int = 3,
    action_id: str | None = None,
    thought_id: str | None = None,
    artifact_id: str | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    if not content:
        raise ToolInputError("Content cannot be empty.", param_name="content")
    try:
        mem_type_enum = MemoryType(memory_type.lower())
    except ValueError as e:
        raise ToolInputError(
            f"Invalid memory_type. Use one of: {', '.join(mt.value for mt in MemoryType)}",
            param_name="memory_type",
        ) from e
    try:
        mem_level_enum = MemoryLevel(memory_level.lower())
    except ValueError as e:
        raise ToolInputError(
            f"Invalid memory_level. Use one of: {', '.join(ml.value for ml in MemoryLevel)}",
            param_name="memory_level",
        ) from e
    if not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0–10.0.", param_name="importance")

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_memory_details(
        conn_fetch: aiosqlite.Connection,
        existing_memory_id: str,
        wf_id: str,
    ) -> Dict[str, Any]:
        """
        Return a clean, backward-compatible payload for a store_memory()
        idempotency hit.

        • Guarantees `tags` is **always** a list of unique, lowercase strings,
          even when legacy rows stored a scalar or malformed JSON.
        • Preserves every original field used by callers; no behaviour removed.
        """
        # ─── reload full row ───
        mem_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM memories WHERE memory_id = ?",
            (existing_memory_id,),
        )
        if mem_row is None:  # highly unlikely but defensive
            raise ToolError(
                f"Failed to re-fetch existing memory {existing_memory_id} on idempotency hit."
            )

        mem_data: Dict[str, Any] = dict(mem_row)

        # ─── robust tag normalisation ───
        raw_tags = await MemoryUtils.deserialize(mem_data.get("tags"))
        norm_tags: list[str] = []
        if isinstance(raw_tags, list):
            seen: set[str] = set()
            for t in raw_tags:
                if not isinstance(t, str):
                    t = str(t)
                tl = t.strip().lower()
                if tl and tl not in seen:
                    seen.add(tl)
                    norm_tags.append(tl)
        elif isinstance(raw_tags, str):
            t = raw_tags.strip().lower()
            if t:
                norm_tags.append(t)
        # anything else (None, dict, number…) is discarded as corrupt
        mem_data["tags"] = norm_tags

        # ─── build return payload ───
        return {
            "success": True,
            "data": {
                "idempotency_hit": True,
                "memory_id": existing_memory_id,
                "workflow_id": wf_id,
                "memory_level": mem_data["memory_level"],
                "memory_type": mem_data["memory_type"],
                "content_preview": (
                    mem_data["content"][:100] + "…"
                    if len(mem_data["content"]) > 100
                    else mem_data["content"]
                ),
                "importance": mem_data["importance"],
                "confidence": mem_data["confidence"],
                "created_at": to_iso_z(mem_data["created_at"]),
                "tags": norm_tags,
                "embedding_id": mem_data.get("embedding_id"),
                "linked_action_id": mem_data.get("action_id"),
                "linked_thought_id": mem_data.get("thought_id"),
                "linked_artifact_id": mem_data.get("artifact_id"),
                "suggested_links": [],  # never recompute on hit
                "processing_time": time.perf_counter() - t0_perf,
            },
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_mem_row = await conn_check.execute_fetchone(
                "SELECT memory_id FROM memories WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_mem_row:
            existing_memory_id = existing_mem_row["memory_id"]
            logger.info(
                f"Idempotency hit for store_memory (key='{idempotency_key}'). Returning existing memory {_fmt_id(existing_memory_id)}."
            )
            async with db.transaction(readonly=True) as conn_details:  # Read-only for fetching
                # Since an idempotency hit implies the memory and its embedding (if any) were already processed,
                # we just return its identifier and core metadata.
                # For a more complete return matching a new store, we'd need to fetch links etc.
                # For now, a simpler payload for idempotency hit.
                return await _fetch_existing_memory_details(
                    conn_details, existing_memory_id, workflow_id
                )

    # No idempotency hit or no key, proceed to create new memory
    memory_id_new = MemoryUtils.generate_id()
    # ... (tag normalization, TTL logic remains the same) ...
    base_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
    final_tags = list({*base_tags, mem_type_enum.value, mem_level_enum.value})
    final_tags_json = json.dumps(final_tags)
    if ttl is None:
        ttl = {
            MemoryLevel.WORKING: agent_memory_config.ttl_working,
            MemoryLevel.EPISODIC: agent_memory_config.ttl_episodic,
        }.get(mem_level_enum, 0)
    else:
        ttl = int(ttl)

    embed_id: str | None = None
    suggested_links_new: list[dict[str, Any]] = []

    async with db.transaction(mode="IMMEDIATE") as conn:
        # 1️⃣  Foreign-key existence checks  ----------------------------------
        _PK_MAP = {
            "workflows": "workflow_id",
            "actions": "action_id",
            "thoughts": "thought_id",
            "artifacts": "artifact_id",
            "memories": "memory_id",
        }

        async def _check_fk(table: str, key_val: Optional[str]) -> None:
            if not key_val:
                return
            tbl = MemoryUtils._validate_sql_identifier(table, "table")
            pk = _PK_MAP.get(tbl)
            if pk is None:
                raise ToolError(f"Unknown table '{tbl}' in FK check.")

            sql, params = (
                (f"SELECT 1 FROM {tbl} WHERE {pk} = ?", (key_val,))
                if tbl == "workflows"
                else (
                    f"SELECT 1 FROM {tbl} WHERE {pk} = ? AND workflow_id = ?",
                    (key_val, workflow_id),
                )
            )
            if await conn.execute_fetchone(sql, params) is None:
                raise ToolInputError(
                    f"{tbl[:-1].capitalize()} {key_val} not found in workflow {workflow_id}.",
                    param_name=f"{tbl[:-1]}_id",
                )

        # -- check all referenced IDs ----------------------------------------
        await _check_fk("workflows", workflow_id)
        await _check_fk("actions", action_id)
        await _check_fk("thoughts", thought_id)
        await _check_fk("artifacts", artifact_id)

        # 2️⃣  INSERT memory
        await conn.execute(
            """INSERT INTO memories (memory_id, workflow_id, content, memory_level, memory_type, importance, confidence, description, reasoning, source, context, tags, created_at, updated_at, last_accessed, access_count, ttl, action_id, thought_id, artifact_id, embedding_id, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",  # MODIFIED: Added idempotency_key
            (
                memory_id_new,
                workflow_id,
                content,
                mem_level_enum.value,
                mem_type_enum.value,
                importance,
                confidence,
                description or "",
                reasoning or "",
                source or "",
                await MemoryUtils.serialize(context_data) if context_data else "{}",
                final_tags_json,
                now_unix,
                now_unix,
                None,
                0,
                ttl,
                action_id,
                thought_id,
                artifact_id,
                idempotency_key,
            ),  # MODIFIED: Added value
        )

        # 3️⃣  Embedding
        if generate_embedding:
            try:
                embed_id = await _store_embedding(
                    conn, memory_id_new, f"{description}: {content}" if description else content
                )
            except Exception as e_embed:
                logger.error(f"Embedding failed for {memory_id_new}: {e_embed}", exc_info=True)

        # 4️⃣  Link suggestions
        if suggest_links and embed_id and max_suggested_links:
            try:
                sims = await _find_similar_memories(
                    conn=conn,
                    query_text=content,
                    workflow_id=workflow_id,
                    limit=max_suggested_links + 1,
                    threshold=link_suggestion_threshold,
                )
                target_ids = [mid for mid, _ in sims if mid != memory_id_new][:max_suggested_links]
                if target_ids:
                    ph = ",".join("?" * len(target_ids))
                    rows = await conn.execute_fetchall(
                        f"""
                        SELECT memory_id,
                               description,
                               memory_type
                        FROM   memories
                        WHERE  memory_id IN ({ph})
                        """,
                        target_ids,
                    )
                    score_map = dict(sims)

                    for row in rows:
                        m_id = row["memory_id"]
                        sim = round(score_map.get(m_id, 0.0), 4)
                        tgt_type_str = row["memory_type"]

                        # normalise to enum for *consistent* comparisons
                        try:
                            tgt_type_enum: MemoryType = MemoryType(tgt_type_str.lower())
                        except ValueError:  # corrupted / legacy row
                            tgt_type_enum = MemoryType.TEXT

                        # --- relationship inference ---------------------------------
                        link_type_val = LinkType.RELATED.value

                        # same-type episodic memories tend to be sequential
                        if (
                            mem_type_enum == tgt_type_enum
                            and mem_level_enum == MemoryLevel.EPISODIC
                        ):
                            link_type_val = LinkType.SEQUENTIAL.value

                        # insight → fact generalises
                        elif (
                            mem_type_enum == MemoryType.INSIGHT and tgt_type_enum == MemoryType.FACT
                        ):
                            link_type_val = LinkType.GENERALIZES.value

                        # populate suggestion list
                        suggested_links_new.append(
                            {
                                "target_memory_id": m_id,
                                "target_description": row["description"],
                                "target_type": tgt_type_enum.value,
                                "similarity": sim,
                                "suggested_link_type": link_type_val,
                            }
                        )
            except Exception as e_link:
                logger.error(
                    f"Link-suggestion failure for {memory_id_new}: {e_link}", exc_info=True
                )

        # 5️⃣  Touch workflow
        await conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now_unix, now_unix, workflow_id),
        )

        # 6️⃣  Operation log
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create",
            memory_id_new,
            action_id,
            {
                "memory_level": mem_level_enum.value,
                "memory_type": mem_type_enum.value,
                "importance": importance,
                "embedding_generated": bool(embed_id),
                "links_suggested": len(suggested_links_new),
                "tags": final_tags,
            },
        )

        # 7️⃣  Record cognitive state change in timeline
        try:
            await _record_cognitive_timeline_state(
                conn,
                workflow_id,
                CognitiveStateType.MEMORY_STORED,
                {
                    "memory_id": memory_id_new,
                    "memory_type": mem_type_enum.value,
                    "memory_level": mem_level_enum.value,
                    "importance": importance,
                    "confidence": confidence,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "has_embedding": bool(embed_id),
                    "links_suggested": len(suggested_links_new),
                },
                f"Stored {mem_type_enum.value} memory (importance: {importance})",
            )
        except Exception as e:
            logger.warning(f"Failed to record cognitive state for memory storage: {e}")

    # 7️⃣  Response
    elapsed_time = time.perf_counter() - t0_perf
    logger.info(
        f"Memory {memory_id_new} stored; {len(suggested_links_new)} links suggested.",
        emoji_key="floppy_disk",
        time=elapsed_time,
    )
    return {
        "success": True,
        "data": {
            "idempotency_hit": False,
            "memory_id": memory_id_new,
            "workflow_id": workflow_id,
            "memory_level": mem_level_enum.value,
            "memory_type": mem_type_enum.value,
            "content_preview": content[:100] + ("…" if len(content) > 100 else ""),
            "importance": importance,
            "confidence": confidence,
            "created_at": to_iso_z(now_unix),
            "tags": final_tags,
            "embedding_id": embed_id,
            "linked_action_id": action_id,
            "linked_thought_id": thought_id,
            "linked_artifact_id": artifact_id,
            "suggested_links": suggested_links_new,
            "processing_time": elapsed_time,
        },
    }


@with_tool_metrics
@with_error_handling
async def get_memory_by_id(
    memory_id: str,
    *,
    include_links: bool = True,
    include_context: bool = True,
    context_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single memory row and its optional graph / semantic context.

    • Access statistics are updated inside the R/W transaction.
    • TTL-expired rows are deleted atomically and reported as errors.
    • All integer timestamps are preserved; ISO strings are appended as *_iso.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    t0 = time.time()
    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction() as conn:  # R/W IMMEDIATE txn
            mem_row = await conn.execute_fetchone(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            )
            if mem_row is None:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

            mem: Dict[str, Any] = dict(mem_row)

            ttl = mem.get("ttl", 0)
            if ttl and mem["created_at"] + ttl <= int(time.time()):
                logger.warning(f"Memory {memory_id} expired; deleting.", emoji_key="wastebasket")
                await conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                raise ToolError(f"Memory {memory_id} has expired and was deleted.")

            mem["tags"] = await MemoryUtils.deserialize(mem.get("tags"))
            mem["context"] = await MemoryUtils.deserialize(mem.get("context"))

            # These operations modify the database:
            await MemoryUtils._update_memory_access(conn, memory_id)
            await MemoryUtils._log_memory_operation(
                conn, mem["workflow_id"], "access_by_id", memory_id
            )

            if include_links:
                mem["outgoing_links"], mem["incoming_links"] = [], []
                # Using execute_fetchall as these are typically small, bounded queries for a single memory_id
                outgoing_rows = await conn.execute_fetchall(
                    """
                    SELECT ml.*, m.description AS target_description,
                           m.memory_type AS target_type
                    FROM memory_links ml
                    JOIN memories m ON m.memory_id = ml.target_memory_id
                    WHERE ml.source_memory_id = ?
                    """,
                    (memory_id,),
                )
                for r in outgoing_rows:
                    link = dict(r)
                    _add_iso(link, ["created_at"])  # Keep original int, add iso
                    # No need for created_at_unix if _add_iso handles it
                    mem["outgoing_links"].append(
                        {
                            "link_id": link["link_id"],
                            "target_memory_id": link["target_memory_id"],
                            "target_description": link["target_description"],
                            "target_type": link["target_type"],
                            "link_type": link["link_type"],
                            "strength": link["strength"],
                            "description": link["description"],
                            "created_at": link["created_at_iso"],  # Use ISO formatted
                        }
                    )

                incoming_rows = await conn.execute_fetchall(
                    """
                    SELECT ml.*, m.description AS source_description,
                           m.memory_type AS source_type
                    FROM memory_links ml
                    JOIN memories m ON m.memory_id = ml.source_memory_id
                    WHERE ml.target_memory_id = ?
                    """,
                    (memory_id,),
                )
                for r in incoming_rows:
                    link = dict(r)
                    _add_iso(link, ["created_at"])
                    mem["incoming_links"].append(
                        {
                            "link_id": link["link_id"],
                            "source_memory_id": link["source_memory_id"],
                            "source_description": link["source_description"],
                            "source_type": link["source_type"],
                            "link_type": link["link_type"],
                            "strength": link["strength"],
                            "description": link["description"],
                            "created_at": link["created_at_iso"],
                        }
                    )
            else:
                mem["outgoing_links"] = mem["incoming_links"] = []

            mem["semantic_context"] = []
            if include_context and mem.get("embedding_id"):
                query = (mem.get("description", "") or "") + ": " + (mem.get("content", "") or "")
                if query.strip():
                    try:
                        sims = await _find_similar_memories(
                            conn=conn,
                            query_text=query,
                            workflow_id=mem["workflow_id"],
                            limit=context_limit
                            + 1,  # Fetch one extra to ensure we can exclude self
                            threshold=agent_memory_config.similarity_threshold * 0.9,
                        )
                        if sims:
                            ids = [i for i, _ in sims if i != memory_id][:context_limit]
                            if ids:
                                ph = ",".join("?" * len(ids))
                                ctx_rows = await conn.execute_fetchall(
                                    f"""
                                    SELECT memory_id, description, memory_type, importance
                                    FROM memories WHERE memory_id IN ({ph})
                                    """,
                                    ids,
                                )
                                score_map = dict(sims)  # Renamed for clarity
                                for r_ctx in sorted(  # Renamed r to r_ctx
                                    ctx_rows,
                                    key=lambda x: score_map.get(
                                        x["memory_id"], 0.0
                                    ),  # Use .get() for safety
                                    reverse=True,
                                ):
                                    mem["semantic_context"].append(
                                        {
                                            "memory_id": r_ctx["memory_id"],
                                            "description": r_ctx["description"],
                                            "memory_type": r_ctx["memory_type"],
                                            "importance": r_ctx["importance"],
                                            "similarity": round(
                                                score_map.get(r_ctx["memory_id"], 0.0), 4
                                            ),  # Use .get()
                                        }
                                    )
                    except Exception as err:
                        logger.warning(
                            f"Semantic context lookup failed for memory {_fmt_id(memory_id)}: {err}",
                            exc_info=True,
                        )

            _add_iso(mem, ["created_at", "updated_at", "last_accessed"])

        processing_time = time.time() - t0
        logger.info(
            f"Memory {_fmt_id(memory_id)} retrieved (links={include_links}, ctx={include_context}) in {processing_time:.3f}s",
            emoji_key="inbox_tray",
        )
        return {
            "success": True,
            "data": mem,
            "processing_time": processing_time,
        }

    except ToolError as te:  # Catch ToolError if memory expired and was deleted
        # Log as info because it's an expected outcome for expired memory
        logger.info(f"get_memory_by_id({_fmt_id(memory_id)}): {te}")
        # Re-raise to signal failure to the caller as the memory is gone
        raise
    except ToolInputError:
        raise  # Let the decorator handle this
    except Exception as exc:
        logger.error(f"get_memory_by_id({_fmt_id(memory_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get memory {memory_id}: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_memory_metadata(
    workflow_id: str,
    memory_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve the context JSON blob associated with a specific memory.

    Args:
        workflow_id: Required workflow scope
        memory_id: Required memory ID
        db_path: Database path

    Returns:
        Dict containing success status and the memory's metadata
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not memory_id or not isinstance(memory_id, str):
        raise ToolInputError("memory_id is required and must be a string.", param_name="memory_id")

    ts_start = time.perf_counter()
    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Fetch memory context
            memory_row = await conn.execute_fetchone(
                "SELECT context FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (memory_id, workflow_id),
            )

            if not memory_row:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            # Deserialize the context field
            context_json = memory_row["context"]
            metadata = await MemoryUtils.deserialize(context_json)

            # If deserialization fails or returns None, use empty dict
            if metadata is None:
                metadata = {}

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Retrieved metadata for memory {memory_id} in workflow {workflow_id}",
            emoji_key="gear",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "memory_id": memory_id,
                "metadata": metadata,
                "processing_time": processing_time,
            },
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve memory metadata: {e}") from e


@with_tool_metrics
@with_error_handling
async def get_memory_tags(
    workflow_id: str,
    memory_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve the list of tags for a specific memory.

    Args:
        workflow_id: Required workflow scope
        memory_id: Required memory ID
        db_path: Database path

    Returns:
        Dict containing success status and the memory's tags
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not memory_id or not isinstance(memory_id, str):
        raise ToolInputError("memory_id is required and must be a string.", param_name="memory_id")

    ts_start = time.perf_counter()
    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Fetch memory tags
            memory_row = await conn.execute_fetchone(
                "SELECT tags FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (memory_id, workflow_id),
            )

            if not memory_row:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            # Deserialize the tags field
            tags_json = memory_row["tags"]
            tags = await MemoryUtils.deserialize(tags_json)

            # Ensure tags is a list
            if tags is None:
                tags = []
            elif not isinstance(tags, list):
                # If it's not a list, try to convert or wrap it
                if isinstance(tags, str):
                    tags = [tags]
                else:
                    tags = []

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Retrieved {len(tags)} tags for memory {memory_id} in workflow {workflow_id}",
            emoji_key="tag",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {"memory_id": memory_id, "tags": tags, "processing_time": processing_time},
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory tags: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve memory tags: {e}") from e


@with_tool_metrics
@with_error_handling
async def update_memory_metadata(
    workflow_id: str,
    memory_id: str,
    metadata: Dict[str, Any],
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Update the entire context JSON blob for a specific memory.

    Args:
        workflow_id: Required workflow scope
        memory_id: Required memory ID
        metadata: The new, complete metadata object to store (replaces existing metadata)
        db_path: Database path

    Returns:
        Dict containing success status and update information
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not memory_id or not isinstance(memory_id, str):
        raise ToolInputError("memory_id is required and must be a string.", param_name="memory_id")

    if not isinstance(metadata, dict):
        raise ToolInputError("metadata must be a dictionary.", param_name="metadata")

    ts_start = time.perf_counter()
    db = DBConnection(db_path)

    try:
        # Serialize metadata to JSON
        serialized_metadata = await MemoryUtils.serialize(metadata)
        if serialized_metadata is None:
            raise ToolInputError("Failed to serialize metadata to JSON.", param_name="metadata")

        now = int(time.time())

        async with db.transaction() as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Update memory context
            result = await conn.execute(
                """
                UPDATE memories 
                SET context = ?, updated_at = ? 
                WHERE memory_id = ? AND workflow_id = ?
                """,
                (serialized_metadata, now, memory_id, workflow_id),
            )

            if result.rowcount == 0:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            # Log the metadata update operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "metadata_update",
                memory_id,
                None,
                {"metadata_keys": list(metadata.keys()), "metadata_size": len(serialized_metadata)},
            )

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Updated metadata for memory {memory_id} in workflow {workflow_id}",
            emoji_key="gear",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "memory_id": memory_id,
                "updated_at_iso": to_iso_z(now),
                "processing_time": processing_time,
            },
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error updating memory metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to update memory metadata: {e}") from e


@with_tool_metrics
@with_error_handling
async def update_memory_link_metadata(
    workflow_id: str,
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,
    metadata: Dict[str, Any],
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Update the description field of a specific memory_links row with new JSON metadata.

    Args:
        workflow_id: Required workflow scope for validation
        source_memory_id: Required source memory ID
        target_memory_id: Required target memory ID
        link_type: Required link type (case-insensitive)
        metadata: The new, complete metadata object to store in link description
        db_path: Database path

    Returns:
        Dict containing success status and update information
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not source_memory_id or not isinstance(source_memory_id, str):
        raise ToolInputError(
            "source_memory_id is required and must be a string.", param_name="source_memory_id"
        )

    if not target_memory_id or not isinstance(target_memory_id, str):
        raise ToolInputError(
            "target_memory_id is required and must be a string.", param_name="target_memory_id"
        )

    if not link_type or not isinstance(link_type, str):
        raise ToolInputError("link_type is required and must be a string.", param_name="link_type")

    if not isinstance(metadata, dict):
        raise ToolInputError("metadata must be a dictionary.", param_name="metadata")

    ts_start = time.perf_counter()

    # Normalize link_type to uppercase for consistency
    normalized_link_type = link_type.upper()

    db = DBConnection(db_path)

    try:
        # Serialize metadata to JSON
        serialized_metadata = await MemoryUtils.serialize(metadata)
        if serialized_metadata is None:
            raise ToolInputError("Failed to serialize metadata to JSON.", param_name="metadata")

        async with db.transaction() as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Validate that both memories belong to the workflow
            source_memory_row = await conn.execute_fetchone(
                "SELECT 1 FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (source_memory_id, workflow_id),
            )
            if not source_memory_row:
                raise ToolInputError(
                    f"Source memory {source_memory_id} not found in workflow {workflow_id}.",
                    param_name="source_memory_id",
                )

            target_memory_row = await conn.execute_fetchone(
                "SELECT 1 FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (target_memory_id, workflow_id),
            )
            if not target_memory_row:
                raise ToolInputError(
                    f"Target memory {target_memory_id} not found in workflow {workflow_id}.",
                    param_name="target_memory_id",
                )

            # Update memory link metadata
            result = await conn.execute(
                """
                UPDATE memory_links 
                SET description = ? 
                WHERE source_memory_id = ? AND target_memory_id = ? AND UPPER(link_type) = ?
                """,
                (serialized_metadata, source_memory_id, target_memory_id, normalized_link_type),
            )

            if result.rowcount == 0:
                raise ToolInputError(
                    f"Memory link from {source_memory_id} to {target_memory_id} with type {normalized_link_type} not found.",
                    param_name="link_type",
                )

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Updated link metadata for {source_memory_id} -> {target_memory_id} (type: {normalized_link_type}) in workflow {workflow_id}",
            emoji_key="link",
            time=processing_time,
        )

        return {
            "data": {"message": "Link metadata updated successfully."},
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error updating memory link metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to update memory link metadata: {e}") from e


@with_tool_metrics
@with_error_handling
async def decay_link_strengths(
    workflow_id: Optional[str] = None,
    *,
    half_life_days: int = 14,
    decay_factor: float = 0.95,
    min_strength_threshold: float = 0.1,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Periodically reduce the strength of older memory links to simulate forgetting.

    Args:
        workflow_id: Optional workflow scope. If provided, decay links where at least
                    one linked memory belongs to this workflow
        half_life_days: Time period after which strength decays (default: 14 days)
        decay_factor: Multiplicative decay factor (default: 0.95)
        min_strength_threshold: Links below this strength will be pruned (default: 0.1)
        db_path: Database path

    Returns:
        Dict containing decay statistics
    """
    start_time = time.time()

    # Input validation
    if workflow_id is not None:
        workflow_id = _validate_uuid_format(workflow_id, "workflow_id")

    if half_life_days <= 0:
        raise ToolInputError("half_life_days must be positive", param_name="half_life_days")

    if not (0.0 < decay_factor <= 1.0):
        raise ToolInputError("decay_factor must be between 0.0 and 1.0", param_name="decay_factor")

    if min_strength_threshold < 0.0:
        raise ToolInputError(
            "min_strength_threshold cannot be negative", param_name="min_strength_threshold"
        )

    logger.info(
        f"Decaying link strengths (half_life: {half_life_days} days, factor: {decay_factor})"
    )

    db = DBConnection(db_path)

    try:
        async with db.transaction() as conn:
            # Validate workflow exists if provided
            if workflow_id:
                workflow_row = await conn.execute_fetchone(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
                )
                if not workflow_row:
                    raise ToolInputError(
                        f"Workflow {workflow_id} not found.", param_name="workflow_id"
                    )

            # Calculate cutoff timestamp
            now_unix = int(time.time())
            cutoff_timestamp = now_unix - (half_life_days * 86400)  # 86400 seconds per day

            # Build decay query
            if workflow_id:
                # Decay links where at least one memory belongs to the workflow
                decay_sql = """
                    UPDATE memory_links 
                    SET strength = strength * ? 
                    WHERE created_at < ? 
                    AND (source_memory_id IN (SELECT memory_id FROM memories WHERE workflow_id = ?)
                         OR target_memory_id IN (SELECT memory_id FROM memories WHERE workflow_id = ?))
                """
                decay_params = (decay_factor, cutoff_timestamp, workflow_id, workflow_id)
            else:
                # Global decay
                decay_sql = """
                    UPDATE memory_links 
                    SET strength = strength * ? 
                    WHERE created_at < ?
                """
                decay_params = (decay_factor, cutoff_timestamp)

            # Execute decay update
            decay_result = await conn.execute(decay_sql, decay_params)
            links_updated_count = decay_result.rowcount

            # Prune weak links
            if workflow_id:
                prune_sql = """
                    DELETE FROM memory_links 
                    WHERE strength < ? 
                    AND (source_memory_id IN (SELECT memory_id FROM memories WHERE workflow_id = ?)
                         OR target_memory_id IN (SELECT memory_id FROM memories WHERE workflow_id = ?))
                """
                prune_params = (min_strength_threshold, workflow_id, workflow_id)
            else:
                prune_sql = "DELETE FROM memory_links WHERE strength < ?"
                prune_params = (min_strength_threshold,)

            # Execute pruning
            prune_result = await conn.execute(prune_sql, prune_params)
            links_pruned_count = prune_result.rowcount

        processing_time = time.time() - start_time

        scope_msg = f"workflow {_fmt_id(workflow_id)}" if workflow_id else "globally"
        logger.info(
            f"Decayed {links_updated_count} links and pruned {links_pruned_count} weak links {scope_msg}",
            emoji_key="hourglass",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "links_strength_updated_count": links_updated_count,
                "links_pruned_count": links_pruned_count,
                "decay_params": {
                    "workflow_id": workflow_id,
                    "half_life_days": half_life_days,
                    "decay_factor": decay_factor,
                    "min_strength_threshold": min_strength_threshold,
                    "cutoff_timestamp": cutoff_timestamp,
                },
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error decaying link strengths: {e}", exc_info=True)
        raise ToolError(f"Failed to decay link strengths: {e}") from e


@with_tool_metrics
@with_error_handling
async def get_similar_memories(
    workflow_id: str,
    memory_id: str,
    *,
    k: int = 5,
    threshold: float = agent_memory_config.similarity_threshold,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Find memories semantically similar to a given source memory.

    Args:
        workflow_id: Required for scoping and ensuring the source memory belongs to this workflow
        memory_id: The ID of the source memory to find similar items for
        k: Number of similar memories to return (default: 5)
        threshold: Minimum similarity score (default: from config)
        memory_level: Optional filter for memory level (e.g., "semantic")
        memory_type: Optional filter for memory type (e.g., "fact")
        db_path: Database path

    Returns:
        Dict containing success status, source memory info, and similar memories list
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not memory_id or not isinstance(memory_id, str):
        raise ToolInputError("memory_id is required and must be a string.", param_name="memory_id")

    if k <= 0:
        raise ToolInputError("k must be positive.", param_name="k")

    if threshold < 0.0 or threshold > 1.0:
        raise ToolInputError("threshold must be between 0.0 and 1.0.", param_name="threshold")

    if memory_level and not isinstance(memory_level, str):
        raise ToolInputError(
            "memory_level must be a string if provided.", param_name="memory_level"
        )

    if memory_type and not isinstance(memory_type, str):
        raise ToolInputError("memory_type must be a string if provided.", param_name="memory_type")

    ts_start = time.perf_counter()
    db = DBConnection(db_path)

    try:
        # First, fetch the source memory details
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Fetch source memory
            source_memory_row = await conn.execute_fetchone(
                """
                SELECT memory_id, workflow_id, content, description, memory_type, memory_level,
                       importance, confidence, created_at
                FROM memories 
                WHERE memory_id = ? AND workflow_id = ?
                """,
                (memory_id, workflow_id),
            )

            if not source_memory_row:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            source_memory = dict(source_memory_row)

            # Create query text from source memory content and description
            query_parts = []
            if source_memory.get("content"):
                query_parts.append(source_memory["content"])
            if source_memory.get("description"):
                query_parts.append(source_memory["description"])

            if not query_parts:
                logger.warning(
                    f"Source memory {memory_id} has no content or description for similarity search"
                )
                return {
                    "success": True,
                    "data": {"source_memory_id": memory_id, "similar_memories": []},
                    "processing_time": time.perf_counter() - ts_start,
                }

            query_text = " ".join(query_parts)

            # Find similar memories using the existing helper function
            # Add 1 to limit to account for potentially including source memory in results
            similar_results = await _find_similar_memories(
                conn=conn,
                query_text=query_text,
                workflow_id=workflow_id,
                limit=k + 1,  # Get one extra to handle potential self-match
                threshold=threshold,
                memory_level=memory_level,
                memory_type=memory_type,
            )

            # Filter out the source memory from results
            filtered_similar = [
                (mem_id, score) for mem_id, score in similar_results if mem_id != memory_id
            ][:k]

            if not filtered_similar:
                return {
                    "success": True,
                    "data": {"source_memory_id": memory_id, "similar_memories": []},
                    "processing_time": time.perf_counter() - ts_start,
                }

            # Fetch details for similar memories
            similar_memory_ids = [mem_id for mem_id, _ in filtered_similar]

            placeholders = ",".join("?" * len(similar_memory_ids))
            similar_details_rows = await conn.execute_fetchall(
                f"""
                SELECT memory_id, description, memory_type, content, importance, confidence,
                       created_at, memory_level
                FROM memories 
                WHERE memory_id IN ({placeholders})
                """,
                similar_memory_ids,
            )

            # Build the response with similar memories in order of similarity
            similar_memories = []
            for mem_id, similarity_score in filtered_similar:
                # Find the corresponding row
                memory_row = next(
                    (row for row in similar_details_rows if row["memory_id"] == mem_id), None
                )
                if memory_row:
                    content = memory_row["content"] if memory_row["content"] else ""
                    content_preview = content[:100] + "..." if len(content) > 100 else content

                    similar_memories.append(
                        {
                            "memory_id": mem_id,
                            "similarity": round(similarity_score, 4),
                            "description": memory_row["description"],
                            "memory_type": memory_row["memory_type"],
                            "memory_level": memory_row["memory_level"],
                            "content_preview": content_preview,
                            "importance": memory_row["importance"],
                            "confidence": memory_row["confidence"],
                            "created_at_iso": to_iso_z(
                                memory_row["created_at"] if memory_row["created_at"] else 0
                            ),
                        }
                    )

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Found {len(similar_memories)} similar memories for memory {memory_id} in workflow {workflow_id}",
            emoji_key="mag_right",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {"source_memory_id": memory_id, "similar_memories": similar_memories},
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error finding similar memories: {e}", exc_info=True)
        raise ToolError(f"Failed to find similar memories: {e}") from e


@with_tool_metrics
@with_error_handling
async def query_graph_by_link_type(
    workflow_id: str,
    link_type: str,
    *,
    limit: int = 100,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Query the memory graph for all links of a specific type within a workflow.

    Args:
        workflow_id: Required workflow scope
        link_type: Required link type (e.g., "CAUSAL", "CONTRADICTS") - case-insensitive
        limit: Max number of linked pairs to return (default: 100)
        offset: For pagination (default: 0)
        db_path: Database path

    Returns:
        Dict containing success status and linked memory pairs
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if not link_type or not isinstance(link_type, str):
        raise ToolInputError("link_type is required and must be a string.", param_name="link_type")

    if limit <= 0:
        raise ToolInputError("limit must be positive.", param_name="limit")

    if offset < 0:
        raise ToolInputError("offset must be non-negative.", param_name="offset")

    ts_start = time.perf_counter()

    # Normalize link_type to uppercase for consistency
    normalized_link_type = link_type.upper()

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # First get total count for pagination info
            count_sql = """
                SELECT COUNT(*) as total
                FROM memory_links ml
                JOIN memories m_source ON ml.source_memory_id = m_source.memory_id
                JOIN memories m_target ON ml.target_memory_id = m_target.memory_id
                WHERE m_source.workflow_id = ? AND m_target.workflow_id = ? 
                AND UPPER(ml.link_type) = ?
            """

            count_row = await conn.execute_fetchone(
                count_sql, (workflow_id, workflow_id, normalized_link_type)
            )
            total_found = count_row["total"] if count_row else 0

            if total_found == 0:
                return {
                    "success": True,
                    "data": {
                        "query_link_type": normalized_link_type,
                        "pairs": [],
                        "total_found": 0,
                        "limit": limit,
                        "offset": offset,
                    },
                    "processing_time": time.perf_counter() - ts_start,
                }

            # Query for the actual links
            links_sql = """
                SELECT ml.source_memory_id, ml.target_memory_id, ml.strength, 
                       ml.description AS link_description, ml.created_at AS link_created_at
                FROM memory_links ml
                JOIN memories m_source ON ml.source_memory_id = m_source.memory_id
                JOIN memories m_target ON ml.target_memory_id = m_target.memory_id
                WHERE m_source.workflow_id = ? AND m_target.workflow_id = ? 
                AND UPPER(ml.link_type) = ?
                ORDER BY ml.created_at DESC
                LIMIT ? OFFSET ?
            """

            links_rows = await conn.execute_fetchall(
                links_sql, (workflow_id, workflow_id, normalized_link_type, limit, offset)
            )

            # Format the results
            pairs = []
            for row in links_rows:
                pairs.append(
                    {
                        "source_memory_id": row["source_memory_id"],
                        "target_memory_id": row["target_memory_id"],
                        "strength": row["strength"],
                        "link_description": row["link_description"],
                        "link_created_at_iso": to_iso_z(row["link_created_at"])
                        if row["link_created_at"]
                        else None,
                    }
                )

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Found {len(pairs)} '{normalized_link_type}' links in workflow {workflow_id} "
            f"(total: {total_found}, offset: {offset})",
            emoji_key="link",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "query_link_type": normalized_link_type,
                "pairs": pairs,
                "total_found": total_found,
                "limit": limit,
                "offset": offset,
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error querying graph by link type: {e}", exc_info=True)
        raise ToolError(f"Failed to query graph by link type: {e}") from e


@with_tool_metrics
@with_error_handling
async def get_contradictions(
    workflow_id: str,
    *,
    limit: int = 20,
    include_resolved: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Server-side detection of contradictory memory pairs within a workflow.

    Args:
        workflow_id: Required workflow scope
        limit: Max contradictions to return (default: 20)
        include_resolved: Whether to include already resolved contradictions (default: False)
        db_path: Database path

    Returns:
        Dict containing success status and list of contradictory memory pairs
    """
    # Validation
    if not workflow_id or not isinstance(workflow_id, str):
        raise ToolInputError(
            "workflow_id is required and must be a string.", param_name="workflow_id"
        )

    if limit <= 0:
        raise ToolInputError("limit must be positive.", param_name="limit")

    ts_start = time.perf_counter()

    try:
        contradictions_found = []

        # Step 1: Find explicit CONTRADICTS links
        contradicts_result = await query_graph_by_link_type(
            workflow_id=workflow_id,
            link_type="CONTRADICTS",
            limit=limit * 2,  # Get more to account for filtering
            offset=0,
            db_path=db_path,
        )

        if contradicts_result.get("success", False):
            explicit_pairs = contradicts_result.get("data", {}).get("pairs", [])

            for pair in explicit_pairs:
                # For now, we'll include all explicit links since we don't have
                # get_memory_link_metadata implemented yet. In the future, this
                # would check for resolved_at timestamp when include_resolved=False

                contradictions_found.append(
                    {
                        "memory_id_a": pair["source_memory_id"],
                        "memory_id_b": pair["target_memory_id"],
                        "reason": "explicit_link",
                        "details": {
                            "link_description": pair.get("link_description"),
                            "strength": pair.get("strength"),
                            "link_created_at": pair.get("link_created_at_iso"),
                        },
                    }
                )

                if len(contradictions_found) >= limit:
                    break

        # Step 2: Simple negation heuristic on recent memories (if we haven't hit limit)
        if len(contradictions_found) < limit:
            try:
                # Get recent working/episodic memories for heuristic analysis
                recent_memories_result = await query_memories(
                    workflow_id=workflow_id,
                    memory_level=None,  # Don't filter by level
                    sort_by="created_at",
                    sort_order="DESC",
                    limit=50,  # Check last 50 memories
                    include_content=True,
                    db_path=db_path,
                )

                if recent_memories_result.get("success", False):
                    memories = recent_memories_result.get("data", {}).get("memories", [])

                    # Track processed pairs to avoid duplicates
                    processed_pairs = set()

                    for i, mem_i in enumerate(memories):
                        if len(contradictions_found) >= limit:
                            break

                        for _j, mem_j in enumerate(memories[i + 1 :], i + 1):
                            if len(contradictions_found) >= limit:
                                break

                            # Create a canonical pair ID to avoid duplicates
                            pair_id = tuple(
                                sorted([mem_i.get("memory_id", ""), mem_j.get("memory_id", "")])
                            )
                            if pair_id in processed_pairs:
                                continue
                            processed_pairs.add(pair_id)

                            # Simple negation heuristic
                            content_i = (mem_i.get("content") or "").lower()
                            content_j = (mem_j.get("content") or "").lower()

                            if not content_i or not content_j:
                                continue

                            # Check for negation patterns
                            has_negation_i = any(
                                neg in content_i
                                for neg in [" not ", " no ", " never ", " cannot ", " can't "]
                            )
                            has_negation_j = any(
                                neg in content_j
                                for neg in [" not ", " no ", " never ", " cannot ", " can't "]
                            )

                            # Look for overlapping concepts between negated and positive statements
                            if has_negation_i or has_negation_j:
                                # Get significant words (simple approach)
                                words_i = set(word for word in content_i.split() if len(word) > 3)
                                words_j = set(word for word in content_j.split() if len(word) > 3)

                                # Check for significant word overlap
                                overlap = words_i.intersection(words_j)
                                if len(overlap) >= 2:  # At least 2 overlapping significant words
                                    contradictions_found.append(
                                        {
                                            "memory_id_a": mem_i.get("memory_id"),
                                            "memory_id_b": mem_j.get("memory_id"),
                                            "reason": "negation_heuristic_v1",
                                            "details": {
                                                "snippet_a": content_i[:50]
                                                + ("..." if len(content_i) > 50 else ""),
                                                "snippet_b": content_j[:50]
                                                + ("..." if len(content_j) > 50 else ""),
                                                "overlapping_words": list(overlap)[
                                                    :5
                                                ],  # Show first 5 overlapping words
                                            },
                                        }
                                    )

            except Exception as e:
                logger.warning(f"Error in negation heuristic analysis: {e}")
                # Continue without heuristic results

        # Limit final results
        contradictions_found = contradictions_found[:limit]

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Found {len(contradictions_found)} contradictions in workflow {workflow_id}",
            emoji_key="warning",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "contradictions_found": contradictions_found,
                "total_found": len(contradictions_found),
                "include_resolved": include_resolved,
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error finding contradictions: {e}", exc_info=True)
        raise ToolError(f"Failed to find contradictions: {e}") from e


@with_tool_metrics
@with_error_handling
async def hybrid_search_memories(
    query: str,
    *,
    workflow_id: str | None = None,
    limit: int = 10,
    offset: int = 0,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    memory_level: str | None = None,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    min_importance: float | None = None,
    max_importance: float | None = None,
    min_confidence: float | None = None,
    min_created_at_unix: int | None = None,
    max_created_at_unix: int | None = None,
    include_content: bool = True,
    include_links: bool = False,
    link_direction: str = "outgoing",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Hybrid (semantic + keyword/FTS) memory search with rich filtering.
    Returns ranked memories, preserves raw timestamps, adds *_iso strings.
    """
    t0 = time.time()

    # ───────── validation ─────────
    if not query:
        raise ToolInputError("Query string cannot be empty.", param_name="query")
    if not 0 <= semantic_weight <= 1:
        raise ToolInputError("semantic_weight 0-1", param_name="semantic_weight")
    if not 0 <= keyword_weight <= 1:
        raise ToolInputError("keyword_weight 0-1", param_name="keyword_weight")
    if semantic_weight + keyword_weight == 0:
        raise ToolInputError("At least one weight must be >0", param_name="semantic_weight")
    if limit < 1:
        raise ToolInputError("limit ≥1", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset ≥0", param_name="offset")
    if memory_level:
        MemoryLevel(memory_level.lower())
    if memory_type:
        MemoryType(memory_type.lower())
    if (ld := link_direction.lower()) not in {"outgoing", "incoming", "both"}:
        raise ToolInputError("link_direction invalid", param_name="link_direction")

    # normalise weights
    w_sum = semantic_weight + keyword_weight
    w_sem = semantic_weight / w_sum
    w_kw = keyword_weight / w_sum

    db = DBConnection(db_path)

    # ranking container
    score_map: dict[str, dict[str, float]] = defaultdict(
        lambda: {"semantic": 0.0, "keyword": 0.0, "hybrid": 0.0}
    )

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ───── semantic phase ─────
        if w_sem:
            try:
                sem_limit = min(max(limit * 10, 100), agent_memory_config.max_semantic_candidates)
                sem_results = await _find_similar_memories(
                    conn=conn,
                    query_text=query,
                    workflow_id=workflow_id,
                    limit=sem_limit,
                    threshold=0.1,
                    memory_level=memory_level,
                    memory_type=memory_type,
                )
                for m_id, s in sem_results:
                    score_map[m_id]["semantic"] = s
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}", exc_info=True)

        # ───── keyword/FTS phase ─────
        if w_kw:
            wh, prm = ["1=1"], []
            joins = ""
            if workflow_id:
                wh.append("m.workflow_id=?")
                prm.append(workflow_id)
            if memory_level:
                wh.append("m.memory_level=?")
                prm.append(memory_level.lower())
            if memory_type:
                wh.append("m.memory_type=?")
                prm.append(memory_type.lower())
            if min_importance is not None:
                wh.append("m.importance>=?")
                prm.append(min_importance)
            if max_importance is not None:
                wh.append("m.importance<=?")
                prm.append(max_importance)
            if min_confidence is not None:
                wh.append("m.confidence>=?")
                prm.append(min_confidence)
            if min_created_at_unix is not None:
                wh.append("m.created_at>=?")
                prm.append(min_created_at_unix)
            if max_created_at_unix is not None:
                wh.append("m.created_at<=?")
                prm.append(max_created_at_unix)

            now = int(time.time())
            wh.append("(m.ttl=0 OR m.created_at+m.ttl>?)")
            prm.append(now)

            if tags:
                tag_json = json.dumps([t.strip().lower() for t in tags if t.strip()])
                wh.append("json_contains_all(m.tags, ?)")
                prm.append(tag_json)

            sanitized_fts_term = re.sub(r'[^a-zA-Z0-9\s*+\-"]', "", query).strip()
            if sanitized_fts_term:
                joins += " JOIN memory_fts f ON m.rowid=f.rowid"
                wh.append("f.memory_fts MATCH ?")
                prm.append(sanitized_fts_term)

            sql_kw = (
                "SELECT m.memory_id, "
                "compute_memory_relevance(m.importance,m.confidence,m.created_at,"
                "IFNULL(m.access_count,0),m.last_accessed) AS kw_rel "
                "FROM memories m" + joins + (" WHERE " + " AND ".join(wh) if wh else "")
            )

            # Ensure UDFs are registered before using compute_memory_relevance
            await _ensure_udfs_registered(conn)

            rows_kw = await conn.execute_fetchall(sql_kw, prm)
            if rows_kw:
                max_rel = max(r["kw_rel"] for r in rows_kw) or 1e-6
                for r in rows_kw:
                    score_map[r["memory_id"]]["keyword"] = min(max(r["kw_rel"] / max_rel, 0.0), 1.0)

        # ───── hybrid scoring & ranking ─────
        for sc in score_map.values():
            sc["hybrid"] = sc["semantic"] * w_sem + sc["keyword"] * w_kw

        ranked = sorted(score_map.items(), key=lambda i: i[1]["hybrid"], reverse=True)
        total_considered = len(ranked)
        ranked_page = ranked[offset : offset + limit]
        ids_page = [m_id for m_id, _ in ranked_page]

        # ───── hydrate rows ─────
        memories: list[dict[str, Any]] = []
        if ids_page:
            cols = (
                "memory_id, workflow_id, memory_level, memory_type, importance, confidence, "
                "description, reasoning, source, tags, created_at, updated_at, last_accessed, "
                "access_count, ttl, action_id, thought_id, artifact_id"
                + (", content" if include_content else "")
            )
            rows = await conn.execute_fetchall(
                f"SELECT {cols} FROM memories WHERE memory_id IN ({','.join('?' * len(ids_page))})",
                ids_page,
            )
            row_map = {r["memory_id"]: dict(r) for r in rows}

            # ───── optional link fetch ─────
            link_map: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
                lambda: {"outgoing": [], "incoming": []}
            )
            if include_links:
                ph = ",".join("?" * len(ids_page))
                if ld in {"outgoing", "both"}:
                    async with conn.execute(
                        f"""
                        SELECT ml.*, t.description AS target_description, t.memory_type AS target_type
                        FROM memory_links ml
                        JOIN memories t ON ml.target_memory_id = t.memory_id
                        WHERE ml.source_memory_id IN ({ph})
                        """,
                        ids_page,
                    ) as cur:
                        async for r in cur:
                            link_map[r["source_memory_id"]]["outgoing"].append(dict(r))
                if ld in {"incoming", "both"}:
                    async with conn.execute(
                        f"""
                        SELECT ml.*, s.description AS source_description, s.memory_type AS source_type
                        FROM memory_links ml
                        JOIN memories s ON ml.source_memory_id = s.memory_id
                        WHERE ml.target_memory_id IN ({ph})
                        """,
                        ids_page,
                    ) as cur:
                        async for r in cur:
                            link_map[r["target_memory_id"]]["incoming"].append(dict(r))

            # ───── build return list & batched stat updates ─────
            upd_params: list[tuple[int, str]] = []
            log_rows: list[tuple] = []
            ts_now = int(time.time())

            for m_id in ids_page:
                row = row_map[m_id]
                sc = score_map[m_id]
                row.update(
                    hybrid_score=round(sc["hybrid"], 4),
                    semantic_score=round(sc["semantic"], 4),
                    keyword_relevance_score=round(sc["keyword"], 4),
                    tags=await MemoryUtils.deserialize(row.get("tags")),
                )
                if include_links:
                    row["links"] = link_map[m_id]
                memories.append(row)

                # prepare access update
                upd_params.append((ts_now, m_id))

                # prepare operation-log row (batched, serialised once)
                op_data_json = await MemoryUtils.serialize(
                    {"query": query[:100], "hybrid_score": row["hybrid_score"]}
                )
                log_rows.append(
                    (
                        MemoryUtils.generate_id(),
                        row["workflow_id"],
                        m_id,
                        None,
                        "hybrid_access",
                        op_data_json,
                        ts_now,
                    )
                )

            # batch write access stats
            if upd_params:
                await conn.executemany(
                    "UPDATE memories "
                    "SET last_accessed=?, access_count=COALESCE(access_count,0)+1 "
                    "WHERE memory_id=?",
                    upd_params,
                )

            # batch insert operation logs  (performance fix)
            if log_rows:
                await conn.executemany(
                    """
                    INSERT INTO memory_operations
                        (operation_log_id, workflow_id, memory_id, action_id,
                         operation, operation_data, timestamp)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    log_rows,
                )

    # ───── timestamp prettification ─────
    def _iso(d: dict[str, Any], key: str) -> None:
        if (v := d.get(key)) is not None:
            d[f"{key}_iso"] = safe_format_timestamp(v)

    for m in memories:
        for k in ("created_at", "updated_at", "last_accessed"):
            _iso(m, k)
        if include_links:
            for dir_ in ("outgoing", "incoming"):
                for ln in m.get("links", {}).get(dir_, []):
                    _iso(ln, "created_at")

    proc_time = time.time() - t0
    logger.info(
        f"Hybrid search ({query[:40]}…) → {len(memories)} rows in {proc_time:.3f}s",
        emoji_key="sparkles",
    )
    return {
        "success": True,
        "data": {
            "memories": memories,
            "total_candidates_considered": total_considered,
        },
        "processing_time": proc_time,
    }


@with_tool_metrics
@with_error_handling
async def create_memory_link(
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,
    *,
    strength: float = 1.0,
    description: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Create – or replace – a typed link between two memories.

    • Enforces UNIQUE(source_memory_id, target_memory_id, link_type) constraint via
      `INSERT OR REPLACE`.
    • Logs the operation in `memory_operations`.
    """
    # ─────── basic validation ───────
    if not source_memory_id:
        raise ToolInputError("Source memory ID required.", param_name="source_memory_id")
    if not target_memory_id:
        raise ToolInputError("Target memory ID required.", param_name="target_memory_id")
    if source_memory_id == target_memory_id:
        raise ToolInputError("Cannot link memory to itself.", param_name="source_memory_id")

    try:
        link_type_enum = LinkType(link_type.lower())
    except ValueError as exc:
        valid = ", ".join(lt.value for lt in LinkType)
        raise ToolInputError(
            f"Invalid link_type. Must be one of: {valid}", param_name="link_type"
        ) from exc

    if not 0.0 <= strength <= 1.0:
        raise ToolInputError("Strength must be 0.0–1.0.", param_name="strength")

    # ─────── prepare ───────
    link_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    started = time.time()

    db = DBConnection(db_path)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            src = await conn.execute_fetchone(
                "SELECT workflow_id FROM memories WHERE memory_id = ?",
                (source_memory_id,),
            )
            if src is None:
                raise ToolInputError(
                    f"Source memory {source_memory_id} not found.", param_name="source_memory_id"
                )
            workflow_id = src["workflow_id"]

            tgt = await conn.execute_fetchone(
                "SELECT 1 FROM memories WHERE memory_id = ?",
                (target_memory_id,),
            )
            if tgt is None:
                raise ToolInputError(
                    f"Target memory {target_memory_id} not found.", param_name="target_memory_id"
                )

            await conn.execute(
                """
                INSERT OR REPLACE INTO memory_links
                    (link_id, source_memory_id, target_memory_id,
                     link_type, strength, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    source_memory_id,
                    target_memory_id,
                    link_type_enum.value,
                    strength,
                    description or "",
                    now_unix,
                ),
            )

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "link_created",
                source_memory_id,
                None,
                {
                    "target_memory_id": target_memory_id,
                    "link_type": link_type_enum.value,
                    "link_id": link_id,
                    "strength": strength,
                    "description": description or "",
                },
            )

        # ─────── success payload ───────
        elapsed = time.time() - started
        result = {
            "link_id": link_id,
            "source_memory_id": source_memory_id,
            "target_memory_id": target_memory_id,
            "link_type": link_type_enum.value,
            "strength": strength,
            "description": description or "",
            "created_at_unix": now_unix,
            "created_at_iso": to_iso_z(now_unix),
            "processing_time": elapsed,
        }
        logger.info(
            f"Memory link {link_id} ⟶ {_fmt_id(target_memory_id)} [{link_type_enum.value}] created.",
            emoji_key="link",
            time=elapsed,
        )
        return {
            "success": True,
            "data": result,
            "processing_time": result.pop("processing_time"),
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"create_memory_link failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to create memory link: {exc}") from exc


# --- 7. Core Memory Retrieval ---
@with_tool_metrics
@with_error_handling
async def query_memories(
    *,
    workflow_id: str | None = None,
    memory_level: str | None = None,
    memory_type: str | None = None,
    search_text: str | None = None,
    tags: list[str] | None = None,
    min_importance: float | None = None,
    max_importance: float | None = None,
    min_confidence: float | None = None,
    min_created_at_unix: int | None = None,
    max_created_at_unix: int | None = None,
    sort_by: str = "relevance",  # relevance, importance, created_at …
    sort_order: str = "DESC",
    include_content: bool = True,
    include_links: bool = False,
    link_direction: str = "outgoing",  # outgoing / incoming / both
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Filter, rank and paginate memories **safely**.

    • ORDER BY now uses a constant mapping → no identifier interpolation.
    • Batched access-stat update **and** batched operation-log insert unchanged.
    • Raw timestamps preserved; *_iso companions appended.
    """
    t0 = time.time()

    # ────────── validation ──────────
    SORTABLE_COLUMNS: dict[str, str] = {
        # user input  →  actual SQL expression / qualified column
        "relevance": "relevance",
        "importance": "m.importance",
        "created_at": "m.created_at",
        "updated_at": "m.updated_at",
        "confidence": "m.confidence",
        "last_accessed": "m.last_accessed",
        "access_count": "m.access_count",
    }
    sort_by_lc = sort_by.lower()
    if sort_by_lc not in SORTABLE_COLUMNS:
        raise ToolInputError(
            f"sort_by must be one of {', '.join(SORTABLE_COLUMNS.keys())}", param_name="sort_by"
        )
    if sort_order.upper() not in {"ASC", "DESC"}:
        raise ToolInputError("sort_order must be 'ASC' or 'DESC'", param_name="sort_order")
    if limit < 1:
        raise ToolInputError("limit must be ≥ 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset must be ≥ 0", param_name="offset")

    if memory_level:
        MemoryLevel(memory_level.lower())
    if memory_type:
        MemoryType(memory_type.lower())

    # ────────── dynamic parts ──────────
    order_clause = f"ORDER BY {SORTABLE_COLUMNS[sort_by_lc]} {sort_order.upper()}"

    sel_cols = [
        "m.memory_id",
        "m.workflow_id",
        "m.memory_level",
        "m.memory_type",
        "m.importance",
        "m.confidence",
        "m.description",
        "m.reasoning",
        "m.source",
        "m.tags",
        "m.created_at",
        "m.updated_at",
        "m.last_accessed",
        "m.access_count",
        "m.ttl",
        "m.action_id",
        "m.thought_id",
        "m.artifact_id",
    ]
    if include_content:
        sel_cols.append("m.content")

    select_clause = ", ".join(
        sel_cols
        + [
            "compute_memory_relevance("
            "m.importance, m.confidence, m.created_at, "
            "IFNULL(m.access_count,0), m.last_accessed) AS relevance"
        ]
    )

    joins: list[str] = []
    where: list[str] = ["1=1"]
    params: list[Any] = []
    fts_params: list[Any] = []

    if workflow_id:
        where.append("m.workflow_id = ?")
        params.append(workflow_id)
    if memory_level:
        where.append("m.memory_level = ?")
        params.append(memory_level.lower())
    if memory_type:
        where.append("m.memory_type = ?")
        params.append(memory_type.lower())
    if min_importance is not None:
        where.append("m.importance >= ?")
        params.append(min_importance)
    if max_importance is not None:
        where.append("m.importance <= ?")
        params.append(max_importance)
    if min_confidence is not None:
        where.append("m.confidence >= ?")
        params.append(min_confidence)
    if min_created_at_unix is not None:
        where.append("m.created_at >= ?")
        params.append(min_created_at_unix)
    if max_created_at_unix is not None:
        where.append("m.created_at <= ?")
        params.append(max_created_at_unix)

    now_int = int(time.time())
    where.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
    params.append(now_int)

    if tags:
        tag_list = [t.strip().lower() for t in tags if t.strip()]
        if tag_list:
            where.append("json_contains_all(m.tags, ?)")
            params.append(json.dumps(tag_list))

    if search_text:
        sanitized = re.sub(r'[^a-zA-Z0-9\s*+\-"]', "", search_text).strip()
        if sanitized:
            joins.append("JOIN memory_fts fts ON fts.rowid = m.rowid")
            where.append("fts.memory_fts MATCH ?")
            fts_params.append(sanitized)

    where_sql = " AND ".join(where)
    join_sql = " ".join(joins)

    base_from = f"FROM memories m {join_sql} WHERE {where_sql}"
    count_sql = f"SELECT COUNT(*) {base_from}"
    data_sql = f"SELECT {select_clause} {base_from}"

    paginated_sql = f"{data_sql} {order_clause} LIMIT ? OFFSET ?"

    db = DBConnection(db_path)

    async with db.transaction() as conn:
        if workflow_id:
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Ensure UDFs are registered before using compute_memory_relevance
            await _ensure_udfs_registered(conn)

        total_matching = (await conn.execute_fetchone(count_sql, params + fts_params))[0]

        rows = await conn.execute_fetchall(paginated_sql, params + fts_params + [limit, offset])

        memories: list[dict[str, Any]] = []
        now_unix = int(time.time())
        access_updates: list[tuple[int, str]] = []
        op_rows: list[tuple[str, str, str, Any, str, str, int]] = []

        for r in rows:
            mem = dict(r)
            mem["tags"] = await MemoryUtils.deserialize(mem.get("tags"))
            memories.append(mem)

            access_updates.append((now_unix, mem["memory_id"]))

            # ---- prepare operation-log row ---------------
            op_data_serialised = await MemoryUtils.serialize(
                {"query_filters": {"sort": sort_by_lc, "limit": limit}}
            )
            op_rows.append(
                (
                    MemoryUtils.generate_id(),  # operation_log_id
                    mem["workflow_id"],
                    mem["memory_id"],
                    None,  # action_id
                    "query_access",
                    op_data_serialised,
                    now_unix,
                )
            )

        if access_updates:
            await conn.executemany(
                "UPDATE memories "
                "SET last_accessed=?, access_count=COALESCE(access_count,0)+1 "
                "WHERE memory_id=?",
                access_updates,
            )

        if op_rows:
            await conn.executemany(
                """
                INSERT INTO memory_operations
                    (operation_log_id, workflow_id, memory_id, action_id,
                     operation, operation_data, timestamp)
                VALUES (?,?,?,?,?,?,?)
                """,
                op_rows,
            )

    # ───────── optional linked memories (read-only) ─────────
    if include_links and memories:

        async def _get_links(mid: str) -> dict[str, Any]:
            try:
                data = await get_linked_memories(
                    memory_id=mid,
                    direction=link_direction.lower(),
                    limit=5,
                    include_memory_details=False,
                    db_path=db_path,
                )
                return data.get("links", {})
            except Exception as e:
                logger.warning(f"link fetch failed for {mid}: {e}")
                return {"error": str(e)}

        link_tasks = {
            m["memory_id"]: asyncio.create_task(_get_links(m["memory_id"])) for m in memories
        }
        for m in memories:
            m["links"] = await link_tasks[m["memory_id"]]

    # ───────── timestamp prettification ─────────
    def _iso(obj: dict[str, Any], ks: Sequence[str]) -> None:
        for k in ks:
            if (v := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(v)

    for m in memories:
        _iso(m, ["created_at", "updated_at", "last_accessed"])

    elapsed = time.time() - t0
    logger.info(
        f"query_memories → {len(memories)}/{total_matching} rows in {elapsed:0.2f}s",
        emoji_key="scroll",
    )

    return {
        "success": True,
        "data": {
            "memories": memories,
            "total_matching_count": total_matching,
        },
        "processing_time": elapsed,
    }


# --- 8. Workflow Listing & Details ---
@with_tool_metrics
@with_error_handling
async def list_workflows(
    *,
    status: str | None = None,
    tag: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Paginated workflow listing without relying on SQLite’s
    “any-row” GROUP BY behaviour.

    • Filters: status (enum), single tag, created_at range.
    • Raw integer timestamps are preserved; *_iso companions added.
    • `total_count` is the number of rows that match *before* LIMIT/OFFSET.
    """
    # ──────────── validation ────────────
    if status:
        try:
            WorkflowStatus(status.lower())
        except ValueError as exc:
            raise ToolInputError("Invalid status", param_name="status") from exc

    def _iso_to_ts(iso: str, field: str) -> int:
        try:
            return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
        except ValueError as exc:
            raise ToolInputError(f"Invalid {field} format", param_name=field) from exc

    after_ts = _iso_to_ts(after_date, "after_date") if after_date else None
    before_ts = _iso_to_ts(before_date, "before_date") if before_date else None

    if limit < 1:
        raise ToolInputError("limit must be ≥1", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset must be ≥0", param_name="offset")

    db = DBConnection(db_path)

    # helper
    def _add_iso(obj: Dict[str, Any], *keys: str) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── WHERE fragments ─────────
            where: list[str] = ["1=1"]
            params: list[Any] = []

            if status:
                where.append("w.status = ?")
                params.append(status.lower())

            if after_ts is not None:
                where.append("w.created_at >= ?")
                params.append(after_ts)

            if before_ts is not None:
                where.append("w.created_at <= ?")
                params.append(before_ts)

            if tag:
                where.append(
                    "EXISTS (SELECT 1 FROM workflow_tags wt "
                    "JOIN tags t ON t.tag_id = wt.tag_id "
                    "WHERE wt.workflow_id = w.workflow_id AND t.name = ?)"
                )
                params.append(tag)

            where_sql = " AND ".join(where)

            # ───────── total count ─────────
            total_sql = f"SELECT COUNT(*) FROM workflows w WHERE {where_sql}"
            total_count = (await conn.execute_fetchone(total_sql, params))[0]

            # ───────── main data query ─────
            data_sql = (
                "SELECT w.workflow_id, w.title, w.description, w.goal, "
                "w.status, w.created_at, w.updated_at, w.completed_at "
                f"FROM workflows w WHERE {where_sql} "
                "ORDER BY w.updated_at DESC LIMIT ? OFFSET ?"
            )
            rows = await conn.execute_fetchall(data_sql, params + [limit, offset])
            workflows: list[Dict[str, Any]] = [dict(r) for r in rows]
            wf_ids = [wf["workflow_id"] for wf in workflows]

            # ───────── attach tags ─────────
            if wf_ids:
                placeholders = ",".join("?" * len(wf_ids))
                tag_rows = await conn.execute_fetchall(
                    f"""SELECT wt.workflow_id, t.name
                        FROM workflow_tags wt
                        JOIN tags t ON t.tag_id = wt.tag_id
                        WHERE wt.workflow_id IN ({placeholders})""",
                    wf_ids,
                )
                tag_map: Dict[str, list[str]] = defaultdict(list)
                for r in tag_rows:
                    tag_map[r["workflow_id"]].append(r["name"])
                for wf in workflows:
                    wf["tags"] = tag_map.get(wf["workflow_id"], [])
            else:
                for wf in workflows:
                    wf["tags"] = []

            # ───────── ISO decoration ──────
            for wf in workflows:
                _add_iso(wf, "created_at", "updated_at", "completed_at")

            logger.info(
                "list_workflows → %d rows (total=%d)",
                len(workflows),
                total_count,
                emoji_key="scroll",
            )
            return {
                "success": True,
                "data": {
                    "workflows": workflows,
                    "total_count": total_count,
                },
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"list_workflows failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to list workflows: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_workflow_details(
    workflow_id: str,
    *,
    include_actions: bool = True,
    include_artifacts: bool = True,
    include_thoughts: bool = True,
    include_memories: bool = False,  # Default from original
    include_cognitive_states: bool = False,  # NEW PARAMETER
    memories_limit: int = 20,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Hydrate a workflow with all optional children, including cognitive states.

    * Raw integer timestamps are preserved.
    * ISO-8601 siblings are added as *_iso.
    * Memory rows include `relevance` from the deterministic UDF.
    * Cognitive states include deserialized JSON fields and ISO timestamps.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0_perf = time.perf_counter()  # For more precise processing time
    db = DBConnection(db_path)

    def _add_iso(row: Dict[str, Any], keys: Sequence[str]) -> None:
        """Append *_iso keys in-place."""
        for k in keys:
            if (ts := row.get(k)) is not None:
                row[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── workflow core ─────────
            wf_row = await conn.execute_fetchone(
                "SELECT * FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
            if wf_row is None:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            details: Dict[str, Any] = dict(wf_row)
            details["metadata"] = await MemoryUtils.deserialize(details.get("metadata"))

            # ───────── tags ─────────
            tag_rows = await conn.execute_fetchall(
                """SELECT t.name
                   FROM tags t
                   JOIN workflow_tags wt ON wt.tag_id = t.tag_id
                   WHERE wt.workflow_id = ?""",
                (workflow_id,),
            )
            details["tags"] = [row["name"] for row in tag_rows]

            # ───────── actions ─────────
            if include_actions:
                details["actions"] = []
                async with conn.execute(
                    """
                    SELECT a.*,
                           GROUP_CONCAT(DISTINCT t.name) AS tags_str
                    FROM   actions a
                           LEFT JOIN action_tags at ON at.action_id = a.action_id
                           LEFT JOIN tags        t  ON t.tag_id      = at.tag_id
                    WHERE  a.workflow_id = ?
                    GROUP  BY a.action_id
                    ORDER  BY a.sequence_number
                    """,
                    (workflow_id,),
                ) as cur:
                    async for row_raw_action in cur:
                        act = dict(row_raw_action)
                        act["tool_args"] = await MemoryUtils.deserialize(act.get("tool_args"))
                        act["tool_result"] = await MemoryUtils.deserialize(act.get("tool_result"))
                        act["tags"] = (
                            row_raw_action["tags_str"].split(",")
                            if row_raw_action["tags_str"]
                            else []
                        )
                        act.pop("tags_str", None)
                        details["actions"].append(act)

            # ───────── artifacts ─────────
            if include_artifacts:
                details["artifacts"] = []
                async with conn.execute(
                    """
                    SELECT a.*,
                           GROUP_CONCAT(DISTINCT t.name) AS tags_str
                    FROM   artifacts a
                           LEFT JOIN artifact_tags att ON att.artifact_id = a.artifact_id
                           LEFT JOIN tags          t   ON t.tag_id        = att.tag_id
                    WHERE  a.workflow_id = ?
                    GROUP  BY a.artifact_id
                    ORDER  BY a.created_at
                    """,
                    (workflow_id,),
                ) as cur:
                    async for row_raw_artifact in cur:
                        art = dict(row_raw_artifact)
                        art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
                        art["is_output"] = bool(art["is_output"])  # Ensure boolean
                        art["tags"] = (
                            row_raw_artifact["tags_str"].split(",")
                            if row_raw_artifact["tags_str"]
                            else []
                        )
                        art.pop("tags_str", None)
                        if (
                            art.get("content") and len(art["content"]) > 200
                        ):  # content_preview logic from original
                            art["content_preview"] = art["content"][:197] + "…"
                        details["artifacts"].append(art)

            # ───────── thought chains / thoughts ─────────
            if include_thoughts:
                details["thought_chains"] = []
                async with conn.execute(
                    "SELECT * FROM thought_chains WHERE workflow_id = ? ORDER BY created_at",
                    (workflow_id,),
                ) as chains_cursor:
                    async for chain_raw in chains_cursor:
                        chain_dict = dict(chain_raw)
                        chain_dict["thoughts"] = []
                        async with conn.execute(
                            "SELECT * FROM thoughts "
                            "WHERE thought_chain_id = ? "
                            "ORDER BY sequence_number",
                            (chain_dict["thought_chain_id"],),
                        ) as thoughts_cursor:
                            async for thought_raw in thoughts_cursor:
                                chain_dict["thoughts"].append(dict(thought_raw))
                        details["thought_chains"].append(chain_dict)

            # ───────── memories (scored) ─────────
            if include_memories:
                details["memories_sample"] = []
                # Ensure UDFs are registered before using compute_memory_relevance
                await _ensure_udfs_registered(conn)

                async with conn.execute(
                    """
                    SELECT memory_id, content, memory_type, memory_level,
                           importance, confidence, access_count,
                           created_at, last_accessed,
                           compute_memory_relevance(
                               importance, confidence, created_at,
                               IFNULL(access_count,0), last_accessed
                           ) AS relevance
                    FROM   memories
                    WHERE  workflow_id = ?
                    ORDER  BY relevance DESC
                    LIMIT  ?
                    """,
                    (workflow_id, memories_limit),
                ) as mems_cursor:
                    async for row_raw_mem in mems_cursor:
                        mem = dict(row_raw_mem)
                        # 'created_at' is already an integer timestamp from DB
                        # _add_iso will handle creating 'created_at_iso' and 'last_accessed_iso'
                        # The original used 'created_at_unix' as a duplicate; this can be omitted
                        # as 'created_at' already holds the Unix timestamp.
                        if (
                            mem.get("content") and len(mem["content"]) > 150
                        ):  # content_preview logic
                            mem["content_preview"] = mem["content"][:147] + "…"
                        details["memories_sample"].append(mem)

            # --- FETCH COGNITIVE STATES (NEW) ---
            if include_cognitive_states:
                details["cognitive_states"] = []
                async with conn.execute(
                    # Fetch ALL cognitive states for the workflow, order by is_latest then created_at
                    # This ensures the 'is_latest=True' one is first if it exists,
                    # otherwise the most recently created one.
                    "SELECT * FROM cognitive_states WHERE workflow_id = ? ORDER BY is_latest DESC, created_at DESC",
                    (workflow_id,),
                ) as cog_states_cursor:
                    async for cs_row_raw in cog_states_cursor:
                        cs_row = dict(cs_row_raw)
                        # Deserialize JSON fields
                        for json_field_cs in [
                            "working_memory",
                            "focus_areas",
                            "context_actions",
                            "current_goals",
                        ]:
                            if cs_row.get(json_field_cs):  # Check if field exists and is not None
                                cs_row[json_field_cs] = await MemoryUtils.deserialize(
                                    cs_row[json_field_cs]
                                )

                        # ISO decoration for timestamps
                        _add_iso(cs_row, ["created_at", "last_active"])

                        # Ensure is_latest is boolean
                        cs_row["is_latest"] = bool(cs_row.get("is_latest", False))

                        details["cognitive_states"].append(cs_row)
            # --- END COGNITIVE STATES FETCH ---

            # ───────── ISO decoration for root + children ─────────
            _add_iso(details, ["created_at", "updated_at", "completed_at", "last_active"])
            for act_item in details.get("actions", []):  # Use different var name
                _add_iso(act_item, ["started_at", "completed_at"])
            for art_item in details.get("artifacts", []):  # Use different var name
                _add_iso(art_item, ["created_at"])
            for ch_item in details.get("thought_chains", []):  # Use different var name
                _add_iso(ch_item, ["created_at"])
                for th_item in ch_item.get("thoughts", []):  # Use different var name
                    _add_iso(th_item, ["created_at"])
            # ISO decoration for memories_sample was already handled by _add_iso within the loop
            # for memories_sample items in the original code. Re-check this.
            # Original code: _add_iso(mem, ["created_at", "last_accessed"]) for memories. This is correct.
            # Let's ensure it is applied to each memory in memories_sample if that list exists.
            if "memories_sample" in details:
                for mem_item in details["memories_sample"]:
                    _add_iso(mem_item, ["created_at", "last_accessed"])

            # ISO decoration for cognitive_states was handled inside its loop.

            processing_time = time.perf_counter() - t0_perf

            logger.info(
                f"Workflow {workflow_id} hydrated. Actions: {len(details.get('actions', []))}, "
                f"Artifacts: {len(details.get('artifacts', []))}, Thoughts: {sum(len(tc.get('thoughts', [])) for tc in details.get('thought_chains', []))}, "
                f"Memories Sample: {len(details.get('memories_sample', [])) if include_memories else 'N/A'}, "
                f"Cognitive States: {len(details.get('cognitive_states', [])) if include_cognitive_states else 'N/A'}. "
                f"Time: {processing_time:.3f}s",
                emoji_key="books",
            )
            return {
                "success": True,
                "data": details,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_workflow_details({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get workflow details: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_workflow_metadata(
    workflow_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve the metadata JSON blob for a specific workflow.

    Args:
        workflow_id: Required workflow ID
        db_path: Database path

    Returns:
        Dict containing workflow metadata
    """
    start_time = time.time()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")

    logger.info(f"Getting metadata for workflow {_fmt_id(workflow_id)}")

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Get workflow metadata
            row = await conn.execute_fetchone(
                "SELECT metadata FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )

            if not row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Deserialize metadata
            try:
                metadata_json = row.get("metadata")
                metadata = await MemoryUtils.deserialize(metadata_json)
                if not isinstance(metadata, dict):
                    metadata = {}
            except Exception as e:
                logger.warning(f"Failed to deserialize metadata for workflow {workflow_id}: {e}")
                metadata = {}

        processing_time = time.time() - start_time

        logger.info(
            f"Retrieved metadata for workflow {_fmt_id(workflow_id)}",
            emoji_key="package",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {"workflow_id": workflow_id, "metadata": metadata},
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving workflow metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve workflow metadata: {e}") from e


@with_tool_metrics
@with_error_handling
async def update_workflow_metadata(
    workflow_id: str,
    metadata: Dict[str, Any],
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Update the entire metadata JSON blob for a specific workflow.

    Args:
        workflow_id: Required workflow ID
        metadata: The new, complete metadata object to replace existing metadata
        db_path: Database path

    Returns:
        Dict containing success status and update information
    """
    start_time = time.time()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")

    if not isinstance(metadata, dict):
        raise ToolInputError("metadata must be a dictionary.", param_name="metadata")

    logger.info(f"Updating metadata for workflow {_fmt_id(workflow_id)}")

    db = DBConnection(db_path)

    try:
        async with db.transaction() as conn:
            # Verify workflow exists first
            existing_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )

            if not existing_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Serialize the metadata
            try:
                serialized_metadata = await MemoryUtils.serialize(metadata)
            except Exception as e:
                raise ToolInputError(
                    f"Failed to serialize metadata: {e}", param_name="metadata"
                ) from e

            # Update the workflow
            now = int(time.time())

            result = await conn.execute(
                "UPDATE workflows SET metadata = ?, updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (serialized_metadata, now, now, workflow_id),
            )

            if result.rowcount == 0:
                raise ToolError(f"Failed to update workflow {workflow_id} metadata")

        processing_time = time.time() - start_time

        logger.info(
            f"Updated metadata for workflow {_fmt_id(workflow_id)}",
            emoji_key="package",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {"workflow_id": workflow_id, "updated_at_iso": to_iso_z(now)},
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to update workflow metadata: {e}") from e


# --- 9. Action Details ---


@with_tool_metrics
@with_error_handling
async def get_recent_actions(
    workflow_id: str,
    *,
    limit: int = 5,
    action_type: str | None = None,
    status: str | None = None,
    include_tool_results: bool = True,
    include_reasoning: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return the *latest* `limit` actions for a workflow, optionally filtered.

    • Raw integer timestamps (`started_at`, `completed_at`) are preserved.
      ISO companions are added under *_iso.
    • Supports `action_type`, `status` filters; validates against enums.
    • Optional columns: `reasoning` and `tool_result`.

    Fix 2025-05 — remove GROUP BY on joined tag table to make result deterministic.
    """
    # ───────────── validation ─────────────
    if not (isinstance(limit, int) and limit > 0):
        raise ToolInputError("limit must be a positive integer", param_name="limit")

    if action_type:
        try:
            ActionType(action_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid action_type '{action_type}'. Allowed: {[t.value for t in ActionType]}",
                param_name="action_type",
            ) from e

    if status:
        try:
            ActionStatus(status.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid status '{status}'. Allowed: {[s.value for s in ActionStatus]}",
                param_name="status",
            ) from e

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───── verify workflow ─────
            wf_row = await conn.execute_fetchone(
                "SELECT title FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if wf_row is None:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            workflow_title = wf_row["title"]

            # ───── build dynamic SELECT (no GROUP BY) ─────
            cols = [
                "a.action_id",
                "a.action_type",
                "a.title",
                "a.tool_name",
                "a.tool_args",
                "a.status",
                "a.started_at",
                "a.completed_at",
                "a.sequence_number",
                "a.parent_action_id",
                "(SELECT GROUP_CONCAT(t.name) "
                "   FROM action_tags at2 "
                "   JOIN tags t ON t.tag_id = at2.tag_id "
                "  WHERE at2.action_id = a.action_id) AS tags_str",
            ]
            if include_reasoning:
                cols.append("a.reasoning")
            if include_tool_results:
                cols.append("a.tool_result")

            sql = f"SELECT {', '.join(cols)} FROM actions a WHERE a.workflow_id = ?"
            params: list[Any] = [workflow_id]

            if action_type:
                sql += " AND a.action_type = ?"
                params.append(action_type.lower())

            if status:
                sql += " AND a.status = ?"
                params.append(status.lower())

            sql += " ORDER BY a.sequence_number DESC LIMIT ?"
            params.append(limit)

            # ───── execute & transform ─────
            actions: list[Dict[str, Any]] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    a = dict(row)

                    a["tags"] = a.pop("tags_str").split(",") if a["tags_str"] else []

                    a["tool_args"] = await MemoryUtils.deserialize(a.get("tool_args"))
                    if include_tool_results and "tool_result" in a:
                        a["tool_result"] = await MemoryUtils.deserialize(a.get("tool_result"))

                    _add_iso(a, ["started_at", "completed_at"])
                    actions.append(a)

            logger.info(
                f"Fetched {len(actions)} recent actions for workflow {workflow_id}",
                emoji_key="rewind",
            )
            return {
                "success": True,
                "data": {
                    "workflow_id": workflow_id,
                    "workflow_title": workflow_title,
                    "actions": actions,
                },
                "processing_time": time.perf_counter() - t0,
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_recent_actions({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get recent actions: {exc}") from exc


# --- 10. Artifact Details ---
@with_tool_metrics
@with_error_handling
async def get_artifacts(
    workflow_id: str,
    *,
    artifact_type: str | None = None,
    tag: str | None = None,
    is_output: bool | None = None,
    include_content: bool = False,
    limit: int = 10,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve artifacts for a workflow with rich filtering.

    • Keeps raw `created_at` integer; adds `created_at_iso`.
    • Content trimmed to a preview unless `include_content=True`.

    Fix 2025-05 — sub-query tag aggregation replaces GROUP BY to guarantee deterministic rows.
    """
    if limit < 1:
        raise ToolInputError("limit must be ≥ 1", param_name="limit")

    if artifact_type:
        try:
            ArtifactType(artifact_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid artifact_type '{artifact_type}'",
                param_name="artifact_type",
            ) from e

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ensure workflow exists
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            # dynamic SQL (no GROUP BY, tag list via scalar sub-query)
            select_cols = (
                "a.artifact_id, a.action_id, a.artifact_type, a.name, a.description, "
                "a.path, a.metadata, a.created_at, a.is_output, "
                "(SELECT GROUP_CONCAT(t.name) "
                "   FROM artifact_tags att2 "
                "   JOIN tags t ON t.tag_id = att2.tag_id "
                "  WHERE att2.artifact_id = a.artifact_id) AS tags_str"
            )
            if include_content:
                select_cols += ", a.content"

            sql = f"SELECT {select_cols} FROM artifacts a WHERE a.workflow_id = ?"
            params: list[Any] = [workflow_id]

            if tag:
                # use EXISTS so we can still keep deterministic main select
                sql += (
                    " AND EXISTS (SELECT 1 FROM artifact_tags att3 "
                    "              JOIN tags t3 ON t3.tag_id = att3.tag_id "
                    "             WHERE att3.artifact_id = a.artifact_id AND t3.name = ?)"
                )
                params.append(tag)

            if artifact_type:
                sql += " AND a.artifact_type = ?"
                params.append(artifact_type.lower())

            if is_output is not None:
                sql += " AND a.is_output = ?"
                params.append(1 if is_output else 0)

            sql += " ORDER BY a.created_at DESC LIMIT ?"
            params.append(limit)

            # fetch + transform
            artifacts: list[Dict[str, Any]] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    art = dict(row)

                    art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
                    art["is_output"] = bool(art["is_output"])
                    art["tags"] = art.pop("tags_str").split(",") if art["tags_str"] else []

                    if not include_content and art.get("content"):
                        if len(art["content"]) > 100:
                            art["content_preview"] = art["content"][:97] + "…"
                        art.pop("content", None)

                    _add_iso(art, ["created_at"])
                    artifacts.append(art)

            logger.info(
                f"Fetched {len(artifacts)} artifacts for workflow {workflow_id}",
                emoji_key="open_file_folder",
            )
            return {
                "success": True,
                "data": {
                    "workflow_id": workflow_id,
                    "artifacts": artifacts,
                },
                "processing_time": time.perf_counter() - t0,
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_artifacts({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get artifacts: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_artifact_by_id(
    artifact_id: str,
    *,
    include_content: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single artifact (and optionally its content) **and**
    record the access against its linked memory row.

    • Preserves raw integer timestamps; adds `created_at_iso`.
    • Tags are returned as a list.
    • If `include_content` is False the `content` key is removed.
    """
    if not artifact_id:
        raise ToolInputError("Artifact ID required.", param_name="artifact_id")

    db = DBConnection(db_path)
    start_time = time.time()

    # Helper for ISO decoration
    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if ts := obj.get(key):
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    async with db.transaction() as conn:  # R/W – we may update memory row
        # ───────── Artifact row ─────────
        artifact_row = await conn.execute_fetchone(
            """
            SELECT a.*, GROUP_CONCAT(t.name) AS tags_str
            FROM artifacts a
            LEFT JOIN artifact_tags att ON att.artifact_id = a.artifact_id
            LEFT JOIN tags          t    ON t.tag_id       = att.tag_id
            WHERE a.artifact_id = ?
            GROUP BY a.artifact_id
            """,
            (artifact_id,),
        )
        if artifact_row is None:
            raise ToolInputError(f"Artifact {artifact_id} not found.", param_name="artifact_id")

        art = dict(artifact_row)
        art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
        art["is_output"] = bool(art["is_output"])
        art["tags"] = art.pop("tags_str").split(",") if artifact_row["tags_str"] else []

        if not include_content:
            art.pop("content", None)

        # ───────── Memory linkage / log ─────────
        mem = await conn.execute_fetchone(
            "SELECT memory_id, workflow_id FROM memories WHERE artifact_id = ?",
            (artifact_id,),
        )
        if mem:
            await MemoryUtils._update_memory_access(conn, mem["memory_id"])
            await MemoryUtils._log_memory_operation(
                conn,
                mem["workflow_id"],
                "access_via_artifact",
                mem["memory_id"],
                None,
                {"artifact_id": artifact_id},
            )

        # ───────── Post-transaction formatting ─────────
        _add_iso(art, "created_at")
        logger.info(f"Artifact {_fmt_id(artifact_id)} fetched.", emoji_key="page_facing_up")
        return {
            "success": True,
            "data": art,
            "processing_time": time.time() - start_time,
        }


# --- 10.5 Goals ---
@with_tool_metrics
@with_error_handling
async def create_goal(
    workflow_id: str,
    description: str,
    *,
    parent_goal_id: str | None = None,
    title: str | None = None,
    priority: int = 3,
    reasoning: str | None = None,
    acceptance_criteria: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    initial_status: str = GoalStatus.ACTIVE.value,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    # ... (validation logic remains the same) ...
    if not description:
        raise ToolInputError("Goal description is required.", param_name="description")
    try:
        status_enum = GoalStatus(initial_status.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid initial_status '{initial_status}'. Must be one of: {', '.join(gs.value for gs in GoalStatus)}",
            param_name="initial_status",
        ) from exc

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_goal_details(
        conn_fetch: aiosqlite.Connection, existing_goal_id: str
    ) -> Dict[str, Any]:
        # Reusing get_goal_details logic, simplified for idempotency return
        row = await conn_fetch.execute_fetchone(
            "SELECT * FROM goals WHERE goal_id = ?", (existing_goal_id,)
        )
        if not row:
            raise ToolError(
                f"Failed to re-fetch existing goal {existing_goal_id} on idempotency hit."
            )

        goal_data = dict(row)
        goal_data["acceptance_criteria"] = await MemoryUtils.deserialize(
            goal_data.get("acceptance_criteria")
        )
        goal_data["metadata"] = await MemoryUtils.deserialize(goal_data.get("metadata"))

        def _add_iso_local(obj: Dict[str, Any], keys: tuple[str, ...]) -> None:
            for k_iso in keys:
                if (ts := obj.get(k_iso)) is not None:
                    obj[f"{k_iso}_iso"] = safe_format_timestamp(ts)

        _add_iso_local(goal_data, ("created_at", "updated_at", "completed_at"))

        return {
            "success": True,
            "data": {
                "goal": goal_data,
                "idempotency_hit": True,
            },
            "processing_time": time.perf_counter() - t0_perf,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_goal_row = await conn_check.execute_fetchone(
                "SELECT goal_id FROM goals WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_goal_row:
            existing_goal_id = existing_goal_row["goal_id"]
            logger.info(
                f"Idempotency hit for create_goal (key='{idempotency_key}'). Returning existing goal {_fmt_id(existing_goal_id)}."
            )
            async with db.transaction(
                readonly=True
            ) as conn_details:  # New transaction for fetching details
                return await _fetch_existing_goal_details(conn_details, existing_goal_id)

    goal_id_new = MemoryUtils.generate_id()

    async with db.transaction(mode="IMMEDIATE") as conn:
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")
        if parent_goal_id and not await conn.execute_fetchone(
            "SELECT 1 FROM goals WHERE goal_id=? AND workflow_id=?",
            (parent_goal_id, workflow_id),
        ):
            raise ToolInputError(
                f"Parent goal {parent_goal_id} not found in workflow {workflow_id}.",
                param_name="parent_goal_id",
            )

        # Calculate sequence number atomically to avoid race conditions
        if parent_goal_id:
            # For child goals, sequence within the parent goal
            sequence_number = await MemoryUtils.get_next_sequence_number(
                conn, parent_goal_id, "goals", "parent_goal_id"
            )
        else:
            # For root goals, sequence within the workflow (where parent_goal_id IS NULL)
            # Use a more robust retry logic that matches the pattern used elsewhere
            for attempt in range(6):  # Retry logic similar to get_next_sequence_number
                seq_row = await conn.execute_fetchone(
                    "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM goals WHERE workflow_id=? AND parent_goal_id IS NULL",
                    (workflow_id,),
                )
                sequence_number = int(seq_row[0] if seq_row and seq_row[0] is not None else 1)

                # Quick pre-check: is this sequence number already taken?
                existing = await conn.execute_fetchone(
                    "SELECT 1 FROM goals WHERE workflow_id=? AND parent_goal_id IS NULL AND sequence_number=? LIMIT 1",
                    (workflow_id, sequence_number),
                )
                if not existing:
                    break  # sequence_number is available

                # Backoff and retry (matches the pattern in get_next_sequence_number)
                await asyncio.sleep(0.02 * (2**attempt) * (0.5 + random.random() / 2))
            else:
                raise ToolError(
                    f"Unable to allocate unique sequence_number for root goal in workflow {workflow_id} after 6 retries."
                )

        await conn.execute(
            """INSERT INTO goals (goal_id, workflow_id, parent_goal_id, title, description, status, priority, reasoning, acceptance_criteria, metadata, created_at, updated_at, sequence_number, completed_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
            (
                goal_id_new,
                workflow_id,
                parent_goal_id,
                title,
                description,
                status_enum.value,
                priority,
                reasoning,
                await MemoryUtils.serialize(acceptance_criteria or []),
                await MemoryUtils.serialize(metadata or {}),
                now_unix,
                now_unix,
                sequence_number,
                idempotency_key,
            ),
        )
        created_row_raw = await conn.execute_fetchone(
            "SELECT * FROM goals WHERE goal_id=?", (goal_id_new,)
        )
        if not created_row_raw:
            raise ToolError("Failed to retrieve goal after insert.")
        created_row = dict(created_row_raw)
        created_row["acceptance_criteria"] = await MemoryUtils.deserialize(
            created_row.get("acceptance_criteria")
        )
        created_row["metadata"] = await MemoryUtils.deserialize(created_row.get("metadata"))
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_goal",
            None,
            None,
            {
                "goal_id": goal_id_new,
                "title": title or description[:50],
                "parent_goal_id": parent_goal_id,
                "status": status_enum.value,
            },
        )

    # ISO decoration for the new row
    def _add_iso_local(o: dict[str, Any], keys: tuple[str, ...]) -> None:  # Local helper
        for k_iso in keys:
            if (ts := o.get(k_iso)) is not None:
                o[f"{k_iso}_iso"] = safe_format_timestamp(ts)

    _add_iso_local(created_row, ("created_at", "updated_at", "completed_at"))

    duration = time.perf_counter() - t0_perf
    logger.info(
        f"Goal '{title or description[:50]}…' ({goal_id_new}) created in workflow {workflow_id}",
        time=duration,
    )
    return {
        "success": True,
        "data": {
            "goal": created_row,
            "idempotency_hit": False,
        },
        "processing_time": duration,
    }


@with_tool_metrics
@with_error_handling
async def update_goal_status(
    goal_id: str,
    status: str,
    *,
    reason: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Change a goal's `status` and return the refreshed record.

    • Raw Unix timestamps are kept; *_iso companions are added.
    • `completed_at` is set for terminal states (completed / failed / abandoned).
    • A memory-operations log row is written inside the same transaction.
    """
    if not goal_id:
        raise ToolInputError("Goal ID is required.", param_name="goal_id")

    try:
        status_enum = GoalStatus(status.lower())
    except ValueError as exc:
        opts = ", ".join(g.value for g in GoalStatus)
        raise ToolInputError(
            f"Invalid goal status '{status}'. Must be one of: {opts}", param_name="status"
        ) from exc

    now = int(time.time())
    t0 = time.time()

    db = DBConnection(db_path)
    upd: Optional[Dict[str, Any]] = None  # Initialize to ensure it's defined
    parent_goal_id: Optional[str] = None
    is_root_finished: bool = False
    workflow_id_for_log: Optional[str] = None

    # helper for ISO decoration
    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k_iso in keys:  # Renamed k to k_iso to avoid conflict
            if (ts := obj.get(k_iso)) is not None:  # Use k_iso
                obj[f"{k_iso}_iso"] = safe_format_timestamp(ts)  # Use k_iso

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            row = await conn.execute_fetchone(
                "SELECT workflow_id, parent_goal_id FROM goals WHERE goal_id=?",
                (goal_id,),
            )
            if row is None:
                raise ToolInputError(f"Goal {goal_id} not found.", param_name="goal_id")

            workflow_id_for_log = row["workflow_id"]  # Capture for logging outside transaction
            parent_goal_id = row["parent_goal_id"]

            set_clauses = ["status=?", "updated_at=?"]
            params: list[Any] = [status_enum.value, now]

            if status_enum in {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED}:
                set_clauses.append("completed_at=?")
                params.append(now)

            params.append(goal_id)

            await conn.execute(
                f"UPDATE goals SET {', '.join(set_clauses)} WHERE goal_id=?",
                params,
            )

            updated_row = await conn.execute_fetchone(
                "SELECT * FROM goals WHERE goal_id=?", (goal_id,)
            )
            if updated_row is None:
                raise ToolError("Failed to retrieve goal after update.")  # Should be very rare

            upd = dict(updated_row)  # Assign to the variable defined outside
            upd["acceptance_criteria"] = await MemoryUtils.deserialize(
                upd.get("acceptance_criteria")
            )
            upd["metadata"] = await MemoryUtils.deserialize(upd.get("metadata"))
            _add_iso(upd, ["created_at", "updated_at", "completed_at"])  # upd is now a dict

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id_for_log,  # Use captured workflow_id
                "update_goal_status",
                None,
                None,
                {
                    "goal_id": goal_id,
                    "new_status": status_enum.value,
                    "reason": reason,
                },
            )

            is_root_finished = parent_goal_id is None and status_enum in {
                GoalStatus.COMPLETED,
                GoalStatus.FAILED,
                GoalStatus.ABANDONED,
            }

        dt = time.time() - t0
        logger.info(f"Goal {_fmt_id(goal_id)} set → {status_enum.value}", time=dt)

        if upd is None:  # Should ideally not be None if transaction succeeded
            raise ToolError("Internal error: Updated goal data not available after transaction.")

        return {
            "success": True,
            "data": {
                "updated_goal_details": upd,  # upd is now guaranteed to be a dict
                "parent_goal_id": parent_goal_id,
                "is_root_finished": is_root_finished,
            },
            "processing_time": dt,
        }

    except ToolInputError:
        # Log before re-raising for better context if needed, or just re-raise
        logger.error(
            f"Input error updating goal {goal_id}: status='{status}'", exc_info=False
        )  # exc_info=False for ToolInputError
        raise
    except Exception as exc:
        # Log with full traceback for unexpected errors
        logger.error(
            f"Unexpected error updating goal {goal_id} (status='{status}'): {exc}", exc_info=True
        )
        raise ToolError(f"Failed to update goal status: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_goal_details(
    goal_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single goal row and return a richly formatted record.

    • Keeps raw integer timestamps and appends *_iso keys
    • Robust JSON deserialisation for `acceptance_criteria` and `metadata`
    • Supports truncated goal IDs (8 hex chars) for convenience
    """
    t0 = time.time()

    db = DBConnection(db_path)

    # Validate and resolve goal ID (supports both full UUIDs and 8-char truncated IDs)
    goal_id = await _validate_and_resolve_id(db, goal_id, "goals", "goal_id", "goal_id")

    def _safe_json(text: str | None, default):
        try:
            return json.loads(text) if text else default
        except (json.JSONDecodeError, TypeError):
            return default

    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if (ts := obj.get(key)) is not None:
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            row = await conn.execute_fetchone(
                """
                SELECT goal_id, workflow_id, parent_goal_id, description, status,
                       created_at, updated_at, completed_at,
                       priority, reasoning, acceptance_criteria, metadata
                FROM goals
                WHERE goal_id = ?
                """,
                (goal_id,),
            )
            if row is None:
                raise ToolInputError(f"Goal '{goal_id}' not found.", param_name="goal_id")

        goal: Dict[str, Any] = dict(row)
        goal["acceptance_criteria"] = _safe_json(goal.get("acceptance_criteria"), [])
        goal["metadata"] = _safe_json(goal.get("metadata"), {})

        # keep raw ints, add ISO companions
        for k in ("created_at", "updated_at", "completed_at"):
            _add_iso(goal, k)

        elapsed = time.time() - t0
        logger.info(f"Goal '{_fmt_id(goal_id)}' loaded.", time=elapsed)

        return {
            "success": True,
            "data": {
                "goal": goal,
            },
            "processing_time": elapsed,
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_goal_details({_fmt_id(goal_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to retrieve goal details: {exc}") from exc


# --- 10.6 Goal Hierarchy ------------------------------------------------------
@with_tool_metrics
@with_error_handling
async def get_goal_stack(
    workflow_id: str,
    *,
    include_completed: bool = True,  # False → only active / in-progress
    include_metadata: bool = False,  # serialise & return goal.metadata JSON
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return **all goals for *workflow_id*** in a *parent → children* nested tree.

    • Rows are ordered by `sequence_number` inside each sibling list.
    • `acceptance_criteria` and `metadata` (optional) are JSON-deserialised.
    • When *include_completed* is *False* the filter excludes terminal states
      (completed / failed / abandoned) **recursively** – i.e. an inactive root
      goal will omit its entire sub-tree.
    • Access is *read-only*; no WAL pressure.

    The payload always contains:

    ```json
    {
        "workflow_id": "...",
        "goal_tree": [ { ...top-level goals… } ],
        "total_goals": <int>,

        "processing_time": <seconds.float>
    }
    ```
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    # -------- helper: ISO decoration -------------------
    def _iso(row: Dict[str, Any], *keys: str) -> None:
        for k in keys:
            if (ts := row.get(k)) is not None:
                row[f"{k}_iso"] = safe_format_timestamp(ts)

    # -------- helper: build tree -----------------------
    def _build_tree(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        by_parent: dict[str | None, list[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_parent[r["parent_goal_id"]].append(r)

        # stable ordering inside siblings
        for lst in by_parent.values():
            lst.sort(key=lambda g: g["sequence_number"])

        def _attach(node: Dict[str, Any]) -> Dict[str, Any]:
            kids = by_parent.get(node["goal_id"], [])
            node["children"] = [_attach(c) for c in kids]
            return node

        return [_attach(root) for root in by_parent.get(None, [])]

    async with db.transaction(readonly=True) as conn:
        # 1️⃣ confirm workflow
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        # 2️⃣ fetch goal rows
        where = ["workflow_id = ?"]
        params: list[Any] = [workflow_id]

        if not include_completed:
            where.append("status NOT IN (?, ?, ?)")
            params += [
                GoalStatus.COMPLETED.value,
                GoalStatus.FAILED.value,
                GoalStatus.ABANDONED.value,
            ]

        rows = await conn.execute_fetchall(
            f"""
            SELECT *
            FROM   goals
            WHERE  {" AND ".join(where)}
            """,
            params,
        )

        goals: list[Dict[str, Any]] = []
        for r in rows:
            g = dict(r)
            # deserialise json cols
            g["acceptance_criteria"] = await MemoryUtils.deserialize(g.get("acceptance_criteria"))
            if include_metadata:
                g["metadata"] = await MemoryUtils.deserialize(g.get("metadata"))
            else:
                g.pop("metadata", None)

            _iso(g, "created_at", "updated_at", "completed_at")
            goals.append(g)

    tree = _build_tree(goals)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Goal stack for {_fmt_id(workflow_id)} returned "
        f"({len(goals)} rows, include_completed={include_completed}).",
        emoji_key="evergreen_tree",
        time=elapsed,
    )

    return {
        "success": True,
        "data": {
            "workflow_id": workflow_id,
            "goal_tree": tree,
            "total_goals": len(goals),
        },
        "processing_time": elapsed,
    }


@with_tool_metrics
@with_error_handling
async def create_thought_chain(
    workflow_id: str,
    *,
    title: str,
    initial_thought_content: str | None = None,
    initial_thought_type: str = ThoughtType.GOAL.value,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    if not workflow_id:
        raise ToolInputError("workflow_id is required.", param_name="workflow_id")
    if not title:
        raise ToolInputError("title is required.", param_name="title")
    if initial_thought_content:
        try:
            _ = ThoughtType(initial_thought_type.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"initial_thought_type must be one of {', '.join(t.value for t in ThoughtType)}",
                param_name="initial_thought_type",
            ) from exc

    now = int(time.time())
    start_perf_t = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_chain_details(
        conn_fetch: aiosqlite.Connection, existing_chain_id: str, wf_id: str
    ) -> Dict[str, Any]:
        # Reusing get_thought_chain logic, simplified
        chain_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM thought_chains WHERE thought_chain_id = ?", (existing_chain_id,)
        )
        if not chain_row:
            raise ToolError(
                f"Failed to re-fetch existing thought_chain {existing_chain_id} on idempotency hit."
            )

        chain_data = dict(chain_row)
        # Fetch initial thought if it was supposed to be created
        initial_thought_id_existing = None
        if initial_thought_content:  # Check if the original call intended an initial thought
            thought_row = await conn_fetch.execute_fetchone(
                "SELECT thought_id FROM thoughts WHERE thought_chain_id = ? ORDER BY sequence_number ASC LIMIT 1",
                (existing_chain_id,),
            )
            if thought_row:
                initial_thought_id_existing = thought_row["thought_id"]

        return {
            "success": True,
            "data": {
                "thought_chain_id": existing_chain_id,
                "workflow_id": wf_id,
                "title": chain_data["title"],
                "created_at_unix": chain_data["created_at"],
                "created_at_iso": safe_format_timestamp(chain_data["created_at"]),
                "initial_thought_id": initial_thought_id_existing,
                "idempotency_hit": True,
            },
            "processing_time": time.perf_counter() - start_perf_t,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_chain_row = await conn_check.execute_fetchone(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_chain_row:
            existing_chain_id = existing_chain_row["thought_chain_id"]
            logger.info(
                f"Idempotency hit for create_thought_chain (key='{idempotency_key}'). Returning existing chain {_fmt_id(existing_chain_id)}."
            )
            async with db.transaction(readonly=True) as conn_details:
                return await _fetch_existing_chain_details(
                    conn_details, existing_chain_id, workflow_id
                )

    chain_id_new = MemoryUtils.generate_id()
    initial_th_id_new: str | None = None

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ... (workflow existence check remains) ...
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        await conn.execute(
            "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at, idempotency_key) VALUES (?,?,?,?,?)",  # MODIFIED
            (chain_id_new, workflow_id, title, now, idempotency_key),  # MODIFIED
        )
        await conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now, now, workflow_id),
        )

        if initial_thought_content:
            # Here, record_thought is called. We assume the agent might generate a *new* idempotency key
            # for this specific thought if it wants that thought to be idempotent independently.
            # If not, it's created as a new thought within this chain.
            th_res = await record_thought(
                workflow_id=workflow_id,
                content=initial_thought_content,
                thought_type=initial_thought_type,
                thought_chain_id=chain_id_new,
                db_path=db_path,
                conn=conn,
                idempotency_key=None,  # Explicitly None for default behavior
            )
            if not th_res.get("success"):
                raise ToolError(th_res.get("error", "Failed to create initial thought"))
            initial_th_id_new = th_res["thought_id"]

    result_data = {
        "thought_chain_id": chain_id_new,
        "workflow_id": workflow_id,
        "title": title,
        "created_at_unix": now,
        "created_at_iso": safe_format_timestamp(now),
        "initial_thought_id": initial_th_id_new,
        "idempotency_hit": False,  # NEW
    }
    logger.info(
        f"New thought-chain {_fmt_id(chain_id_new)} for workflow {_fmt_id(workflow_id)}{' with seed ' + _fmt_id(initial_th_id_new) if initial_th_id_new else ''}.",
        emoji_key="thought_balloon",
    )
    return {
        "success": True,
        "data": result_data,
        "processing_time": round(time.perf_counter() - start_perf_t, 4),
    }


@with_tool_metrics
@with_error_handling
async def get_thought_chain(
    thought_chain_id: str,
    *,
    include_thoughts: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single thought–chain plus (optionally) its ordered thoughts.

    • Raw integer timestamps are preserved.
    • ISO companions are added as *_iso for human consumption.
    """
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if (ts := obj.get(key)) is not None:
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── Chain row ─────────
            row = await conn.execute_fetchone(
                "SELECT * FROM thought_chains WHERE thought_chain_id = ?",
                (thought_chain_id,),
            )
            if row is None:
                raise ToolInputError(
                    f"Thought chain {thought_chain_id} not found.",
                    param_name="thought_chain_id",
                )

            chain: Dict[str, Any] = dict(row)
            _add_iso(chain, "created_at")

            # ───────── Thoughts ─────────
            chain["thoughts"] = []
            if include_thoughts:
                async with conn.execute(
                    """
                    SELECT *
                    FROM thoughts
                    WHERE thought_chain_id = ?
                    ORDER BY sequence_number
                    """,
                    (thought_chain_id,),
                ) as cur:
                    async for t in cur:
                        th = dict(t)
                        _add_iso(th, "created_at")
                        chain["thoughts"].append(th)

            logger.info(
                f"Retrieved thought chain {thought_chain_id} ({len(chain['thoughts'])} thoughts)",
                emoji_key="left_speech_bubble",
            )
            return {
                "success": True,
                "data": chain,
                "processing_time": time.perf_counter() - t0,
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_thought_chain({thought_chain_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get thought chain: {exc}") from exc


# ======================================================
# Helper Function for Working Memory Management
# ======================================================
async def _add_to_active_memories(
    conn: aiosqlite.Connection,
    context_id: str,
    memory_id: str,
) -> bool:
    """
    Add *memory_id* to the working-memory list for *context_id* while enforcing
    `agent_memory_config.max_working_memory_size` with proper race condition protection.

    Uses atomic read-modify-write pattern with retry logic to prevent concurrent
    modification issues in working memory capacity management.
    """
    try:
        max_retries = 3
        backoff_base = 0.01

        for attempt in range(max_retries):
            # ─────────────────────── atomic fetch with FOR UPDATE ───────────────────────
            state_row = await conn.execute_fetchone(
                "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            )
            if state_row is None:
                logger.warning(f"Context {context_id} not found when adding {memory_id}.")
                return False

            workflow_id: str = state_row["workflow_id"]
            current_ids: list[str] = (
                await MemoryUtils.deserialize(state_row["working_memory"]) or []
            )

            # ─────────────────────── quick-exit if already present ───────────────
            if memory_id in current_ids:
                logger.debug(f"{_fmt_id(memory_id)} already present in context {context_id}.")
                return True

            # ─────────────────────── validate *memory_id* exists ────────────────
            if (
                await conn.execute_fetchone(
                    "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
                )
                is None
            ):
                logger.warning(f"Memory {memory_id} missing; cannot add to context {context_id}.")
                return False

            # ─────────────────────── purge stale IDs (self-healing) ──────────────
            if current_ids:
                placeholders = ",".join("?" * len(current_ids))
                alive_rows = await conn.execute_fetchall(
                    f"SELECT memory_id FROM memories WHERE memory_id IN ({placeholders})",
                    current_ids,
                )
                alive_set = {r["memory_id"] for r in alive_rows}
                if len(alive_set) != len(current_ids):
                    stale_ids = set(current_ids) - alive_set
                    current_ids = [mid for mid in current_ids if mid in alive_set]
                    logger.info(
                        f"Purged {len(stale_ids)} stale ID(s) from working memory of "
                        f"{context_id}: {', '.join(map(_fmt_id, stale_ids))}"
                    )

            # ─────────────────────── prepare final list with capacity enforcement ──────────
            limit_cfg = getattr(agent_memory_config, "max_working_memory_size", 1)
            limit = max(1, int(limit_cfg))  # ← fix #6: never < 1

            # Create target list by appending new memory
            target_ids = current_ids + [memory_id]
            removed_id: str | None = None

            # If we exceed capacity, remove least relevant memory
            if len(target_ids) > limit and len(target_ids) > 1:
                # Ensure UDFs are registered before using compute_memory_relevance
                await _ensure_udfs_registered(conn)

                placeholders = ",".join("?" * len(current_ids))
                least_row = await conn.execute_fetchone(
                    f"""
                    SELECT memory_id
                    FROM   memories
                    WHERE  memory_id IN ({placeholders})
                    ORDER BY compute_memory_relevance(
                               importance, confidence, created_at,
                               IFNULL(access_count,0), last_accessed
                             ) ASC
                    LIMIT 1
                    """,
                    current_ids,
                )

                # ▸ fix #5 — if every current_id vanished between purge & query
                if least_row is None:
                    logger.warning(
                        f"All working-memory IDs for context {context_id} vanished concurrently; "
                        "resetting list."
                    )
                    target_ids = [memory_id]  # start fresh with just the new memory

                else:
                    removed_id = least_row["memory_id"]
                    if removed_id in target_ids:
                        target_ids.remove(removed_id)
                        logger.debug(
                            f"Will remove {_fmt_id(removed_id)} from context {context_id} "
                            f"(capacity {limit})."
                        )

            # ─────────────────────── atomic update with optimistic concurrency ────────────
            serialized_target = await MemoryUtils.serialize(target_ids)
            now_unix = int(time.time())

            # Use the original working_memory value as optimistic lock
            update_result = await conn.execute(
                """
                UPDATE cognitive_states 
                SET working_memory = ?, last_active = ? 
                WHERE state_id = ? AND working_memory = ?
                """,
                (serialized_target, now_unix, context_id, state_row["working_memory"]),
            )

            # Check if update succeeded (rowcount > 0 means we updated exactly one row)
            rowcount = (
                update_result.rowcount
                if hasattr(update_result, "rowcount")
                else (await conn.execute("SELECT changes()")).fetchone()[0]
            )

            if rowcount > 0:
                # Success! Log operations and return
                if removed_id:
                    await MemoryUtils._log_memory_operation(
                        conn,
                        workflow_id,
                        "remove_from_working",
                        removed_id,
                        None,
                        {
                            "context_id": context_id,
                            "reason": "working_memory_limit",
                        },
                    )

                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "add_to_working",
                    memory_id,
                    None,
                    {"context_id": context_id},
                )

                logger.debug(
                    f"Added {_fmt_id(memory_id)} to working memory for {context_id}; "
                    f"size={len(target_ids)}/{limit}"
                    f"{f', removed {_fmt_id(removed_id)}' if removed_id else ''}."
                )
                return True

            else:
                # Concurrent modification detected, retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff_base * (2**attempt))
                    logger.debug(
                        f"Working memory concurrent modification detected for {context_id}, "
                        f"retrying ({attempt + 1}/{max_retries})"
                    )
                    continue
                else:
                    logger.warning(
                        f"Failed to add {_fmt_id(memory_id)} to working memory for {context_id} "
                        f"after {max_retries} attempts due to concurrent modifications."
                    )
                    return False

        return False

    except Exception as exc:
        logger.error(
            f"_add_to_active_memories({context_id}, {memory_id}) failed: {exc}",
            exc_info=True,
        )
        return False


# --- 12. Working Memory Management ---
@with_tool_metrics
@with_error_handling
async def get_working_memory(
    context_id: str,
    *,
    include_content: bool = True,
    include_links: bool = True,
    update_access: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return the current working-memory set for *context_id*.

    • Opens a **read-only snapshot** when no writes are requested
      (i.e. ``update_access=False``) to maximise concurrency.
    • Otherwise falls back to the original ``mode="IMMEDIATE"`` write lock,
      preserving the exact access-count / audit-log semantics.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")

    t0 = time.time()
    result: Dict[str, Any] = {
        "context_id": context_id,
        "workflow_id": None,
        "focal_memory_id": None,
        "working_memories": [],
        "processing_time": 0.0,
    }

    db = DBConnection(db_path)

    # ───────── helpers ─────────
    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    def _chunk(seq: Sequence[Any], size: int = 900) -> Iterable[Sequence[Any]]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    # ───────── dynamic transaction mode ─────────
    txn_ctx = (
        db.transaction(readonly=True) if not update_access else db.transaction(mode="IMMEDIATE")
    )

    try:
        async with txn_ctx as conn:
            # 1️⃣ fetch cognitive state ------------------------------------------------
            state_row = await conn.execute_fetchone(
                "SELECT * FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            )
            if state_row is None:
                processing_time = time.time() - t0
                logger.warning(f"Context {context_id} not found; returning empty set.")
                return {
                    "success": True,
                    "data": result,
                    "processing_time": processing_time,
                }

            wf_id = state_row["workflow_id"]
            result["workflow_id"] = wf_id
            result["focal_memory_id"] = state_row["focal_memory_id"]

            mem_ids: list[str] = await MemoryUtils.deserialize(state_row["working_memory"]) or []

            if not (mem_ids and wf_id):
                processing_time = time.time() - t0
                return {
                    "success": True,
                    "data": result,
                    "processing_time": processing_time,
                }

            # 2️⃣ fetch memory rows (chunk-safe) --------------------------------------
            cols = [
                "memory_id",
                "workflow_id",
                "description",
                "memory_type",
                "memory_level",
                "importance",
                "confidence",
                "created_at",
                "updated_at",
                "last_accessed",
                "tags",
                "action_id",
                "thought_id",
                "artifact_id",
                "reasoning",
                "source",
                "context",
                "access_count",
                "ttl",
                "embedding_id",
            ]
            if include_content:
                cols.append("content")

            mem_rows: list[aiosqlite.Row] = []
            for chunk in _chunk(mem_ids):
                ph = ",".join("?" * len(chunk))
                mem_rows += await conn.execute_fetchall(
                    f"SELECT {', '.join(cols)} FROM memories "
                    f"WHERE memory_id IN ({ph}) AND workflow_id = ?",
                    (*chunk, wf_id),
                )

            mem_map: Dict[str, Dict[str, Any]] = {}
            for r in mem_rows:
                m = dict(r)
                m["tags"] = await MemoryUtils.deserialize(m.get("tags"))
                m["context"] = await MemoryUtils.deserialize(m.get("context"))
                if include_content and m.get("content") and len(m["content"]) > 150:
                    m["content_preview"] = m["content"][:147] + "…"
                _add_iso(m, ["created_at", "updated_at", "last_accessed"])
                m["links"] = {"outgoing": [], "incoming": []}
                mem_map[m["memory_id"]] = m

            # 3️⃣ optional link hydration ----------------------------------------------
            if include_links and mem_map:
                seen: set[str] = set()

                async def _gather(sql_tpl: str, ids: Sequence[str]) -> list[aiosqlite.Row]:
                    out: list[aiosqlite.Row] = []
                    for chunk in _chunk(ids):
                        ph = ",".join("?" * len(chunk))
                        out += await conn.execute_fetchall(sql_tpl.format(ph=ph), (*chunk, wf_id))
                    return out

                out_sql = (
                    "SELECT ml.link_id, ml.source_memory_id, ml.target_memory_id, ml.link_type, "
                    "ml.strength, ml.description AS link_description, ml.created_at, "
                    "tm.description AS target_description, tm.memory_type AS target_type "
                    "FROM memory_links ml "
                    "JOIN memories tm ON tm.memory_id = ml.target_memory_id "
                    "WHERE ml.source_memory_id IN ({ph}) AND tm.workflow_id = ?"
                )
                in_sql = (
                    "SELECT ml.link_id, ml.source_memory_id, ml.target_memory_id, ml.link_type, "
                    "ml.strength, ml.description AS link_description, ml.created_at, "
                    "sm.description AS source_description, sm.memory_type AS source_type "
                    "FROM memory_links ml "
                    "JOIN memories sm ON sm.memory_id = ml.source_memory_id "
                    "WHERE ml.target_memory_id IN ({ph}) AND sm.workflow_id = ?"
                )

                for lr in await _gather(out_sql, mem_ids):
                    if lr["link_id"] in seen:
                        continue
                    seen.add(lr["link_id"])
                    row = dict(lr)
                    _add_iso(row, ["created_at"])
                    mem_map[row["source_memory_id"]]["links"]["outgoing"].append(row)

                for lr in await _gather(in_sql, mem_ids):
                    if lr["link_id"] in seen:
                        continue
                    seen.add(lr["link_id"])
                    row = dict(lr)
                    _add_iso(row, ["created_at"])
                    mem_map[row["target_memory_id"]]["links"]["incoming"].append(row)

            # 4️⃣ order, access-stats & audit (conditional) ----------------------------
            ordered: list[Dict[str, Any]] = []
            upd_params: list[tuple[int, str, str]] = []
            now_ts = int(time.time())

            for mid in mem_ids:
                if mid in mem_map:
                    ordered.append(mem_map[mid])
                    if update_access:
                        upd_params.append((now_ts, mid, wf_id))

            if update_access and upd_params:
                await conn.executemany(
                    """
                    UPDATE memories
                    SET last_accessed = ?, access_count = COALESCE(access_count,0)+1
                    WHERE memory_id = ? AND workflow_id = ?
                    """,
                    upd_params,
                )
                for _, mem_id, wf in upd_params:
                    await MemoryUtils._log_memory_operation(
                        conn,
                        wf,
                        "access_working",
                        mem_id,
                        None,
                        {"context_id": context_id},
                    )

            result["working_memories"] = ordered

        processing_time = time.time() - t0
        logger.info(
            f"Working memory for {context_id} returned "
            f"({len(result['working_memories'])} items, update_access={update_access}).",
            emoji_key="brain",
            time=processing_time,
        )
        return {
            "success": True,
            "data": result,
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_working_memory({context_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get working memory: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def focus_memory(
    memory_id: str,
    context_id: str,
    *,
    add_to_working: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Make *memory_id* the focal memory for *context_id*.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")

    t_start = time.time()
    db = DBConnection(db_path)
    focus_changed = False
    added_to_wm: bool | None = None  # None = not attempted

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            # ─── validation (single write-lock) ───
            mem_row = await conn.execute_fetchone(
                "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
            )
            if mem_row is None:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
            mem_wf = mem_row["workflow_id"]

            ctx_row = await conn.execute_fetchone(
                "SELECT workflow_id, focal_memory_id FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            )
            if ctx_row is None:
                raise ToolInputError(f"Context {context_id} not found.", param_name="context_id")
            ctx_wf = ctx_row["workflow_id"]
            prev_focal = ctx_row["focal_memory_id"]

            if mem_wf != ctx_wf:
                raise ToolInputError(
                    f"Memory {_fmt_id(memory_id)} belongs to workflow {_fmt_id(mem_wf)}, "
                    f"not {_fmt_id(ctx_wf)} of context {context_id}",
                    param_name="memory_id",
                )

            # ─── optionally push into working memory ───
            if add_to_working:
                added_to_wm = await _add_to_active_memories(conn, context_id, memory_id)
                if not added_to_wm:
                    # hard-fail so no orphaned focal reference is written
                    raise ToolError(
                        f"Unable to add {_fmt_id(memory_id)} to working-memory set for "
                        f"context {context_id}; focal change aborted."
                    )

            # ─── update focal pointer only if it changed ───
            if memory_id != prev_focal:
                now_unix = int(time.time())
                await conn.execute(
                    "UPDATE cognitive_states "
                    "SET focal_memory_id = ?, last_active = ? "
                    "WHERE state_id = ?",
                    (memory_id, now_unix, context_id),
                )

                # touch parent workflow timestamps
                await conn.execute(
                    "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                    (now_unix, now_unix, mem_wf),
                )
                focus_changed = True

                await MemoryUtils._log_memory_operation(
                    conn,
                    mem_wf,
                    "focus",
                    memory_id,
                    None,  # action_id
                    {
                        "context_id": context_id,
                        "previous_focal_id": prev_focal,
                        "added_to_working": bool(added_to_wm),
                    },
                )

        elapsed = time.time() - t_start
        logger.info(
            f"focus_memory: {_fmt_id(memory_id)} → context {context_id} "
            f"(changed={focus_changed}, added_to_WM={added_to_wm})",
            emoji_key="target",
            time=elapsed,
        )
        return {
            "success": True,
            "data": {
                "context_id": context_id,
                "focused_memory_id": memory_id,
                "workflow_id": mem_wf,
                "focus_changed": focus_changed,
                "added_to_working": added_to_wm,
            },
            "processing_time": elapsed,
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"focus_memory({context_id}, {memory_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Unable to focus memory: {exc}") from exc


# --- 12. Working Memory Management ---
@with_tool_metrics
@with_error_handling
async def optimize_working_memory(
    context_id: str,
    *,
    target_size: int = agent_memory_config.max_working_memory_size,
    strategy: str = "balanced",  # balanced | importance | recency | diversity
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Shrink the context’s working-memory list to *target_size* using the chosen
    *strategy* **while preserving list order** and purging orphaned IDs.
    """
    # ───────────────────────────── validation ─────────────────────────────
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if not isinstance(target_size, int) or target_size < 0:
        raise ToolInputError("Target size must be a non-negative integer.", "target_size")
    strategies = {"balanced", "importance", "recency", "diversity"}
    if strategy not in strategies:
        raise ToolInputError(
            f"Strategy must be one of: {', '.join(sorted(strategies))}", "strategy"
        )

    t0 = time.time()
    db = DBConnection(db_path)

    # ───── Phase 1: fetch context row ─────
    async with db.transaction(readonly=True) as conn:
        st = await conn.execute_fetchone(
            "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        )
        if st is None:
            logger.error(
                f"INVARIANT VIOLATION: Context '{context_id}' missing in cognitive_states!",
            )
            raise ToolError(
                f"Context '{context_id}' not found in cognitive_states – fatal contract breach."
            )
        workflow_id = st["workflow_id"]
        wm_ids: list[str] = await MemoryUtils.deserialize(st["working_memory"]) or []
        before_cnt = len(wm_ids)

    # ───── fast-path exits ─────
    if before_cnt == 0 or before_cnt <= target_size:
        await _log_optimization_event(
            db,
            workflow_id,
            op="calculate_wm_optimization_skipped",
            payload={
                "context_id": context_id,
                "strategy": strategy,
                "target_size": target_size,
                "before_count": before_cnt,
                "reason": "empty" if before_cnt == 0 else "already_optimal_size",
            },
        )
        return {
            "success": True,
            "data": {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "strategy_used": strategy,
                "target_size": target_size,
                "before_count": before_cnt,
                "after_count": before_cnt,
                "removed_count": 0,
                "retained_memories": wm_ids,
                "removed_memories": [],
            },
            "processing_time": time.time() - t0,
        }

    # ───── Phase 2: fetch memory details ─────
    async with db.transaction(readonly=True) as conn:
        placeholders = ", ".join("?" * before_cnt)
        mem_rows = await conn.execute_fetchall(
            f"""
            SELECT memory_id,
                   memory_type,
                   importance,
                   confidence,
                   created_at,
                   last_accessed,
                   access_count
            FROM   memories
            WHERE  memory_id IN ({placeholders})
              AND  workflow_id = ?
            """,
            (*wm_ids, workflow_id),
        )

    fetched_ids = {r["memory_id"] for r in mem_rows}

    # ───── Phase 3: score memories ─────
    now = int(time.time())
    scored: list[dict[str, Any]] = []

    for row in mem_rows:  # only existing rows
        imp = row["importance"] if row["importance"] is not None else 5.0
        conf = row["confidence"] if row["confidence"] is not None else 0.5
        acc = row["access_count"] if row["access_count"] is not None else 0
        created = row["created_at"] or now
        last_acc = row["last_accessed"] or None

        rel = _compute_memory_relevance(imp, conf, created, acc, last_acc)
        recency = 1.0 / (1.0 + (now - (last_acc or created)) / 86_400)

        if strategy == "balanced":
            score = rel
        elif strategy == "importance":
            score = imp * 0.6 + conf * 0.2 + rel * 0.1 + recency * 0.1
        elif strategy == "recency":
            score = recency * 0.5 + min(1.0, acc / 5.0) * 0.2 + rel * 0.3
        else:  # diversity
            score = rel

        scored.append(
            {
                "id": row["memory_id"],
                "score": float(score),
                "type": (row["memory_type"] or "unknown"),
            }
        )

    # ─── Phase 3b: treat orphans as lowest-score stubs ───
    for orphan_id in (id_ for id_ in wm_ids if id_ not in fetched_ids):
        scored.append({"id": orphan_id, "score": -1.0, "type": "orphan"})

    # ───── Phase 4: select IDs to keep ─────
    retained_ids: list[str]

    if strategy == "diversity":
        from collections import defaultdict

        buckets: dict[str, list[dict]] = defaultdict(list)
        for rec in scored:
            buckets[rec["type"]].append(rec)
        for lst in buckets.values():
            lst.sort(key=lambda d: d["score"], reverse=True)
        iters = {t: iter(lst) for t, lst in buckets.items()}
        active = list(iters.keys())
        retained_ids = []
        while len(retained_ids) < target_size and active:
            t = active.pop(0)
            try:
                retained_ids.append(next(iters[t])["id"])
                active.append(t)
            except StopIteration:
                pass
    else:
        scored.sort(key=lambda d: d["score"], reverse=True)
        retained_ids = [rec["id"] for rec in scored[:target_size]]

    retained_set = set(retained_ids)

    # ---- Phase 5: restore original order & compute removed ----
    final_retained = [mid for mid in wm_ids if mid in retained_set]
    removed_ids = [mid for mid in wm_ids if mid not in retained_set]

    # ---- Phase 6: persist if changed ----
    if final_retained != wm_ids:
        async with db.transaction(mode="IMMEDIATE") as conn:
            await conn.execute(
                """
                UPDATE cognitive_states
                SET    working_memory = ?, last_active = ?
                WHERE  state_id = ?
                """,
                (
                    await MemoryUtils.serialize(final_retained),
                    int(time.time()),
                    context_id,
                ),
            )

    # ---- Phase 7: audit log ----
    await _log_optimization_event(
        db,
        workflow_id,
        op="calculate_wm_optimization",
        payload={
            "context_id": context_id,
            "strategy": strategy,
            "target_size": target_size,
            "before_count": before_cnt,
            "after_count": len(final_retained),
            "removed_count": len(removed_ids),
            "retained_ids_sample": final_retained[:5],
            "removed_ids_sample": removed_ids[:5],
        },
    )

    # ---- Phase 8: response ----
    return {
        "success": True,
        "data": {
            "context_id": context_id,
            "workflow_id": workflow_id,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": before_cnt,
            "after_count": len(final_retained),
            "removed_count": len(removed_ids),
            "retained_memories": final_retained,
            "removed_memories": removed_ids,
        },
        "processing_time": time.time() - t0,
    }


# ───────────────────── helper : single logging call ─────────────────────
async def _log_optimization_event(
    db: DBConnection | aiosqlite.Connection,
    workflow_id: str | None,
    *,
    op: str,
    payload: dict[str, Any] | list[Any] | str,
    conn: Optional[aiosqlite.Connection] = None,
) -> None:
    """
    Write one row to **memory_operations** without double-encoding JSON.

    Behavior
    ---------
    • *payload* is forwarded **raw** to ``MemoryUtils._log_memory_operation``,
      which already takes care of serialisation; this removes the
      double-encoding bug that produced escaped JSON strings in the log.
    • If *conn* is supplied (or *db* itself is an active ``aiosqlite.Connection``)
      the log entry is inserted through that handle so it is committed /
      rolled-back together with the caller.
    • Otherwise we fall back to a short autocommit transaction exactly as before.
    • Any exception during logging is swallowed after emitting an error so that
      instrumentation can never break primary control-flow.
    """
    if workflow_id is None:  # graceful no-op for orphaned optimisation attempts
        return

    try:
        # ─── path A: caller already holds an open transaction ───
        if conn is not None:
            await MemoryUtils._log_memory_operation(conn, workflow_id, op, None, None, payload)
            return

        # ─── path B: *db* is itself a live aiosqlite connection ───
        if isinstance(db, aiosqlite.Connection):
            await MemoryUtils._log_memory_operation(db, workflow_id, op, None, None, payload)
            return

        # ─── path C: independent, short autocommit transaction ───
        async with db.transaction() as tx_conn:
            await MemoryUtils._log_memory_operation(tx_conn, workflow_id, op, None, None, payload)

    except Exception as exc:  # never let logging break the caller
        logger.error(
            f"Unable to log WM optimisation event '{op}' for workflow {workflow_id}: {exc}",
            exc_info=True,
        )


# --- 13. Cognitive State Persistence ---
@with_tool_metrics
@with_error_handling
async def save_cognitive_state(
    workflow_id: str,
    title: str,
    working_memory_ids: list[str],
    *,
    focus_area_ids: list[str] | None = None,
    context_action_ids: list[str] | None = None,
    current_goals: list[str] | None = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Persist the agent’s *latest* cognitive snapshot.
    """
    if not title:
        raise ToolInputError("State title required.", param_name="title")
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    state_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    t0 = time.time()

    focus_area_ids = list(focus_area_ids or [])
    working_memory_ids = list(working_memory_ids or [])
    context_action_ids = list(context_action_ids or [])
    current_goals = list(current_goals or [])

    # ------------------------------------------------------------------ helpers
    def _first_nonblank(seq: list[str]) -> str | None:
        """Return first truthy/nonnull element or None."""
        return next((s for s in seq if s), None)

    db = DBConnection(db_path)

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ───── 1. validate workflow ─────
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(
                f"Workflow {_fmt_id(workflow_id)} not found.", param_name="workflow_id"
            )

        # ───── 2. gather ALL referenced IDs for FK checking ─────
        memory_ids_to_check: set[str] = {*(working_memory_ids or []), *(focus_area_ids or [])}
        action_ids_to_check: set[str] = set(context_action_ids)
        goal_ids_to_check: set[str] = set(current_goals)

        # ---- validate memories ----
        if memory_ids_to_check:
            ph = ",".join("?" * len(memory_ids_to_check))
            rows = await conn.execute_fetchall(
                f"SELECT memory_id FROM memories WHERE memory_id IN ({ph}) AND workflow_id = ?",
                (*memory_ids_to_check, workflow_id),
            )
            found = {r["memory_id"] for r in rows}
            missing = memory_ids_to_check - found
            if missing:
                example = ", ".join(_fmt_id(m) for m in list(missing)[:5])
                raise ToolInputError(
                    f"Memory IDs not found / wrong workflow: {example}", "working_memory_ids"
                )

        # ---- validate actions ----
        if action_ids_to_check:
            ph = ",".join("?" * len(action_ids_to_check))
            rows = await conn.execute_fetchall(
                f"SELECT action_id FROM actions WHERE action_id IN ({ph}) AND workflow_id = ?",
                (*action_ids_to_check, workflow_id),
            )
            found = {r["action_id"] for r in rows}
            missing = action_ids_to_check - found
            if missing:
                example = ", ".join(_fmt_id(a) for a in list(missing)[:5])
                raise ToolInputError(
                    f"Action IDs not found / wrong workflow: {example}", "context_action_ids"
                )

        # ---- validate goals ----
        if goal_ids_to_check:
            ph = ",".join("?" * len(goal_ids_to_check))
            rows = await conn.execute_fetchall(
                f"""
                SELECT goal_id
                FROM goals
                WHERE goal_id IN ({ph}) AND workflow_id = ?
                """,
                (*goal_ids_to_check, workflow_id),
            )
            found = {r["goal_id"] for r in rows}
            missing = goal_ids_to_check - found
            if missing:
                example = ", ".join(_fmt_id(goal) for goal in list(missing)[:5])
                raise ToolInputError(
                    f"Goal IDs not found / wrong workflow: {example}", "current_goals"
                )

        # ───── 3. SERIALISE PAYLOAD *after* validation ─────
        wm_json = await MemoryUtils.serialize(working_memory_ids)
        fa_json = await MemoryUtils.serialize(focus_area_ids)
        ca_json = await MemoryUtils.serialize(context_action_ids)
        cg_json = await MemoryUtils.serialize(current_goals)

        # ───── 4. focal-memory pick (robust + invariant: **must be in WM set**) ─────
        #
        #   1. Prefer first focus-area ID         (if any and also in working-memory list)
        #   2. Else first working-memory ID       (guaranteed in list)
        #   3. Else None                          (no focal pointer)
        #
        # This guarantees the persisted `focal_memory_id` is either NULL or a member
        # of the *serialised* working-memory vector, preventing later dereference
        # failures in `focus_memory` / `auto_update_focus`.
        #
        focal_memory_id_selected: str | None = None

        cand = _first_nonblank(focus_area_ids)
        if cand and cand in working_memory_ids:
            focal_memory_id_selected = cand
        elif working_memory_ids:
            focal_memory_id_selected = working_memory_ids[0]  # already non-blank by FK check
        # else - remains None

        # ───── 5. atomically insert new state and mark as latest ─────
        # FIXED: Prevent race condition by using INSERT first, then UPDATE others
        await conn.execute(
            """
            INSERT INTO cognitive_states (
                state_id, workflow_id, title,
                working_memory, focus_areas, context_actions, current_goals,
                created_at, is_latest, focal_memory_id, last_active
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                state_id,
                workflow_id,
                title,
                wm_json,
                fa_json,
                ca_json,
                cg_json,
                now_unix,
                1,  # is_latest = 1 for new state
                focal_memory_id_selected,
                now_unix,
            ),
        )

        # Now atomically mark all OTHER states as non-latest (prevents race condition)
        await conn.execute(
            "UPDATE cognitive_states SET is_latest = 0 WHERE workflow_id = ? AND state_id != ?",
            (workflow_id, state_id),
        )

        # touch workflow timestamps
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # audit-log
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "save_state",
            None,
            None,
            {
                "state_id": state_id,
                "title": title,
                "working_memory_count": len(working_memory_ids),
                "focus_areas_count": len(focus_area_ids),
                "context_actions_count": len(context_action_ids),
                "current_goals_count": len(current_goals),
                "focal_memory_id_used": _fmt_id(focal_memory_id_selected)
                if focal_memory_id_selected
                else None,
            },
        )

    # ───── 7. response ─────
    elapsed = time.time() - t0
    logger.info(
        f"Saved cognitive state {_fmt_id(state_id)} for workflow {_fmt_id(workflow_id)}.",
        emoji_key="save",
        time=elapsed,
    )
    return {
        "success": True,
        "data": {
            "state_id": state_id,
            "workflow_id": workflow_id,
            "title": title,
            "created_at": to_iso_z(now_unix),
        },
        "processing_time": elapsed,
    }


# --- 13. Cognitive State Persistence ---
@with_tool_metrics
@with_error_handling
async def load_cognitive_state(
    workflow_id: str,
    *,
    state_id: str | None = None,  # None → latest
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve a saved cognitive-state snapshot.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t_start = time.time()
    db = DBConnection(db_path)

    # ───────────────────────── helper ─────────────────────────
    def _empty_payload(msg: str) -> Dict[str, Any]:
        return {
            "state_id": None,
            "workflow_id": workflow_id,
            "title": None,
            "working_memory_ids": [],
            "focus_areas": [],
            "context_action_ids": [],
            "current_goals": [],
            "created_at_unix": None,
            "created_at_iso": None,
            "focal_memory_id": None,
            "message": msg,
            "processing_time": time.time() - t_start,
        }

    # ─────────────────────── main transaction ───────────────────────
    async with db.transaction() as conn:
        # 1️⃣  workflow existence check
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        # 2️⃣  fetch state row (deterministic → LIMIT 1)
        if state_id:
            row = await conn.execute_fetchone(
                """
                SELECT *
                FROM   cognitive_states
                WHERE  state_id    = ?
                  AND  workflow_id = ?
                LIMIT  1
                """,
                (state_id, workflow_id),
            )
        else:
            row = await conn.execute_fetchone(
                """
                SELECT *
                FROM   cognitive_states
                WHERE  workflow_id = ?
                ORDER  BY is_latest DESC, created_at DESC
                LIMIT  1
                """,
                (workflow_id,),
            )

        # 3️⃣  graceful “not found”
        if row is None:
            msg = (
                f"State {state_id} not found for workflow {workflow_id}."
                if state_id
                else f"No cognitive state saved for workflow {workflow_id}."
            )
            logger.warning(f"load_cognitive_state: {msg}")
            return _empty_payload(msg)

        # 4️⃣  convert to plain dict *before* leaving TX
        state: Dict[str, Any] = dict(row)

        # 5️⃣  audit-log inside the same TX
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "load_state",
            None,
            None,
            {
                "state_id": state["state_id"],
                "title": state["title"],
                "requested_state_id": state_id,
            },
        )

    # ─────────────────────── post-transaction ───────────────────────
    created_ts: int | None = state.get("created_at")
    result: Dict[str, Any] = {
        "state_id": state["state_id"],
        "workflow_id": state["workflow_id"],
        "title": state["title"],
        "working_memory_ids": await MemoryUtils.deserialize(state.get("working_memory")) or [],
        "focus_areas": await MemoryUtils.deserialize(state.get("focus_areas")) or [],
        "context_action_ids": await MemoryUtils.deserialize(state.get("context_actions")) or [],
        "current_goals": await MemoryUtils.deserialize(state.get("current_goals")) or [],
        "created_at_unix": created_ts,
        "created_at_iso": safe_format_timestamp(created_ts) if created_ts else None,
        "focal_memory_id": state.get("focal_memory_id"),
        "processing_time": time.time() - t_start,
    }

    # ─── CLEAN, CONSISTENT LOG MESSAGE (no mixed styles) ───
    logger.info(
        f"Loaded cognitive state {_fmt_id(result['state_id'])} "
        f"for workflow {_fmt_id(workflow_id)} (title='{result['title']}').",
        emoji_key="inbox_tray",
        time=result["processing_time"],
    )
    return {
        "success": True,
        "data": result,
        "processing_time": result.pop("processing_time"),
    }


# --- 14. Comprehensive Context Retrieval ---
@with_tool_metrics
@with_error_handling
async def get_workflow_context(
    workflow_id: str,
    *,
    recent_actions_limit: int = 10,
    important_memories_limit: int = 5,
    key_thoughts_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Compact snapshot of a workflow’s “working set” **without risking RO-vs-RW
    dead-locks**:

    1. A *single* **read-only** snapshot **only** gathers data that never
       triggers writes (core workflow row + key-thoughts).
    2. All helper calls that may perform writes (`load_cognitive_state`,
       `query_memories`, …) are executed **after** that transaction commits,
       so no writer is blocked by our read lock.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0 = time.time()
    db = DBConnection(db_path)

    # ────────────────────────────── helpers ──────────────────────────────
    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if key in obj and obj[key] is not None:
            obj[f"{key}_iso"] = safe_format_timestamp(obj[key])

    # ───────────────────── Phase 1: read-only snapshot ───────────────────
    async with db.transaction(readonly=True) as conn:
        wf = await conn.execute_fetchone(
            "SELECT title, goal, status FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )
        if wf is None:
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        ctx: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_title": wf["title"],
            "workflow_goal": wf["goal"],
            "workflow_status": wf["status"],
        }

        # ---- key thoughts (pure read) -----------------------------------
        chain_id_row = await conn.execute_fetchone(
            "SELECT thought_chain_id "
            "FROM thought_chains WHERE workflow_id = ? "
            "ORDER BY created_at ASC LIMIT 1",
            (workflow_id,),
        )

        if chain_id_row:
            thought_rows = await conn.execute_fetchall(
                """
                SELECT thought_type, content, sequence_number, created_at
                FROM   thoughts
                WHERE  thought_chain_id = ?
                  AND  thought_type IN (?, ?, ?, ?, ?, ?)
                ORDER  BY sequence_number DESC
                LIMIT  ?
                """,
                (
                    chain_id_row["thought_chain_id"],
                    ThoughtType.GOAL.value,
                    ThoughtType.DECISION.value,
                    ThoughtType.REASONING.value,
                    ThoughtType.ANALYSIS.value,
                    ThoughtType.SUMMARY.value,
                    ThoughtType.REFLECTION.value,
                    key_thoughts_limit,
                ),
            )
            ctx["key_thoughts"] = [dict(r) for r in thought_rows]
            for th in ctx["key_thoughts"]:
                _add_iso(th, "created_at")
        else:
            ctx["key_thoughts"] = []

    # ───────────────────── Phase 2: helper calls (may write) ─────────────
    # NOTE: now *outside* any open read transaction → no RW dead-lock risk.

    # ---- latest cognitive state ----------------------------------------
    try:
        latest_state = await load_cognitive_state(
            workflow_id=workflow_id,
            state_id=None,
            db_path=db_path,
        )
        latest_state.pop("success", None)
        latest_state.pop("processing_time", None)
        ctx["latest_cognitive_state"] = latest_state
    except ToolInputError:
        ctx["latest_cognitive_state"] = None
    except Exception as exc:
        logger.warning(f"load_cognitive_state failed: {exc}")
        ctx["latest_cognitive_state"] = {"error": str(exc)}

    # ---- recent actions (RO helper) ------------------------------------
    try:
        ra = await get_recent_actions(
            workflow_id=workflow_id,
            limit=recent_actions_limit,
            include_reasoning=False,
            include_tool_results=False,
            db_path=db_path,
        )
        ctx["recent_actions"] = ra.get("actions", [])
    except Exception as exc:
        logger.warning(f"get_recent_actions failed: {exc}")
        ctx["recent_actions"] = [{"error": str(exc)}]

    # ---- important memories --------------------------------------------
    try:
        mems = await query_memories(
            workflow_id=workflow_id,
            limit=important_memories_limit,
            sort_by="importance",
            sort_order="DESC",
            include_content=False,
            db_path=db_path,
        )
        ctx["important_memories"] = [
            {
                "memory_id": m["memory_id"],
                "description": m.get("description"),
                "memory_type": m.get("memory_type"),
                "importance": m.get("importance"),
            }
            for m in mems.get("memories", [])
        ]
    except Exception as exc:
        logger.warning(f"query_memories failed: {exc}")
        ctx["important_memories"] = [{"error": str(exc)}]

    # ─────────────────────────── finalise ───────────────────────────────
    processing_time = time.time() - t0
    logger.info(
        f"Context summary for {_fmt_id(workflow_id)} ready",
        emoji_key="clipboard",
        time=processing_time,
    )
    return {
        "success": True,
        "data": ctx,
        "processing_time": processing_time,
    }


# --- Helper: Scoring for Focus ---
def _calculate_focus_score_internal_ums(
    memory: Dict[str, Any],
    recent_action_ids: list[str],
    now_unix: int,
) -> float:
    """
    Internal scoring function mirrored from the agent-side logic.

    Robust against legacy rows containing NULL for confidence / importance /
    access_count.
    """
    importance = memory.get("importance") if memory.get("importance") is not None else 5.0
    confidence = memory.get("confidence") if memory.get("confidence") is not None else 1.0
    access_cnt = memory.get("access_count") if memory.get("access_count") is not None else 0

    base_relevance = _compute_memory_relevance(
        importance,
        confidence,
        memory.get("created_at", now_unix),
        access_cnt,
        memory.get("last_accessed"),
    )

    # Use additive scoring for interpretability - all components are weighted bonuses
    score = base_relevance  # Start with base relevance (0-10 range)

    # Contextual bonuses (as multipliers of base relevance for proportional scaling)
    if memory.get("action_id") in recent_action_ids:
        score += base_relevance * 0.5  # 50% boost for immediate contextuality

    # Type-based bonuses (fixed values for specific memory types)
    if memory.get("memory_type") in {
        MemoryType.QUESTION.value,
        MemoryType.PLAN.value,
        MemoryType.INSIGHT.value,
    }:
        score += 2.0  # Fixed bonus for important types

    # Level-based bonuses (encourage higher-level memories)
    lvl = memory.get("memory_level")
    if lvl == MemoryLevel.SEMANTIC.value:
        score += 1.0
    elif lvl == MemoryLevel.PROCEDURAL.value:
        score += 1.5

    return max(score, 0.0)


# --- Tool: Promote Memory Level ---
@with_tool_metrics
@with_error_handling
async def promote_memory_level(
    memory_id: str,
    *,
    target_level: str | None = None,
    min_access_count_episodic: int = 5,
    min_confidence_episodic: float = 0.8,
    min_access_count_semantic: int = 10,
    min_confidence_semantic: float = 0.9,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Attempt to promote a memory’s cognitive level (episodic → semantic → procedural).
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    # ─────── validate explicit target (optional) ───────
    explicit_target: MemoryLevel | None = None
    if target_level:
        try:
            explicit_target = MemoryLevel(target_level.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"Invalid target_level. Use one of: {', '.join(ml.value for ml in MemoryLevel)}",
                param_name="target_level",
            ) from exc

    db = DBConnection(db_path)
    t0 = time.time()

    # ─────── 1. fetch current memory row ───────
    async with db.transaction(readonly=True) as conn:
        row = await conn.execute_fetchone(
            """
            SELECT workflow_id, memory_level, memory_type,
                   access_count, confidence, importance
            FROM   memories
            WHERE  memory_id = ?
            """,
            (memory_id,),
        )
    if row is None:
        raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

    current_level = MemoryLevel(row["memory_level"])
    # ---- FIX #18: tolerate NULL legacy values ---------------------------------
    raw_type_val = (row["memory_type"] or MemoryType.FACT.value).lower()
    try:
        mem_type = MemoryType(raw_type_val)
    except ValueError:  # in case of unexpected old enum strings
        mem_type = MemoryType.FACT
    # --------------------------------------------------------------------------
    access_count = row["access_count"] or 0
    confidence = row["confidence"] or 0.0
    workflow_id = row["workflow_id"]

    # ─────── 2. derive automatic target (if any) ───────
    # ---- FIX #19 --------------------------------------------------------------
    if current_level == MemoryLevel.EPISODIC:
        auto_next: MemoryLevel | None = MemoryLevel.SEMANTIC
    elif current_level == MemoryLevel.SEMANTIC and mem_type in {
        MemoryType.PROCEDURE,
        MemoryType.SKILL,
    }:
        auto_next = MemoryLevel.PROCEDURAL
    else:
        auto_next = None
    # --------------------------------------------------------------------------
    candidate: MemoryLevel | None = explicit_target or auto_next

    promoted = False
    new_level = current_level
    explanatory_msg = "Criteria not met or level already maximal."

    # ─────── 3. evaluate promotion eligibility ───────
    # Define proper level hierarchy for ordinal comparison
    level_hierarchy = {
        MemoryLevel.WORKING: 0,
        MemoryLevel.EPISODIC: 1,
        MemoryLevel.SEMANTIC: 2,
        MemoryLevel.PROCEDURAL: 3,
    }

    if candidate and level_hierarchy[candidate] > level_hierarchy[current_level]:
        if candidate == MemoryLevel.SEMANTIC:
            ok = access_count >= min_access_count_episodic and confidence >= min_confidence_episodic
            explanatory_msg = (
                f"access_count {access_count}/{min_access_count_episodic}, "
                f"confidence {confidence:.2f}/{min_confidence_episodic}"
            )
        elif candidate == MemoryLevel.PROCEDURAL:
            ok = (
                mem_type in {MemoryType.PROCEDURE, MemoryType.SKILL}
                and access_count >= min_access_count_semantic
                and confidence >= min_confidence_semantic
            )
            explanatory_msg = (
                f"type {mem_type.value}, "
                f"access_count {access_count}/{min_access_count_semantic}, "
                f"confidence {confidence:.2f}/{min_confidence_semantic}"
            )
        else:
            ok = False

        if ok:
            promoted = True
            new_level = candidate
            explanatory_msg = f"Promoted: {explanatory_msg}"
        else:
            explanatory_msg = f"Not promoted: {explanatory_msg}"
    elif candidate:
        explanatory_msg = f"Already at or above {candidate.value}."

    # ─────── 4. apply update (RW) ───────
    if promoted:
        async with db.transaction(mode="IMMEDIATE") as conn:
            now = int(time.time())
            await conn.execute(
                "UPDATE memories SET memory_level = ?, updated_at = ? WHERE memory_id = ?",
                (new_level.value, now, memory_id),
            )
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "promote_level",
                memory_id,
                None,
                {
                    "previous_level": current_level.value,
                    "new_level": new_level.value,
                    "reason": explanatory_msg,
                },
            )

        logger.info(
            f"{_fmt_id(memory_id)}: {current_level.value} → {new_level.value}",
            emoji_key="arrow_up",
        )
    else:
        logger.info(f"{_fmt_id(memory_id)} not promoted – {explanatory_msg}")

    # ─────── 5. response ───────
    return {
        "success": True,
        "data": {
            "memory_id": memory_id,
            "promoted": promoted,
            "previous_level": current_level.value,
            "new_level": new_level.value if promoted else None,
            "reason": explanatory_msg,
        },
        "processing_time": time.time() - t0,
    }


# --- 15. Memory Update ---
@with_tool_metrics
@with_error_handling
async def update_memory(
    memory_id: str,
    *,
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    description: str | None = None,
    reasoning: str | None = None,
    tags: list[str] | None = None,
    ttl: int | None = None,
    memory_level: str | None = None,
    regenerate_embedding: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Patch any subset of a memory row and (optionally) regenerate its embedding.
    """
    # ───────── basic validations ─────────
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if importance is not None and not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0-10.0.", param_name="importance")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ToolInputError("Confidence must be 0.0-1.0.", param_name="confidence")

    # ───────── normalise & prepare tags (BUG #20) ─────────
    final_tags_json: Optional[str] = None
    if tags is not None:  # explicit intent, even if empty list
        # strip → lower → unique → *preserve order for human diffability*
        norm = []
        seen = set()
        for t in tags:
            s = str(t).strip().lower()
            if s and s not in seen:
                seen.add(s)
                norm.append(s)
        final_tags_json = json.dumps(norm)  # may be `"[]"` ⇒ clears tags

    # ───────── validate / map memory_level ─────────
    memory_level_value: Optional[str] = None
    if memory_level:
        try:
            memory_level_value = MemoryLevel(memory_level.lower()).value
        except ValueError as exc:
            raise ToolInputError(
                f"Invalid memory_level. Must be one of: "
                f"{', '.join(ml.value for ml in MemoryLevel)}",
                param_name="memory_level",
            ) from exc

    # ───────── dynamic SET clause assembly ─────────
    update_clauses: list[str] = []
    params: list[Any] = []
    touched: list[str] = []

    def _add(field: str, value: Any | None) -> None:
        """append field to SQL SET if value explicitly supplied"""
        if value is not None:
            update_clauses.append(f"{field} = ?")
            params.append(value)
            touched.append(field)

    _add("content", content)
    _add("importance", importance)
    _add("confidence", confidence)
    _add("description", description)
    _add("reasoning", reasoning)
    if final_tags_json is not None:  # ← always update when param given
        _add("tags", final_tags_json)
    _add("ttl", ttl)
    if memory_level_value is not None:
        _add("memory_level", memory_level_value)

    if not update_clauses and not regenerate_embedding:
        raise ToolInputError(
            "No fields provided to update and regenerate_embedding is False. "
            "Provide at least one field to update (content, importance, confidence, etc.) "
            "or set regenerate_embedding=True.",
            param_name="regenerate_embedding",
        )

    now_ts = int(time.time())
    if update_clauses:
        update_clauses.append("updated_at = ?")
        params.append(now_ts)

    embedding_regenerated = False
    new_embedding_id: str | None = None
    start_time = time.time()
    db = DBConnection(db_path)

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ─── ensure memory exists ───
        mem_row = await conn.execute_fetchone(
            "SELECT workflow_id, description, content FROM memories WHERE memory_id = ?",
            (memory_id,),
        )
        if mem_row is None:
            raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

        workflow_id = mem_row["workflow_id"]
        db_desc, db_content = mem_row["description"], mem_row["content"]

        # ─── apply UPDATE if needed ───
        if update_clauses:
            await conn.execute(
                f"UPDATE memories SET {', '.join(update_clauses)} WHERE memory_id = ?",
                (*params, memory_id),
            )

        # ─── embedding regeneration (BUG #21) ───
        if regenerate_embedding:
            eff_desc = description if "description" in touched else db_desc
            eff_content = content if "content" in touched else db_content
            text_for_embed = (f"{eff_desc}: {eff_content}" if eff_desc else eff_content) or ""

            if not text_for_embed.strip():
                raise ToolError(
                    "Cannot regenerate embedding: effective text is empty after applying updates."
                )

            try:
                new_embedding_id = await _store_embedding(conn, memory_id, text_for_embed)
                if new_embedding_id:
                    embedding_regenerated = True
                    if "updated_at" not in touched:  # ensure updated_at bump at least once
                        await conn.execute(
                            "UPDATE memories SET updated_at = ? WHERE memory_id = ?",
                            (now_ts, memory_id),
                        )
                    logger.info(
                        f"Embedding regenerated for {_fmt_id(memory_id)} (id={new_embedding_id}).",
                        emoji_key="brain",
                    )
            except Exception as exc:
                logger.error(
                    f"Embedding regen failed for {_fmt_id(memory_id)}: {exc}",
                    exc_info=True,
                )
                # Do NOT raise: keep memory patch successful even if embedding fails

        # ─── operation log ───
        log_payload = {
            "updated_fields": touched,
            "embedding_regenerated": embedding_regenerated,
            "new_embedding_id": new_embedding_id,
        }
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "update_memory",
            memory_id,
            None,
            log_payload,
        )

    proc_time = time.time() - start_time
    logger.info(
        f"Memory {_fmt_id(memory_id)} patched (fields: {touched or 'none'}, embedding:{'yes' if embedding_regenerated else 'no'})",
        emoji_key="pencil2",
        time=proc_time,
    )

    return {
        "success": True,
        "data": {
            "memory_id": memory_id,
            "updated_fields": touched,
            "embedding_regenerated": embedding_regenerated,
            "new_embedding_id": new_embedding_id,
            "updated_at": to_iso_z(now_ts),
        },
        "processing_time": proc_time,
    }


# ======================================================
# Linked Memories Retrieval
# ======================================================
@with_tool_metrics
@with_error_handling
async def get_linked_memories(
    memory_id: str,
    *,
    direction: str = "both",  # "outgoing" | "incoming" | "both"
    link_type: str | None = None,  # optional filter (case-insensitive)
    limit: int = 10,
    include_memory_details: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch links touching *memory_id* with a **stable result shape**:
        {
            "memory_id": …,
            "links": {
                "outgoing": [...],   # always present
                "incoming": [...],   # always present
            },

            "processing_time": …
        }

    • Keeps existing behaviour (access logging, ISO decoration, detail hydration).
    • Ensures BOTH `"outgoing"` and `"incoming"` keys are **always** present,
      even when a single direction is requested or when zero rows match.
    """
    # ─── validation ───
    if not memory_id:
        raise ToolInputError("Memory ID is required.", param_name="memory_id")

    direction = direction.lower().strip()
    if direction not in {"outgoing", "incoming", "both"}:
        raise ToolInputError(
            "direction must be 'outgoing', 'incoming', or 'both'", param_name="direction"
        )

    if link_type:
        try:
            LinkType(link_type.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"Invalid link_type — allowed: {', '.join(lt.value for lt in LinkType)}",
                param_name="link_type",
            ) from exc

    if limit < 1:
        raise ToolInputError("limit must be a positive integer.", param_name="limit")

    t0 = time.time()
    payload: Dict[str, Any] = {
        "memory_id": memory_id,
        "links": {"outgoing": [], "incoming": []},  # ← guaranteed keys
        "processing_time": 0.0,
    }

    def _add_iso(obj: Dict[str, Any], *keys: str) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    db = DBConnection(db_path)
    async with db.transaction() as conn:  # single R/W txn (updates access stats)
        # ─── confirm source memory exists & capture workflow_id ───
        src_row = await conn.execute_fetchone(
            "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
        )
        if src_row is None:
            raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
        workflow_id: str = src_row["workflow_id"]

        # ─── helper to optionally hydrate memory details ───
        async def _hydrate_mem(mid: str) -> Dict[str, Any] | None:
            if not include_memory_details:
                return None
            row = await conn.execute_fetchone(
                """
                SELECT memory_id, memory_type, memory_level, importance, confidence,
                       description, tags, created_at, updated_at
                FROM   memories
                WHERE  memory_id = ?
                """,
                (mid,),
            )
            if not row:
                return None
            m = dict(row)
            m["tags"] = await MemoryUtils.deserialize(m.get("tags"))
            _add_iso(m, "created_at", "updated_at")
            return m

        # ─── build SQL fragments once ───
        type_filter_sql = " AND ml.link_type = ?" if link_type else ""

        # Common SELECT columns
        base_cols = "ml.link_id, ml.link_type, ml.strength, ml.description AS link_description, ml.created_at"

        # ─── OUTGOING ───
        if direction in {"outgoing", "both"}:
            out_params: list[Any] = [memory_id]
            if link_type:
                out_params.append(link_type.lower())
            out_params.append(limit)  # ← limit LAST
            out_rows = await conn.execute_fetchall(
                f"""
                SELECT {base_cols},
                       ml.target_memory_id,
                       t.description AS target_description,
                       t.memory_type AS target_type
                FROM   memory_links ml
                JOIN   memories t ON t.memory_id = ml.target_memory_id
                WHERE  ml.source_memory_id = ?{type_filter_sql}
                ORDER  BY ml.created_at DESC
                LIMIT  ?
                """,
                out_params,
            )
            for r in out_rows:
                link = dict(r)
                _add_iso(link, "created_at")
                if include_memory_details:
                    link["target_memory"] = await _hydrate_mem(link["target_memory_id"])
                payload["links"]["outgoing"].append(link)

        # ─── INCOMING ───
        if direction in {"incoming", "both"}:
            in_params: list[Any] = [memory_id]
            if link_type:
                in_params.append(link_type.lower())
            in_params.append(limit)  # ← limit LAST
            in_rows = await conn.execute_fetchall(
                f"""
                SELECT {base_cols},
                       ml.source_memory_id,
                       s.description AS source_description,
                       s.memory_type AS source_type
                FROM   memory_links ml
                JOIN   memories s ON s.memory_id = ml.source_memory_id
                WHERE  ml.target_memory_id = ?{type_filter_sql}
                ORDER  BY ml.created_at DESC
                LIMIT  ?
                """,
                in_params,
            )
            for r in in_rows:
                link = dict(r)
                _add_iso(link, "created_at")
                if include_memory_details:
                    link["source_memory"] = await _hydrate_mem(link["source_memory_id"])
                payload["links"]["incoming"].append(link)

        # ─── update access stats & audit log ───
        await MemoryUtils._update_memory_access(conn, memory_id)
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "access_links",
            memory_id,
            None,
            {
                "direction": direction,
                "link_type_filter": link_type.lower() if link_type else None,
                "returned_outgoing": len(payload["links"]["outgoing"]),
                "returned_incoming": len(payload["links"]["incoming"]),
            },
        )

    processing_time = time.time() - t0
    logger.info(
        f"get_linked_memories({_fmt_id(memory_id)}) dir={direction} type={link_type or 'any'} → {len(payload['links']['outgoing'])} out / {len(payload['links']['incoming'])} in",
        emoji_key="link",
        time=processing_time,
    )
    return {
        "success": True,
        "data": payload,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def get_subgraph(
    workflow_id: str,
    start_node_id: Optional[str] = None,
    *,
    # NetworkX Traversal (always used)
    algorithm: str = "ego_graph",  # ego_graph, bfs_tree, dfs_tree, component, full_graph
    max_hops: int = 1,
    max_nodes: int = 50,
    link_type_filter: Optional[List[str]] = None,
    # NetworkX Analysis (optional)
    compute_centrality: bool = False,
    centrality_algorithms: Optional[
        List[str]
    ] = None,  # ["pagerank", "betweenness", "closeness", "degree"]
    detect_communities: bool = False,
    community_algorithm: str = "louvain",
    compute_graph_metrics: bool = False,
    include_shortest_paths: bool = False,
    shortest_path_targets: Optional[List[str]] = None,
    # Output Control
    include_node_content: bool = False,
    centrality_top_k: int = 10,
    min_community_size: int = 3,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Extract and analyze subgraphs using NetworkX algorithms.

    This tool builds a NetworkX graph from the memory database and uses sophisticated
    graph algorithms for both traversal and analysis. All operations are NetworkX-based
    for consistency and advanced capabilities.

    Args:
        workflow_id: The workflow scope for the subgraph
        start_node_id: Memory ID to start traversal from (None for full graph analysis)

        # Traversal Control
        algorithm: NetworkX algorithm to use:
            - "ego_graph": Get neighborhood around start_node_id
            - "bfs_tree": Breadth-first tree from start_node_id
            - "dfs_tree": Depth-first tree from start_node_id
            - "component": Connected component containing start_node_id
            - "full_graph": Return entire workflow graph (ignores start_node_id)
        max_hops: Maximum traversal depth (for ego_graph, bfs_tree, dfs_tree)
        max_nodes: Maximum nodes to return
        link_type_filter: Only traverse these link types

        # Analysis Options
        compute_centrality: Calculate node importance scores
        centrality_algorithms: Which centrality measures to compute
        detect_communities: Find clusters of related memories
        community_algorithm: Community detection algorithm
        compute_graph_metrics: Calculate global graph statistics
        include_shortest_paths: Analyze paths between nodes
        shortest_path_targets: Specific nodes for path analysis

        # Output Options
        include_node_content: Include full memory content vs preview
        centrality_top_k: Return only top K central nodes
        min_community_size: Minimum size for detected communities

    Returns:
        Dict with subgraph nodes, edges, and optional analysis results
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")
    if start_node_id:
        start_node_id = _validate_uuid_format(start_node_id, "start_node_id")

    valid_algorithms = {"ego_graph", "bfs_tree", "dfs_tree", "component", "full_graph"}
    if algorithm not in valid_algorithms:
        raise ToolInputError(
            f"algorithm must be one of: {valid_algorithms}", param_name="algorithm"
        )

    if algorithm != "full_graph" and not start_node_id:
        raise ToolInputError(
            f"start_node_id required for algorithm '{algorithm}'", param_name="start_node_id"
        )

    if max_hops < 1 or max_hops > 10:
        raise ToolInputError("max_hops must be between 1 and 10", param_name="max_hops")

    if max_nodes <= 0 or max_nodes > 2000:
        raise ToolInputError("max_nodes must be between 1 and 2000", param_name="max_nodes")

    # Validate analysis parameters
    if centrality_algorithms is None:
        centrality_algorithms = ["pagerank"] if compute_centrality else []

    valid_centrality = {"pagerank", "betweenness", "closeness", "degree", "eigenvector", "katz"}
    invalid_centrality = set(centrality_algorithms) - valid_centrality
    if invalid_centrality:
        raise ToolInputError(
            f"Invalid centrality algorithms: {invalid_centrality}",
            param_name="centrality_algorithms",
        )

    valid_community = {"louvain", "leiden", "greedy_modularity", "label_propagation"}
    if community_algorithm not in valid_community:
        raise ToolInputError(
            f"community_algorithm must be one of: {valid_community}",
            param_name="community_algorithm",
        )

    # Normalize link types
    normalized_link_types = None
    if link_type_filter:
        normalized_link_types = [lt.upper() for lt in link_type_filter if lt]

    logger.info(f"Building NetworkX subgraph for workflow {_fmt_id(workflow_id)} using {algorithm}")

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Step 1: Build full NetworkX graph from database
            full_graph = await _build_networkx_graph_from_db(
                conn,
                workflow_id,
                normalized_link_types,
                max_nodes * 2,  # Load more for better traversal
            )

            if not full_graph.nodes():
                return {
                    "success": True,
                    "data": {
                        "workflow_id": workflow_id,
                        "start_node_id": start_node_id,
                        "algorithm": algorithm,
                        "nodes": [],
                        "edges": [],
                        "node_count": 0,
                        "edge_count": 0,
                        "message": "No memories found in workflow",
                    },
                    "processing_time": time.perf_counter() - ts_start,
                }

            # Step 2: Apply NetworkX traversal algorithm
            subgraph = await _apply_networkx_algorithm(
                full_graph, algorithm, start_node_id, max_hops, max_nodes
            )

            # Step 3: Convert subgraph to output format
            nodes_data, edges_data = await _extract_subgraph_data(
                conn, subgraph, include_node_content
            )

            # Step 4: Run optional NetworkX analysis
            analysis_results = {}
            if any(
                [
                    compute_centrality,
                    detect_communities,
                    compute_graph_metrics,
                    include_shortest_paths,
                ]
            ):
                analysis_results = await _perform_networkx_analysis(
                    subgraph,
                    centrality_algorithms=centrality_algorithms,
                    detect_communities=detect_communities,
                    community_algorithm=community_algorithm,
                    compute_graph_metrics=compute_graph_metrics,
                    include_shortest_paths=include_shortest_paths,
                    shortest_path_targets=shortest_path_targets,
                    centrality_top_k=centrality_top_k,
                    min_community_size=min_community_size,
                )

        processing_time = time.perf_counter() - ts_start

        result = {
            "success": True,
            "data": {
                "workflow_id": workflow_id,
                "start_node_id": start_node_id,
                "algorithm": algorithm,
                "traversal_params": {
                    "max_hops": max_hops,
                    "max_nodes": max_nodes,
                    "link_type_filter": link_type_filter,
                },
                "nodes": nodes_data,
                "edges": edges_data,
                "node_count": len(nodes_data),
                "edge_count": len(edges_data),
                **analysis_results,  # Include all analysis results
            },
            "processing_time": processing_time,
        }

        logger.info(
            f"NetworkX subgraph complete: {len(nodes_data)} nodes, {len(edges_data)} edges, "
            f"algorithm: {algorithm}, analysis: {bool(analysis_results)}, time: {processing_time:.3f}s",
            emoji_key="graph",
            time=processing_time,
        )

        return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error building NetworkX subgraph: {e}", exc_info=True)
        raise ToolError(f"Failed to build NetworkX subgraph: {e}") from e


async def _build_networkx_graph_from_db(
    conn,
    workflow_id: str,
    link_type_filter: Optional[List[str]] = None,
    node_limit: Optional[int] = None,
) -> nx.DiGraph:
    """Build NetworkX directed graph directly from database."""

    G = nx.DiGraph()

    # Get nodes (memories)
    node_sql = """
        SELECT memory_id, description, memory_type, memory_level,
               importance, confidence, created_at, last_accessed,
               CASE 
                   WHEN LENGTH(content) > 100 THEN SUBSTR(content, 1, 100) || '...'
                   ELSE content
               END as content_preview
        FROM memories 
        WHERE workflow_id = ?
        ORDER BY importance DESC, created_at DESC
    """
    node_params = [workflow_id]

    if node_limit:
        node_sql += " LIMIT ?"
        node_params.append(node_limit)

    node_rows = await conn.execute_fetchall(node_sql, node_params)

    # Add nodes to graph
    for row in node_rows:
        G.add_node(
            row["memory_id"],
            description=row["description"],
            content_preview=row["content_preview"],
            memory_type=row["memory_type"],
            memory_level=row["memory_level"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at_iso=to_iso_z(row["created_at"]) if row["created_at"] else None,
            last_accessed_iso=to_iso_z(row["last_accessed"]) if row["last_accessed"] else None,
        )

    # Get edges (memory links)
    edge_sql = """
        SELECT ml.source_memory_id, ml.target_memory_id, ml.link_type, 
               ml.strength, ml.description as link_description
        FROM memory_links ml
        JOIN memories m1 ON ml.source_memory_id = m1.memory_id
        JOIN memories m2 ON ml.target_memory_id = m2.memory_id
        WHERE m1.workflow_id = ? AND m2.workflow_id = ?
    """
    edge_params = [workflow_id, workflow_id]

    if link_type_filter:
        placeholders = ",".join("?" * len(link_type_filter))
        edge_sql += f" AND UPPER(ml.link_type) IN ({placeholders})"
        edge_params.extend(link_type_filter)

    edge_rows = await conn.execute_fetchall(edge_sql, edge_params)

    # Add edges to graph (only if both nodes exist)
    for row in edge_rows:
        source = row["source_memory_id"]
        target = row["target_memory_id"]

        if G.has_node(source) and G.has_node(target):
            G.add_edge(
                source,
                target,
                link_type=row["link_type"],
                strength=row["strength"],
                link_description=row["link_description"] or "",
            )

    return G


async def _apply_networkx_algorithm(
    full_graph: nx.DiGraph,
    algorithm: str,
    start_node_id: Optional[str],
    max_hops: int,
    max_nodes: int,
) -> nx.DiGraph:
    """Apply the specified NetworkX algorithm to get subgraph."""

    if algorithm == "full_graph":
        # Return the full graph (possibly limited by size)
        if len(full_graph) <= max_nodes:
            return full_graph
        else:
            # Return highest importance nodes
            nodes_by_importance = sorted(
                full_graph.nodes(data=True), key=lambda x: x[1].get("importance", 0), reverse=True
            )
            top_nodes = [node_id for node_id, _ in nodes_by_importance[:max_nodes]]
            return full_graph.subgraph(top_nodes).copy()

    # Validate start node exists
    if start_node_id not in full_graph:
        raise ToolInputError(
            f"Start node {start_node_id} not found in graph", param_name="start_node_id"
        )

    if algorithm == "ego_graph":
        subgraph = nx.ego_graph(full_graph, start_node_id, radius=max_hops)

    elif algorithm == "bfs_tree":
        subgraph = nx.bfs_tree(full_graph, start_node_id, depth_limit=max_hops)

    elif algorithm == "dfs_tree":
        subgraph = nx.dfs_tree(full_graph, start_node_id, depth_limit=max_hops)

    elif algorithm == "component":
        # Get weakly connected component containing start_node
        if full_graph.is_directed():
            components = nx.weakly_connected_components(full_graph)
        else:
            components = nx.connected_components(full_graph)

        # Find component containing start_node
        component_nodes = None
        for component in components:
            if start_node_id in component:
                component_nodes = component
                break

        if component_nodes:
            subgraph = full_graph.subgraph(component_nodes).copy()
        else:
            # Fallback to single node
            subgraph = full_graph.subgraph([start_node_id]).copy()

    else:
        raise ToolInputError(f"Unknown algorithm: {algorithm}", param_name="algorithm")

    # Limit size if needed
    if len(subgraph) > max_nodes:
        # Keep nodes closest to start_node by shortest path
        try:
            distances = nx.single_source_shortest_path_length(
                subgraph.to_undirected(), start_node_id, cutoff=max_hops
            )
            # Sort by distance, then by importance
            nodes_by_priority = sorted(
                distances.keys(),
                key=lambda n: (distances[n], -subgraph.nodes[n].get("importance", 0)),
            )
            selected_nodes = nodes_by_priority[:max_nodes]
            subgraph = subgraph.subgraph(selected_nodes).copy()
        except Exception:
            # Fallback: take highest importance nodes
            nodes_by_importance = sorted(
                subgraph.nodes(data=True), key=lambda x: x[1].get("importance", 0), reverse=True
            )
            top_nodes = [node_id for node_id, _ in nodes_by_importance[:max_nodes]]
            subgraph = subgraph.subgraph(top_nodes).copy()

    return subgraph


async def _extract_subgraph_data(
    conn, subgraph: nx.DiGraph, include_full_content: bool
) -> Tuple[List[Dict], List[Dict]]:
    """Extract nodes and edges data from NetworkX subgraph."""

    nodes_data = []
    edges_data = []

    # Extract nodes
    for node_id in subgraph.nodes():
        node_attrs = subgraph.nodes[node_id]
        node_dict = {
            "memory_id": node_id,
            "description": node_attrs.get("description", ""),
            "memory_type": node_attrs.get("memory_type", "unknown"),
            "memory_level": node_attrs.get("memory_level", "working"),
            "importance": node_attrs.get("importance", 5.0),
            "confidence": node_attrs.get("confidence", 1.0),
            "created_at_iso": node_attrs.get("created_at_iso"),
            "last_accessed_iso": node_attrs.get("last_accessed_iso"),
        }

        if include_full_content:
            # Get full content from database
            content_res = await conn.execute_fetchone(
                "SELECT content FROM memories WHERE memory_id = ?", (node_id,)
            )
            if content_res:
                node_dict["content"] = content_res["content"]
        else:
            node_dict["content_preview"] = node_attrs.get("content_preview", "")

        nodes_data.append(node_dict)

    # Extract edges
    for source, target in subgraph.edges():
        edge_attrs = subgraph.edges[source, target]
        edges_data.append(
            {
                "source": source,
                "target": target,
                "link_type": edge_attrs.get("link_type", "RELATED"),
                "strength": edge_attrs.get("strength", 1.0),
                "link_description": edge_attrs.get("link_description", ""),
            }
        )

    return nodes_data, edges_data


async def _perform_networkx_analysis(
    G: nx.DiGraph,
    centrality_algorithms: List[str],
    detect_communities: bool,
    community_algorithm: str,
    compute_graph_metrics: bool,
    include_shortest_paths: bool,
    shortest_path_targets: Optional[List[str]],
    centrality_top_k: int,
    min_community_size: int,
) -> Dict[str, Any]:
    """Perform comprehensive NetworkX analysis."""

    analysis = {}

    # Centrality Analysis
    if centrality_algorithms:
        centrality_scores = {}

        for algorithm in centrality_algorithms:
            try:
                if algorithm == "pagerank":
                    scores = nx.pagerank(G, alpha=0.85, max_iter=100)
                elif algorithm == "betweenness":
                    scores = nx.betweenness_centrality(G, k=min(100, len(G)))
                elif algorithm == "closeness":
                    scores = nx.closeness_centrality(G)
                elif algorithm == "degree":
                    scores = dict(G.degree())
                    max_degree = max(scores.values()) if scores else 1
                    scores = {k: v / max_degree for k, v in scores.items()}
                elif algorithm == "eigenvector":
                    try:
                        scores = nx.eigenvector_centrality(G, max_iter=100)
                    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                        scores = nx.eigenvector_centrality_numpy(G)
                elif algorithm == "katz":
                    try:
                        scores = nx.katz_centrality(G, alpha=0.1, max_iter=100)
                    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                        scores = nx.katz_centrality_numpy(G, alpha=0.1)

                # Get top K nodes
                top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                    :centrality_top_k
                ]
                centrality_scores[algorithm] = {
                    "top_nodes": dict(top_nodes),
                    "stats": {
                        "total_nodes": len(scores),
                        "max_score": max(scores.values()) if scores else 0,
                        "min_score": min(scores.values()) if scores else 0,
                        "mean_score": sum(scores.values()) / len(scores) if scores else 0,
                    },
                }

            except Exception as e:
                logger.warning(f"Centrality calculation failed for {algorithm}: {e}")
                centrality_scores[algorithm] = {"error": str(e)}

        analysis["centrality"] = centrality_scores

    # Community Detection
    if detect_communities and len(G) > 2:
        try:
            G_undirected = G.to_undirected()

            if community_algorithm == "louvain":
                try:
                    import community as community_louvain

                    partition = community_louvain.best_partition(G_undirected)
                except ImportError:
                    communities_gen = nx.community.greedy_modularity_communities(G_undirected)
                    partition = {}
                    for i, community in enumerate(communities_gen):
                        for node in community:
                            partition[node] = i

            elif community_algorithm == "greedy_modularity":
                communities_gen = nx.community.greedy_modularity_communities(G_undirected)
                partition = {}
                for i, community in enumerate(communities_gen):
                    for node in community:
                        partition[node] = i

            elif community_algorithm == "label_propagation":
                communities_gen = nx.community.label_propagation_communities(G_undirected)
                partition = {}
                for i, community in enumerate(communities_gen):
                    for node in community:
                        partition[node] = i

            else:  # leiden - fallback to greedy modularity
                communities_gen = nx.community.greedy_modularity_communities(G_undirected)
                partition = {}
                for i, community in enumerate(communities_gen):
                    for node in community:
                        partition[node] = i

            # Process communities
            communities_by_id = defaultdict(list)
            for node, comm_id in partition.items():
                communities_by_id[comm_id].append(node)

            # Filter by size and format
            communities = [
                {
                    "community_id": comm_id,
                    "members": members,
                    "size": len(members),
                    "avg_importance": sum(G.nodes[node].get("importance", 5.0) for node in members)
                    / len(members),
                }
                for comm_id, members in communities_by_id.items()
                if len(members) >= min_community_size
            ]

            # Calculate modularity
            try:
                modularity = nx.community.modularity(G_undirected, communities_by_id.values())
            except Exception:
                modularity = None

            analysis["communities"] = {
                "algorithm": community_algorithm,
                "communities": communities,
                "total_communities": len(communities),
                "modularity": modularity,
                "largest_community_size": max((c["size"] for c in communities), default=0),
            }

        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            analysis["communities"] = {"error": str(e)}

    # Graph Metrics
    if compute_graph_metrics:
        try:
            metrics = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "is_weakly_connected": nx.is_weakly_connected(G),
                "number_weakly_connected_components": nx.number_weakly_connected_components(G),
            }

            # Additional metrics for reasonable-sized graphs
            if 0 < len(G) <= 500:
                try:
                    G_undirected = G.to_undirected()
                    if nx.is_connected(G_undirected):
                        metrics["average_shortest_path_length"] = nx.average_shortest_path_length(
                            G_undirected
                        )

                    metrics["average_clustering"] = nx.average_clustering(G_undirected)

                    # Degree statistics
                    degrees = dict(G.degree())
                    if degrees:
                        degree_values = list(degrees.values())
                        metrics["degree_stats"] = {
                            "mean": sum(degree_values) / len(degree_values),
                            "max": max(degree_values),
                            "min": min(degree_values),
                        }

                except Exception as e:
                    logger.debug(f"Advanced graph metrics failed: {e}")

            analysis["graph_metrics"] = metrics

        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
            analysis["graph_metrics"] = {"error": str(e)}

    # Shortest Paths Analysis
    if include_shortest_paths and shortest_path_targets:
        try:
            paths_analysis = {}

            for target in shortest_path_targets:
                if target in G:
                    try:
                        # Paths TO this target
                        paths_to = nx.single_target_shortest_path_length(G, target, cutoff=5)

                        # Paths FROM this target
                        paths_from = nx.single_source_shortest_path_length(G, target, cutoff=5)

                        paths_analysis[target] = {
                            "reachable_from": len(paths_to),
                            "can_reach": len(paths_from),
                            "avg_distance_to": sum(paths_to.values()) / len(paths_to)
                            if paths_to
                            else 0,
                            "avg_distance_from": sum(paths_from.values()) / len(paths_from)
                            if paths_from
                            else 0,
                            "max_distance_to": max(paths_to.values()) if paths_to else 0,
                            "max_distance_from": max(paths_from.values()) if paths_from else 0,
                        }
                    except Exception as e:
                        paths_analysis[target] = {"error": str(e)}

            analysis["shortest_paths"] = paths_analysis

        except Exception as e:
            logger.warning(f"Shortest paths analysis failed: {e}")
            analysis["shortest_paths"] = {"error": str(e)}

    return analysis


@with_tool_metrics
@with_error_handling
async def get_memory_link_metadata(
    workflow_id: str,
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve metadata stored in the description field of a specific memory link.

    Args:
        workflow_id: Workflow scope for validation (consistent API)
        source_memory_id: The source memory ID of the link
        target_memory_id: The target memory ID of the link
        link_type: The type of link (case-insensitive)
        db_path: Database path

    Returns:
        Dict containing success status and the link's metadata
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")
    source_memory_id = _validate_uuid_format(source_memory_id, "source_memory_id")
    target_memory_id = _validate_uuid_format(target_memory_id, "target_memory_id")

    if not link_type or not isinstance(link_type, str):
        raise ToolInputError("link_type is required and must be a string.", param_name="link_type")

    # Normalize link_type to uppercase for consistency
    normalized_link_type = link_type.upper()

    logger.info(
        f"Retrieving metadata for link {_fmt_id(source_memory_id)} -> {_fmt_id(target_memory_id)} ({normalized_link_type})"
    )

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Validate that source memory belongs to the specified workflow for scoping
            source_memory_row = await conn.execute_fetchone(
                "SELECT 1 FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (source_memory_id, workflow_id),
            )
            if not source_memory_row:
                raise ToolInputError(
                    f"Source memory {source_memory_id} not found in workflow {workflow_id}.",
                    param_name="source_memory_id",
                )

            # Fetch the memory link
            link_row = await conn.execute_fetchone(
                """
                SELECT description, strength, created_at
                FROM memory_links 
                WHERE source_memory_id = ? AND target_memory_id = ? AND UPPER(link_type) = ?
                """,
                (source_memory_id, target_memory_id, normalized_link_type),
            )

            if not link_row:
                raise ToolInputError(
                    f"Memory link not found: {source_memory_id} -> {target_memory_id} ({normalized_link_type})",
                    param_name="link_type",
                )

            # Deserialize the description field (which may contain JSON metadata)
            description_json = link_row["description"]
            metadata = await MemoryUtils.deserialize(description_json)

            # If deserialization fails or returns None, use empty dict
            if metadata is None:
                metadata = {}

            # If the description was plain text (not JSON), store it as a text field in metadata
            if isinstance(metadata, str):
                metadata = {"description_text": metadata}
            elif not isinstance(metadata, dict):
                # If it's some other type, convert to string and store
                metadata = {"raw_description": str(metadata)}

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Retrieved metadata for link {_fmt_id(source_memory_id)} -> {_fmt_id(target_memory_id)} ({normalized_link_type})",
            emoji_key="link",
            time=processing_time,
        )

        return {
            "data": {
                "source_memory_id": source_memory_id,
                "target_memory_id": target_memory_id,
                "link_type": normalized_link_type,
                "metadata": metadata,
                "strength": link_row["strength"],
                "created_at_iso": to_iso_z(link_row["created_at"])
                if link_row["created_at"]
                else None,
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory link metadata: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve memory link metadata: {e}") from e


@with_tool_metrics
@with_error_handling
async def add_tag_to_memory(
    workflow_id: str,
    memory_id: str,
    tag: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Add a new tag to a memory's existing list of tags, ensuring uniqueness and case-insensitivity.

    Args:
        workflow_id: Required workflow scope
        memory_id: Required memory ID
        tag: The tag to add (will be normalized to lowercase and stripped)
        db_path: Database path

    Returns:
        Dict containing success status and updated tags list
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")
    memory_id = _validate_uuid_format(memory_id, "memory_id")

    if not tag or not isinstance(tag, str):
        raise ToolInputError("tag is required and must be a non-empty string.", param_name="tag")

    # Normalize tag: lowercase and strip whitespace
    normalized_tag = tag.strip().lower()
    if not normalized_tag:
        raise ToolInputError("tag cannot be empty after normalization.", param_name="tag")

    # Basic validation for tag content (no special characters that could break serialization)
    if any(char in normalized_tag for char in ['"', "'", "\n", "\r", "\t"]):
        raise ToolInputError("tag cannot contain quotes, newlines, or tabs.", param_name="tag")

    logger.info(
        f"Adding tag '{normalized_tag}' to memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}"
    )

    db = DBConnection(db_path)

    try:
        now = int(time.time())

        async with db.transaction() as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Fetch current tags for the memory (with FOR UPDATE for safe concurrent access)
            memory_row = await conn.execute_fetchone(
                "SELECT tags FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (memory_id, workflow_id),
            )

            if not memory_row:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            # Deserialize existing tags
            existing_tags_json = memory_row["tags"]
            current_tags = await MemoryUtils.deserialize(existing_tags_json)

            # If deserialization fails or returns None, start with empty list
            if not isinstance(current_tags, list):
                current_tags = []

            # Normalize existing tags for comparison (but preserve original case in storage)
            normalized_existing = [t.lower() for t in current_tags if isinstance(t, str)]

            # Check if the normalized tag already exists
            if normalized_tag in normalized_existing:
                # Tag already exists, no update needed
                logger.info(f"Tag '{normalized_tag}' already exists on memory {_fmt_id(memory_id)}")
                updated_tags = current_tags
            else:
                # Add the new tag (preserve original case from input after normalization)
                updated_tags = current_tags + [normalized_tag]

                # Serialize updated tags back to JSON
                serialized_tags = await MemoryUtils.serialize(updated_tags)
                if serialized_tags is None:
                    raise ToolInputError(
                        "Failed to serialize updated tags to JSON.", param_name="tag"
                    )

                # Update the memory with new tags and timestamp
                result = await conn.execute(
                    """
                    UPDATE memories 
                    SET tags = ?, updated_at = ? 
                    WHERE memory_id = ? AND workflow_id = ?
                    """,
                    (serialized_tags, now, memory_id, workflow_id),
                )

                if result.rowcount == 0:
                    raise ToolInputError(
                        f"Failed to update memory {memory_id} - memory may have been deleted.",
                        param_name="memory_id",
                    )

                logger.info(
                    f"Successfully added tag '{normalized_tag}' to memory {_fmt_id(memory_id)}"
                )

            # Log the tag addition operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "tag_added",
                memory_id,
                None,
                {
                    "added_tag": normalized_tag,
                    "total_tags_count": len(updated_tags),
                    "was_duplicate": normalized_tag in normalized_existing,
                },
            )

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Tag operation completed for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}",
            emoji_key="tag",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "memory_id": memory_id,
                "updated_tags": updated_tags,
                "added_tag": normalized_tag,
                "was_duplicate": normalized_tag in normalized_existing,
                "total_tags_count": len(updated_tags),
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error adding tag to memory: {e}", exc_info=True)
        raise ToolError(f"Failed to add tag to memory: {e}") from e


@with_tool_metrics
@with_error_handling
async def create_embedding(
    workflow_id: str,
    memory_id: str,
    *,
    force_recreation: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Explicitly trigger the creation (or re-creation) and storage of an embedding for a memory.

    Args:
        workflow_id: Required workflow scope
        memory_id: Required memory ID
        force_recreation: If True, create even if one exists (default: False)
        db_path: Database path

    Returns:
        Dict containing success status and embedding details
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")
    memory_id = _validate_uuid_format(memory_id, "memory_id")

    logger.info(
        f"Creating embedding for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)} (force={force_recreation})"
    )

    db = DBConnection(db_path)

    try:
        async with db.transaction() as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Check if embedding already exists (unless force_recreation is True)
            if not force_recreation:
                existing_embedding_row = await conn.execute_fetchone(
                    """
                    SELECT e.id as embedding_id, e.model, e.dimension, e.created_at,
                           LENGTH(e.embedding) as vector_size
                    FROM embeddings e
                    JOIN memories m ON m.embedding_id = e.id
                    WHERE m.memory_id = ? AND m.workflow_id = ?
                    """,
                    (memory_id, workflow_id),
                )

                if existing_embedding_row:
                    # Fetch the actual vector data
                    vector_row = await conn.execute_fetchone(
                        "SELECT embedding FROM embeddings WHERE id = ?",
                        (existing_embedding_row["embedding_id"],),
                    )

                    if vector_row:
                        # Deserialize the vector from binary format
                        import numpy as np

                        vector = np.frombuffer(vector_row["embedding"], dtype=np.float32).tolist()

                        processing_time = time.perf_counter() - ts_start

                        logger.info(
                            f"Retrieved existing embedding for memory {_fmt_id(memory_id)} "
                            f"(model: {existing_embedding_row['model']}, dim: {existing_embedding_row['dimension']})",
                            emoji_key="recycle",
                            time=processing_time,
                        )

                        return {
                            "success": True,
                            "data": {
                                "memory_id": memory_id,
                                "embedding_id": existing_embedding_row["embedding_id"],
                                "vector": vector,
                                "model_used": existing_embedding_row["model"],
                                "dimension": existing_embedding_row["dimension"],
                                "status": "retrieved_existing",
                                "created_at_iso": to_iso_z(existing_embedding_row["created_at"])
                                if existing_embedding_row["created_at"]
                                else None,
                            },
                            "processing_time": processing_time,
                        }

            # Fetch memory content and description
            memory_row = await conn.execute_fetchone(
                "SELECT content, description FROM memories WHERE memory_id = ? AND workflow_id = ?",
                (memory_id, workflow_id),
            )

            if not memory_row:
                raise ToolInputError(
                    f"Memory {memory_id} not found in workflow {workflow_id}.",
                    param_name="memory_id",
                )

            # Construct text to embed
            content = memory_row["content"] if memory_row["content"] else ""
            description = memory_row["description"] if memory_row["description"] else ""

            if description and content:
                text_to_embed = f"{description}: {content}"
            elif description:
                text_to_embed = description
            elif content:
                text_to_embed = content
            else:
                raise ToolInputError(
                    f"Memory {memory_id} has no content or description to embed.",
                    param_name="memory_id",
                )

            # Create the embedding using existing internal function
            embedding_db_id = await _store_embedding(conn, memory_id, text_to_embed)

            if not embedding_db_id:
                raise ToolError("Failed to create embedding - embedding service may be unavailable")

            # Fetch the newly created embedding details
            new_embedding_row = await conn.execute_fetchone(
                """
                SELECT id, model, dimension, created_at, embedding
                FROM embeddings 
                WHERE id = ?
                """,
                (embedding_db_id,),
            )

            if not new_embedding_row:
                raise ToolError("Failed to retrieve newly created embedding")

            # Deserialize the vector from binary format
            import numpy as np

            vector = np.frombuffer(new_embedding_row["embedding"], dtype=np.float32).tolist()

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Created new embedding for memory {_fmt_id(memory_id)} "
            f"(model: {new_embedding_row['model']}, dim: {new_embedding_row['dimension']})",
            emoji_key="sparkles",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "memory_id": memory_id,
                "embedding_id": embedding_db_id,
                "vector": vector,
                "model_used": new_embedding_row["model"],
                "dimension": new_embedding_row["dimension"],
                "status": "created",
                "created_at_iso": to_iso_z(new_embedding_row["created_at"])
                if new_embedding_row["created_at"]
                else None,
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error creating embedding: {e}", exc_info=True)
        raise ToolError(f"Failed to create embedding: {e}") from e


@with_tool_metrics
@with_error_handling
async def get_embedding(
    workflow_id: str,
    memory_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve the stored embedding vector for a memory.

    Args:
        workflow_id: Required workflow scope for auth/scoping
        memory_id: Required memory ID
        db_path: Database path

    Returns:
        Dict containing success status and the memory's embedding vector
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")
    memory_id = _validate_uuid_format(memory_id, "memory_id")

    logger.info(
        f"Retrieving embedding for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}"
    )

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Fetch embedding data with memory validation
            embedding_row = await conn.execute_fetchone(
                """
                SELECT e.embedding, e.model, e.dimension
                FROM embeddings e
                JOIN memories m ON e.memory_id = m.memory_id
                WHERE m.memory_id = ? AND m.workflow_id = ?
                """,
                (memory_id, workflow_id),
            )

            if not embedding_row:
                # Check if memory exists but has no embedding
                memory_exists = await conn.execute_fetchone(
                    "SELECT 1 FROM memories WHERE memory_id = ? AND workflow_id = ?",
                    (memory_id, workflow_id),
                )

                if not memory_exists:
                    raise ToolInputError(
                        f"Memory {memory_id} not found in workflow {workflow_id}.",
                        param_name="memory_id",
                    )

                # Memory exists but no embedding
                processing_time = time.perf_counter() - ts_start
                logger.info(f"No embedding found for memory {_fmt_id(memory_id)}")

                return {
                    "success": True,
                    "data": {
                        "memory_id": memory_id,
                        "vector": None,
                        "model": None,
                        "dimension": None,
                    },
                    "processing_time": processing_time,
                }

            # Deserialize the embedding vector
            embedding_blob = embedding_row.get("embedding")
            model = embedding_row.get("model")
            dimension = embedding_row.get("dimension")

            vector = None
            if embedding_blob:
                try:
                    # Deserialize the embedding BLOB into a list of floats
                    vector = await MemoryUtils.deserialize(embedding_blob)

                    # Ensure it's a list of numbers
                    if not isinstance(vector, list):
                        logger.warning(f"Embedding for memory {_fmt_id(memory_id)} is not a list")
                        vector = None
                    elif vector and not all(isinstance(x, (int, float)) for x in vector):
                        logger.warning(
                            f"Embedding for memory {_fmt_id(memory_id)} contains non-numeric values"
                        )
                        vector = None

                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize embedding for memory {_fmt_id(memory_id)}: {e}"
                    )
                    vector = None

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Retrieved embedding for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)} "
            f"(dimension: {dimension}, model: {model})",
            emoji_key="brain",
            time=processing_time,
        )

        return {
            "success": True,
            "data": {
                "memory_id": memory_id,
                "vector": vector,
                "model": model,
                "dimension": dimension,
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving embedding: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve embedding: {e}") from e


@with_tool_metrics
@with_error_handling
async def query_goals(
    workflow_id: str,
    *,
    status: Optional[str] = None,
    priority: Optional[int] = None,
    min_priority: Optional[int] = None,
    max_priority: Optional[int] = None,
    title_contains: Optional[str] = None,
    description_contains: Optional[str] = None,
    created_after_unix: Optional[int] = None,
    created_before_unix: Optional[int] = None,
    is_root_goal: Optional[bool] = None,
    sort_by: str = "priority,sequence_number",
    sort_order: str = "ASC",
    limit: int = 25,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Provide flexible querying for goals with advanced filtering and sorting.

    Args:
        workflow_id: Required workflow scope
        status: Filter by GoalStatus (e.g., "active", "completed")
        priority: Filter by exact priority level
        min_priority: Filter by minimum priority (inclusive)
        max_priority: Filter by maximum priority (inclusive)
        title_contains: Search in title using LIKE pattern
        description_contains: Search in description using LIKE pattern
        created_after_unix: Filter goals created after this Unix timestamp
        created_before_unix: Filter goals created before this Unix timestamp
        is_root_goal: If true, only root goals. If false, only non-root goals
        sort_by: Sort field(s), comma-separated. Allowed: priority, sequence_number, created_at, updated_at, title
        sort_order: Sort direction: ASC or DESC
        limit: Maximum number of goals to return (default: 25)
        offset: Number of goals to skip for pagination (default: 0)
        db_path: Database path

    Returns:
        Dict containing success status, goals list, and pagination info
    """
    ts_start = time.perf_counter()

    # Validate inputs
    workflow_id = _validate_uuid_format(workflow_id, "workflow_id")

    # Validate status if provided
    if status is not None:
        if not isinstance(status, str):
            raise ToolInputError("status must be a string.", param_name="status")
        try:
            GoalStatus(status.lower())
            normalized_status = status.lower()
        except ValueError as e:
            valid_statuses = [s.value for s in GoalStatus]
            raise ToolInputError(
                f"Invalid status '{status}'. Must be one of: {valid_statuses}", param_name="status"
            ) from e
    else:
        normalized_status = None

    # Validate priority filters
    if priority is not None and not isinstance(priority, int):
        raise ToolInputError("priority must be an integer.", param_name="priority")
    if min_priority is not None and not isinstance(min_priority, int):
        raise ToolInputError("min_priority must be an integer.", param_name="min_priority")
    if max_priority is not None and not isinstance(max_priority, int):
        raise ToolInputError("max_priority must be an integer.", param_name="max_priority")

    # Validate timestamp filters
    if created_after_unix is not None and not isinstance(created_after_unix, int):
        raise ToolInputError(
            "created_after_unix must be an integer.", param_name="created_after_unix"
        )
    if created_before_unix is not None and not isinstance(created_before_unix, int):
        raise ToolInputError(
            "created_before_unix must be an integer.", param_name="created_before_unix"
        )

    # Validate sort parameters
    allowed_sort_fields = {"priority", "sequence_number", "created_at", "updated_at", "title"}
    sort_fields = [field.strip() for field in sort_by.split(",")]
    for field in sort_fields:
        if field not in allowed_sort_fields:
            raise ToolInputError(
                f"Invalid sort field '{field}'. Must be one of: {sorted(allowed_sort_fields)}",
                param_name="sort_by",
            )

    if sort_order.upper() not in ("ASC", "DESC"):
        raise ToolInputError("sort_order must be 'ASC' or 'DESC'.", param_name="sort_order")

    # Validate pagination
    if limit <= 0:
        raise ToolInputError("limit must be positive.", param_name="limit")
    if limit > 1000:
        raise ToolInputError("limit cannot exceed 1000.", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset cannot be negative.", param_name="offset")

    logger.info(f"Querying goals for workflow {_fmt_id(workflow_id)} with filters")

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # Validate workflow exists
            workflow_row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not workflow_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Build WHERE clause dynamically
            where_clauses = ["workflow_id = ?"]
            params = [workflow_id]

            # Add filters
            if normalized_status is not None:
                where_clauses.append("status = ?")
                params.append(normalized_status)

            if priority is not None:
                where_clauses.append("priority = ?")
                params.append(priority)

            if min_priority is not None:
                where_clauses.append("priority >= ?")
                params.append(min_priority)

            if max_priority is not None:
                where_clauses.append("priority <= ?")
                params.append(max_priority)

            if title_contains is not None:
                where_clauses.append("title LIKE ?")
                params.append(f"%{title_contains}%")

            if description_contains is not None:
                where_clauses.append("description LIKE ?")
                params.append(f"%{description_contains}%")

            if created_after_unix is not None:
                where_clauses.append("created_at >= ?")
                params.append(created_after_unix)

            if created_before_unix is not None:
                where_clauses.append("created_at <= ?")
                params.append(created_before_unix)

            if is_root_goal is not None:
                if is_root_goal:
                    where_clauses.append("parent_goal_id IS NULL")
                else:
                    where_clauses.append("parent_goal_id IS NOT NULL")

            where_sql = " AND ".join(where_clauses)

            # Build ORDER BY clause
            order_clauses = []
            for field in sort_fields:
                order_clauses.append(f"{field} {sort_order.upper()}")
            order_sql = ", ".join(order_clauses)

            # Get total count
            count_sql = f"SELECT COUNT(*) as total FROM goals WHERE {where_sql}"
            count_result = await conn.execute_fetchone(count_sql, params)
            total_matching = count_result["total"] if count_result else 0

            # Execute main query
            sql = f"""
                SELECT goal_id, parent_goal_id, title, description, status, priority, 
                       sequence_number, created_at, updated_at, completed_at, 
                       acceptance_criteria, metadata, reasoning
                FROM goals 
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
            """

            query_params = params + [limit, offset]
            goal_rows = await conn.execute_fetchall(sql, query_params)

            # Process results
            goals = []
            for row in goal_rows:
                goal_dict = {
                    "goal_id": row["goal_id"],
                    "parent_goal_id": row["parent_goal_id"],
                    "title": row["title"],
                    "description": row["description"],
                    "status": row["status"],
                    "priority": row["priority"],
                    "sequence_number": row["sequence_number"],
                    "reasoning": row["reasoning"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "completed_at": row["completed_at"],
                }

                # Deserialize JSON fields
                try:
                    acceptance_criteria_json = row.get("acceptance_criteria")
                    acceptance_criteria = await MemoryUtils.deserialize(acceptance_criteria_json)
                    if not isinstance(acceptance_criteria, list):
                        acceptance_criteria = []
                    goal_dict["acceptance_criteria"] = acceptance_criteria
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize acceptance_criteria for goal {row['goal_id']}: {e}"
                    )
                    goal_dict["acceptance_criteria"] = []

                try:
                    metadata_json = row.get("metadata")
                    metadata = await MemoryUtils.deserialize(metadata_json)
                    if not isinstance(metadata, dict):
                        metadata = {}
                    goal_dict["metadata"] = metadata
                except Exception as e:
                    logger.warning(f"Failed to deserialize metadata for goal {row['goal_id']}: {e}")
                    goal_dict["metadata"] = {}

                # Add ISO timestamps
                goal_dict["created_at_iso"] = (
                    to_iso_z(row["created_at"]) if row["created_at"] else None
                )
                goal_dict["updated_at_iso"] = (
                    to_iso_z(row["updated_at"]) if row["updated_at"] else None
                )
                goal_dict["completed_at_iso"] = (
                    to_iso_z(row["completed_at"]) if row["completed_at"] else None
                )

                goals.append(goal_dict)

        processing_time = time.perf_counter() - ts_start

        logger.info(
            f"Queried {len(goals)} goals (of {total_matching} total matching) for workflow {_fmt_id(workflow_id)}",
            emoji_key="target",
            time=processing_time,
        )

        return {
            "data": {
                "goals": goals,
                "total_matching": total_matching,
                "limit_applied": limit,
                "offset_applied": offset,
                "query_params": {
                    "workflow_id": workflow_id,
                    "status": normalized_status,
                    "priority": priority,
                    "min_priority": min_priority,
                    "max_priority": max_priority,
                    "title_contains": title_contains,
                    "description_contains": description_contains,
                    "created_after_unix": created_after_unix,
                    "created_before_unix": created_before_unix,
                    "is_root_goal": is_root_goal,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                },
            },
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error querying goals: {e}", exc_info=True)
        raise ToolError(f"Failed to query goals: {e}") from e


# ======================================================
# Meta-Cognition Tools
# ======================================================


# --- Helper: Generate Consolidation Prompt (FULL INSTRUCTIONS) ---
def _generate_consolidation_prompt(memories: List[Dict], consolidation_type: str) -> str:
    """Generates a prompt for memory consolidation based on the type, with full instructions."""
    # Configurable limits for prompt generation
    MAX_MEMORIES_IN_PROMPT = 20
    MAX_CONTENT_PREVIEW_CHARS = 300
    MAX_DESCRIPTION_CHARS = 80

    # Format memories as text (Limit input memories and content length for prompt size)
    memory_texts = []
    # Limit source memories included in prompt to avoid excessive length
    for i, memory in enumerate(memories[:MAX_MEMORIES_IN_PROMPT], 1):
        desc = (memory.get("description") or "")[:MAX_DESCRIPTION_CHARS]
        # Limit content preview to keep prompts manageable
        content_preview = (memory.get("content", "") or "")[:MAX_CONTENT_PREVIEW_CHARS]
        mem_type = memory.get("memory_type", "N/A")
        importance = memory.get("importance", 5.0)
        confidence = memory.get("confidence", 1.0)
        created_ts = memory.get("created_at", 0)
        created_dt_str = (
            datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")
            if created_ts
            else "Unknown Date"
        )
        mem_id_short = memory.get("memory_id", "UNKNOWN")[:8]

        formatted = f"--- MEMORY #{i} (ID: {mem_id_short}..., Type: {mem_type}, Importance: {importance:.1f}, Confidence: {confidence:.1f}, Date: {created_dt_str}) ---\n"
        if desc:
            formatted += f"Description: {desc}\n"
        formatted += f"Content Preview: {content_preview}"
        # Indicate truncation
        if len(memory.get("content", "")) > MAX_CONTENT_PREVIEW_CHARS:
            formatted += "...\n"
        else:
            formatted += "\n"
        memory_texts.append(formatted)

    memories_str = "\n".join(memory_texts)

    # Base prompt template
    base_prompt = f"""You are an advanced cognitive system processing and consolidating memories for an AI agent. Below are {len(memories)} memory items containing information, observations, and insights relevant to a task. Your goal is to perform a specific type of consolidation: '{consolidation_type}'.

Analyze the following memories carefully:

{memories_str}
--- END OF MEMORIES ---

"""

    # Add specific instructions based on consolidation type (FULL INSTRUCTIONS)
    if consolidation_type == "summary":
        base_prompt += """TASK: Create a comprehensive and coherent summary that synthesizes the key information and context from ALL the provided memories. Your summary should:
1.  Distill the most critical facts, findings, and core ideas presented across the memories.
2.  Organize the information logically, perhaps chronologically or thematically, creating a clear narrative flow.
3.  Highlight significant connections, relationships, or developments revealed by considering the memories together.
4.  Eliminate redundancy while preserving essential details and nuances.
5.  Be objective and accurately reflect the content of the source memories.
6.  Be well-structured and easy to understand for someone reviewing the workflow's progress.

Generate ONLY the summary content based on the provided memories.

CONSOLIDATED SUMMARY:"""

    elif consolidation_type == "insight":
        base_prompt += """TASK: Generate high-level insights by identifying significant patterns, implications, conclusions, or discrepancies emerging from the provided memories. Your insights should:
1.  Go beyond simple summarization to reveal non-obvious patterns, trends, or relationships connecting different memories.
2.  Draw meaningful conclusions or formulate hypotheses that are supported by the collective information but may not be explicit in any single memory.
3.  Explicitly highlight any contradictions, tensions, or unresolved issues found between memories.
4.  Identify the broader significance, potential impact, or actionable implications of the combined information.
5.  Be stated clearly and concisely, using cautious language where certainty is limited (e.g., "It appears that...", "This might suggest...").
6.  Focus on the most impactful and novel understandings gained from analyzing these memories together.

Generate ONLY the list of insights based on the provided memories.

CONSOLIDATED INSIGHTS:"""

    elif consolidation_type == "procedural":
        base_prompt += """TASK: Formulate a generalized procedure, method, or set of steps based on the actions, outcomes, and observations described in the memories. Your procedure should:
1.  Identify recurring sequences of actions or steps that appear to lead to successful or notable outcomes.
2.  Generalize from specific instances described in the memories to create a potentially reusable approach or workflow pattern.
3.  Clearly outline the sequence of steps involved in the procedure.
4.  Note important conditions, prerequisites, inputs, outputs, or constraints associated with the procedure or its steps.
5.  Highlight decision points, potential variations, or common failure points if identifiable from the memories.
6.  Be structured as a clear set of instructions or a logical flow that could guide similar future situations.

Generate ONLY the procedure based on the provided memories.

CONSOLIDATED PROCEDURE:"""

    elif consolidation_type == "question":
        base_prompt += """TASK: Identify the most important and actionable questions that arise from analyzing these memories. Your questions should:
1.  Target significant gaps in knowledge, understanding, or information revealed by the memories.
2.  Highlight areas of uncertainty, ambiguity, or contradiction that require further investigation or clarification.
3.  Focus on issues that are critical for achieving the implied or stated goals related to these memories.
4.  Be specific and well-defined enough to guide further research, analysis, or action.
5.  Be prioritized, starting with the most critical or foundational questions.
6.  Avoid questions that are already answered within the provided memory content.

Generate ONLY the list of questions based on the provided memories.

CONSOLIDATED QUESTIONS:"""

    # The final marker like "CONSOLIDATED SUMMARY:" is added by the TASK instruction itself.
    return base_prompt


# Helper for reflection prompt (similar structure to consolidation)
def _generate_reflection_prompt(
    workflow_name: str,
    workflow_desc: Optional[str],
    operations: List[Dict],
    memories: Dict[str, Dict],
    reflection_type: str,
) -> str:
    """
    Generate a rich prompt for reflective analysis.
    """
    # ───────── format recent operations ─────────
    MAX_OPERATIONS_IN_PROMPT = 30
    MAX_OP_DATA_CHARS = 20
    MAX_MEMORY_DESC_CHARS = 40

    formatted_ops: list[str] = []
    for idx, op in enumerate(
        operations[:MAX_OPERATIONS_IN_PROMPT], 1
    ):  # cap to keep prompt size sane
        ts_unix: int = op.get("timestamp", 0) or 0
        ts_human = (
            datetime.fromtimestamp(ts_unix).strftime("%Y-m-d %H:%M:%S") if ts_unix else "Unknown-TS"
        )
        op_type = op.get("operation", "UNKNOWN").upper()
        mem_id: str | None = op.get("memory_id")
        act_id: str | None = op.get("action_id")

        parts: list[str] = [f"OP #{idx} ({ts_human})", f"Type: {op_type}"]

        # ─── memory reference ───
        if mem_id:
            mem_meta = memories.get(mem_id)
            if mem_meta:
                desc = (mem_meta.get("description") or "")[:MAX_MEMORY_DESC_CHARS] or "N/A"
                mtype = mem_meta.get("memory_type") or "N/A"
                parts.append(f"Memory(id≈{mem_id[:6]}, type={mtype}, desc={desc})")
            else:
                # log for engineers, redact for LLM
                logger.warning(
                    f"Reflection-prompt: memory_id '{mem_id}' referenced in op #{idx} "
                    "has no metadata; redacting from prompt.",
                )
                parts.append("(memory details unavailable)")

        # ─── action reference ───
        if act_id:
            parts.append(f"Action(id≈{act_id[:6]})")

        # ─── trimmed op-data details ───
        op_data_raw = op.get("operation_data")
        if op_data_raw:
            try:
                op_data = json.loads(op_data_raw)
            except (TypeError, json.JSONDecodeError):
                op_data = {"raw_snippet": str(op_data_raw)[:MAX_OP_DATA_CHARS]}

            kv_snippets = [
                f"{k}={str(v)[:MAX_OP_DATA_CHARS]}"  # keep very short
                for k, v in op_data.items()
                if k not in {"content", "prompt", "embedding"}
            ]
            if kv_snippets:
                parts.append("Data(" + ", ".join(kv_snippets) + ")")

        formatted_ops.append(" | ".join(parts))

    operations_block = "\n".join(formatted_ops)

    # ───────── assemble prompt header ─────────
    prompt_header = (
        f'You are an advanced meta-cognitive module analysing the workflow "{workflow_name}".\n'
        f"Workflow description: {workflow_desc or 'N/A'}\n"
        f"Your task: produce a **{reflection_type.upper()}** reflection based solely on the "
        f"recent operations below.\n\n"
        "RECENT OPERATIONS (newest first):\n"
        f"{operations_block}\n\n"
    )

    # ───────── task-specific instructions ─────────
    task_instructions: dict[str, str] = {
        "summary": "TASK: Provide a reflective summary highlighting key developments, insights, "
        "and current state. Focus on clarity and completeness.\n\nREFLECTIVE SUMMARY:",
        "progress": "TASK: Analyse progress toward goals, citing concrete evidence from the "
        "operations. Identify milestones and obstacles.\n\nPROGRESS ANALYSIS:",
        "gaps": "TASK: Detect knowledge gaps, contradictions, or unanswered questions. Formulate "
        "specific follow-up questions.\n\nKNOWLEDGE GAPS ANALYSIS:",
        "strengths": "TASK: Identify successful patterns, effective strategies, and strengths "
        "demonstrated in the operations.\n\nSTRENGTHS ANALYSIS:",
        "plan": "TASK: Propose the next strategic steps the agent should take, grounded in the "
        "operations log.\n\nSTRATEGIC PLAN:",
    }

    return prompt_header + task_instructions.get(reflection_type, "").rstrip()


# --- Tool: Consolidate Memories ---
@with_tool_metrics
@with_error_handling
async def consolidate_memories(
    *,
    workflow_id: str | None = None,
    target_memories: list[str] | None = None,
    consolidation_type: str = "summary",
    query_filter: dict[str, Any] | None = None,
    max_source_memories: int = 20,
    prompt_override: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    store_result: bool = True,
    store_as_level: str = MemoryLevel.SEMANTIC.value,
    store_as_type: str | None = None,
    max_tokens: int = 1_000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Synthesise multiple memories into *summary/insight/procedural/question* content.
    """
    t0 = time.time()
    valid_types = {"summary", "insight", "procedural", "question"}
    if consolidation_type not in valid_types:
        raise ToolInputError("Invalid consolidation_type.", param_name="consolidation_type")

    db = DBConnection(db_path)
    source_rows: list[dict[str, Any]] = []
    effective_wf = workflow_id  # may be inferred

    # ─────────────────────── 1. Select source memories ────────────────────────
    async with db.transaction(readonly=True) as conn:
        # 1-a. Explicit list supplied
        if target_memories:
            if len(target_memories) < 2:
                raise ToolInputError("Need at least two target_memories.", "target_memories")

            ph = ",".join("?" * len(target_memories))
            rows = await conn.execute_fetchall(
                f"SELECT * FROM memories WHERE memory_id IN ({ph})", target_memories
            )
            found = {r["memory_id"]: dict(r) for r in rows}
            if not rows:
                raise ToolInputError("No target_memories found.", "target_memories")

            effective_wf = effective_wf or rows[0]["workflow_id"]

            issues: list[str] = []
            for mid in target_memories:
                r = found.get(mid)
                if not r:
                    issues.append(f"{mid} not found")
                elif r["workflow_id"] != effective_wf:
                    issues.append(
                        f"{mid} is in workflow {r['workflow_id']}, expected {effective_wf}"
                    )
                else:
                    source_rows.append(r)
            if issues:
                raise ToolInputError("; ".join(issues), "target_memories")

            # ─── Fix-15: cap list length here too ────────────────────────────
            if len(source_rows) > max_source_memories:
                # replicate ORDER BY importance DESC, created_at DESC
                source_rows.sort(
                    key=lambda r: (
                        -(r.get("importance") or 0.0),
                        -(r.get("created_at") or 0),
                    )
                )
                source_rows = source_rows[:max_source_memories]

        # 1-b. Query-filter branch (already capped by LIMIT ?)
        elif query_filter:
            if not effective_wf:
                raise ToolInputError("workflow_id required with query_filter.", "workflow_id")

            where, params = ["workflow_id = ?"], [effective_wf]
            for k, v in (query_filter or {}).items():
                match k:
                    case "memory_level" | "memory_type" | "source" if v:
                        where.append(f"{k} = ?")
                        params.append(str(v).lower())
                    case "min_importance" if v is not None:
                        where.append("importance >= ?")
                        params.append(float(v))
                    case "min_confidence" if v is not None:
                        where.append("confidence >= ?")
                        params.append(float(v))
            nowu = int(time.time())
            where.append("(ttl = 0 OR created_at + ttl > ?)")
            params.append(nowu)
            params.append(max_source_memories)  # LIMIT param

            sql = (
                f"SELECT * FROM memories WHERE {' AND '.join(where)} "
                "ORDER BY importance DESC, created_at DESC LIMIT ?"
            )
            source_rows = [dict(r) async for r in conn.execute(sql, params)]

        # 1-c. Fallback: whole-workflow query (was already limited)
        else:
            if not effective_wf:
                raise ToolInputError("workflow_id required.", "workflow_id")
            nowu = int(time.time())
            sql = (
                "SELECT * FROM memories WHERE workflow_id = ? "
                "AND (ttl = 0 OR created_at + ttl > ?) "
                "ORDER BY importance DESC, created_at DESC LIMIT ?"
            )
            source_rows = [
                dict(r) async for r in conn.execute(sql, (effective_wf, nowu, max_source_memories))
            ]

    if len(source_rows) < 2:
        raise ToolError("Need ≥ 2 source memories after filtering/capping.")
    source_ids = [r["memory_id"] for r in source_rows]

    # ─────────────────────── 2. Build LLM prompt ───────────────────────────────
    prompt = prompt_override or _generate_consolidation_prompt(source_rows, consolidation_type)

    cfg = get_config()
    provider_name = provider or cfg.default_provider or LLMGatewayProvider.OPENAI.value
    provider_inst = await get_provider(provider_name)
    if not provider_inst:
        raise ToolError(f"LLM provider '{provider_name}' unavailable.")
    model_name = model or provider_inst.get_default_model()

    # ─────────────────────── 3. Call LLM ───────────────────────────────────────
    try:
        llm_resp = await provider_inst.generate_completion(
            prompt=prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.6,
        )
        consolidated = llm_resp.text.strip()
    except Exception as e:
        logger.error("LLM error in consolidation.", exc_info=True)
        raise ToolError(f"LLM error: {e}") from e

    # ─────────────────────── 4. Optionally store result ───────────────────────
    stored_id: str | None = None
    async with db.transaction() as wconn:
        if store_result and consolidated:
            mtype = (
                store_as_type
                or {
                    "summary": MemoryType.SUMMARY.value,
                    "insight": MemoryType.INSIGHT.value,
                    "procedural": MemoryType.PROCEDURE.value,
                    "question": MemoryType.QUESTION.value,
                }[consolidation_type]
            )
            try:
                mlevel = MemoryLevel(store_as_level.lower())
            except ValueError:
                mlevel = MemoryLevel.SEMANTIC

            # ---- derive scoring ----
            src_imp = [r.get("importance", 5.0) for r in source_rows]
            src_conf = [r.get("confidence", 0.5) for r in source_rows]
            imp = min(max(max(src_imp) + 0.5, 0.0), 10.0)

            #  Fix-16 → ensure ≥ MIN_CONFIDENCE_SEMANTIC
            conf_raw = max(0.1, min(sum(src_conf) / len(src_conf), 1.0))
            conf = round(max(conf_raw, MIN_CONFIDENCE_SEMANTIC), 3)

            res = await store_memory(
                workflow_id=effective_wf,
                content=consolidated,
                memory_type=mtype,
                memory_level=mlevel.value,
                importance=round(imp, 2),
                confidence=conf,
                description=f"Consolidated {consolidation_type} from {len(source_ids)} memories.",
                source=f"consolidation_{consolidation_type}",
                tags=["consolidated", consolidation_type, mtype, mlevel.value],
                context_data={
                    "source_memories": source_ids,
                    "consolidation_type": consolidation_type,
                },
                generate_embedding=True,
                suggest_links=True,
                db_path=db_path,
            )
            stored_id = res.get("memory_id")

            # ---- link back to sources (fire-and-forget) ----
            if stored_id:
                await asyncio.gather(
                    *(
                        create_memory_link(
                            source_memory_id=stored_id,
                            target_memory_id=sid,
                            link_type=LinkType.GENERALIZES.value,
                            description=f"Source for {consolidation_type}",
                            db_path=db_path,
                        )
                        for sid in source_ids
                    ),
                    return_exceptions=True,
                )

        # ---- operation log ----
        await MemoryUtils._log_memory_operation(
            wconn,
            effective_wf,
            "consolidate",
            None,
            None,
            {
                "consolidation_type": consolidation_type,
                "source_count": len(source_ids),
                "llm_provider": provider_name,
                "llm_model": model_name or "provider_default",
                "stored_memory_id": stored_id,
                "content_length": len(consolidated),
            },
        )

    elapsed = time.time() - t0
    logger.info(
        f"Consolidated {len(source_ids)} memories → {stored_id if stored_id else 'not-stored'} in {elapsed:.2f}s",
        emoji_key="sparkles",
    )
    return {
        "success": True,
        "data": {
            "consolidated_content": consolidated or "LLM produced no content.",
            "consolidation_type": consolidation_type,
            "source_memory_ids": source_ids,
            "workflow_id": effective_wf,
            "stored_memory_id": stored_id,
        },
        "processing_time": elapsed,
    }


@with_tool_metrics
@with_error_handling
async def generate_reflection(
    workflow_id: str,
    *,
    reflection_type: str = "summary",  # summary | progress | gaps | strengths | plan
    recent_ops_limit: int = 30,
    provider: str = LLMGatewayProvider.OPENAI.value,
    model: str | None = None,
    max_tokens: int = 1_000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Run a meta-cognitive LLM pass over recent workflow activity and persist the result.
    """
    t0 = time.time()
    if reflection_type not in {"summary", "progress", "gaps", "strengths", "plan"}:
        raise ToolInputError(
            "Invalid reflection_type. Must be one of: summary, progress, gaps, strengths, plan",
            param_name="reflection_type",
        )

    db = DBConnection(db_path)

    # ───────────────────────── 1. fetch context (read-only) ─────────────────────────
    ops: list[dict[str, Any]]
    mem_ids: set[str] = set()
    mem_meta: dict[str, dict[str, Any]] = {}

    async with db.transaction(readonly=True) as conn:
        wf_row = await conn.execute_fetchone(
            "SELECT title, description FROM workflows WHERE workflow_id=?",
            (workflow_id,),
        )
        if wf_row is None:
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        wf_title, wf_desc = wf_row["title"], wf_row["description"]

        raw_ops_rows = await conn.execute_fetchall(
            "SELECT * FROM memory_operations WHERE workflow_id=? ORDER BY timestamp DESC LIMIT ?",
            (workflow_id, recent_ops_limit),
        )
        # Ensure ops is a list of dicts for consistent access
        ops = [dict(row) for row in raw_ops_rows]

        if not ops:
            raise ToolError("No memory_operations to analyse for this workflow.")

        for op in ops:  # Now op is a dictionary
            if op.get(
                "memory_id"
            ):  # Use .get() for safety, though direct access would also work now
                mem_ids.add(op["memory_id"])

        if mem_ids:
            placeholders = ",".join("?" * len(mem_ids))
            async with conn.execute(
                f"""
                SELECT memory_id, description, memory_type
                FROM memories
                WHERE memory_id IN ({placeholders})
                """,
                list(mem_ids),
            ) as cur:
                async for row in cur:
                    mem_meta[row["memory_id"]] = dict(row)

    # ───────────────────────── 2. build prompt ─────────────────────────
    prompt = _generate_reflection_prompt(wf_title, wf_desc, ops, mem_meta, reflection_type)

    # ───────────────────────── 3. call LLM ────────────────────────────
    provider_cfg = get_config()
    provider_name = provider or provider_cfg.default_provider or LLMGatewayProvider.OPENAI.value
    prov = await get_provider(provider_name)
    if prov is None:
        raise ToolError(f"Could not initialise LLM provider '{provider_name}'.")

    model_name = model or prov.get_default_model()
    try:
        llm_out = await prov.generate_completion(
            prompt=prompt, model=model_name, max_tokens=max_tokens, temperature=0.7
        )
    except Exception as llm_err:
        logger.error("LLM failure during reflection", exc_info=True)
        raise ToolError(f"Reflection failed (LLM): {llm_err}") from llm_err

    content = llm_out.text.strip()
    if not content:
        raise ToolError("LLM returned empty reflection.")

    # ───────────────────────── 4. persist reflection (write txn) ─────────────────────
    refl_id = MemoryUtils.generate_id()
    now = int(time.time())
    title = (content.split("\n", 1)[0].lstrip("# ").strip() or reflection_type.title())[:100]

    async with db.transaction() as wconn:
        await wconn.execute(
            """
            INSERT INTO reflections
                  (reflection_id, workflow_id, title, content, reflection_type,
                   created_at, referenced_memories)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                refl_id,
                workflow_id,
                title,
                content,
                reflection_type,
                now,
                json.dumps(list(mem_ids)),
            ),
        )
        await MemoryUtils._log_memory_operation(
            wconn,
            workflow_id,
            "reflect",
            None,  # memory_id
            None,  # action_id
            {
                "reflection_id": refl_id,
                "reflection_type": reflection_type,
                "ops_analyzed": len(ops),
                "title": title,
            },
        )

    elapsed = time.time() - t0
    logger.info(
        f"Reflection '{title}' ({_fmt_id(refl_id)}) generated for workflow {_fmt_id(workflow_id)} "
        f"in {elapsed:0.2f}s",
        emoji_key="mirror",
    )
    return {
        "success": True,
        "data": {
            "reflection_id": refl_id,
            "workflow_id": workflow_id,
            "reflection_type": reflection_type,
            "title": title,
            "content": content,
            "operations_analyzed": len(ops),
        },
        "processing_time": elapsed,
    }


@with_tool_metrics
@with_error_handling
async def diagnose_file_access_issues(
    path_to_check: Optional[str] = None,
    operation_type: str = "database",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Diagnose and provide solutions for file access denied issues.

    This tool helps agents understand why file operations are failing and provides
    specific guidance on harmonizing UMS and filesystem.py validation systems.

    Args:
        path_to_check: Specific path to check (defaults to database path)
        operation_type: Type of operation ('database', 'artifacts', 'logs')
        db_path: Database path to validate

    Returns:
        Dict with comprehensive diagnosis and actionable recommendations
    """
    start_time = time.time()
    import os
    import tempfile

    check_path = path_to_check or db_path

    diagnosis = {
        "original_path": check_path,
        "operation_type": operation_type,
        "issues_found": [],
        "recommendations": [],
        "safe_alternatives": [],
        "current_permissions": {},
        "filesystem_integration": {
            "filesystem_tools_available": False,
            "allowed_directories_configured": False,
            "path_in_allowed_dirs": False,
            "validation_harmonized": False,
        },
        "ums_validation": {"passed": False, "fallback_used": False, "final_path": None},
    }

    try:
        path_obj = Path(check_path).resolve()
        diagnosis["resolved_path"] = str(path_obj)

        # Test UMS validation first
        try:
            validated_ums_path = validate_and_secure_db_path(check_path)
            diagnosis["ums_validation"]["passed"] = True
            diagnosis["ums_validation"]["final_path"] = validated_ums_path
            diagnosis["ums_validation"]["fallback_used"] = validated_ums_path != str(path_obj)

            if diagnosis["ums_validation"]["fallback_used"]:
                diagnosis["issues_found"].append(
                    f"UMS validation rejected original path and used fallback: {validated_ums_path}"
                )
        except Exception as e:
            diagnosis["ums_validation"]["error"] = str(e)
            diagnosis["issues_found"].append(f"UMS validation failed: {e}")

        # Check filesystem tools integration
        try:
            from ultimate_mcp_server.tools.filesystem import get_allowed_directories

            diagnosis["filesystem_integration"]["filesystem_tools_available"] = True

            try:
                allowed_dirs = get_allowed_directories()
                diagnosis["filesystem_integration"]["allowed_directories_configured"] = (
                    len(allowed_dirs) > 0
                )
                diagnosis["filesystem_integration"]["allowed_directories"] = allowed_dirs

                # Check if path is in allowed directories
                for allowed_dir in allowed_dirs:
                    try:
                        allowed_path = Path(allowed_dir).resolve()
                        if str(path_obj).startswith(str(allowed_path) + os.sep) or str(
                            path_obj
                        ) == str(allowed_path):
                            diagnosis["filesystem_integration"]["path_in_allowed_dirs"] = True
                            break
                    except Exception:
                        continue

            except Exception as e:
                diagnosis["issues_found"].append(f"Error getting allowed directories: {e}")

            # Check filesystem validation availability (lightweight check)
            try:
                # Just test if validation function is available without calling it
                diagnosis["filesystem_integration"]["validation_available"] = True

                # For harmonization check, compare the actual paths used by both systems
                if diagnosis["ums_validation"]["final_path"]:
                    # Check if UMS final path would be in allowed directories
                    ums_final = Path(diagnosis["ums_validation"]["final_path"]).resolve()
                    for allowed_dir in allowed_dirs:
                        try:
                            allowed_path = Path(allowed_dir).resolve()
                            if str(ums_final).startswith(str(allowed_path) + os.sep) or str(
                                ums_final
                            ) == str(allowed_path):
                                diagnosis["filesystem_integration"]["validation_harmonized"] = True
                                break
                        except Exception:
                            continue
                    else:
                        # UMS path is not in allowed directories
                        diagnosis["filesystem_integration"]["validation_harmonized"] = False
                        diagnosis["issues_found"].append(
                            f"UMS path not in filesystem allowed directories. "
                            f"UMS uses: {diagnosis['ums_validation']['final_path']}, "
                            f"Allowed dirs: {allowed_dirs}"
                        )

            except Exception as e:
                diagnosis["filesystem_integration"]["validation_available"] = False
                diagnosis["issues_found"].append(f"Filesystem validation check failed: {e}")

        except ImportError:
            diagnosis["issues_found"].append(
                "Filesystem tools not available for unified validation"
            )
            diagnosis["recommendations"].append(
                "Enable filesystem.py tools integration for consistent path validation"
            )

        # Check basic path properties
        if path_obj.exists():
            diagnosis["path_exists"] = True
            try:
                diagnosis["current_permissions"] = {
                    "readable": os.access(str(path_obj), os.R_OK),
                    "writable": os.access(str(path_obj), os.W_OK),
                    "executable": os.access(str(path_obj), os.X_OK),
                }
            except Exception as e:
                diagnosis["current_permissions"]["error"] = str(e)
        else:
            diagnosis["path_exists"] = False
            # Check parent directory permissions
            parent = path_obj.parent
            if parent.exists():
                try:
                    diagnosis["current_permissions"] = {
                        "parent_writable": os.access(str(parent), os.W_OK),
                        "parent_readable": os.access(str(parent), os.R_OK),
                        "parent_executable": os.access(str(parent), os.X_OK),
                    }
                except Exception as e:
                    diagnosis["current_permissions"]["error"] = str(e)

        # Generate specific recommendations
        if diagnosis["ums_validation"]["final_path"]:
            diagnosis["safe_alternatives"].append(diagnosis["ums_validation"]["final_path"])

        # Include filesystem-aware alternatives
        if diagnosis["filesystem_integration"]["filesystem_tools_available"]:
            for allowed_dir in diagnosis["filesystem_integration"].get("allowed_directories", []):
                candidate = Path(allowed_dir) / "ultimate_mcp_server" / f"{operation_type}"
                if str(candidate) not in diagnosis["safe_alternatives"]:
                    diagnosis["safe_alternatives"].append(str(candidate))

        # Traditional safe alternatives
        traditional_alternatives = [
            str(Path.home() / ".ultimate_mcp_server" / Path(check_path).name),
            str(Path.cwd() / "data" / Path(check_path).name),
            str(Path(tempfile.gettempdir()) / "ultimate_mcp_server" / Path(check_path).name),
        ]

        for alt in traditional_alternatives:
            if alt not in diagnosis["safe_alternatives"]:
                diagnosis["safe_alternatives"].append(alt)

        # Provide integration-specific recommendations
        if not diagnosis["filesystem_integration"]["filesystem_tools_available"]:
            diagnosis["recommendations"].append(
                "Enable filesystem.py tools for unified validation across all file operations"
            )
        elif not diagnosis["filesystem_integration"]["allowed_directories_configured"]:
            diagnosis["recommendations"].append(
                "Configure filesystem.allowed_directories in your config to specify safe locations"
            )
        elif not diagnosis["filesystem_integration"]["path_in_allowed_dirs"]:
            diagnosis["recommendations"].append(
                f"Move {operation_type} files to one of the allowed directories for consistency"
            )

        if not diagnosis["filesystem_integration"]["validation_harmonized"]:
            diagnosis["recommendations"].append(
                "Update configuration to ensure UMS and filesystem validation use the same rules"
            )

        # Provide specific recommendations based on operation type
        if operation_type == "database":
            diagnosis["recommendations"].extend(
                [
                    "Configure database path in a user-writable directory",
                    "Ensure the UMS config points to ~/.ultimate_mcp_server/",
                    "Avoid system directories like /var, /etc, /usr",
                    "Consider using a path that's also in filesystem.allowed_directories",
                ]
            )
        elif operation_type == "artifacts":
            diagnosis["recommendations"].extend(
                [
                    "Store artifacts in user home directory or project data folder",
                    "Use relative paths within the project workspace",
                    "Ensure artifact paths are within filesystem.allowed_directories",
                ]
            )
        elif operation_type == "logs":
            diagnosis["recommendations"].extend(
                [
                    "Configure log directory to ~/.ultimate_mcp_server/logs/",
                    "Ensure log rotation is properly configured",
                    "Verify log directory is accessible to both UMS and filesystem tools",
                ]
            )

        # Determine overall status
        issues_count = len(diagnosis["issues_found"])
        if issues_count == 0:
            diagnosis["status"] = "HEALTHY"
        elif issues_count <= 2:
            diagnosis["status"] = "NEEDS_ATTENTION"
        else:
            diagnosis["status"] = "CRITICAL"

        # Add summary message
        harmony_status = (
            "harmonized"
            if diagnosis["filesystem_integration"]["validation_harmonized"]
            else "conflicted"
        )
        diagnosis["summary"] = (
            f"File access diagnosis: {diagnosis['status']}. Validation systems: {harmony_status}."
        )

    except Exception as e:
        diagnosis["status"] = "ERROR"
        diagnosis["error"] = str(e)
        diagnosis["recommendations"].append(
            "Use fallback temporary directory for file operations until issues are resolved"
        )

    diagnosis["execution_time"] = time.time() - start_time
    diagnosis["timestamp"] = datetime.now(timezone.utc).isoformat()

    return {
        "success": True,
        "data": diagnosis,
        "processing_time": diagnosis.pop("execution_time"),
    }


@with_tool_metrics
@with_error_handling
async def start_batch_operation(
    workflow_id: str,
    batch_description: str = "Multi-tool agent operation",
    expected_tools: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Start a batch of UMS tool operations for an agent turn.

    This tool should be called at the beginning of a turn where the agent
    plans to make multiple UMS tool calls in sequence.

    Args:
        workflow_id: The workflow context for this batch
        batch_description: Human-readable description of what this batch will do
        expected_tools: Optional list of tool names that will be called
        metadata: Optional metadata to associate with this batch
        db_path: Database path

    Returns:
        Dict with batch_id and context information
    """
    start_time = time.time()

    # Generate a unique batch ID
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"

    # Get current batch context or create new one
    current_batch = get_current_batch()
    if current_batch:
        logger.warning(
            f"Batch operation already in progress ({current_batch['batch_id']}), "
            f"starting nested batch {batch_id}",
            emoji_key="warning",
        )

    # Start the batch context
    batch_context = UMSBatchContext(batch_id=batch_id, metadata=metadata)  # noqa: F841

    # Log the batch start
    logger.info(
        f"Started UMS batch operation: {batch_description} (ID: {batch_id})",
        emoji_key="batch_start",
    )

    if expected_tools:
        logger.debug(f"Expected tools in batch: {', '.join(expected_tools)}")

    # Store batch info for tracking
    try:
        async with DBConnection(db_path).transaction() as conn:
            # Log this as a memory operation for tracking
            operation_data = {
                "batch_id": batch_id,
                "description": batch_description,
                "expected_tools": expected_tools,
                "metadata": metadata,
                "start_time": start_time,
            }

            operation_id = await MemoryUtils._log_memory_operation(  # noqa: F841
                conn, workflow_id, "batch_start", operation_data=operation_data
            )

    except Exception as e:
        logger.warning(f"Failed to log batch start operation: {e}")
        # Don't fail the entire operation for logging issues

    return {
        "success": True,
        "data": {
            "batch_id": batch_id,
            "workflow_id": workflow_id,
            "description": batch_description,
            "expected_tools": expected_tools or [],
            "started_at": to_iso_z(start_time),
            "message": f"Batch operation {batch_id} started. Ready for multi-tool operations.",
        },
    }


@with_tool_metrics
@with_error_handling
async def get_rich_context_package(
    workflow_id: str,
    focus_goal_id: Optional[str] = None,
    context_id: Optional[str] = None,
    current_plan_step_description: Optional[str] = None,
    focal_memory_id_hint: Optional[str] = None,
    fetch_limits: Optional[Dict[str, int]] = None,
    show_limits: Optional[Dict[str, int]] = None,
    include_core_context: bool = True,
    include_working_memory: bool = True,
    include_proactive_memories: bool = True,
    include_relevant_procedures: bool = True,
    include_contextual_links: bool = True,
    include_graph: bool = True,
    include_recent_actions: bool = True,
    include_contradictions: bool = True,
    max_memories: int = 20,
    include_goal_stack: bool = False,
    compression_token_threshold: Optional[int] = None,
    compression_target_tokens: Optional[int] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Assemble a *rich* context package for the agent master-loop (AML).

    Uses direct database queries within a single transaction context instead of calling other UMS tools,
    improving performance and error handling. Returns proper success/error envelope format.

    Args:
        workflow_id: Required workflow scope
        focus_goal_id: Optional goal ID to focus context around
        context_id: Optional context ID for working memory
        current_plan_step_description: Description of current plan step
        focal_memory_id_hint: Memory ID to use as focal point for links
        fetch_limits: Limits for fetching various types of data
        show_limits: Limits for showing summarized data
        include_core_context: Include core workflow context
        include_working_memory: Include working memory if context_id provided
        include_proactive_memories: Include proactive memory search
        include_relevant_procedures: Include procedural memory search
        include_contextual_links: Include memory links
        include_graph: Include graph snapshot/subgraph data
        include_recent_actions: Include recent actions data
        include_contradictions: Include contradiction detection
        max_memories: Maximum number of memories to include in proactive search
        include_goal_stack: Include goal hierarchy/tree data
        compression_token_threshold: Token threshold for compression
        compression_target_tokens: Target tokens after compression
        db_path: Database path
    """
    start_time = time.time()
    retrieval_ts = datetime.now(timezone.utc).isoformat()

    # Check if we're in a batch operation
    current_batch = get_current_batch()
    if current_batch:
        logger.info(
            f"get_rich_context_package called within batch {current_batch['batch_id']} "
            f"for workflow {workflow_id}",
            emoji_key="package",
        )
        # Record this tool call in the batch context
        current_batch["tool_calls"].append(
            {
                "tool_name": "get_rich_context_package",
                "workflow_id": workflow_id,
                "focus_goal_id": focus_goal_id,
                "timestamp": start_time,
            }
        )
    else:
        logger.debug(f"get_rich_context_package called for workflow {workflow_id} (standalone)")

    assembled: Dict[str, Any] = {"retrieval_timestamp_ums_package": retrieval_ts}
    errors: List[str] = []
    focal_mem_id_for_links: Optional[str] = focal_memory_id_hint

    # Add batch info to the response if we're in a batch
    if current_batch:
        assembled["batch_context"] = {
            "batch_id": current_batch["batch_id"],
            "is_multi_tool_turn": True,
            "call_number": len(current_batch["tool_calls"]),
        }

    # Set up limits - use max_memories for proactive search
    fetch_limits = fetch_limits or {}
    show_limits = show_limits or {}

    lim_actions = fetch_limits.get("recent_actions", UMS_PKG_DEFAULT_FETCH_RECENT_ACTIONS)
    lim_imp_mems = fetch_limits.get("important_memories", UMS_PKG_DEFAULT_FETCH_IMPORTANT_MEMORIES)
    lim_key_thts = fetch_limits.get("key_thoughts", UMS_PKG_DEFAULT_FETCH_KEY_THOUGHTS)
    lim_proactive = min(
        max_memories, fetch_limits.get("proactive_memories", UMS_PKG_DEFAULT_FETCH_PROACTIVE)
    )
    lim_procedural = fetch_limits.get("procedural_memories", UMS_PKG_DEFAULT_FETCH_PROCEDURAL)
    lim_links = fetch_limits.get("link_traversal", UMS_PKG_DEFAULT_FETCH_LINKS)

    lim_show_links_summary = show_limits.get("link_traversal", UMS_PKG_DEFAULT_SHOW_LINKS_SUMMARY)

    # Helper function to safely format timestamps
    def to_iso_z(ts):
        return safe_format_timestamp(ts)

    db = DBConnection(db_path)

    try:
        async with db.transaction(readonly=True) as conn:
            # 0. Validate workflow
            wf_check_row = await conn.execute_fetchone(
                "SELECT title, goal, status FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if wf_check_row is None:
                # This is a critical failure for this tool.
                return {
                    "success": False,
                    "error_message": f"Target workflow_id '{workflow_id}' not found in UMS.",
                    "error_type": "WorkflowNotFound",
                    "processing_time": time.time() - start_time,
                }

            # 1. Core context (direct query)
            if include_core_context:
                try:
                    # Fetch core workflow details directly
                    core_wf_data = dict(wf_check_row)

                    # Fetch key thoughts directly
                    key_thoughts_list = []
                    chain_id_row_core = await conn.execute_fetchone(
                        "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                        (workflow_id,),
                    )
                    if chain_id_row_core:
                        thought_rows_core = await conn.execute_fetchall(
                            """SELECT thought_type, content, sequence_number, created_at FROM thoughts
                               WHERE thought_chain_id = ? AND thought_type IN (?, ?, ?, ?, ?, ?)
                               ORDER BY sequence_number DESC LIMIT ?""",
                            (
                                chain_id_row_core["thought_chain_id"],
                                ThoughtType.GOAL.value,
                                ThoughtType.DECISION.value,
                                ThoughtType.REASONING.value,
                                ThoughtType.ANALYSIS.value,
                                ThoughtType.SUMMARY.value,
                                ThoughtType.REFLECTION.value,
                                lim_key_thts,
                            ),
                        )
                        key_thoughts_list = [dict(r) for r in thought_rows_core]
                        for th_core in key_thoughts_list:  # Add ISO timestamps
                            if th_core.get("created_at"):
                                th_core["created_at_iso"] = to_iso_z(th_core["created_at"])

                    # Fetch important memories directly
                    important_memories_list = []
                    mem_rows_imp = await conn.execute_fetchall(
                        """SELECT memory_id, description, memory_type, importance FROM memories
                           WHERE workflow_id = ? AND (ttl = 0 OR created_at + ttl > ?)
                           ORDER BY importance DESC LIMIT ?""",
                        (workflow_id, int(time.time()), lim_imp_mems),
                    )
                    important_memories_list = [dict(r) for r in mem_rows_imp]

                    assembled["core_context"] = {
                        "workflow_id": workflow_id,
                        "workflow_title": core_wf_data["title"],
                        "workflow_goal": core_wf_data["goal"],
                        "workflow_status": core_wf_data["status"],
                        "key_thoughts": key_thoughts_list,
                        "important_memories": important_memories_list,
                        "retrieved_at": retrieval_ts,
                    }
                except Exception as exc_core:
                    msg = f"Core context direct retrieval failed: {exc_core}"
                    errors.append(f"UMS Package: {msg}")
                    logger.error(msg, exc_info=True)
                    assembled["core_context"] = {"error": msg}

            # 2. Working memory (direct query if context_id provided)
            if include_working_memory and context_id:
                try:
                    state_row_wm = await conn.execute_fetchone(
                        "SELECT working_memory, focal_memory_id FROM cognitive_states WHERE state_id = ? AND workflow_id = ?",
                        (context_id, workflow_id),
                    )
                    if state_row_wm:
                        working_memory_ids_json = state_row_wm["working_memory"]
                        focal_memory_id_from_state = state_row_wm["focal_memory_id"]
                        working_memory_ids_list = (
                            await MemoryUtils.deserialize(working_memory_ids_json) or []
                        )

                        fetched_wm_details = []
                        if working_memory_ids_list:
                            placeholders_wm = ",".join("?" * len(working_memory_ids_list))
                            mem_rows_wm = await conn.execute_fetchall(
                                f"""SELECT memory_id, description, memory_type, content, importance 
                                   FROM memories WHERE memory_id IN ({placeholders_wm}) AND workflow_id = ?""",
                                (*working_memory_ids_list, workflow_id),
                            )
                            mem_map_wm = {r["memory_id"]: dict(r) for r in mem_rows_wm}
                            for (
                                mem_id_wm
                            ) in working_memory_ids_list:  # Preserve order from cognitive state
                                if mem_id_wm in mem_map_wm:
                                    detail = mem_map_wm[mem_id_wm]
                                    if detail.get("content") and len(detail["content"]) > 150:
                                        detail["content_preview"] = detail["content"][:147] + "..."
                                    fetched_wm_details.append(detail)

                        assembled["current_working_memory"] = {
                            "retrieved_at": retrieval_ts,
                            "context_id": context_id,
                            "focal_memory_id": focal_memory_id_from_state,
                            "working_memories": fetched_wm_details,
                        }
                        if focal_memory_id_from_state:
                            focal_mem_id_for_links = focal_memory_id_from_state
                    else:
                        msg = f"Working memory context_id '{context_id}' not found for workflow '{workflow_id}'."
                        errors.append(f"UMS Package: {msg}")
                        logger.warning(msg)
                        assembled["current_working_memory"] = {"error": msg, "working_memories": []}
                except Exception as exc_wm:
                    msg = (
                        f"Exception fetching working memory directly for context package: {exc_wm}"
                    )
                    errors.append(f"UMS Package: {msg}")
                    logger.error(msg, exc_info=True)
                    assembled["current_working_memory"] = {"error": msg, "working_memories": []}

            # 3. Recent Actions (direct query) - include in core_context structure
            if include_recent_actions:
                try:
                    action_rows = await conn.execute_fetchall(
                        """SELECT action_id, action_type, title, status, started_at 
                           FROM actions WHERE workflow_id = ? ORDER BY sequence_number DESC LIMIT ?""",
                        (workflow_id, lim_actions),
                    )
                    recent_actions_list = []
                    for r_act in action_rows:
                        act_dict = dict(r_act)
                        if act_dict.get("started_at"):
                            act_dict["started_at_iso"] = to_iso_z(act_dict["started_at"])
                        recent_actions_list.append(act_dict)
                    # Add recent_actions to core_context AND as separate key for agent compatibility
                    if assembled.get("core_context"):
                        assembled["core_context"]["recent_actions"] = recent_actions_list
                    assembled["recent_actions"] = recent_actions_list
                except Exception as exc_actions:
                    msg = f"Recent actions direct retrieval failed: {exc_actions}"
                    errors.append(f"UMS Package: {msg}")
                    logger.error(msg, exc_info=True)
                    error_result = [{"error": msg}]
                    if assembled.get("core_context"):
                        assembled["core_context"]["recent_actions"] = error_result
                    assembled["recent_actions"] = error_result

    except Exception as outer_exc:
        # Catch errors during the database transaction itself
        logger.error(
            f"UMS Package: Database transaction error for get_rich_context_package: {outer_exc}",
            exc_info=True,
        )
        return {
            "success": False,
            "error_message": f"Failed to assemble context due to DB error: {outer_exc}",
            "error_type": "DatabaseError",
            "processing_time": time.time() - start_time,
        }

    # After the transaction completes, call other UMS tools that require their own transaction contexts
    search_source = current_plan_step_description or "current agent objectives"
    if focus_goal_id:
        # Don't include the actual goal ID in search terms as it can break FTS parsing
        search_source += f" (focused on goal: {_fmt_id(focus_goal_id)})"

    # 4. Proactive memories (using hybrid search)
    if include_proactive_memories:
        try:
            q = f"Information relevant to current task or goal: {search_source}"
            pr_res = await hybrid_search_memories(
                query=q,
                workflow_id=workflow_id,
                limit=lim_proactive,
                include_content=False,
                semantic_weight=0.7,
                keyword_weight=0.3,
                db_path=db_path,
            )
            if pr_res.get("success"):
                data = pr_res.get("data", {})
                assembled["proactive_memories"] = {
                    "retrieved_at": retrieval_ts,
                    "query_used": q,
                    "memories": data.get("memories", []),
                }
            else:
                errors.append(
                    f"UMS Package: Proactive search failed: {pr_res.get('error_message', 'Unknown error')}"
                )
        except Exception as exc:
            errors.append(f"UMS Package: Proactive search exception: {exc}")
            logger.error("Proactive search error", exc_info=True)

    # 5. Procedural memories (using hybrid search)
    if include_relevant_procedures:
        try:
            q = f"How to accomplish, perform, or execute: {search_source}"
            proc_res = await hybrid_search_memories(
                query=q,
                workflow_id=workflow_id,
                limit=lim_procedural,
                memory_level=MemoryLevel.PROCEDURAL.value,
                include_content=False,
                db_path=db_path,
            )
            if proc_res.get("success"):
                data = proc_res.get("data", {})
                assembled["relevant_procedures"] = {
                    "retrieved_at": retrieval_ts,
                    "query_used": q,
                    "procedures": data.get("memories", []),
                }
            else:
                errors.append(
                    f"UMS Package: Procedural search failed: {proc_res.get('error_message', 'Unknown error')}"
                )
        except Exception as exc:
            errors.append(f"UMS Package: Procedural search exception: {exc}")
            logger.error("Procedural search error", exc_info=True)

    # 6. Graph snapshot
    if include_graph:
        try:
            # Try to find a good starting node for the subgraph
            start_node_for_graph = focal_mem_id_for_links
            if not start_node_for_graph:
                # Use the first important memory as starting point
                imp_mems = assembled.get("core_context", {}).get("important_memories", [])
                if imp_mems:
                    start_node_for_graph = imp_mems[0].get("memory_id")

            if start_node_for_graph:
                # Use the NetworkX-powered get_subgraph with enhanced analysis
                graph_res = await get_subgraph(
                    workflow_id=workflow_id,
                    start_node_id=start_node_for_graph,
                    algorithm="ego_graph",
                    max_hops=2,
                    max_nodes=min(30, max_memories + 10),
                    link_type_filter=None,
                    compute_centrality=True,
                    centrality_algorithms=["pagerank"],
                    compute_graph_metrics=True,
                    detect_communities=False,
                    include_node_content=False,
                    centrality_top_k=10,
                    db_path=db_path,
                )

                if graph_res.get("success") and graph_res.get("data"):
                    graph_data = graph_res["data"]
                    assembled["graph_snapshot"] = {
                        "retrieved_at": retrieval_ts,
                        "start_node_id": start_node_for_graph,
                        "algorithm_used": graph_data.get("algorithm"),
                        "nodes": graph_data.get("nodes", []),
                        "edges": graph_data.get("edges", []),
                        "node_count": graph_data.get("node_count", 0),
                        "edge_count": graph_data.get("edge_count", 0),
                        "centrality": graph_data.get("centrality", {}),
                        "graph_metrics": graph_data.get("graph_metrics", {}),
                    }
                else:
                    errors.append(
                        f"UMS Package: Graph snapshot failed: {graph_res.get('error_message', 'Unknown error')}"
                    )
            else:
                # Try full graph analysis if no starting node found
                try:
                    graph_res = await get_subgraph(
                        workflow_id=workflow_id,
                        start_node_id=None,
                        algorithm="full_graph",
                        max_nodes=min(20, max_memories),
                        compute_centrality=True,
                        centrality_algorithms=["pagerank"],
                        compute_graph_metrics=True,
                        include_node_content=False,
                        db_path=db_path,
                    )

                    if graph_res.get("success") and graph_res.get("data"):
                        graph_data = graph_res["data"]
                        assembled["graph_snapshot"] = {
                            "retrieved_at": retrieval_ts,
                            "start_node_id": None,
                            "algorithm_used": "full_graph",
                            "nodes": graph_data.get("nodes", []),
                            "edges": graph_data.get("edges", []),
                            "node_count": graph_data.get("node_count", 0),
                            "edge_count": graph_data.get("edge_count", 0),
                            "centrality": graph_data.get("centrality", {}),
                            "graph_metrics": graph_data.get("graph_metrics", {}),
                            "note": "Full graph sample - no specific starting node",
                        }
                    else:
                        assembled["graph_snapshot"] = {
                            "retrieved_at": retrieval_ts,
                            "nodes": [],
                            "edges": [],
                            "node_count": 0,
                            "edge_count": 0,
                            "note": "No suitable starting node found and full graph failed",
                        }
                except Exception as fallback_exc:
                    assembled["graph_snapshot"] = {
                        "retrieved_at": retrieval_ts,
                        "nodes": [],
                        "edges": [],
                        "node_count": 0,
                        "edge_count": 0,
                        "note": f"Graph analysis failed: {str(fallback_exc)}",
                    }

        except Exception as exc:
            errors.append(f"UMS Package: Graph snapshot exception: {exc}")
            logger.error("Graph snapshot error", exc_info=True)

    # 7. Contradiction detection
    if include_contradictions:
        try:
            contradiction_res = await get_contradictions(
                workflow_id=workflow_id,
                limit=min(10, max_memories // 2),
                include_resolved=False,
                db_path=db_path,
            )
            if contradiction_res.get("success"):
                contradictions_data = contradiction_res.get("data", {})
                assembled["contradictions"] = {
                    "retrieved_at": retrieval_ts,
                    "contradictions_found": contradictions_data.get("contradictions_found", []),
                    "total_found": contradictions_data.get("total_found", 0),
                }
            else:
                errors.append(
                    f"UMS Package: Contradiction detection failed: {contradiction_res.get('error_message', 'Unknown error')}"
                )
        except Exception as exc:
            errors.append(f"UMS Package: Contradiction detection exception: {exc}")
            logger.error("Contradiction detection error", exc_info=True)

    # 8. Goal Stack (Tree)
    if include_goal_stack:
        try:
            goal_stack_res = await get_goal_stack(
                workflow_id=workflow_id,
                include_completed=False,
                include_metadata=False,
                db_path=db_path,
            )
            if goal_stack_res.get("success"):
                data = goal_stack_res.get("data", {})
                assembled["goal_stack"] = {
                    "retrieved_at": retrieval_ts,
                    "goal_tree": data.get("goal_tree", []),
                    "total_goals": data.get("total_goals", 0),
                }
            else:
                errors.append(
                    f"UMS Package: Goal stack retrieval failed: {goal_stack_res.get('error_message', 'Unknown error')}"
                )
        except Exception as exc:
            errors.append(f"UMS Package: Goal stack retrieval exception: {exc}")
            logger.error("Goal stack retrieval error", exc_info=True)

    # 9. Contextual links
    if include_contextual_links:
        link_seed = focal_mem_id_for_links
        if link_seed is None:
            imp_mems = assembled.get("core_context", {}).get("important_memories", [])
            if imp_mems:
                link_seed = imp_mems[0].get("memory_id")

        if link_seed:
            try:
                link_res = await get_linked_memories(
                    memory_id=link_seed,
                    direction="both",
                    limit=lim_links,
                    include_memory_details=False,
                    db_path=db_path,
                )
                if link_res.get("success"):
                    data = link_res.get("data", {})
                    payload = data.get("links", {})
                    asm = {
                        "source_memory_id": link_seed,
                        "outgoing_count": len(payload.get("outgoing", [])),
                        "incoming_count": len(payload.get("incoming", [])),
                        "top_outgoing_links_summary": [
                            {
                                "target_memory_id": _fmt_id(link["target_memory_id"]),
                                "link_type": link["link_type"],
                                "description": (link.get("description") or "")[:70] + "…",
                            }
                            for link in payload.get("outgoing", [])[:lim_show_links_summary]
                        ],
                        "top_incoming_links_summary": [
                            {
                                "source_memory_id": _fmt_id(link["source_memory_id"]),
                                "link_type": link["link_type"],
                                "description": (link.get("description") or "")[:70] + "…",
                            }
                            for link in payload.get("incoming", [])[:lim_show_links_summary]
                        ],
                    }
                    assembled["contextual_links"] = {"retrieved_at": retrieval_ts, "summary": asm}
                else:
                    errors.append(
                        f"UMS Package: Link retrieval failed: {link_res.get('error_message', 'Unknown error')}"
                    )
            except Exception as exc:
                errors.append(f"UMS Package: Link retrieval exception: {exc}")
                logger.error("Link retrieval error", exc_info=True)

    # 10. Compression
    if compression_token_threshold is not None and compression_target_tokens is not None:
        try:
            pkg_json = json.dumps(assembled, default=str)
            tok_est = count_tokens(pkg_json)
            if tok_est > compression_token_threshold:
                logger.info(
                    f"Context {workflow_id} at {tok_est} tokens exceeds {compression_token_threshold}; compressing."
                )
                # heuristic selection of a large list to summarise
                cand = {
                    "core_context.recent_actions": assembled.get("core_context", {}).get(
                        "recent_actions"
                    ),
                    "core_context.important_memories": assembled.get("core_context", {}).get(
                        "important_memories"
                    ),
                    "proactive_memories.memories": assembled.get("proactive_memories", {}).get(
                        "memories"
                    ),
                    "relevant_procedures.procedures": assembled.get("relevant_procedures", {}).get(
                        "procedures"
                    ),
                }
                target_key, target_txt, max_tok = None, "", 0
                thresh = compression_target_tokens * 0.5

                for k, v in cand.items():
                    if v and isinstance(v, list) and len(v) > 3:
                        s = json.dumps(v, default=str)
                        t = count_tokens(s)
                        if t > thresh and t > max_tok:
                            target_key, target_txt, max_tok = k, s, t

                if target_key:
                    sum_res = await summarize_text(
                        text_to_summarize=target_txt,
                        target_tokens=int(compression_target_tokens * 0.6),
                        context_type=f"ums_package_component:{target_key}",
                        workflow_id=workflow_id,
                        record_summary=False,
                        db_path=db_path,
                    )
                    if sum_res.get("success") and sum_res.get("data", {}).get("summary"):
                        # replace component with marker + preview
                        keys = target_key.split(".")
                        ref = assembled
                        for k in keys[:-1]:
                            ref = ref.setdefault(k, {})
                        existing_value = ref.get(keys[-1])
                        if isinstance(existing_value, dict):
                            original_rt = existing_value.get("retrieved_at", retrieval_ts)
                        else:
                            original_rt = retrieval_ts
                        ref[keys[-1]] = {
                            "retrieved_at": original_rt,
                            "_ums_compressed_": True,
                            "original_token_estimate": max_tok,
                            "summary_preview": sum_res["data"]["summary"][:150] + "…",
                        }
                        assembled.setdefault("ums_compression_details", {})[target_key] = {
                            "summary_content": sum_res["data"]["summary"],
                            "retrieved_at": retrieval_ts,
                        }
                        logger.info(f"Compressed component '{target_key}' in context package.")
                    else:
                        errors.append(
                            f"UMS Package: Compression of '{target_key}' failed: {sum_res.get('error_message', 'Unknown error')}"
                        )
            else:
                logger.debug(f"Context {workflow_id} size {tok_est} within threshold.")
        except Exception as exc:
            errors.append(f"UMS Package: Compression exception: {exc}")
            logger.error("Compression error", exc_info=True)

    # Final check for essential components
    if not assembled.get("core_context"):
        errors.append("UMS Package Critical: Core context could not be assembled.")
    if (
        include_working_memory
        and context_id
        and not assembled.get("current_working_memory", {}).get("working_memories")
        and "error" not in assembled.get("current_working_memory", {})
    ):
        # Only add error if working memory was expected but is empty AND no error was already logged for it
        errors.append(
            "UMS Package Warning: Working memory was requested but is empty or could not be retrieved."
        )

    if "UMS Package Critical: Core context could not be assembled." in errors:
        final_error_msg = "; ".join(errors)
        logger.error(f"Rich context package for {workflow_id} critically failed: {final_error_msg}")
        return {
            "success": False,
            "error_message": final_error_msg,
            "error_type": "ContextAssemblyCriticalFailure",
            "data": {"context_package": assembled},  # Return partial for debug
            "processing_time": time.time() - start_time,
        }

    # If successful or only minor errors:
    resp = {
        "success": True,
        "data": {"context_package": assembled},
        "processing_time": time.time() - start_time,
    }
    if errors:
        # Add warnings to the context_package itself if it's otherwise "successful"
        assembled["assembly_warnings"] = errors  # Add to the package being returned
        # Also add to the top-level error_message for client visibility if some data is missing
        resp["error_message"] = f"Context package assembled with warnings: {'; '.join(errors)}"
        logger.warning(
            f"get_rich_context_package for {workflow_id} completed with warnings: {errors}"
        )
    else:
        logger.info(f"get_rich_context_package for {workflow_id} succeeded cleanly.")

    return resp


def _mermaid_escape(text: str) -> str:
    """Escapes characters problematic for Mermaid node labels."""
    if not isinstance(text, str):
        text = str(text)
    # Replace quotes first, then other potentially problematic characters
    text = text.replace('"', "#quot;")
    text = text.replace("(", "#40;")
    text = text.replace(")", "#41;")
    text = text.replace("[", "#91;")
    text = text.replace("]", "#93;")
    text = text.replace("{", "#123;")
    text = text.replace("}", "#125;")
    text = text.replace(":", "#58;")
    text = text.replace(";", "#59;")
    text = text.replace("<", "#lt;")
    text = text.replace(">", "#gt;")
    # Replace newline with <br> for multiline labels if needed, or just space
    text = text.replace("\n", "<br>")
    return text


async def _generate_mermaid_diagram(workflow: Dict[str, Any]) -> str:
    """Generates a detailed Mermaid flowchart representation of the workflow."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            # Generate a unique fallback for missing IDs to avoid collisions
            return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
        # Replace hyphens which are problematic in unquoted Mermaid node IDs
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "flowchart TD"]  # Top-Down flowchart

    # --- Workflow Node ---
    wf_node_id = sanitize_mermaid_id(workflow.get("workflow_id"), "W")  # Use sanitized full ID
    wf_title = _mermaid_escape(workflow.get("title", "Workflow"))
    wf_status_class = f":::{workflow.get('status', 'active')}"  # Style based on status
    diagram.append(f'    {wf_node_id}("{wf_title}"){wf_status_class}')
    diagram.append("")  # Spacer

    # --- Action Nodes & Links ---
    action_nodes = {}  # Map action_id to mermaid_node_id
    parent_links = {}  # Map child_action_id to parent_action_id
    sequential_links = {}  # Map sequence_number to action_id for sequential linking if no parent

    actions = sorted(workflow.get("actions", []), key=lambda a: a.get("sequence_number", 0))

    for i, action in enumerate(actions):
        action_id = action.get("action_id")
        if not action_id:
            continue  # Skip actions somehow missing an ID

        node_id = sanitize_mermaid_id(action_id, "A")  # Use sanitized full ID
        action_nodes[action_id] = node_id
        sequence_number = action.get("sequence_number", i)  # Use sequence number if available

        # Label: Include type, title, and potentially tool name
        action_type = action.get("action_type", "Action").capitalize()
        action_title = _mermaid_escape(action.get("title", action_type))
        label = f"<b>{action_type} #{sequence_number}</b><br/>{action_title}"
        if action.get("tool_name"):
            label += f"<br/><i>Tool: {_mermaid_escape(action['tool_name'])}</i>"

        # Node shape/style based on status
        status = action.get("status", ActionStatus.PLANNED.value)
        node_style = f":::{status}"  # Use status directly for class name

        # Node Definition
        diagram.append(f'    {node_id}["{label}"]{node_style}')

        # Store parent relationship
        parent_action_id = action.get("parent_action_id")
        if parent_action_id:
            parent_links[action_id] = parent_action_id
        else:
            sequential_links[sequence_number] = action_id

    diagram.append("")  # Spacer

    # Draw Links: Parent/Child first, then sequential for roots
    linked_actions = set()
    # Parent->Child links
    for child_id, parent_id in parent_links.items():
        if child_id in action_nodes and parent_id in action_nodes:
            child_node = action_nodes[child_id]
            parent_node = action_nodes[parent_id]
            diagram.append(f"    {parent_node} --> {child_node}")
            linked_actions.add(child_id)  # Mark child as linked

    # Sequential links for actions without explicit parents
    last_sequential_node = wf_node_id  # Start sequence from workflow node
    sorted_sequences = sorted(sequential_links.keys())
    for seq_num in sorted_sequences:
        action_id = sequential_links[seq_num]
        if action_id in action_nodes:  # Ensure action node exists
            node_id = action_nodes[action_id]
            diagram.append(f"    {last_sequential_node} --> {node_id}")
            last_sequential_node = node_id  # Chain sequential actions
            linked_actions.add(action_id)  # Mark root as linked

    # Link any remaining unlinked actions (e.g., if parents were missing/invalid) sequentially
    for action in actions:
        action_id = action.get("action_id")
        if action_id and action_id not in linked_actions and action_id in action_nodes:
            node_id = action_nodes[action_id]
            # Link from workflow if no other link established
            diagram.append(f"    {wf_node_id} -.-> {node_id} :::orphanLink")
            logger.debug(f"Linking orphan action {action_id} to workflow.")

    diagram.append("")  # Spacer

    # --- Artifact Nodes & Links ---
    artifacts = workflow.get("artifacts", [])
    if artifacts:
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id")
            if not artifact_id:
                continue  # Skip artifacts missing ID

            node_id = sanitize_mermaid_id(artifact_id, "F")  # Use sanitized full ID
            artifact_name = _mermaid_escape(artifact.get("name", "Artifact"))
            artifact_type = _mermaid_escape(artifact.get("artifact_type", "file"))
            label = f"📄<br/><b>{artifact_name}</b><br/>({artifact_type})"

            # Node shape/style based on type/output status
            node_shape_start, node_shape_end = "[(", ")]"  # Default: capsule for artifacts
            node_style = ":::artifact"
            if artifact.get("is_output"):
                node_style = ":::artifact_output"  # Style final outputs differently

            diagram.append(f'    {node_id}{node_shape_start}"{label}"{node_shape_end}{node_style}')

            # Link from creating action or workflow
            creator_action_id = artifact.get("action_id")
            if creator_action_id and creator_action_id in action_nodes:
                creator_node = action_nodes[creator_action_id]
                diagram.append(f"    {creator_node} -- Creates --> {node_id}")
            else:
                # Link artifact to workflow if no specific action created it
                diagram.append(f"    {wf_node_id} -.-> {node_id}")

    # --- Class Definitions (Full Set) ---
    diagram.append("\n    %% Stylesheets")
    diagram.append("    classDef workflow fill:#e7f0fd,stroke:#0056b3,stroke-width:2px,color:#000")
    # Action Statuses
    diagram.append(
        "    classDef completed fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724"
    )
    diagram.append("    classDef failed fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24")
    diagram.append(
        "    classDef skipped fill:#e2e3e5,stroke:#383d41,stroke-width:1px,color:#383d41"
    )
    diagram.append(
        "    classDef in_progress fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404"
    )
    diagram.append(
        "    classDef planned fill:#fefefe,stroke:#6c757d,stroke-width:1px,color:#343a40,stroke-dasharray: 3 3"
    )
    # Artifacts
    diagram.append("    classDef artifact fill:#fdfae7,stroke:#b3a160,stroke-width:1px,color:#333")
    diagram.append(
        "    classDef artifact_output fill:#e7fdf4,stroke:#2e855d,stroke-width:2px,color:#000"
    )
    diagram.append("    classDef orphanLink stroke:#ccc,stroke-dasharray: 2 2")

    diagram.append("```")
    return "\n".join(diagram)


async def _generate_thought_chain_mermaid(thought_chain: Dict[str, Any]) -> str:
    """Generates a detailed Mermaid flowchart of a thought chain."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "graph TD"]  # Top-Down graph

    # --- Header Node ---
    chain_node_id = sanitize_mermaid_id(
        thought_chain.get("thought_chain_id"), "TC"
    )  # Use sanitized full ID
    chain_title = _mermaid_escape(thought_chain.get("title", "Thought Chain"))
    diagram.append(f'    {chain_node_id}("{chain_title}"):::header')
    diagram.append("")

    # --- Thought Nodes ---
    thoughts = sorted(thought_chain.get("thoughts", []), key=lambda t: t.get("sequence_number", 0))
    thought_nodes = {}  # Map thought_id to mermaid_node_id
    parent_links = {}  # Map child_thought_id to parent_thought_id

    if thoughts:
        for thought in thoughts:
            thought_id = thought.get("thought_id")
            if not thought_id:
                continue

            node_id = sanitize_mermaid_id(thought_id, "T")  # Use sanitized full ID
            thought_nodes[thought_id] = node_id
            thought_type = thought.get("thought_type", "thought").lower()

            # Node shape and style based on thought type
            shapes = {
                "goal": ("([", "])"),
                "question": ("{{", "}}"),
                "decision": ("[/", "\\]"),
                "summary": ("[(", ")]"),
                "constraint": ("[[", "]]"),
                "hypothesis": ("( ", " )"),
            }
            node_shape_start, node_shape_end = shapes.get(
                thought_type, ("[", "]")
            )  # Default rectangle
            node_style = f":::type{thought_type}"

            # Label content
            content = _mermaid_escape(thought.get("content", "..."))
            label = f"<b>{thought_type.capitalize()} #{thought.get('sequence_number')}</b><br/>{content}"

            diagram.append(f'    {node_id}{node_shape_start}"{label}"{node_shape_end}{node_style}')

            # Store parent relationship
            parent_id = thought.get("parent_thought_id")
            if parent_id:
                parent_links[thought_id] = parent_id

    diagram.append("")

    # --- Draw Links ---
    linked_thoughts = set()
    # Parent -> Child links
    for child_id, parent_id in parent_links.items():
        if child_id in thought_nodes and parent_id in thought_nodes:
            child_node = thought_nodes[child_id]
            parent_node = thought_nodes[parent_id]
            diagram.append(f"    {parent_node} --> {child_node}")
            linked_thoughts.add(child_id)

    # Link root thoughts (no parent or parent not found) sequentially from the header
    last_root_node = chain_node_id  # Use sanitized chain node ID
    for thought in thoughts:
        thought_id = thought.get("thought_id")
        if thought_id and thought_id not in linked_thoughts and thought_id in thought_nodes:
            # Check if its parent exists in the fetched thoughts; if not, treat as root for linking
            parent_id = parent_links.get(thought_id)
            if not parent_id or parent_id not in thought_nodes:
                node_id = thought_nodes[thought_id]
                diagram.append(f"    {last_root_node} --> {node_id}")
                last_root_node = node_id  # Chain subsequent roots
                linked_thoughts.add(thought_id)

    # --- External Links (Actions/Artifacts/Memories) ---
    if thoughts:
        diagram.append("")
        for thought in thoughts:
            thought_id = thought.get("thought_id")
            if not thought_id or thought_id not in thought_nodes:
                continue  # Skip if thought or its node wasn't created
            node_id = thought_nodes[thought_id]

            # Link to relevant action
            rel_action_id = thought.get("relevant_action_id")
            if rel_action_id:
                ext_node_id = sanitize_mermaid_id(rel_action_id, "ExtA")  # Sanitize external ID
                diagram.append(f'    {ext_node_id}["Action: {rel_action_id[:8]}..."]:::action')
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

            # Link to relevant artifact
            rel_artifact_id = thought.get("relevant_artifact_id")
            if rel_artifact_id:
                ext_node_id = sanitize_mermaid_id(rel_artifact_id, "ExtF")  # Sanitize external ID
                diagram.append(
                    f'    {ext_node_id}[("Artifact: {rel_artifact_id[:8]}...")]:::artifact'
                )
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

            # Link to relevant memory
            rel_memory_id = thought.get("relevant_memory_id")
            if rel_memory_id:
                ext_node_id = sanitize_mermaid_id(rel_memory_id, "ExtM")  # Sanitize external ID
                diagram.append(f'    {ext_node_id}("Memory: {rel_memory_id[:8]}..."):::memory')
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

    # --- Class Definitions (Full Set) ---
    diagram.append("\n    %% Stylesheets")
    diagram.append(
        "    classDef header fill:#666,stroke:#333,color:#fff,stroke-width:2px,font-weight:bold;"
    )
    # Thought Types
    diagram.append("    classDef typegoal fill:#d4edda,stroke:#155724,color:#155724;")
    diagram.append("    classDef typequestion fill:#cce5ff,stroke:#004085,color:#004085;")
    diagram.append("    classDef typehypothesis fill:#e2e3e5,stroke:#383d41,color:#383d41;")
    diagram.append("    classDef typeinference fill:#fff3cd,stroke:#856404,color:#856404;")
    diagram.append("    classDef typeevidence fill:#d1ecf1,stroke:#0c5460,color:#0c5460;")
    diagram.append("    classDef typeconstraint fill:#f8d7da,stroke:#721c24,color:#721c24;")
    diagram.append("    classDef typeplan fill:#d6d8f8,stroke:#3f4d9a,color:#3f4d9a;")
    diagram.append(
        "    classDef typedecision fill:#ffe6f5,stroke:#97114c,color:#97114c,font-weight:bold;"
    )
    diagram.append("    classDef typereflection fill:#f5f5f5,stroke:#5a5a5a,color:#5a5a5a;")
    diagram.append("    classDef typecritique fill:#feeed8,stroke:#a34e00,color:#a34e00;")
    diagram.append("    classDef typesummary fill:#cfe2ff,stroke:#0a3492,color:#0a3492;")
    # External Links
    diagram.append(
        "    classDef action fill:#f9f2f4,stroke:#c7254e,color:#c7254e,stroke-dasharray: 5 5;"
    )
    diagram.append(
        "    classDef artifact fill:#f3f6f9,stroke:#367fa9,color:#367fa9,stroke-dasharray: 5 5;"
    )
    diagram.append("    classDef memory fill:#f0f0f0,stroke:#777,color:#333,stroke-dasharray: 2 2;")

    diagram.append("```")
    return "\n".join(diagram)


# --- 19. Reporting ---
@with_tool_metrics
@with_error_handling
async def generate_workflow_report(
    workflow_id: str,
    report_format: str = "markdown",  # markdown | html | json | mermaid
    include_details: bool = True,
    include_thoughts: bool = True,
    include_artifacts: bool = True,
    style: Optional[str] = "professional",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Generates a comprehensive report for the specified workflow.

    All functional behaviour (formats, styles, helper calls, error handling,
    timing metadata, HTML assembly, etc.) is preserved 100 %.
    The only revision is the read-only DB access pattern, now routed through
    the new DBConnection.read-only snapshot
    """
    # -------- Validation (unchanged) -----------------
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    valid_formats = ["markdown", "html", "json", "mermaid"]
    report_format_lower = (report_format or "markdown").lower()
    if report_format_lower not in valid_formats:
        raise ToolInputError(
            f"Invalid format '{report_format}'. Must be one of: {valid_formats}",
            param_name="report_format",
        )

    valid_styles = ["professional", "concise", "narrative", "technical"]
    style_lower = (style or "professional").lower()
    if report_format_lower in ["markdown", "html"] and style_lower not in valid_styles:
        raise ToolInputError(
            f"Invalid style '{style}'. Must be one of: {valid_styles}",
            param_name="style",
        )

    start_time = time.time()

    try:
        # -------- READ-ONLY data hydration ------------
        # We open a read-only snapshot explicitly;
        async with DBConnection(db_path).transaction(readonly=True) as _:
            workflow_data = await get_workflow_details(
                workflow_id=workflow_id,
                include_actions=True,
                include_artifacts=include_artifacts,
                include_thoughts=include_thoughts,
                include_memories=False,
                db_path=db_path,  # propagated unchanged
            )

        if not workflow_data.get("success"):
            raise ToolError(
                f"Failed to retrieve workflow details for report generation (ID: {workflow_id})."
            )

        # -------- Report generation (unchanged) -------
        report_content = None
        markdown_report_content = None

        if report_format_lower in ["markdown", "html"]:
            # Style-specific markdown
            if style_lower == "concise":
                markdown_report_content = await _generate_concise_report(
                    workflow_data, include_details
                )
            elif style_lower == "narrative":
                markdown_report_content = await _generate_narrative_report(
                    workflow_data, include_details
                )
            elif style_lower == "technical":
                markdown_report_content = await _generate_technical_report(
                    workflow_data, include_details
                )
            else:  # professional
                markdown_report_content = await _generate_professional_report(
                    workflow_data, include_details
                )

            if report_format_lower == "markdown":
                report_content = markdown_report_content
            else:  # html
                try:
                    html_body = markdown.markdown(
                        markdown_report_content,
                        extensions=["tables", "fenced_code", "codehilite"],
                    )
                    pygments_css = ""
                    try:
                        formatter = HtmlFormatter(style="default")
                        pygments_css = f"<style>{formatter.get_style_defs('.codehilite')}</style>"
                    except Exception as css_err:
                        logger.warning(f"Failed to generate Pygments CSS: {css_err}")
                        pygments_css = "<!-- Pygments CSS generation failed -->"
                    report_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Workflow Report: {workflow_data.get("title", "Untitled")}</title>
    {pygments_css}
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        pre code {{ display: block; padding: 10px; background-color: #f5f5f5;
                   border: 1px solid #ddd; border-radius: 4px; }}
        .codehilite pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
                except Exception as md_err:
                    logger.error("Markdown → HTML conversion failed", exc_info=True)
                    raise ToolError(f"Failed to convert report to HTML: {md_err}") from md_err

        elif report_format_lower == "json":
            try:
                clean_data = {
                    k: v
                    for k, v in workflow_data.items()
                    if k not in ["success", "processing_time"]
                }
                report_content = json.dumps(clean_data, indent=2, ensure_ascii=False)
            except Exception as json_err:
                logger.error("JSON serialization failed for report", exc_info=True)
                raise ToolError(
                    f"Failed to serialize workflow data to JSON: {json_err}"
                ) from json_err

        else:  # mermaid
            report_content = await _generate_mermaid_diagram(workflow_data)

        if report_content is None:
            raise ToolError(
                f"Report content generation failed unexpectedly for format '{report_format_lower}' and style '{style_lower}'."
            )

        # -------- Assemble result ---------------------
        result = {
            "workflow_id": workflow_id,
            "title": workflow_data.get("title", "Workflow Report"),
            "report": report_content,
            "format": report_format_lower,
            "style_used": style_lower if report_format_lower in ["markdown", "html"] else None,
            "generated_at": to_iso_z(datetime.now(timezone.utc).timestamp()),
        }
        logger.info(
            f"Generated {report_format_lower} report (style: {style_lower if report_format_lower in ['markdown', 'html'] else 'N/A'}) for workflow {workflow_id}",
            emoji_key="newspaper",
        )
        return {
            "success": True,
            "data": result,
            "processing_time": time.time() - start_time,
        }

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report for {workflow_id}: {e}", exc_info=True)
        raise ToolError(
            f"Failed to generate workflow report due to an unexpected error: {e}"
        ) from e


async def _generate_professional_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a professional-style report with formal structure and comprehensive details."""
    report_lines = [f"# Workflow Report: {workflow.get('title', 'Untitled Workflow')}"]

    # --- Executive Summary ---
    report_lines.append("\n## Executive Summary\n")
    report_lines.append(f"**Status:** {workflow.get('status', 'N/A').capitalize()}")
    if workflow.get("goal"):
        report_lines.append(f"**Goal:** {workflow['goal']}")
    if workflow.get("description"):
        report_lines.append(f"\n{workflow['description']}")

    # Use safe_format_timestamp with the correct keys
    report_lines.append(f"\n**Created:** {safe_format_timestamp(workflow.get('created_at'))}")
    report_lines.append(f"**Last Updated:** {safe_format_timestamp(workflow.get('updated_at'))}")
    if workflow.get("completed_at"):
        report_lines.append(f"**Completed:** {safe_format_timestamp(workflow.get('completed_at'))}")
    if workflow.get("tags"):
        report_lines.append(f"**Tags:** {', '.join(workflow['tags'])}")

    # --- Progress Overview ---
    actions = workflow.get("actions", [])
    if actions:
        total_actions = len(actions)
        completed_actions_count = sum(
            1 for a in actions if a.get("status") == ActionStatus.COMPLETED.value
        )
        completion_percentage = (
            int((completed_actions_count / total_actions) * 100) if total_actions > 0 else 0
        )
        report_lines.append("\n## Progress Overview\n")
        report_lines.append(
            f"Overall completion: **{completion_percentage}%** ({completed_actions_count}/{total_actions} actions completed)"
        )
        bar_filled = "#" * (completion_percentage // 5)
        bar_empty = " " * (20 - (completion_percentage // 5))
        report_lines.append(f"\n```\n[{bar_filled}{bar_empty}] {completion_percentage}%\n```")

    # --- Key Actions and Steps ---
    if actions and include_details:
        report_lines.append("\n## Key Actions and Steps\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
        for i, action in enumerate(sorted_actions):
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "in_progress": "⏳",
                "planned": "🗓️",
            }.get(action.get("status"), "❓")
            title = action.get("title", action.get("action_type", "Action")).strip()
            report_lines.append(f"### {i + 1}. {status_emoji} {title}\n")
            report_lines.append(f"**Action ID:** `{action.get('action_id')}`")
            report_lines.append(f"**Type:** {action.get('action_type', 'N/A').capitalize()}")
            report_lines.append(f"**Status:** {action.get('status', 'N/A').capitalize()}")
            report_lines.append(f"**Started:** {safe_format_timestamp(action.get('started_at'))}")
            if action.get("completed_at"):
                report_lines.append(
                    f"**Completed:** {safe_format_timestamp(action['completed_at'])}"
                )

            if action.get("reasoning"):
                report_lines.append(f"\n**Reasoning:**\n```\n{action['reasoning']}\n```")
            if action.get("tool_name"):
                report_lines.append(f"\n**Tool Used:** `{action['tool_name']}`")
                # tool_args might already be deserialized by get_workflow_details
                tool_args = action.get("tool_args")
                if tool_args:
                    try:
                        # Attempt to format as JSON if it's dict/list
                        if isinstance(tool_args, (dict, list)):
                            args_str = json.dumps(tool_args, indent=2)
                            lang = "json"
                        else:
                            args_str = str(tool_args)
                            lang = ""
                    except Exception:  # Catch potential errors during dump
                        args_str = str(tool_args)
                        lang = ""
                    report_lines.append(f"**Arguments:**\n```{lang}\n{args_str}\n```")

                # tool_result might already be deserialized by get_workflow_details
                tool_result = action.get("tool_result")
                if tool_result is not None:  # Check for None explicitly
                    result_repr = tool_result
                    try:
                        # Attempt to format as JSON if it's dict/list
                        if isinstance(result_repr, (dict, list)):
                            result_str = json.dumps(result_repr, indent=2)
                            lang = "json"
                        else:
                            result_str = str(result_repr)
                            lang = ""
                    except Exception:  # Catch potential errors during dump
                        result_str = str(result_repr)
                        lang = ""

                    if len(result_str) > 500:
                        result_str = result_str[:497] + "..."
                    report_lines.append(f"**Result Preview:**\n```{lang}\n{result_str}\n```")

            if action.get("tags"):
                report_lines.append(f"**Tags:** {', '.join(action['tags'])}")
            report_lines.append("\n---")  # Separator

    # --- Key Findings & Insights (from Thoughts) ---
    thought_chains = workflow.get("thought_chains", [])
    if thought_chains and include_details:
        report_lines.append("\n## Key Findings & Insights (from Reasoning)\n")
        for i, chain in enumerate(thought_chains):
            report_lines.append(f"### Reasoning Chain {i + 1}: {chain.get('title', 'Untitled')}\n")
            # Ensure sorting key exists or provide default
            thoughts = sorted(
                chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float("inf"))
            )
            if not thoughts:
                report_lines.append("_No thoughts recorded in this chain._")
            else:
                for thought in thoughts:
                    is_key_thought = thought.get("thought_type") in [
                        "goal",
                        "decision",
                        "summary",
                        "hypothesis",
                        "inference",
                        "reflection",
                        "critique",
                    ]
                    prefix = "**" if is_key_thought else ""
                    suffix = "**" if is_key_thought else ""
                    type_label = thought.get("thought_type", "Thought").capitalize()
                    # Use safe_format_timestamp for thought timestamps
                    thought_time = safe_format_timestamp(thought.get("created_at"))
                    report_lines.append(
                        f"- {prefix}{type_label}{suffix} ({thought_time}): {thought.get('content', '')}"
                    )
                    links = []
                    if thought.get("relevant_action_id"):
                        links.append(f"Action `{thought['relevant_action_id'][:8]}`")
                    if thought.get("relevant_artifact_id"):
                        links.append(f"Artifact `{thought['relevant_artifact_id'][:8]}`")
                    if thought.get("relevant_memory_id"):
                        links.append(f"Memory `{thought['relevant_memory_id'][:8]}`")
                    if links:
                        report_lines.append(f"  *Related to:* {', '.join(links)}")
            report_lines.append("")

    # --- Artifacts & Outputs ---
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Artifacts & Outputs\n")
        report_lines.append(
            "| Name | Type | Description | Path/Preview | Created | Tags | Output? |"
        )
        report_lines.append(
            "| ---- | ---- | ----------- | ------------ | ------- | ---- | ------- |"
        )
        for artifact in artifacts:
            name = artifact.get("name", "N/A")
            atype = artifact.get("artifact_type", "N/A")
            desc = (artifact.get("description", "") or "")[:50]
            path_or_preview = artifact.get("path", "") or (
                artifact.get("content_preview", "") or ""
            )
            path_or_preview = (
                f"`{path_or_preview}`" if artifact.get("path") else path_or_preview[:60]
            )
            # Use safe_format_timestamp for artifact timestamps
            created_time = safe_format_timestamp(artifact.get("created_at"))
            tags = ", ".join(artifact.get("tags", []))
            is_output = "Yes" if artifact.get("is_output") else "No"
            report_lines.append(
                f"| {name} | {atype} | {desc} | {path_or_preview} | {created_time} | {tags} | {is_output} |"
            )

    # --- Conclusion / Next Steps ---
    report_lines.append("\n## Conclusion & Next Steps\n")
    status = workflow.get("status", "N/A")
    if status == WorkflowStatus.COMPLETED.value:
        report_lines.append("Workflow marked as **Completed**.")
    elif status == WorkflowStatus.FAILED.value:
        report_lines.append("Workflow marked as **Failed**.")
    elif status == WorkflowStatus.ABANDONED.value:
        report_lines.append("Workflow marked as **Abandoned**.")
    elif status == WorkflowStatus.PAUSED.value:
        report_lines.append("Workflow is currently **Paused**.")
    else:  # Active
        report_lines.append("Workflow is **Active**. Potential next steps include:")
        last_action = (
            sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))[-1]
            if actions
            else None
        )
        if last_action and last_action.get("status") == ActionStatus.IN_PROGRESS.value:
            report_lines.append(f"- Completing action: '{last_action.get('title', 'Last Action')}'")
        elif last_action:
            report_lines.append(
                f"- Planning the next action after '{last_action.get('title', 'Last Action')}'"
            )
        else:
            report_lines.append("- Defining the initial actions for the workflow goal.")

    # Footer
    report_lines.append(
        "\n---\n*Report generated on "
        + safe_format_timestamp(datetime.now(timezone.utc).timestamp())
        + "*"
    )
    return "\n".join(report_lines)


async def _generate_concise_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a concise report focusing on key information."""
    report_lines = [
        f"# {workflow.get('title', 'Untitled Workflow')} (`{workflow.get('workflow_id', '')[:8]}`)"
    ]
    report_lines.append(f"**Status:** {workflow.get('status', 'N/A').capitalize()}")
    if workflow.get("goal"):
        report_lines.append(f"**Goal:** {workflow.get('goal', '')[:100]}...")

    actions = workflow.get("actions", [])
    if actions:
        total = len(actions)
        completed = sum(1 for a in actions if a.get("status") == ActionStatus.COMPLETED.value)
        perc = int((completed / total) * 100) if total > 0 else 0
        report_lines.append(f"**Progress:** {perc}% ({completed}/{total} actions)")

    # Recent/Current Actions
    if actions:
        report_lines.append("\n**Recent Activity:**")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(
            actions, key=lambda a: a.get("sequence_number", float("inf")), reverse=True
        )
        for action in sorted_actions[:3]:  # Show top 3 recent
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "in_progress": "⏳",
                "planned": "🗓️",
            }.get(action.get("status"), "❓")
            report_lines.append(f"- {status_emoji} {action.get('title', 'Action')[:50]}")

    # Outputs
    artifacts = workflow.get("artifacts", [])
    outputs = [a for a in artifacts if a.get("is_output")]
    if outputs:
        report_lines.append("\n**Outputs:**")
        for output in outputs[:5]:  # Limit outputs listed
            report_lines.append(
                f"- {output.get('name', 'N/A')} (`{output.get('artifact_type', 'N/A')}`)"
            )

    return "\n".join(report_lines)


async def _generate_narrative_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a narrative-style report as a story."""
    report_lines = [f"# The Journey of: {workflow.get('title', 'Untitled Workflow')}"]

    # Introduction
    report_lines.append("\n## Our Quest Begins\n")
    # Use safe_format_timestamp with the correct key
    start_time = safe_format_timestamp(workflow.get("created_at"))
    if workflow.get("goal"):
        report_lines.append(
            f"We embarked on a mission around {start_time}: **{workflow['goal']}**."
        )
    else:
        report_lines.append(
            f"Our story started on {start_time}, aiming to understand or create '{workflow.get('title', 'something interesting')}'"
        )
    if workflow.get("description"):
        report_lines.append(f"> {workflow['description']}\n")

    # The Path
    actions = workflow.get("actions", [])
    if actions:
        report_lines.append("## The Path Unfolds\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
        for action in sorted_actions:
            title = action.get("title", action.get("action_type", "A step"))
            # Use safe_format_timestamp for action start time
            start_time_action = safe_format_timestamp(action.get("started_at"))
            if action.get("status") == ActionStatus.COMPLETED.value:
                report_lines.append(
                    f"Then, around {start_time_action}, we successfully **{title}**."
                )
                if include_details and action.get("reasoning"):
                    report_lines.append(f"  *Our reasoning was: {action['reasoning'][:150]}...*")
            elif action.get("status") == ActionStatus.FAILED.value:
                report_lines.append(
                    f"Around {start_time_action}, we encountered trouble when trying to **{title}**."
                )
            elif action.get("status") == ActionStatus.IN_PROGRESS.value:
                report_lines.append(
                    f"Starting around {start_time_action}, we are working on **{title}**."
                )
            # Add other statuses if needed (skipped, planned)
            elif action.get("status") == ActionStatus.SKIPPED.value:
                report_lines.append(
                    f"Around {start_time_action}, we decided to skip the step: **{title}**."
                )
            elif action.get("status") == ActionStatus.PLANNED.value:
                report_lines.append(f"The plan included the step: **{title}** (not yet started).")
            report_lines.append("")  # Add spacing between actions

    # Discoveries
    thoughts = [
        t for chain in workflow.get("thought_chains", []) for t in chain.get("thoughts", [])
    ]
    key_thoughts = [
        t
        for t in thoughts
        if t.get("thought_type") in ["decision", "insight", "hypothesis", "summary", "reflection"]
    ]
    if key_thoughts and include_details:
        report_lines.append("## Moments of Clarity\n")
        # Ensure sorting key exists or provide default
        sorted_thoughts = sorted(key_thoughts, key=lambda t: t.get("sequence_number", float("inf")))
        for thought in sorted_thoughts[:7]:
            # Use safe_format_timestamp for thought timestamp
            thought_time = safe_format_timestamp(thought.get("created_at"))
            report_lines.append(
                f"- Around {thought_time}, a key **{thought.get('thought_type')}** emerged: *{thought.get('content', '')[:150]}...*"
            )

    # Treasures
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Treasures Found\n")
        outputs = [a for a in artifacts if a.get("is_output")]
        other_artifacts = [a for a in artifacts if not a.get("is_output")]
        # Ensure sorting key exists or provide default, sort by creation time
        outputs.sort(key=lambda a: a.get("created_at", 0))
        other_artifacts.sort(key=lambda a: a.get("created_at", 0))
        # Combine lists for display, respecting limits
        display_artifacts = outputs[:3] + other_artifacts[: max(0, 5 - len(outputs))]
        display_artifacts.sort(key=lambda a: a.get("created_at", 0))  # Sort combined list

        for artifact in display_artifacts:
            marker = "🏆 Final Result:" if artifact.get("is_output") else "📌 Item Created:"
            # Use safe_format_timestamp for artifact timestamp
            artifact_time = safe_format_timestamp(artifact.get("created_at"))
            report_lines.append(
                f"- {marker} Around {artifact_time}, **{artifact.get('name')}** ({artifact.get('artifact_type')}) was produced."
            )
            if artifact.get("description"):
                report_lines.append(f"  *{artifact['description'][:100]}...*")

    # Current Status/Ending
    status = workflow.get("status", "active")
    report_lines.append(
        f"\n## {"Journey's End" if status == 'completed' else 'The Story So Far...'}\n"
    )
    if status == WorkflowStatus.COMPLETED.value:
        report_lines.append("Our quest is complete! We achieved our objectives.")
    elif status == WorkflowStatus.FAILED.value:
        report_lines.append(
            "Alas, this chapter ends here, marked by challenges we could not overcome."
        )
    elif status == WorkflowStatus.ABANDONED.value:
        report_lines.append("We chose to leave this path, perhaps to return another day.")
    elif status == WorkflowStatus.PAUSED.value:
        report_lines.append("We pause here, taking stock before continuing the adventure.")
    else:
        report_lines.append("The journey continues...")

    # Footer timestamp formatting
    report_lines.append(
        "\n---\n*Narrative recorded on "
        + safe_format_timestamp(datetime.now(timezone.utc).timestamp())
        + "*"
    )
    return "\n".join(report_lines)


async def _generate_technical_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a technical report with data-oriented structure."""
    report_lines = [f"# Technical Report: {workflow.get('title', 'Untitled Workflow')}"]

    # --- Metadata ---
    report_lines.append("\n## Workflow Metadata\n```yaml")
    report_lines.append(f"workflow_id: {workflow.get('workflow_id')}")
    report_lines.append(f"title: {workflow.get('title')}")
    report_lines.append(f"status: {workflow.get('status')}")
    report_lines.append(f"goal: {workflow.get('goal') or 'N/A'}")
    # Use safe_format_timestamp for workflow timestamps
    report_lines.append(f"created_at: {safe_format_timestamp(workflow.get('created_at'))}")
    report_lines.append(f"updated_at: {safe_format_timestamp(workflow.get('updated_at'))}")
    if workflow.get("completed_at"):
        report_lines.append(f"completed_at: {safe_format_timestamp(workflow.get('completed_at'))}")
    if workflow.get("tags"):
        report_lines.append(f"tags: {workflow['tags']}")
    report_lines.append("```")

    # --- Metrics ---
    actions = workflow.get("actions", [])
    if actions:
        report_lines.append("\n## Execution Metrics\n")
        total = len(actions)
        counts = defaultdict(int)
        for a in actions:
            counts[a.get("status", "unknown")] += 1
        report_lines.append("**Action Status Counts:**")
        for status, count in counts.items():
            report_lines.append(f"- {status.capitalize()}: {count} ({int(count / total * 100)}%)")
        type_counts = defaultdict(int)
        for a in actions:
            type_counts[a.get("action_type", "unknown")] += 1
        report_lines.append("\n**Action Type Counts:**")
        for atype, count in type_counts.items():
            report_lines.append(f"- {atype.capitalize()}: {count} ({int(count / total * 100)}%)")

    # --- Action Log ---
    if actions and include_details:
        report_lines.append("\n## Action Log\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
        for action in sorted_actions:
            report_lines.append(f"### Action Sequence: {action.get('sequence_number')}\n```yaml")
            report_lines.append(f"action_id: {action.get('action_id')}")
            report_lines.append(f"title: {action.get('title')}")
            report_lines.append(f"type: {action.get('action_type')}")
            report_lines.append(f"status: {action.get('status')}")
            # Use safe_format_timestamp for action timestamps
            report_lines.append(f"started_at: {safe_format_timestamp(action.get('started_at'))}")
            if action.get("completed_at"):
                report_lines.append(
                    f"completed_at: {safe_format_timestamp(action.get('completed_at'))}"
                )
            if action.get("tool_name"):
                report_lines.append(f"tool_name: {action['tool_name']}")
            # Use the already deserialized data if present
            tool_args_repr = str(action.get("tool_args", "N/A"))
            tool_result_repr = str(action.get("tool_result", "N/A"))
            report_lines.append(f"tool_args_preview: {tool_args_repr[:100]}...")
            report_lines.append(f"tool_result_preview: {tool_result_repr[:100]}...")
            report_lines.append("```")
            if action.get("reasoning"):
                report_lines.append(f"**Reasoning:**\n```\n{action['reasoning']}\n```")

    # --- Artifacts ---
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Artifacts\n```json")
        artifact_list_repr = []
        for artifact in artifacts:
            repr_dict = {
                k: artifact.get(k)
                for k in [
                    "artifact_id",
                    "name",
                    "artifact_type",
                    "description",
                    "path",
                    "is_output",
                    "tags",
                    "created_at",
                ]
            }
            # Format timestamp safely
            if "created_at" in repr_dict:
                repr_dict["created_at"] = safe_format_timestamp(repr_dict["created_at"])
            artifact_list_repr.append(repr_dict)
        # Use default=str for safe JSON dumping
        report_lines.append(json.dumps(artifact_list_repr, indent=2, default=str))
        report_lines.append("```")

    # --- Thoughts ---
    thought_chains = workflow.get("thought_chains", [])
    if thought_chains and include_details:
        report_lines.append("\n## Thought Chains\n")
        for chain in thought_chains:
            report_lines.append(
                f"### Chain: {chain.get('title')} (`{chain.get('thought_chain_id')}`)\n```json"
            )
            # Ensure sorting key exists or provide default
            thoughts = sorted(
                chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float("inf"))
            )
            formatted_thoughts = []
            for thought in thoughts:
                fmt_thought = dict(thought)
                # Format timestamp safely
                if fmt_thought.get("created_at"):
                    fmt_thought["created_at"] = safe_format_timestamp(fmt_thought["created_at"])
                formatted_thoughts.append(fmt_thought)
            # Use default=str for safe JSON dumping
            report_lines.append(json.dumps(formatted_thoughts, indent=2, default=str))
            report_lines.append("```")

    return "\n".join(report_lines)


async def _generate_memory_network_mermaid(
    memories: List[Dict], links: List[Dict], center_memory_id: Optional[str] = None
) -> str:
    """Helper function to generate Mermaid graph syntax for a memory network."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            # Generate a unique fallback for missing IDs to avoid collisions
            # Ensure MemoryUtils is available if needed
            # return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
            return f"{prefix}_MISSING_{str(uuid.uuid4()).replace('-', '_')}"  # Use uuid directly
        # Replace hyphens which are problematic in unquoted Mermaid node IDs
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "graph TD"]  # Top-Down graph direction

    # Node Definitions
    diagram.append("\n    %% Memory Nodes")
    memory_id_to_node_id = {}  # Map full memory ID to sanitized Mermaid node ID
    for memory in memories:
        mem_id = memory.get("memory_id")
        if not mem_id:
            continue

        node_id = sanitize_mermaid_id(mem_id, "M")  # Use sanitized full ID
        memory_id_to_node_id[mem_id] = node_id  # Store mapping

        # Label content: Type, Description (truncated), Importance
        mem_type = memory.get("memory_type", "memory").capitalize()
        # Ensure _mermaid_escape is available
        desc = _mermaid_escape(memory.get("description", mem_id))  # Use full ID if no desc
        if len(desc) > 40:
            desc = desc[:37] + "..."
        importance = memory.get("importance", 5.0)
        label = f"<b>{mem_type}</b><br/>{desc}<br/><i>(I: {importance:.1f})</i>"

        # Node shape/style based on level (e.g., Semantic=rectangle, Episodic=rounded, Procedural=subroutine)
        # Ensure MemoryLevel enum is available
        level = memory.get("memory_level", MemoryLevel.EPISODIC.value)
        shape_start, shape_end = "[", "]"  # Default rectangle (Semantic)
        if level == MemoryLevel.EPISODIC.value:
            shape_start, shape_end = "(", ")"  # Rounded rectangle
        elif level == MemoryLevel.PROCEDURAL.value:
            shape_start, shape_end = "[[", "]]"  # Subroutine shape
        elif level == MemoryLevel.WORKING.value:
            shape_start, shape_end = "([", "])"  # Capsule shape for working memory?

        # Style based on level + highlight center node
        node_style = f":::level{level}"
        if mem_id == center_memory_id:
            node_style += " :::centerNode"  # Add specific style for center

        diagram.append(f'    {node_id}{shape_start}"{label}"{shape_end}{node_style}')

    # Edge Definitions
    diagram.append("\n    %% Memory Links")
    for link in links:
        source_mem_id = link.get("source_memory_id")
        target_mem_id = link.get("target_memory_id")
        link_type = link.get("link_type", "related")

        # Only draw links where both source and target are in the visualized node set
        if source_mem_id in memory_id_to_node_id and target_mem_id in memory_id_to_node_id:
            source_node = memory_id_to_node_id[source_mem_id]
            target_node = memory_id_to_node_id[target_mem_id]
            # Add link type as label, style based on strength? (Keep simple for now)
            diagram.append(f"    {source_node} -- {link_type} --> {target_node}")

    # Class Definitions for Styling
    diagram.append("\n    %% Stylesheets")
    diagram.append(
        "    classDef levelworking fill:#e3f2fd,stroke:#2196f3,color:#1e88e5,stroke-width:1px;"
    )  # Light blue
    diagram.append(
        "    classDef levelepisodic fill:#e8f5e9,stroke:#4caf50,color:#388e3c,stroke-width:1px;"
    )  # Light green
    diagram.append(
        "    classDef levelsemantic fill:#fffde7,stroke:#ffc107,color:#ffa000,stroke-width:1px;"
    )  # Light yellow
    diagram.append(
        "    classDef levelprocedural fill:#fce4ec,stroke:#e91e63,color:#c2185b,stroke-width:1px;"
    )  # Light pink
    diagram.append(
        "    classDef centerNode stroke-width:3px,stroke:#0d47a1,font-weight:bold;"
    )  # Darker blue border for center

    diagram.append("```")
    return "\n".join(diagram)


@with_tool_metrics
@with_error_handling
async def vector_similarity(
    vec_a: List[float],
    vec_b: List[float],
    workflow_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Calculate cosine similarity between two provided vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector
        workflow_id: Optional workflow context for logging
        db_path: Path to database

    Returns:
        Dict containing cosine similarity score
    """
    start_time = time.time()

    # Input validation
    if not vec_a or not vec_b:
        raise ToolInputError("Both vectors must be non-empty lists")

    if len(vec_a) != len(vec_b):
        raise ToolInputError(f"Vector dimensions must match: {len(vec_a)} vs {len(vec_b)}")

    if not all(isinstance(x, (int, float)) for x in vec_a + vec_b):
        raise ToolInputError("All vector elements must be numeric")

    try:
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_a * norm_b)

        processing_time = time.time() - start_time

        if workflow_id:
            logger.info(f"Calculated vector similarity {similarity:.4f} for workflow {workflow_id}")

        return {
            "data": {"cosine_similarity": float(similarity)},
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Error calculating vector similarity: {str(e)}")
        raise ToolError(f"Failed to calculate vector similarity: {str(e)}") from e


# ======================================================
# Exports
# ======================================================

__all__ = [
    # Workflow
    "create_workflow",
    "get_workflow_details",
    # Actions
    "record_action_start",
    "record_action_completion",
    "get_recent_actions",
    # Thoughts
    "get_thought_chain",
    # Core Memory
    "store_memory",
    "get_memory_by_id",
    "get_memory_metadata",
    "get_memory_tags",
    "update_memory_metadata",
    "update_memory_link_metadata",
    "create_memory_link",
    "get_workflow_metadata",
    "get_contradictions",
    "query_memories",
    "update_memory",
    "get_linked_memories",
    "add_tag_to_memory",
    "create_embedding",
    "get_embedding",
    # Context & State
    "get_working_memory",
    "focus_memory",
    "optimize_working_memory",
    "save_cognitive_state",
    "load_cognitive_state",
    # Automated Cognitive Management
    "decay_link_strengths",
    "generate_reflection",
    "get_rich_context_package",
    "get_goal_details",
    "create_goal",
    "vector_similarity",
    "record_artifact",
    "get_artifact_by_id",
    "get_similar_memories",
    "query_goals",
    "consolidate_memories",
    "diagnose_file_access_issues",
    "generate_workflow_report",
]
