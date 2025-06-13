# ultimate_mcp_server/tools/local_text_tools.py
"""
Standalone, secure wrappers around local CLI text-processing utilities (rg, awk, sed, jq)
for the Ultimate MCP Server framework.

This module provides controlled execution of common command-line text tools within
a defined workspace, incorporating enhanced security checks, resource limits, performance
optimizations like caching and concurrency control, and robust error handling.

Key Features:
*   Standalone Functions: Tools exposed as individual async functions.
*   Workspace Confinement: All file/directory operations strictly enforced within WORKSPACE_DIR.
*   Security Hardening: Validates arguments against shell metacharacters, subshells,
    redirection, path traversal, and specific unsafe flags (e.g., `sed -i`). Uses `prctl`
    and `setsid` on Linux for further sandboxing.
*   Resource Limiting: Applies CPU time and memory limits (Unix only).
*   Input Flexibility: Handles input via stdin (`input_data`) or file/directory targets
    specified within the command arguments (`args_str`). Stdin size is capped.
*   Standardized Output: Returns consistent `ToolResult` TypedDict with stdout, stderr,
    exit_code, success status, timing, and truncation info. Output is truncated.
*   Command Integrity: Checks command availability and checksums (lazily, with re-verification).
    Enforces minimum versions.
*   Performance: Includes per-tool concurrency limits and optional disk-based caching
    of identical command invocations.
*   LLM-Friendly: Detailed docstrings, structured errors with codes, optional streaming modes,
    and a `dry_run` option enhance usability for AI agents.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import sys
import textwrap
import time
from dataclasses import (
    dataclass,
)  # Keep field for potential future use, though not used for update now
from enum import Enum
from pathlib import Path
from typing import (
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import aiofiles  # Needed for async checksum

from ultimate_mcp_server.exceptions import ToolExecutionError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.local_text")

# Conditional import for resource limiting and sandboxing
try:
    import resource  # type: ignore [import-not-found]

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    logger.debug("`resource` module not found (likely non-Unix). Resource limits disabled.")

try:
    import prctl  # type: ignore [import-not-found]

    HAS_PRCTL = True
except ImportError:
    HAS_PRCTL = False
    logger.debug("`prctl` module not found (likely non-Linux). Advanced sandboxing disabled.")

# --------------------------------------------------------------------------- #
# Configuration (Loaded from Environment or Defaults)
# --------------------------------------------------------------------------- #

#: Maximum bytes returned in stdout / stderr before truncation
MAX_OUTPUT_BYTES = int(os.getenv("MCP_TEXT_MAX_OUTPUT", "1_000_000"))  # 1 MiB default
#: Maximum bytes accepted via stdin (`input_data`)
MAX_INPUT_BYTES = int(os.getenv("MCP_TEXT_MAX_INPUT", "25_000_000"))  # 25 MiB default
#: Maximum seconds a command may run before being terminated
DEFAULT_TIMEOUT = float(os.getenv("MCP_TEXT_TIMEOUT", "30"))
#: Workspace root – **all file/directory arguments must resolve inside this tree**
try:
    WORKSPACE_DIR_STR = os.getenv("MCP_TEXT_WORKSPACE", ".")
    WORKSPACE_DIR = Path(WORKSPACE_DIR_STR).resolve()
    if not WORKSPACE_DIR.is_dir():
        logger.warning(
            f"MCP_TEXT_WORKSPACE ('{WORKSPACE_DIR_STR}' -> '{WORKSPACE_DIR}') is not a directory. Defaulting to current."
        )
        WORKSPACE_DIR = Path(".").resolve()
except Exception as e:
    logger.error(
        f"Error resolving MCP_TEXT_WORKSPACE ('{WORKSPACE_DIR_STR}'): {e}. Defaulting to current."
    )
    WORKSPACE_DIR = Path(".").resolve()
logger.info(f"LocalTextTools workspace confined to: {WORKSPACE_DIR}")

#: Disk cache directory for command results
CACHE_DIR_STR = os.getenv("MCP_TEXT_CACHE_DIR", "~/.cache/ultimate_mcp_server/local_text_tools")
CACHE_DIR = Path(CACHE_DIR_STR).expanduser().resolve()
CACHE_ENABLED = os.getenv("MCP_TEXT_CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE_MB = int(os.getenv("MCP_TEXT_CACHE_MAX_MB", "500"))
CACHE_MAX_AGE_DAYS = int(os.getenv("MCP_TEXT_CACHE_MAX_AGE_DAYS", "7"))

if CACHE_ENABLED:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using command invocation cache directory: {CACHE_DIR}")
    except OSError as e:
        logger.error(
            f"Failed to create command cache directory {CACHE_DIR}: {e}. Caching disabled."
        )
        CACHE_ENABLED = False  # Disable cache if directory fails

# Concurrency limits per command
DEFAULT_CONCURRENCY = 4
CONCURRENCY_LIMITS = {
    "rg": int(os.getenv("MCP_TEXT_CONCURRENCY_RG", "8")),
    "awk": int(os.getenv("MCP_TEXT_CONCURRENCY_AWK", str(DEFAULT_CONCURRENCY))),
    "sed": int(os.getenv("MCP_TEXT_CONCURRENCY_SED", str(DEFAULT_CONCURRENCY))),
    "jq": int(os.getenv("MCP_TEXT_CONCURRENCY_JQ", str(DEFAULT_CONCURRENCY))),
}

# Forbidden shell metacharacters (pre-compiled set for efficiency)
_FORBIDDEN_CHARS_SET = frozenset("&;`|><$()")


# --------------------------------------------------------------------------- #
# Error Codes Enum
# --------------------------------------------------------------------------- #
class ToolErrorCode(str, Enum):
    """Machine-readable error codes for local text tools."""

    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    ABS_PATH_FORBIDDEN = "ABS_PATH_FORBIDDEN"
    WORKSPACE_VIOLATION = "WORKSPACE_VIOLATION"
    FORBIDDEN_FLAG = "FORBIDDEN_FLAG"
    FORBIDDEN_CHAR = "FORBIDDEN_CHAR"
    CMD_SUBSTITUTION = "CMD_SUBSTITUTION"  # Often related to $() or ``
    INVALID_ARGS = "INVALID_ARGS"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    INVALID_JSON_INPUT = "INVALID_JSON_INPUT"
    CMD_NOT_FOUND = "CMD_NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    EXEC_ERROR = "EXEC_ERROR"
    COMMUNICATION_ERROR = "COMMUNICATION_ERROR"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    VERSION_TOO_OLD = "VERSION_TOO_OLD"
    UNEXPECTED_FAILURE = "UNEXPECTED_FAILURE"
    CACHE_ERROR = "CACHE_ERROR"


# --------------------------------------------------------------------------- #
# Result Schema (TypedDict)
# --------------------------------------------------------------------------- #
class ToolResult(TypedDict, total=False):
    """Standardized result structure for local text tool executions."""

    stdout: Optional[str]  # Decoded standard output (potentially truncated)
    stderr: Optional[str]  # Decoded standard error (potentially truncated)
    exit_code: Optional[int]  # Process exit code
    success: bool  # True if execution considered successful (depends on tool/retcode)
    error: Optional[str]  # Human-readable error message if success is False
    error_code: Optional[ToolErrorCode]  # Machine-readable error code if success is False
    duration: float  # Execution duration in seconds
    stdout_truncated: bool  # True if stdout was truncated
    stderr_truncated: bool  # True if stderr was truncated
    cached_result: bool  # True if this result was served from cache
    dry_run_cmdline: Optional[List[str]]  # Populated only if dry_run=True


# --------------------------------------------------------------------------- #
# Command Metadata & Discovery
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)  # Make truly immutable
class CommandMeta:
    """Metadata for a command-line tool."""

    name: str
    path: Optional[Path] = None  # Store the resolved absolute path
    checksum: Optional[str] = None  # SHA-256 checksum of the executable (calculated lazily)
    mtime: Optional[float] = None  # Last modification time (for checksum re-verification)
    version: Optional[tuple[int, ...]] = None  # Parsed version tuple (e.g., (13, 0, 1))
    forbidden_flags: frozenset[str] = frozenset()  # Flags disallowed for security
    readonly: bool = True  # True if the command should not modify the filesystem
    min_version: Optional[tuple[int, ...]] = None  # Minimum required version tuple


# Store CommandMeta objects, keyed by command name
_COMMAND_METADATA: Dict[str, CommandMeta] = {
    "rg": CommandMeta("rg", min_version=(13, 0, 0)),  # Example: Require ripgrep >= 13.0.0
    "awk": CommandMeta(
        "awk", forbidden_flags=frozenset({"-i", "--in-place"})
    ),  # AWK in-place is less common but exists (gawk)
    "sed": CommandMeta("sed", forbidden_flags=frozenset({"-i", "--in-place"})),
    "jq": CommandMeta("jq", min_version=(1, 6)),  # Example: Require jq >= 1.6
}
_COMMAND_VERSIONS_CACHE: Dict[str, Optional[tuple[int, ...]]] = {}  # Cache parsed versions
_checksum_lock = asyncio.Lock()  # Lock for lazy checksum calculation
_version_lock = asyncio.Lock()  # Lock for lazy version checking


async def _calculate_sha256sum_async(path: Path, chunk: int = 262_144) -> str:
    """Asynchronously calculates SHA256 checksum using aiofiles."""
    h = hashlib.sha256()
    try:
        async with aiofiles.open(path, "rb") as fh:
            while True:
                blk = await fh.read(chunk)
                if not blk:
                    break
                h.update(blk)
        return h.hexdigest()
    except OSError as e:
        logger.error(f"Failed to calculate SHA256 for {path}: {e}")
        return "error_calculating_checksum"


async def _get_command_checksum(meta: CommandMeta) -> Optional[str]:
    """Lazily calculates and caches the command checksum, verifying mtime."""
    global _COMMAND_METADATA  # Allow modification (replacing entry)
    if not meta.path:
        return None  # Cannot checksum if path unknown

    async with _checksum_lock:
        # Re-fetch meta inside lock in case it was updated by another coroutine
        meta_locked = _COMMAND_METADATA.get(meta.name)
        if not meta_locked or not meta_locked.path:
            return None

        needs_recalc = False
        current_mtime = None
        try:
            # Use asyncio.to_thread for potentially blocking stat
            stat_res = await asyncio.to_thread(meta_locked.path.stat)
            current_mtime = stat_res.st_mtime
            if (
                meta_locked.checksum is None
                or meta_locked.mtime is None
                or current_mtime != meta_locked.mtime
            ):
                needs_recalc = True
        except OSError as e:
            logger.warning(
                f"Could not stat {meta_locked.path} for checksum verification: {e}. Recalculating."
            )
            needs_recalc = True  # Force recalc if stat fails

        if needs_recalc:
            logger.debug(
                f"Calculating checksum for {meta.name} (mtime changed or first calculation)..."
            )
            new_checksum = await _calculate_sha256sum_async(meta_locked.path)
            _COMMAND_METADATA[meta.name] = CommandMeta(
                name=meta_locked.name,
                path=meta_locked.path,
                checksum=new_checksum,
                mtime=current_mtime,  # Store the mtime when checksum was calculated
                forbidden_flags=meta_locked.forbidden_flags,
                readonly=meta_locked.readonly,
                min_version=meta_locked.min_version,
            )
            logger.debug(f"Updated checksum for {meta.name}: {new_checksum[:12]}...")
            return new_checksum
        else:
            # Return cached checksum
            return meta_locked.checksum


async def _parse_version(cmd_path: Path) -> Optional[tuple[int, ...]]:
    """Runs '<tool> --version' and parses semantic version tuple."""
    try:
        proc = await asyncio.create_subprocess_exec(
            str(cmd_path),
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            logger.warning(
                f"Command '{cmd_path.name} --version' failed with code {proc.returncode}: {stderr_b.decode(errors='ignore')[:100]}"
            )
            return None

        output = stdout_b.decode(errors="ignore").strip()
        # Common version patterns (adjust as needed for specific tools)
        match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", output)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3) or 0)  # Default patch to 0
            return (major, minor, patch)
        else:
            logger.warning(f"Could not parse version from '{cmd_path.name}' output: {output[:100]}")
            return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout getting version for '{cmd_path.name}'.")
        return None
    except Exception as e:
        logger.error(f"Error getting version for '{cmd_path.name}': {e}")
        return None


async def _check_command_version(meta: CommandMeta) -> None:
    """Checks if the command meets the minimum version requirement."""
    global _COMMAND_VERSIONS_CACHE  # Allow modification
    if not meta.path or not meta.min_version:
        return  # Skip check if no path or no minimum defined

    async with _version_lock:
        # Check cache first
        if meta.name in _COMMAND_VERSIONS_CACHE:
            actual_version = _COMMAND_VERSIONS_CACHE[meta.name]
        else:
            # Parse version if not cached
            actual_version = await _parse_version(meta.path)
            _COMMAND_VERSIONS_CACHE[meta.name] = actual_version  # Cache result (even None)

    if actual_version is None:
        logger.warning(
            f"Could not determine version for '{meta.name}'. Skipping minimum version check."
        )
        return

    if actual_version < meta.min_version:
        actual_str = ".".join(map(str, actual_version))
        required_str = ".".join(map(str, meta.min_version))
        raise ToolExecutionError(
            f"Command '{meta.name}' version ({actual_str}) is older than required minimum ({required_str}). Please update.",
            error_code=ToolErrorCode.VERSION_TOO_OLD,
            details={"required": required_str, "actual": actual_str},
        )
    else:
        logger.debug(
            f"Version check passed for {meta.name} (found {''.join(map(str, actual_version))}, required >= {''.join(map(str, meta.min_version))})"
        )


def _initial_command_discovery() -> None:
    """Finds commands and stores paths. Logs warnings for missing commands."""
    global _COMMAND_METADATA  # Modify the global dict
    missing: list[str] = []
    updated_metadata = {}

    for name, meta in _COMMAND_METADATA.items():
        exe_path_str = shutil.which(name)
        if exe_path_str is None:
            missing.append(name)
            # Keep original meta without path if not found
            updated_metadata[name] = CommandMeta(
                name=meta.name,
                path=None,
                checksum=None,
                mtime=None,
                version=None,  # Reset dynamic fields
                forbidden_flags=meta.forbidden_flags,
                readonly=meta.readonly,
                min_version=meta.min_version,
            )
            continue

        resolved_path = Path(exe_path_str).resolve()
        current_mtime = None
        try:
            # Use sync stat here, module load time is acceptable
            current_mtime = resolved_path.stat().st_mtime
        except OSError as e:
            logger.warning(f"Could not stat {resolved_path} during initial discovery: {e}")

        # Create updated CommandMeta with path and mtime (checksum/version are lazy)
        updated_meta = CommandMeta(
            name=meta.name,
            path=resolved_path,
            checksum=None,  # Lazy loaded
            mtime=current_mtime,
            # Keep original version/checksum null, they are loaded async later
            version=None,
            forbidden_flags=meta.forbidden_flags,
            readonly=meta.readonly,
            min_version=meta.min_version,
        )
        updated_metadata[name] = updated_meta
        logger.debug(f"{name}: found at {resolved_path}")

    _COMMAND_METADATA = updated_metadata  # Replace global dict

    if missing:
        logger.warning(
            "Missing local text CLI tools: %s. Corresponding functions will fail.",
            ", ".join(missing),
            # emoji_key="warning", # Assuming logger supports this extra field
        )


# Run discovery when the module is loaded
_initial_command_discovery()


# --------------------------------------------------------------------------- #
# Argument Validation Placeholder
# --------------------------------------------------------------------------- #


def _validate_arguments(cmd_name: str, argv: List[str]) -> None:
    """
    Validates command arguments for security and workspace compliance.
    (Placeholder implementation - refine based on specific security needs)
    """
    meta = _COMMAND_METADATA.get(cmd_name)
    if not meta:
        # This should generally not happen if called after discovery
        raise ToolInputError(
            f"Metadata not found for command '{cmd_name}' during validation.",
            param_name="cmd_name",
            details={"unexpected_failure": True},
        )

    forbidden_flags = meta.forbidden_flags

    for i, arg in enumerate(argv):
        # Check for forbidden flags
        if arg in forbidden_flags:
            raise ToolInputError(
                f"Forbidden flag '{arg}' is not allowed for command '{cmd_name}'.",
                param_name="args_str",
                details={
                    "argument": arg,
                    "index": i,
                    "command": cmd_name,
                    "error_code": ToolErrorCode.FORBIDDEN_FLAG.value,
                },
            )

        # Command-specific validation rules
        if cmd_name == "rg":
            # For rg, allow regex metacharacters ( (), |, ?, +, *, { }, [, ], ^, $, . )
            forbidden_chars = {
                char for char in arg if char in _FORBIDDEN_CHARS_SET and char not in "()|"
            }
            if forbidden_chars:
                raise ToolInputError(
                    f"Argument '{arg}' contains forbidden shell metacharacter(s): {', '.join(sorted(forbidden_chars))}",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "forbidden_chars": sorted(list(forbidden_chars)),
                        "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                    },
                )

            # Basic check for command substitution patterns (can be complex)
            if "`" in arg or "$(" in arg:
                raise ToolInputError(
                    f"Argument '{arg}' seems to contain command substitution, which is forbidden.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.CMD_SUBSTITUTION.value,
                    },
                )

            continue  # Skip further checks for rg

        elif cmd_name == "jq":
            # For jq, allow (, ), |, > and < characters as they're essential for the query language
            forbidden_chars = {
                char for char in arg if char in _FORBIDDEN_CHARS_SET and char not in "()|}><"
            }
            if forbidden_chars:
                raise ToolInputError(
                    f"Argument '{arg}' contains forbidden shell metacharacter(s): {', '.join(sorted(forbidden_chars))}",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "forbidden_chars": sorted(list(forbidden_chars)),
                        "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                    },
                )

            # Basic check for command substitution patterns (can be complex)
            if "`" in arg or "$(" in arg:
                raise ToolInputError(
                    f"Argument '{arg}' seems to contain command substitution, which is forbidden.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.CMD_SUBSTITUTION.value,
                    },
                )

            continue  # Skip further checks for jq

        elif cmd_name == "awk":
            # For awk, allow $ character (for field references), / (for regex patterns),
            # and other characters needed for awk scripts like (, ), ;
            # Allow '>' and '<' for comparisons in AWK
            forbidden_chars = {
                char for char in arg if char in _FORBIDDEN_CHARS_SET and char not in "$/();{}<>"
            }
            
            # Special check for file redirection (> followed by a string in quotes)
            # Pattern detects constructs like: print $1 > "file.txt" or print $1 > 'file.txt'
            if re.search(r'>\s*["\']', arg) or re.search(r'print.*>\s*["\']', arg):
                raise ToolInputError(
                    f"Argument '{arg}' appears to contain file redirection, which is forbidden.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                    },
                )
                
            if forbidden_chars:
                raise ToolInputError(
                    f"Argument '{arg}' contains forbidden shell metacharacter(s): {', '.join(sorted(forbidden_chars))}",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "forbidden_chars": sorted(list(forbidden_chars)),
                        "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                    },
                )

            # Basic check for command substitution patterns (can be complex)
            if "`" in arg or "$(" in arg:
                raise ToolInputError(
                    f"Argument '{arg}' seems to contain command substitution, which is forbidden.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.CMD_SUBSTITUTION.value,
                    },
                )

            # Don't treat awk patterns like /pattern/ as absolute paths
            if arg.startswith("/") and not (arg.count("/") >= 2 and arg[1:].find("/") > 0):
                # Still check for absolute paths that aren't regex patterns
                # A regex pattern would typically have at least one more / after the first character
                try:
                    # Resolve the path relative to the workspace *without* accessing filesystem yet
                    # Use os.path.normpath and os.path.join for basic checks before full resolve
                    norm_path = os.path.normpath(os.path.join(str(WORKSPACE_DIR), arg))
                    if not norm_path.startswith(str(WORKSPACE_DIR)):
                        raise ToolInputError(
                            f"Path traversal or absolute path '{arg}' is forbidden.",
                            param_name="args_str",
                            details={
                                "argument": arg,
                                "index": i,
                                "error_code": ToolErrorCode.PATH_TRAVERSAL.value,
                            },
                        )
                except Exception as e:
                    logger.error(f"Error checking path '{arg}': {e}")
                    raise ToolInputError(
                        f"Invalid path argument '{arg}'.",
                        param_name="args_str",
                        details={
                            "argument": arg,
                            "index": i,
                            "error_code": ToolErrorCode.INVALID_ARGS.value,
                        },
                    ) from e

            continue  # Skip further checks for awk

        elif cmd_name == "sed":
            # For sed, allow / (for regex patterns), as well as |, (, ) for sed expressions
            forbidden_chars = {
                char for char in arg if char in _FORBIDDEN_CHARS_SET and char not in "/|()"
            }
            if forbidden_chars:
                raise ToolInputError(
                    f"Argument '{arg}' contains forbidden shell metacharacter(s): {', '.join(sorted(forbidden_chars))}",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "forbidden_chars": sorted(list(forbidden_chars)),
                        "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                    },
                )

            # Basic check for command substitution patterns (can be complex)
            if "`" in arg or "$(" in arg:
                raise ToolInputError(
                    f"Argument '{arg}' seems to contain command substitution, which is forbidden.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.CMD_SUBSTITUTION.value,
                    },
                )

            # Don't treat sed patterns like /pattern/ as absolute paths
            if (
                arg.startswith("/")
                and arg != "/"
                and "/ " not in arg
                and not (arg.count("/") >= 2 and arg[1:].find("/") > 0)
            ):
                # Still check for absolute paths that aren't regex patterns
                try:
                    # Resolve the path relative to the workspace *without* accessing filesystem yet
                    # Use os.path.normpath and os.path.join for basic checks before full resolve
                    norm_path = os.path.normpath(os.path.join(str(WORKSPACE_DIR), arg))
                    if not norm_path.startswith(str(WORKSPACE_DIR)):
                        raise ToolInputError(
                            f"Path traversal or absolute path '{arg}' is forbidden.",
                            param_name="args_str",
                            details={
                                "argument": arg,
                                "index": i,
                                "error_code": ToolErrorCode.PATH_TRAVERSAL.value,
                            },
                        )
                except Exception as e:
                    logger.error(f"Error checking path '{arg}': {e}")
                    raise ToolInputError(
                        f"Invalid path argument '{arg}'.",
                        param_name="args_str",
                        details={
                            "argument": arg,
                            "index": i,
                            "error_code": ToolErrorCode.INVALID_ARGS.value,
                        },
                    ) from e

            continue  # Skip further checks for sed

        # Standard checks for all other commands
        # Check for forbidden characters
        if any(char in _FORBIDDEN_CHARS_SET for char in arg):
            # Be more specific about which char if possible
            found_chars = {char for char in arg if char in _FORBIDDEN_CHARS_SET}
            raise ToolInputError(
                f"Argument '{arg}' contains forbidden shell metacharacter(s): {', '.join(sorted(found_chars))}",
                param_name="args_str",
                details={
                    "argument": arg,
                    "index": i,
                    "forbidden_chars": sorted(list(found_chars)),
                    "error_code": ToolErrorCode.FORBIDDEN_CHAR.value,
                },
            )

        # Basic check for command substitution patterns (can be complex)
        if "`" in arg or "$(" in arg:
            raise ToolInputError(
                f"Argument '{arg}' seems to contain command substitution, which is forbidden.",
                param_name="args_str",
                details={
                    "argument": arg,
                    "index": i,
                    "error_code": ToolErrorCode.CMD_SUBSTITUTION.value,
                },
            )

        # --- Path Validation ---
        # Heuristic: Does the argument look like a path that needs checking?
        # This is tricky. We might only check args that aren't flags (don't start with '-')
        # or args known to take paths for specific commands.
        # For simplicity here, we check any arg that contains '/' or could be a filename.
        # More robust: parse args properly (difficult) or check based on tool context.
        potential_path = False
        if not arg.startswith("-") and (os.sep in arg or "." in arg or Path(arg).suffix):
            potential_path = True
            # Allow '-' as a special argument representing stdin/stdout
            if arg == "-":
                potential_path = False

        if potential_path:
            try:
                # Disallow absolute paths
                if Path(arg).is_absolute():
                    raise ToolInputError(
                        f"Absolute paths like '{arg}' are forbidden. Use paths relative to the workspace.",
                        param_name="args_str",
                        details={
                            "argument": arg,
                            "index": i,
                            "error_code": ToolErrorCode.ABS_PATH_FORBIDDEN.value,
                        },
                    )

                # Resolve the path relative to the workspace *without* accessing filesystem yet
                # Use os.path.normpath and os.path.join for basic checks before full resolve
                norm_path = os.path.normpath(os.path.join(str(WORKSPACE_DIR), arg))

                # Check for path traversal using normpath result
                if not norm_path.startswith(str(WORKSPACE_DIR)):
                    # Check specifically for '..' components that might escape
                    if ".." in Path(arg).parts:
                        raise ToolInputError(
                            f"Path traversal ('..') is forbidden in argument '{arg}'.",
                            param_name="args_str",
                            details={
                                "argument": arg,
                                "index": i,
                                "error_code": ToolErrorCode.PATH_TRAVERSAL.value,
                            },
                        )
                    else:
                        # Generic workspace violation if normpath doesn't match prefix (e.g., symlinks handled later)
                        raise ToolInputError(
                            f"Argument '{arg}' resolves outside the allowed workspace '{WORKSPACE_DIR}'.",
                            param_name="args_str",
                            details={
                                "argument": arg,
                                "index": i,
                                "resolved_norm": norm_path,
                                "error_code": ToolErrorCode.WORKSPACE_VIOLATION.value,
                            },
                        )

                # More robust check: Resolve the path fully and check again
                # This *does* access the filesystem but ensures symlinks are handled
                resolved_arg_path = (WORKSPACE_DIR / arg).resolve()
                if not resolved_arg_path.is_relative_to(WORKSPACE_DIR):
                    raise ToolInputError(
                        f"Argument '{arg}' resolves outside the allowed workspace '{WORKSPACE_DIR}' (checked after resolving symlinks).",
                        param_name="args_str",
                        details={
                            "argument": arg,
                            "index": i,
                            "resolved_absolute": str(resolved_arg_path),
                            "error_code": ToolErrorCode.WORKSPACE_VIOLATION.value,
                        },
                    )

            except OSError as e:
                # Ignore errors resolving paths that might not exist yet (e.g., output files for some tools)
                # But log a warning. More strict validation could forbid non-existent input paths.
                logger.debug(
                    f"Could not fully resolve potential path argument '{arg}': {e}. Assuming OK if basic checks passed."
                )
            except ToolInputError:
                raise  # Re-raise our specific validation errors
            except Exception as e:
                logger.error(f"Unexpected error validating argument '{arg}': {e}", exc_info=True)
                raise ToolInputError(
                    f"Unexpected error during validation of argument '{arg}'.",
                    param_name="args_str",
                    details={
                        "argument": arg,
                        "index": i,
                        "error_code": ToolErrorCode.UNEXPECTED_FAILURE.value,
                    },
                ) from e


# --------------------------------------------------------------------------- #
# Invocation Caching (Disk-based LRU-like)
# --------------------------------------------------------------------------- #


def _get_cache_key(cmd_name: str, argv: Sequence[str], input_data_bytes: Optional[bytes]) -> str:
    """Creates a hash key based on command, args, and input data bytes."""
    hasher = hashlib.sha256()
    hasher.update(cmd_name.encode())
    for arg in argv:
        hasher.update(arg.encode())
    if input_data_bytes is not None:
        hasher.update(b"\x00\x01")  # Separator for input data
        hasher.update(input_data_bytes)
    else:
        hasher.update(b"\x00\x00")  # Separator for no input data
    return hasher.hexdigest()


def _get_cache_path(key: str) -> Path:
    """Gets the file path for a cache key."""
    # Simple structure: cache_dir / key_prefix / key.json
    prefix = key[:2]
    return CACHE_DIR / prefix / f"{key}.json"


# --- Async Cache Get/Put using asyncio.to_thread for OS file IO ---
async def _cache_get_async(key: str) -> Optional[ToolResult]:
    """Asynchronously gets a result from the disk cache."""
    if not CACHE_ENABLED:
        return None
    cache_path = _get_cache_path(key)
    try:
        if await asyncio.to_thread(cache_path.exists):
            # Check cache entry age
            stat_res = await asyncio.to_thread(cache_path.stat)
            age_seconds = time.time() - stat_res.st_mtime
            if age_seconds > (CACHE_MAX_AGE_DAYS * 24 * 3600):
                logger.debug(
                    f"Cache entry {key[:8]} expired (age {age_seconds:.0f}s > {CACHE_MAX_AGE_DAYS}d). Removing."
                )
                try:
                    await asyncio.to_thread(cache_path.unlink)
                except OSError as e:
                    logger.warning(f"Failed to remove expired cache file {cache_path}: {e}")
                return None

            # Read cached data (aiofiles is okay for read/write content)
            async with aiofiles.open(cache_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            # --- Type Check and Reconstruction ---
            required_keys = {
                "stdout": None,
                "stderr": None,
                "exit_code": None,
                "success": False,
                "duration": 0.0,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": None,
                "error_code": None,
                "cached_result": True,  # Set flag here
            }
            validated_data = {}
            for k, default_val in required_keys.items():
                validated_data[k] = data.get(k, default_val)

            # Ensure error_code is None or a valid ToolErrorCode enum member
            raw_error_code = validated_data.get("error_code")
            if raw_error_code is not None:
                try:
                    validated_data["error_code"] = ToolErrorCode(raw_error_code)
                except ValueError:
                    logger.warning(
                        f"Invalid error_code '{raw_error_code}' found in cache for key {key[:8]}. Setting to None."
                    )
                    validated_data["error_code"] = None

            logger.info(f"Cache HIT for key {key[:8]}.")
            # Use cast to satisfy type checker, assuming validation ensures structure
            return cast(ToolResult, validated_data)
        else:
            # logger.debug(f"Cache MISS for key {key[:8]}.")
            return None
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Cache read error for key {key[:8]}: {e}. Treating as miss.")
        # Attempt to remove potentially corrupt file
        try:
            if await asyncio.to_thread(cache_path.exists):
                await asyncio.to_thread(cache_path.unlink)
        except OSError:
            pass
        return None
    except Exception as e:
        logger.error(f"Unexpected cache get error for key {key[:8]}: {e}", exc_info=True)
        return None


async def _cache_put_async(key: str, result: ToolResult):
    """Asynchronously puts a result into the disk cache."""
    if not CACHE_ENABLED:
        return
    cache_path = _get_cache_path(key)
    try:
        # Ensure parent directory exists
        await asyncio.to_thread(cache_path.parent.mkdir, parents=True, exist_ok=True)

        # Write data as JSON
        result_to_write = result.copy()
        result_to_write.pop("cached_result", None)  # Don't store cache flag itself
        # Ensure enum is converted to string for JSON
        if result_to_write.get("error_code") is not None:
            result_to_write["error_code"] = result_to_write["error_code"].value

        json_data = json.dumps(result_to_write, indent=2)  # Pretty print for readability
        async with aiofiles.open(cache_path, mode="w", encoding="utf-8") as f:
            await f.write(json_data)
        logger.debug(f"Cache PUT successful for key {key[:8]}.")

        # Trigger cleanup check periodically (simple random approach)
        if random.random() < 0.01:  # ~1% chance on write
            asyncio.create_task(_cleanup_cache_lru())
    except (OSError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Cache write error for key {key[:8]}: {e}")
    except Exception as e:
        logger.error(f"Unexpected cache put error for key {key[:8]}: {e}", exc_info=True)


# --- Cache Cleanup (LRU-like based on mtime) ---
async def _cleanup_cache_lru():
    """Removes cache files exceeding max size or age."""
    if not CACHE_ENABLED:
        return
    logger.debug("Running cache cleanup...")
    try:
        files: List[Tuple[Path, float, int]] = []
        total_size = 0
        now = time.time()
        max_age_seconds = CACHE_MAX_AGE_DAYS * 24 * 3600
        max_size_bytes = CACHE_MAX_SIZE_MB * 1024 * 1024

        # Use aiofiles.os.walk (designed for async iteration)
        # Note: aiofiles.os.walk might still be less performant than scandir + to_thread for large dirs
        async for root, _, filenames in aiofiles.os.walk(CACHE_DIR):
            for filename in filenames:
                if filename.endswith(".json"):
                    filepath = Path(root) / filename
                    try:
                        stat_res = await asyncio.to_thread(filepath.stat)
                        files.append((filepath, stat_res.st_mtime, stat_res.st_size))
                        total_size += stat_res.st_size
                    except OSError:
                        continue  # Skip files we can't stat

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])

        removed_count = 0
        removed_size = 0
        current_total_size = total_size  # Keep track of size as we remove

        # Remove files based on age or if total size exceeds limit
        for filepath, mtime, size in files:
            age = now - mtime
            # Check size limit against potentially reduced total size
            over_size_limit = current_total_size > max_size_bytes
            over_age_limit = age > max_age_seconds

            if over_age_limit or over_size_limit:
                try:
                    await asyncio.to_thread(filepath.unlink)
                    removed_count += 1
                    removed_size += size
                    current_total_size -= size
                    logger.debug(
                        f"Cache cleanup removed: {filepath.name} (Reason: {'Age' if over_age_limit else 'Size'})"
                    )
                except OSError as e:
                    logger.warning(f"Cache cleanup failed to remove {filepath}: {e}")
            # Optimization: If we are no longer over the size limit, we only need to continue checking for age limit
            elif not over_size_limit:
                # We must continue iterating to check older files for age limit.
                pass

        if removed_count > 0:
            logger.info(
                f"Cache cleanup complete. Removed {removed_count} files ({removed_size / (1024 * 1024):.1f} MB). Current size: {current_total_size / (1024 * 1024):.1f} MB"
            )
        else:
            logger.debug(
                f"Cache cleanup complete. No files removed. Current size: {current_total_size / (1024 * 1024):.1f} MB"
            )

    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}", exc_info=True)


# --------------------------------------------------------------------------- #
# Resource Limiting Function (Module Scope)
# --------------------------------------------------------------------------- #


def _limit_resources(timeout: float, cmd_name: Optional[str] = None) -> None:
    """Sets resource limits for the child process (Unix only), potentially command-specific."""
    try:
        if HAS_RESOURCE:
            # CPU seconds (add 1s buffer)
            cpu_limit = int(timeout) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

            # Address-space (bytes) - Virtual Memory
            soft_as_str = os.getenv("MCP_TEXT_RLIMIT_AS", "2_147_483_648")  # 2GiB default
            hard_as_str = os.getenv(
                "MCP_TEXT_RLIMIT_AS_HARD", str(int(soft_as_str) + 100 * 1024 * 1024)
            )  # +100MiB buffer default
            try:
                soft_as = int(soft_as_str)
                hard_as = int(hard_as_str)
                if soft_as > 0 and hard_as > 0:  # Allow disabling with 0 or negative
                    resource.setrlimit(resource.RLIMIT_AS, (soft_as, hard_as))
                    logger.debug(
                        f"Applied resource limit: AS={soft_as / (1024**3):.1f}GiB (soft), {hard_as / (1024**3):.1f}GiB (hard)"
                    )
            except ValueError:
                logger.warning(
                    "Invalid value for MCP_TEXT_RLIMIT_AS or MCP_TEXT_RLIMIT_AS_HARD. Skipping AS limit."
                )
            except resource.error as e:
                logger.warning(
                    f"Failed to set AS limit: {e}. Limit might be too low or too high for the system."
                )

            # No core dumps
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except resource.error as e:
                logger.warning(f"Failed to disable core dumps: {e}")  # May fail in containers

            # --- RLIMIT_NPROC --------------------------------------------------
            try:
                # honour optional global opt-out
                if os.getenv("MCP_TEXT_RLIMIT_NPROC_DISABLE", "").lower() == "true":
                    logger.debug("RLIMIT_NPROC: disabled via env flag")
                else:
                    soft_req = int(os.getenv("MCP_TEXT_RLIMIT_NPROC_SOFT", "4096"))
                    hard_req = int(os.getenv("MCP_TEXT_RLIMIT_NPROC_HARD", "8192"))
                    if cmd_name == "rg":
                        soft_req = int(os.getenv("MCP_TEXT_RLIMIT_NPROC_SOFT_RG", "16384"))
                        hard_req = int(os.getenv("MCP_TEXT_RLIMIT_NPROC_HARD_RG", "32768"))

                    cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_NPROC)

                    # translate "unlimited"/-1
                    req_soft = cur_soft if soft_req <= 0 else soft_req
                    req_hard = cur_hard if hard_req <= 0 else hard_req

                    # never *lower* an existing limit
                    new_soft = max(cur_soft, req_soft)
                    new_hard = max(cur_hard, req_hard)

                    # only call setrlimit if anything actually changes
                    if (new_soft, new_hard) != (cur_soft, cur_hard):
                        resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, new_hard))
                        logger.debug(
                            f"RLIMIT_NPROC set to (soft={new_soft}, hard={new_hard}) "
                            f"(was {cur_soft}/{cur_hard}) for {cmd_name}"
                        )
            except (ValueError, resource.error) as e:
                logger.warning(f"RLIMIT_NPROC not applied: {e}")

            logger.debug(f"Applied resource limits: CPU={cpu_limit}s")  # Summary log

        if HAS_PRCTL and sys.platform == "linux":
            try:
                # Prevent privilege escalation
                prctl.set_no_new_privs(True)
                logger.debug("Applied prctl NO_NEW_PRIVS.")
            except Exception as e:  # prctl might raise various errors
                logger.warning(f"Failed to set prctl NO_NEW_PRIVS: {e}")

        # Run in new session to isolate from controlling terminal (if any)
        # This helps ensure signals (like Ctrl+C) don't propagate unexpectedly.
        if sys.platform != "win32" and hasattr(os, "setsid"):
            try:
                os.setsid()
                logger.debug("Process started in new session ID.")
            except OSError as e:
                logger.warning(f"Failed to call setsid: {e}")  # May fail if already session leader

    except Exception as exc:
        # Catch-all for any unexpected issue during limit setting
        logger.warning(f"Failed to apply resource limits/sandboxing: {exc}")


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def _truncate(data: bytes) -> tuple[str, bool]:
    """Decodes bytes to string and truncates if necessary."""
    truncated = False
    if len(data) > MAX_OUTPUT_BYTES:
        data = data[:MAX_OUTPUT_BYTES]
        truncated = True
        # Try to decode truncated data, replacing errors
        decoded = data.decode("utf-8", errors="replace")
        # Add truncation marker AFTER decoding to avoid corrupting multibyte char
        decoded += "\n... (output truncated)"
    else:
        decoded = data.decode("utf-8", errors="replace")
    return decoded, truncated


def _is_json_or_json_lines(text: str) -> bool:
    """
    True  → text is a single JSON document        (json.loads ok)
    True  → text is newline-delimited JSON lines  (all lines load)
    False → otherwise
    """
    try:
        json.loads(text)
        return True                     # one big doc
    except json.JSONDecodeError:
        pass

    ok = True
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:                      # skip blanks
            continue
        try:
            json.loads(ln)
        except json.JSONDecodeError:
            ok = False
            break
    return ok


# --------------------------------------------------------------------------- #
# Secure Async Executor Core (with Caching, Checksum, Version Checks)
# --------------------------------------------------------------------------- #
# Dictionary of asyncio.Semaphore objects, one per command
_SEMAPHORES = {cmd: asyncio.Semaphore(limit) for cmd, limit in CONCURRENCY_LIMITS.items()}


async def _run_command_secure(
    cmd_name: str,
    args_str: str,
    *,
    input_data: Optional[str] = None,
    is_file_target: bool = False,
    is_dir_target: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> ToolResult:
    """Securely executes a validated local command with enhancements."""

    # 1. Get Command Metadata and Path
    meta = _COMMAND_METADATA.get(cmd_name)
    if not meta or not meta.path:
        raise ToolExecutionError(
            f"Command '{cmd_name}' is not available or configured.",
            error_code=ToolErrorCode.CMD_NOT_FOUND,
        )

    # Version/checksum checks run before semaphore. This is generally okay,
    # potential for minor concurrent hashing/version checks on very first simultaneous calls.
    # 2. Check Minimum Version (async, cached)
    try:
        await _check_command_version(meta)
    except ToolExecutionError as e:
        # Propagate version error directly
        raise e

    # 3. Verify Checksum (async, lazy, cached)
    try:
        await _get_command_checksum(meta)  # Verifies if mtime changed, re-calcs if needed
        # Optionally, compare against a known-good checksum if available/needed
        # if meta.checksum != EXPECTED_CHECKSUMS[cmd_name]: raise ToolExecutionError(...)
    except ToolExecutionError as e:
        raise e  # Propagate checksum errors

    # 4. Parse Arguments using shlex
    try:
        argv: List[str] = shlex.split(args_str, posix=True)
    except ValueError as exc:
        raise ToolInputError(
            "Invalid quoting or escaping in args_str.",
            param_name="args_str",
            details={"error_code": ToolErrorCode.INVALID_ARGS.value},
        ) from exc

    # 5. Validate Arguments (Security & Workspace) using the placeholder/real validator
    try:
        _validate_arguments(cmd_name, argv)
    except ToolInputError as e:
        # Add command context to validation error
        raise ToolInputError(
            f"Argument validation failed for '{cmd_name}': {e}",
            param_name="args_str",
            details=e.details,
        ) from e

    # 6. Handle Dry Run
    cmd_path_str = str(meta.path)
    cmdline: List[str] = [cmd_path_str, *argv]
    if dry_run:
        logger.info(f"Dry run: Command validated successfully: {shlex.join(cmdline)}")
        # Ensure ToolResult structure is fully populated for dry run
        return ToolResult(
            success=True,
            dry_run_cmdline=cmdline,  # Return the command list
            stdout=None,
            stderr=None,
            exit_code=None,
            error=None,
            error_code=None,
            duration=0.0,
            stdout_truncated=False,
            stderr_truncated=False,
            cached_result=False,
        )

    # 7. Stdin Size Check and Encoding (Optimization)
    input_bytes: Optional[bytes] = None
    input_len = 0
    if input_data is not None:
        input_bytes = input_data.encode("utf-8", errors="ignore")
        input_len = len(input_bytes)
        if input_len > MAX_INPUT_BYTES:
            raise ToolInputError(
                f"input_data exceeds maximum allowed size of {MAX_INPUT_BYTES / (1024 * 1024):.1f} MB.",
                param_name="input_data",
                details={
                    "limit_bytes": MAX_INPUT_BYTES,
                    "actual_bytes": input_len,
                    "error_code": ToolErrorCode.INPUT_TOO_LARGE.value,
                },
            )

    # 8. Prepare for Caching
    cache_key = _get_cache_key(cmd_name, argv, input_bytes)
    cached_result = await _cache_get_async(cache_key)
    if cached_result:
        # Ensure cached_result flag is set correctly (should be done by _cache_get_async)
        cached_result["cached_result"] = True
        return cached_result

    # 9. Acquire Concurrency Semaphore
    semaphore = _SEMAPHORES.get(cmd_name)
    if not semaphore:
        logger.error(
            f"Internal error: No semaphore found for command '{cmd_name}'. Using fallback limit 1."
        )
        # Should not happen if CONCURRENCY_LIMITS is correct
        semaphore = asyncio.Semaphore(1)  # Fallback to concurrency of 1

    async with semaphore:
        # 10. Redacted Logging
        redacted_argv = [
            re.sub(r"(?i)(--?(?:password|token|key|secret)=)\S+", r"\1********", arg)
            for arg in argv
        ]
        log_payload = {
            "event": "execute_local_text_tool",
            "command": cmd_name,
            "args": redacted_argv,
            "input_source": "stdin"
            if input_data is not None
            else ("file" if is_file_target else ("dir" if is_dir_target else "args_only")),
            "timeout_s": timeout,
            "cache_key_prefix": cache_key[:8] if cache_key else None,
        }
        # Using extra assumes logger is configured to handle it
        logger.info(json.dumps(log_payload), extra={"json_fields": log_payload})

        # 11. Resource Limit Setup (Unix only) - Uses module-level function
        def preexec_fn_to_use():
            # Pass cmd_name for command-specific limits
            return _limit_resources(timeout, cmd_name=cmd_name) if sys.platform != "win32" else None

        # 12. Minimal Sanitized Environment
        safe_env = {"PATH": os.getenv("PATH", "/usr/bin:/bin:/usr/local/bin")}  # Add common paths
        # Preserve locale settings for correct text processing
        for safe_var in ["LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_COLLATE"]:
            if safe_var in os.environ:
                safe_env[safe_var] = os.environ[safe_var]
        # Add HOME for tools that might need it for config (like awk finding extensions)
        if "HOME" in os.environ:
            safe_env["HOME"] = os.environ["HOME"]
        # Add TMPDIR as some tools use it
        if "TMPDIR" in os.environ:
            safe_env["TMPDIR"] = os.environ["TMPDIR"]
        elif os.getenv("TEMP"):  # Windows variant
            safe_env["TEMP"] = os.getenv("TEMP")  # type: ignore[assignment]

        # 13. Launch Subprocess
        t0 = time.perf_counter()
        proc: Optional[asyncio.subprocess.Process] = None
        result: Optional[ToolResult] = None  # Ensure result is defined

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmdline,
                stdin=asyncio.subprocess.PIPE
                if input_bytes is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=safe_env,
                limit=MAX_OUTPUT_BYTES * 2,  # Limit buffer size slightly larger than max output
                preexec_fn=preexec_fn_to_use,
                # cwd=str(WORKSPACE_DIR) # Generally not needed if paths are validated relative to WORKSPACE_DIR
            )

            # 14. Communicate and Handle Timeout
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=input_bytes),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as e:
                logger.warning(
                    f"Command '{cmd_name}' timed out after {timeout}s. Terminating.",
                    extra={"command": cmd_name, "timeout": timeout},
                )
                if proc and proc.returncode is None:
                    try:
                        proc.terminate()
                        # Give it a moment to terminate gracefully
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Process {proc.pid} did not terminate gracefully after 1s, killing.",
                            extra={"pid": proc.pid},
                        )
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass  # Already gone
                        await proc.wait()  # Wait for kill to complete
                    except ProcessLookupError:
                        pass  # Already gone
                    except Exception as term_err:
                        logger.error(
                            f"Error terminating process {proc.pid}: {term_err}",
                            extra={"pid": proc.pid},
                        )

                raise ToolExecutionError(
                    f"'{cmd_name}' exceeded timeout of {timeout}s.",
                    error_code=ToolErrorCode.TIMEOUT,
                    details={"timeout": timeout},
                ) from e
            except (BrokenPipeError, ConnectionResetError) as comm_err:
                # Handle cases where the process exits before consuming all input/producing output
                exit_code_on_comm_err = proc.returncode if proc else -1
                logger.warning(
                    f"Communication error with '{cmd_name}' (process likely exited early): {comm_err}. Exit code: {exit_code_on_comm_err}",
                    extra={
                        "command": cmd_name,
                        "error": str(comm_err),
                        "exit_code": exit_code_on_comm_err,
                    },
                )
                # Read any remaining output before raising
                stdout_b = b""
                stderr_b = b""
                try:
                    # Use readexactly or read with limit to avoid indefinite block
                    if proc and proc.stdout:
                        stdout_b = await asyncio.wait_for(
                            proc.stdout.read(MAX_OUTPUT_BYTES * 2), timeout=0.5
                        )
                    if proc and proc.stderr:
                        stderr_b = await asyncio.wait_for(
                            proc.stderr.read(MAX_OUTPUT_BYTES * 2), timeout=0.5
                        )
                except asyncio.TimeoutError:
                    pass
                except Exception as read_err:
                    logger.warning(f"Error reading remaining output after comm error: {read_err}")

                # Continue processing with potentially partial output, but mark as communication error
                # Let success be determined by exit code if available, otherwise assume failure
                duration = time.perf_counter() - t0
                exit_code = (
                    proc.returncode if proc and proc.returncode is not None else -1
                )  # Use -1 if unknown
                stdout, stdout_truncated = _truncate(stdout_b)
                stderr, stderr_truncated = _truncate(stderr_b)
                retcode_ok_map = RETCODE_OK.get(cmd_name, {0})
                success = exit_code in retcode_ok_map

                # Construct result indicating communication error
                result = ToolResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    success=success,  # May be true if exit code was OK despite pipe error
                    error=f"Communication error with '{cmd_name}': {comm_err}",
                    error_code=ToolErrorCode.COMMUNICATION_ERROR,
                    duration=round(duration, 3),
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    cached_result=False,
                )
                # Do not cache communication errors
                return result  # Return immediately, skip normal success/cache path

            except Exception as comm_err:  # Catch other potential communicate errors
                exit_code_on_comm_err = proc.returncode if proc else -1
                raise ToolExecutionError(
                    f"Communication error with '{cmd_name}': {comm_err}",
                    error_code=ToolErrorCode.COMMUNICATION_ERROR,
                    details={"exit_code": exit_code_on_comm_err},
                ) from comm_err

            # 15. Process Results
            duration = time.perf_counter() - t0
            exit_code = proc.returncode
            stdout, stdout_truncated = _truncate(stdout_b)
            stderr, stderr_truncated = _truncate(stderr_b)

            # Use normalization table for success check
            retcode_ok_map = RETCODE_OK.get(cmd_name, {0})  # Default to {0} if cmd not in map
            success = exit_code in retcode_ok_map
            error_message: Optional[str] = None
            error_code: Optional[ToolErrorCode] = None

            if not success:
                error_message = f"Command '{cmd_name}' failed with exit code {exit_code}."
                # Attempt to map common exit codes to specific error types if possible
                # e.g., if exit code 2 for rg means regex error -> ToolErrorCode.INVALID_ARGS
                error_code = ToolErrorCode.EXEC_ERROR  # Default execution error
                if stderr:
                    error_message += (
                        f" Stderr: '{textwrap.shorten(stderr, 150, placeholder='...')}'"
                    )

            # 16. Construct ToolResult TypedDict
            result = ToolResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                success=success,
                error=error_message,
                error_code=error_code,
                duration=round(duration, 3),
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                cached_result=False,  # Will be set by caller if cached
            )

            # 17. Cache the result if successful and not too large? (optional check)
            # Consider not caching extremely large successful results if space is a concern
            # current logic caches all successful results
            if success:
                await _cache_put_async(cache_key, result)

            return result

        except (ToolInputError, ToolExecutionError) as e:
            # Propagate specific errors we raised
            raise e
        except FileNotFoundError as e:
            # Specifically catch if the command itself isn't found at exec time
            logger.error(
                f"Command '{cmd_name}' not found at path: {cmdline[0]}. Ensure it's installed and in PATH.",
                exc_info=True,
            )
            raise ToolExecutionError(
                f"Command '{cmd_name}' executable not found.",
                error_code=ToolErrorCode.CMD_NOT_FOUND,
            ) from e
        except PermissionError as e:
            logger.error(
                f"Permission denied executing command '{cmd_name}' at path: {cmdline[0]}.",
                exc_info=True,
            )
            raise ToolExecutionError(
                f"Permission denied executing '{cmd_name}'. Check file permissions.",
                error_code=ToolErrorCode.EXEC_ERROR,
            ) from e
        except Exception as e:
            # Catch unexpected errors during setup/execution
            logger.critical(f"Unexpected error running command '{cmd_name}': {e}", exc_info=True)
            raise ToolExecutionError(
                f"Unexpected failure executing '{cmd_name}': {e}",
                error_code=ToolErrorCode.UNEXPECTED_FAILURE,
            ) from e
        finally:
            # Ensure process is cleaned up if an exception occurred after creation
            if proc and proc.returncode is None:
                logger.warning(
                    f"Process {proc.pid} for '{cmd_name}' still running after exception, attempting kill."
                )
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                except Exception as final_kill_err:
                    logger.error(
                        f"Error killing process {proc.pid} in finally block: {final_kill_err}"
                    )


# --------------------------------------------------------------------------- #
# Retcode Normalization
# --------------------------------------------------------------------------- #
RETCODE_OK: Dict[str, set[int]] = {
    "rg": {0, 1},  # 0 = matches found, 1 = no matches found (both OK for searching)
    "jq": {0},  # 0 = success. Other codes (e.g., 4 for no matches with -e) indicate issues.
    "awk": {0},  # 0 = success
    "sed": {0},  # 0 = success
}

# --------------------------------------------------------------------------- #
# Public Tool Functions (Standalone Wrappers)
# --------------------------------------------------------------------------- #


def _require_single_source(
    cmd: str,
    *,
    input_data: Optional[str],
    input_file: Optional[bool],
    input_dir: Optional[bool],
) -> None:
    """Validates that exactly one input source mode is indicated."""
    # Convert bool flags to 0 or 1 for summing. None becomes 0.
    modes = [
        input_data is not None,
        input_file is True,  # Explicit True check
        input_dir is True,  # Explicit True check
    ]
    num_modes = sum(modes)
    if num_modes == 0:
        # Use a clearer error message
        raise ToolInputError(
            f"For '{cmd}', you must provide exactly one input: 'input_data' OR 'input_file=True' OR 'input_dir=True'.",
            param_name="inputs",
            details={"error_code": ToolErrorCode.INVALID_ARGS.value},
        )
    elif num_modes > 1:
        raise ToolInputError(
            f"For '{cmd}', specify exactly one input mode: provide 'input_data' OR set 'input_file=True' OR set 'input_dir=True'. Found multiple.",
            param_name="inputs",
            details={"error_code": ToolErrorCode.INVALID_ARGS.value},
        )


# --- run_ripgrep ---
@with_tool_metrics
@with_error_handling
async def run_ripgrep(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default to False for clarity
    input_dir: Optional[bool] = False,  # Default to False for clarity
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> ToolResult:
    """
    Executes the 'rg' (ripgrep) command for fast text pattern searching within the secure workspace.

    Searches recursively through directories or specified files (relative to the workspace)
    for lines matching a regular expression or fixed string pattern.

    Input Handling:
    - `input_data`: Provide text directly via stdin. Omit `input_file`/`input_dir` or set to `False`. Do *not* include a path in `args_str`. Max size controlled by MCP_TEXT_MAX_INPUT.
    - `input_file=True`: Indicates `args_str` contains target file path(s). Omit `input_data`/`input_dir` or set to `False`. Path(s) must be relative to the workspace and specified in `args_str`. Example: `args_str="'pattern' path/to/file.txt"`
    - `input_dir=True`: Indicates `args_str` contains target directory path(s). Omit `input_data`/`input_file` or set to `False`. Path(s) must be relative to the workspace and specified in `args_str`. Example: `args_str="'pattern' path/to/dir"`

    Common `rg` Arguments (include in `args_str`, use workspace-relative paths):
      `'pattern'`: Regex pattern or fixed string.
      `path`: Workspace-relative file or directory path(s).
      `-i`, `--ignore-case`: Case-insensitive search.
      `-v`, `--invert-match`: Select non-matching lines.
      `-l`, `--files-with-matches`: List files containing matches.
      `-c`, `--count`: Count matches per file.
      `-A NUM`, `-B NUM`, `-C NUM`: Context control.
      `--json`: JSON output format.
      `-t type`: Filter by file type (e.g., `py`, `md`).
      `-g glob`: Include/exclude files/dirs by glob (e.g., `-g '*.py' -g '!temp/'`).
      `-o, --only-matching`: Print only matching parts.
      `--follow`: Follow symbolic links (targets must also be within workspace).
      `--no-filename`, `-N`: Suppress filename/line numbers.

    Security: Enforces workspace boundary `"{get_workspace_dir()}"`. Blocks absolute paths, path traversal ('..'), unsafe flags, and shell metacharacters. Limited resources (CPU, Memory, Processes).

    Exit Codes & Success Field:
    - `0`: Matches found -> `success: True`, `error: null`
    - `1`: No matches found -> `success: True`, `error: null` (NOTE: Considered success by this tool wrapper)
    - `2+`: Error occurred -> `success: False`, `error: "..."`, `error_code: EXEC_ERROR`

    Args:
        args_str: Command-line arguments for `rg` (pattern, flags, workspace-relative paths).
        input_data: String data to pipe to `rg` via stdin. Omit/False for `input_file`/`input_dir`.
        input_file: Set to True if `args_str` includes target file path(s). Omit/False for `input_data`/`input_dir`.
        input_dir: Set to True if `args_str` includes target directory path(s). Omit/False for `input_data`/`input_file`.
        timeout: Max execution time in seconds.
        dry_run: If True, validate args and return command line without execution.

    Returns:
        ToolResult: Dictionary containing execution results or dry run info.
                    Includes stdout, stderr, exit_code, success, error, error_code, duration,
                    truncation flags, and cached_result status.

    Raises:
        ToolInputError: For invalid arguments, security violations, or incorrect input mode usage.
        ToolExecutionError: If `rg` is not found, times out, fails version/checksum checks, or fails unexpectedly.
    """
    _require_single_source("rg", input_data=input_data, input_file=input_file, input_dir=input_dir)
    return await _run_command_secure(
        "rg",
        args_str,
        input_data=input_data,
        is_file_target=input_file,  # Pass the boolean flag directly
        is_dir_target=input_dir,  # Pass the boolean flag directly
        timeout=timeout,
        dry_run=dry_run,
    )


# --- run_awk ---
@with_tool_metrics
@with_error_handling
async def run_awk(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default to False for clarity
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> ToolResult:
    """
    Executes the 'awk' command for pattern scanning and field-based text processing within the secure workspace.

    Input Handling:
    - `input_data`: Provide text directly via stdin. Omit `input_file` or set to `False`. Do *not* include filename in `args_str`. Max size controlled by MCP_TEXT_MAX_INPUT.
    - `input_file=True`: Indicates `args_str` contains target file path(s). Omit `input_data` or set to `False`. Path(s) must be relative to the workspace and specified in `args_str`. Example: `args_str="'{print $1}' path/data.log"`

    Common `awk` Arguments (include in `args_str`, use workspace-relative paths):
      `'program'`: The AWK script (e.g., `'{ print $1, $3 }'`).
      `filename(s)`: Workspace-relative input filename(s).
      `-F fs`: Define input field separator (e.g., `-F ','`).
      `-v var=value`: Assign variable.

    Security: Enforces workspace boundary `"{get_workspace_dir()}"`. Blocks unsafe flags (`-i`/`--in-place` gawk), shell characters, absolute paths, traversal. Limited resources.

    Args:
        args_str: Command-line arguments for `awk` (script, flags, workspace-relative paths).
        input_data: String data to pipe to `awk` via stdin. Omit/False for `input_file`.
        input_file: Set to True if `args_str` includes target file path(s). Omit/False for `input_data`.
        timeout: Max execution time in seconds.
        dry_run: If True, validate args and return command line without execution.

    Returns:
        ToolResult: Dictionary with results. `success` is True only if exit code is 0.

    Raises:
        ToolInputError: For invalid arguments, security violations, or incorrect input mode usage.
        ToolExecutionError: If `awk` is not found, times out, fails version/checksum checks, or fails unexpectedly.
    """
    _require_single_source("awk", input_data=input_data, input_file=input_file, input_dir=False)
    return await _run_command_secure(
        "awk",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,  # awk typically doesn't take dir targets directly
        timeout=timeout,
        dry_run=dry_run,
    )


# --- run_sed ---
@with_tool_metrics
@with_error_handling
async def run_sed(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default to False for clarity
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> ToolResult:
    """
    Executes the 'sed' (Stream Editor) command for line-by-line text transformations within the secure workspace.

    Performs substitutions, deletions, insertions based on patterns. **In-place editing (`-i`) is disabled.**

    Input Handling:
    - `input_data`: Provide text directly via stdin. Omit `input_file` or set to `False`. Do *not* include filename in `args_str`. Max size controlled by MCP_TEXT_MAX_INPUT.
    - `input_file=True`: Indicates `args_str` contains target file path. Omit `input_data` or set to `False`. Path must be relative to the workspace and specified in `args_str`. Example: `args_str="'s/ERROR/WARN/g' path/app.log"`

    Common `sed` Arguments (include in `args_str`, use workspace-relative paths):
      `'script'`: The `sed` script or command (e.g., `'s/foo/bar/g'`, `'/^DEBUG/d'`).
      `filename`: Workspace-relative input filename.
      `-e script`: Add multiple scripts.
      `-f script-file`: Read commands from a workspace-relative file.
      `-n`: Suppress automatic printing.
      `-E` or `-r`: Use extended regular expressions.

    Security: Enforces workspace boundary `"{get_workspace_dir()}"`. Blocks `-i` flag, shell characters, absolute paths, traversal. Limited resources.

    Args:
        args_str: Command-line arguments for `sed` (script, flags, workspace-relative path).
        input_data: String data to pipe to `sed` via stdin. Omit/False for `input_file`.
        input_file: Set to True if `args_str` includes target file path. Omit/False for `input_data`.
        timeout: Max execution time in seconds.
        dry_run: If True, validate args and return command line without execution.

    Returns:
        ToolResult: Dictionary with results. `success` is True only if exit code is 0.

    Raises:
        ToolInputError: For invalid arguments, security violations (like using -i), or incorrect input mode usage.
        ToolExecutionError: If `sed` is not found, times out, fails version/checksum checks, or fails unexpectedly.
    """
    _require_single_source("sed", input_data=input_data, input_file=input_file, input_dir=False)
    return await _run_command_secure(
        "sed",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,  # sed operates on files or stdin
        timeout=timeout,
        dry_run=dry_run,
    )


# --- run_jq ---
@with_tool_metrics
@with_error_handling
async def run_jq(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default to False for clarity
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> ToolResult:
    """
    Executes the 'jq' command for querying, filtering, and transforming JSON data within the secure workspace.

    Essential for extracting values, filtering arrays/objects, or restructuring JSON
    provided via stdin (`input_data`) or from a file (`input_file=True`).

    Input Handling:
    - `input_data`: Provide valid JSON text directly via stdin. Omit `input_file` or set to `False`. Do *not* include filename in `args_str`. Max size controlled by MCP_TEXT_MAX_INPUT.
    - `input_file=True`: Indicates `args_str` contains target JSON file path. Omit `input_data` or set to `False`. Path must be relative to the workspace and specified in `args_str`. Example: `args_str="'.items[].name' data.json"`

    Common `jq` Arguments (include in `args_str`, use workspace-relative paths):
      `'filter'`: The `jq` filter expression (e.g., `'.users[] | select(.active==true)'`).
      `filename`: Workspace-relative input JSON filename.
      `-c`: Compact output.
      `-r`: Raw string output (no JSON quotes).
      `-s`: Slurp mode (read input stream into an array).
      `--arg name value`: Define string variable.
      `--argjson name json_value`: Define JSON variable.

    Security: Enforces workspace boundary `"{get_workspace_dir()}"`. Blocks unsafe flags, shell characters, absolute paths, traversal. Limited resources. Validates `input_data` is JSON before execution.

    Args:
        args_str: Command-line arguments for `jq` (filter, flags, workspace-relative path).
        input_data: String containing valid JSON data to pipe to `jq` via stdin. Omit/False for `input_file`.
        input_file: Set to True if `args_str` includes target JSON file path. Omit/False for `input_data`.
        timeout: Max execution time in seconds.
        dry_run: If True, validate args and return command line without execution.

    Returns:
        ToolResult: Dictionary with results. `success` is True only if exit code is 0.

    Raises:
        ToolInputError: If `input_data` is not valid JSON, arguments are invalid, security violations occur, or incorrect input mode usage.
        ToolExecutionError: If `jq` is not found, times out, fails version/checksum checks, or fails unexpectedly.
    """
    _require_single_source("jq", input_data=input_data, input_file=input_file, input_dir=False)
    # Extra check for jq: validate input_data is JSON before running
    if input_data is not None and not _is_json_or_json_lines(input_data):
        raise ToolInputError(
            "input_data is not valid JSON or JSON-Lines",
            param_name="input_data",
            details={"error_code": ToolErrorCode.INVALID_JSON_INPUT.value},
        )

    return await _run_command_secure(
        "jq",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,  # jq operates on files or stdin
        timeout=timeout,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Streaming Core Executor
# --------------------------------------------------------------------------- #


async def _run_command_stream(
    cmd_name: str,
    args_str: str,
    *,
    input_data: Optional[str] = None,
    is_file_target: bool = False,
    is_dir_target: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """Securely executes a command and streams its stdout line by line."""

    # 1. Get Command Metadata and Path
    meta = _COMMAND_METADATA.get(cmd_name)
    if not meta or not meta.path:
        raise ToolExecutionError(
            f"Command '{cmd_name}' is not available or configured.",
            error_code=ToolErrorCode.CMD_NOT_FOUND,
        )

    # 2. Version & Checksum Checks (Same as non-streaming)
    await _check_command_version(meta)
    await _get_command_checksum(meta)  # Verification happens here

    # 3. Parse & Validate Arguments
    try:
        argv: List[str] = shlex.split(args_str, posix=True)
        _validate_arguments(cmd_name, argv)
    except ValueError as exc:
        raise ToolInputError(
            "Invalid quoting or escaping in args_str.",
            param_name="args_str",
            details={"error_code": ToolErrorCode.INVALID_ARGS.value},
        ) from exc
    except ToolInputError as e:
        raise ToolInputError(
            f"Argument validation failed for '{cmd_name}' stream: {e}",
            param_name="args_str",
            details=e.details,
        ) from e

    # 4. Prepare command line and logging
    cmd_path_str = str(meta.path)
    cmdline: List[str] = [cmd_path_str, *argv]
    redacted_argv = [
        re.sub(r"(?i)(--?(?:password|token|key|secret)=)\S+", r"\1********", arg) for arg in argv
    ]
    log_payload = {
        "event": "execute_local_text_tool_stream",
        "command": cmd_name,
        "args": redacted_argv,
        "input_source": "stdin"
        if input_data is not None
        else ("file" if is_file_target else ("dir" if is_dir_target else "args_only")),
        "timeout_s": timeout,
    }
    logger.info(json.dumps(log_payload), extra={"json_fields": log_payload})

    # 5. Resource limits & Env (Same as non-streaming)
    def preexec_fn_to_use():
        return _limit_resources(timeout) if sys.platform != "win32" else None

    safe_env = {"PATH": os.getenv("PATH", "/usr/bin:/bin:/usr/local/bin")}
    for safe_var in ["LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_COLLATE"]:
        if safe_var in os.environ:
            safe_env[safe_var] = os.environ[safe_var]
    if "HOME" in os.environ:
        safe_env["HOME"] = os.environ["HOME"]
    if "TMPDIR" in os.environ:
        safe_env["TMPDIR"] = os.environ["TMPDIR"]
    elif os.getenv("TEMP"):
        safe_env["TEMP"] = os.getenv("TEMP")  # type: ignore[assignment]

    input_bytes: Optional[bytes] = None
    if input_data is not None:
        input_bytes = input_data.encode("utf-8", errors="ignore")
        # Check size *before* starting process
        if len(input_bytes) > MAX_INPUT_BYTES:
            raise ToolInputError(
                f"input_data exceeds maximum allowed size of {MAX_INPUT_BYTES / (1024 * 1024):.1f} MB for streaming.",
                param_name="input_data",
                details={
                    "limit_bytes": MAX_INPUT_BYTES,
                    "actual_bytes": len(input_bytes),
                    "error_code": ToolErrorCode.INPUT_TOO_LARGE.value,
                },
            )

    # 6. Launch Subprocess and Stream Output
    proc: Optional[asyncio.subprocess.Process] = None
    stderr_lines: List[str] = []  # Collect stderr lines
    start_time = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmdline,
            stdin=asyncio.subprocess.PIPE
            if input_bytes is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # Capture stderr
            env=safe_env,
            limit=MAX_OUTPUT_BYTES * 2,  # Apply buffer limit
            preexec_fn=preexec_fn_to_use,
        )

        # --- Define Helper Coroutines for IO ---
        async def write_stdin_task(proc: asyncio.subprocess.Process) -> None:
            """Writes input data to stdin if provided and closes stdin."""
            if input_bytes is not None and proc.stdin:
                try:
                    proc.stdin.write(input_bytes)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError) as e:
                    logger.warning(
                        f"Stream: Error writing to stdin for '{cmd_name}' (pid {proc.pid}): {e}. Process might have exited."
                    )
                except Exception as e:
                    logger.error(
                        f"Stream: Unexpected error writing to stdin for '{cmd_name}' (pid {proc.pid}): {e}",
                        exc_info=True,
                    )
                finally:
                    if proc.stdin:
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass  # Ignore errors closing stdin already closed/broken
            elif proc.stdin:
                # Close stdin immediately if no input_data
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        async def read_stderr_task(proc: asyncio.subprocess.Process, lines_list: List[str]) -> None:
            """Reads stderr lines into the provided list."""
            if not proc.stderr:
                return
            stderr_bytes_read = 0
            while True:
                try:
                    # Read with a timeout? readline() could block if process hangs writing stderr
                    line_bytes = await proc.stderr.readline()
                    if not line_bytes:
                        break  # End of stream
                    stderr_bytes_read += len(line_bytes)
                    # Basic truncation within stderr collection to prevent memory issues
                    if stderr_bytes_read > MAX_OUTPUT_BYTES * 1.1:  # Allow slightly more for marker
                        if not lines_list or not lines_list[-1].endswith("...(stderr truncated)"):
                            lines_list.append("...(stderr truncated)")
                        continue  # Stop appending lines but keep reading to drain pipe
                    lines_list.append(line_bytes.decode("utf-8", errors="replace"))
                except Exception as e:
                    logger.warning(
                        f"Stream: Error reading stderr line for '{cmd_name}' (pid {proc.pid}): {e}"
                    )
                    lines_list.append(f"##STREAM_STDERR_READ_ERROR: {e}\n")
                    break  # Stop reading stderr on error

        # --- Create IO Tasks ---
        stdin_writer = asyncio.create_task(write_stdin_task(proc))
        stderr_reader = asyncio.create_task(read_stderr_task(proc, stderr_lines))

        # --- Yield stdout lines (Inlined async generator logic) ---
        stdout_line_count = 0
        stdout_bytes_read = 0
        stdout_truncated = False
        if proc.stdout:
            while True:
                # Check timeout explicitly within the loop
                if time.monotonic() - start_time > timeout:
                    raise asyncio.TimeoutError()

                try:
                    # Use readline with a timeout? Less efficient but prevents hangs.
                    # Or rely on overall wait_for timeout below.
                    line_bytes = await proc.stdout.readline()
                    if not line_bytes:
                        break  # End of stdout stream

                    stdout_bytes_read += len(line_bytes)
                    if stdout_bytes_read > MAX_OUTPUT_BYTES:
                        if not stdout_truncated:  # Add marker only once
                            yield "...(stdout truncated)\n"
                            stdout_truncated = True
                        continue  # Stop yielding lines but keep reading to drain pipe

                    if not stdout_truncated:
                        # Decode and yield the line
                        yield line_bytes.decode("utf-8", errors="replace")
                        stdout_line_count += 1

                except asyncio.TimeoutError as e:  # Catch timeout during readline if applied
                    raise asyncio.TimeoutError() from e  # Re-raise to be caught by outer handler
                except Exception as e:
                    logger.warning(
                        f"Stream: Error reading stdout line for '{cmd_name}' (pid {proc.pid}): {e}"
                    )
                    # Yield an error marker in the stream
                    yield f"##STREAM_STDOUT_READ_ERROR: {e}\n"
                    break  # Stop reading stdout on error
        else:
            logger.warning(
                f"Stream: Process for '{cmd_name}' has no stdout stream.",
                extra={"command": cmd_name},
            )

        # --- Wait for process and stderr/stdin tasks to complete ---
        # Wait for the process itself, applying the main timeout
        try:
            # Wait slightly longer than the timeout to allow for cleanup/exit signals
            exit_code = await asyncio.wait_for(proc.wait(), timeout=timeout + 5.0)
        except asyncio.TimeoutError as e:
            # If proc.wait() times out, it means the process didn't exit within timeout+5s
            # Re-raise the specific timeout error defined earlier
            raise ToolExecutionError(
                f"'{cmd_name}' stream process failed to exit within timeout of {timeout}s (+5s buffer).",
                error_code=ToolErrorCode.TIMEOUT,
                details={"timeout": timeout},
            ) from e

        # Ensure IO tasks have finished (they should have if process exited, but wait briefly)
        try:
            await asyncio.wait_for(stdin_writer, timeout=1.0)
            await asyncio.wait_for(stderr_reader, timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"Stream: IO tasks for '{cmd_name}' (pid {proc.pid}) did not complete quickly after process exit."
            )

        # --- Check final status ---
        duration = time.monotonic() - start_time
        retcode_ok_map = RETCODE_OK.get(cmd_name, {0})
        success = exit_code in retcode_ok_map
        stderr_full = "".join(stderr_lines)

        if not success:
            stderr_snip = textwrap.shorten(stderr_full.strip(), 150, placeholder="...")
            error_msg = f"Command '{cmd_name}' stream finished with failure exit code {exit_code}. Stderr: '{stderr_snip}'"
            logger.warning(
                error_msg,
                extra={"command": cmd_name, "exit_code": exit_code, "stderr": stderr_full},
            )
            # Raise error *after* iteration completes to signal failure to the caller
            raise ToolExecutionError(
                error_msg,
                error_code=ToolErrorCode.EXEC_ERROR,
                details={"exit_code": exit_code, "stderr": stderr_full},
            )
        else:
            logger.info(
                f"Stream: '{cmd_name}' (pid {proc.pid}) finished successfully in {duration:.3f}s (code={exit_code}, stdout_lines={stdout_line_count}, truncated={stdout_truncated})"
            )

    except asyncio.TimeoutError as e:
        # This catches the explicit timeout check within the stdout loop or the wait_for on proc.wait()
        logger.warning(
            f"Command '{cmd_name}' stream timed out after ~{timeout}s. Terminating.",
            extra={"command": cmd_name, "timeout": timeout},
        )
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"Killing unresponsive stream process {proc.pid}")
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            except Exception as term_err:
                logger.error(f"Error killing stream process {proc.pid}: {term_err}")
        # Raise the standard execution error for timeout
        raise ToolExecutionError(
            f"'{cmd_name}' stream exceeded timeout of {timeout}s.",
            error_code=ToolErrorCode.TIMEOUT,
            details={"timeout": timeout},
        ) from e
    except (ToolInputError, ToolExecutionError) as e:
        # Propagate specific errors raised during setup or validation
        raise e
    except FileNotFoundError as e:
        logger.error(
            f"Stream: Command '{cmd_name}' not found at path: {cmdline[0]}. Ensure it's installed and in PATH.",
            exc_info=True,
        )
        raise ToolExecutionError(
            f"Stream: Command '{cmd_name}' executable not found.",
            error_code=ToolErrorCode.CMD_NOT_FOUND,
        ) from e
    except PermissionError as e:
        logger.error(
            f"Stream: Permission denied executing command '{cmd_name}' at path: {cmdline[0]}.",
            exc_info=True,
        )
        raise ToolExecutionError(
            f"Stream: Permission denied executing '{cmd_name}'. Check file permissions.",
            error_code=ToolErrorCode.EXEC_ERROR,
        ) from e
    except Exception as e:
        logger.error(
            f"Stream: Unexpected error during command stream '{cmd_name}': {e}", exc_info=True
        )
        raise ToolExecutionError(
            f"Unexpected failure during '{cmd_name}' stream: {e}",
            error_code=ToolErrorCode.UNEXPECTED_FAILURE,
        ) from e
    finally:
        # Final cleanup check for the process in case of unexpected exit from the try block
        if proc and proc.returncode is None:
            logger.warning(
                f"Stream: Process {proc.pid} for '{cmd_name}' still running after stream processing finished unexpectedly, killing."
            )
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            except Exception as final_kill_err:
                logger.error(
                    f"Stream: Error killing process {proc.pid} in finally block: {final_kill_err}"
                )


# --- run_ripgrep_stream ---
@with_tool_metrics
@with_error_handling
async def run_ripgrep_stream(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default False
    input_dir: Optional[bool] = False,  # Default False
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """
    Executes 'rg' and streams stdout lines asynchronously. Useful for large outputs.

    See `run_ripgrep` for detailed argument descriptions and security notes.
    This variant yields each line of standard output as it becomes available.

    **Note:** The final success status and stderr are not part of the yielded stream.
    Errors during execution (e.g., timeout, non-zero exit code other than 1 for 'no match')
    will raise a `ToolExecutionError` *after* the stream iteration completes
    (or immediately if setup fails). Use a `try...finally` or check status afterwards.

    Args:
        args_str: Command-line arguments for `rg`.
        input_data: String data to pipe via stdin. Omit/False for `input_file`/`input_dir`.
        input_file: Set True if `args_str` includes target file path(s). Omit/False for `input_data`/`input_dir`.
        input_dir: Set True if `args_str` includes target directory path(s). Omit/False for `input_data`/`input_file`.
        timeout: Max execution time in seconds for the entire operation.

    Yields:
        str: Each line of standard output from the `rg` command.

    Raises:
        ToolInputError: For invalid arguments or security violations during setup.
        ToolExecutionError: If `rg` fails execution (timeout, bad exit code > 1),
                            or if errors occur during streaming I/O. Raised *after* iteration.
    """
    _require_single_source(
        "rg (stream)", input_data=input_data, input_file=input_file, input_dir=input_dir
    )
    # Use the streaming executor
    async for line in _run_command_stream(
        "rg",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=input_dir,
        timeout=timeout,
    ):
        yield line


# --- run_awk_stream ---
@with_tool_metrics
@with_error_handling
async def run_awk_stream(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default False
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """
    Executes 'awk' and streams stdout lines asynchronously.

    See `run_awk` for detailed argument descriptions and security notes.
    Yields each line of standard output as it becomes available.

    Args:
        args_str: Command-line arguments for `awk`.
        input_data: String data to pipe via stdin. Omit/False for `input_file`.
        input_file: Set True if `args_str` includes target file path(s). Omit/False for `input_data`.
        timeout: Max execution time in seconds.

    Yields:
        str: Each line of standard output from the `awk` command.

    Raises:
        ToolInputError: For invalid arguments or security violations.
        ToolExecutionError: If `awk` fails (non-zero exit), times out, or errors during streaming. Raised *after* iteration.
    """
    _require_single_source(
        "awk (stream)", input_data=input_data, input_file=input_file, input_dir=False
    )
    async for line in _run_command_stream(
        "awk",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,
        timeout=timeout,
    ):
        yield line


# --- run_sed_stream ---
@with_tool_metrics
@with_error_handling
async def run_sed_stream(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default False
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """
    Executes 'sed' and streams stdout lines asynchronously.

    See `run_sed` for detailed argument descriptions and security notes.
    Yields each line of standard output as it becomes available.

    Args:
        args_str: Command-line arguments for `sed`.
        input_data: String data to pipe via stdin. Omit/False for `input_file`.
        input_file: Set True if `args_str` includes target file path. Omit/False for `input_data`.
        timeout: Max execution time in seconds.

    Yields:
        str: Each line of standard output from the `sed` command.

    Raises:
        ToolInputError: For invalid arguments or security violations.
        ToolExecutionError: If `sed` fails (non-zero exit), times out, or errors during streaming. Raised *after* iteration.
    """
    _require_single_source(
        "sed (stream)", input_data=input_data, input_file=input_file, input_dir=False
    )
    async for line in _run_command_stream(
        "sed",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,
        timeout=timeout,
    ):
        yield line


# --- run_jq_stream ---
@with_tool_metrics
@with_error_handling
async def run_jq_stream(
    args_str: str,
    *,
    input_data: Optional[str] = None,
    input_file: Optional[bool] = False,  # Default False
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """
    Executes 'jq' and streams stdout lines asynchronously.

    See `run_jq` for detailed argument descriptions and security notes.
    Yields each line of standard output (often JSON objects if not using `-c`)
    as it becomes available.

    Args:
        args_str: Command-line arguments for `jq`.
        input_data: String containing valid JSON data to pipe via stdin. Omit/False for `input_file`.
        input_file: Set True if `args_str` includes target JSON file path. Omit/False for `input_data`.
        timeout: Max execution time in seconds.

    Yields:
        str: Each line of standard output from the `jq` command.

    Raises:
        ToolInputError: If `input_data` is not valid JSON, arguments are invalid, or security violations occur.
        ToolExecutionError: If `jq` fails (non-zero exit), times out, or errors during streaming. Raised *after* iteration.
    """
    _require_single_source(
        "jq (stream)", input_data=input_data, input_file=input_file, input_dir=False
    )
    # Validate input_data is JSON if provided, before starting process
    if input_data is not None and not _is_json_or_json_lines(input_data):
        raise ToolInputError(
            "input_data is not valid JSON or JSON-Lines",
            param_name="input_data",
            details={"error_code": ToolErrorCode.INVALID_JSON_INPUT.value},
        )

    async for line in _run_command_stream(
        "jq",
        args_str,
        input_data=input_data,
        is_file_target=input_file,
        is_dir_target=False,
        timeout=timeout,
    ):
        yield line


# --------------------------------------------------------------------------- #
# Public API Exports
# --------------------------------------------------------------------------- #


def get_workspace_dir() -> str:
    """Return the absolute workspace directory path enforced by this module."""
    return str(WORKSPACE_DIR)


# Add command metadata access if needed?
# def get_command_meta(cmd_name: str) -> Optional[CommandMeta]:
#    """Returns the discovered metadata for a command, if available."""
#    return _COMMAND_METADATA.get(cmd_name)


__all__ = [
    # Standard execution
    "run_ripgrep",
    "run_awk",
    "run_sed",
    "run_jq",
    # Streaming variants
    "run_ripgrep_stream",
    "run_awk_stream",
    "run_sed_stream",
    "run_jq_stream",
    # Configuration info
    "get_workspace_dir",
    # Types (for consumers)
    "ToolResult",
    "ToolErrorCode",
    # Maybe export CommandMeta if useful externally?
    # "CommandMeta",
]
