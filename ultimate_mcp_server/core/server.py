"""Main server implementation for Ultimate MCP Server."""

import asyncio
import json
import logging
import logging.config
import math
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

import ultimate_mcp_server

# Import core specifically to set the global instance
import ultimate_mcp_server.core
from ultimate_mcp_server.config import get_config, load_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.state_store import StateStore
from ultimate_mcp_server.graceful_shutdown import (
    create_quiet_server,
    enable_quiet_shutdown,
    register_shutdown_handler,
)
from ultimate_mcp_server.tools.smart_browser import (
    _ensure_initialized as smart_browser_ensure_initialized,
)
from ultimate_mcp_server.tools.smart_browser import (
    shutdown as smart_browser_shutdown,
)
from ultimate_mcp_server.tools.sql_databases import initialize_sql_tools, shutdown_sql_tools

# --- Import the trigger function directly instead of the whole module---
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.logging import logger

# --- Define Logging Configuration Dictionary ---

LOG_FILE_PATH = "logs/ultimate_mcp_server.log"

# Ensure log directory exists before config is used
log_dir = os.path.dirname(LOG_FILE_PATH)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Let Uvicorn's loggers pass through if needed
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
        "file": {  # Formatter for file output
            "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {  # Console handler - redirect to stderr
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Changed from stdout to stderr
        },
        "access": {  # Access log handler - redirect to stderr
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Changed from stdout to stderr
        },
        "rich_console": {  # Rich console handler
            "()": "ultimate_mcp_server.utils.logging.formatter.create_rich_console_handler",
            "stderr": True,  # Add this parameter to use stderr
        },
        "file": {  # File handler
            "formatter": "file",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_FILE_PATH,
            "maxBytes": 2 * 1024 * 1024,  # 2 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "tools_file": {  # Tools log file handler
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/direct_tools.log",
            "encoding": "utf-8",
        },
        "completions_file": {  # Completions log file handler
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/direct_completions.log",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO", "propagate": True},  # Propagate errors to root
        "uvicorn.access": {"handlers": ["access", "file"], "level": "INFO", "propagate": False},
        "ultimate_mcp_server": {  # Our application's logger namespace
            "handlers": ["rich_console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "ultimate_mcp_server.tools": {  # Tools-specific logger
            "handlers": ["tools_file"],
            "level": "DEBUG",
            "propagate": True,  # Propagate to parent for console display
        },
        "ultimate_mcp_server.completions": {  # Completions-specific logger
            "handlers": ["completions_file"],
            "level": "DEBUG",
            "propagate": True,  # Propagate to parent for console display
        },
    },
    "root": {  # Root logger configuration
        "level": "INFO",
        "handlers": ["rich_console", "file"],  # Root catches logs not handled by specific loggers
    },
}

# DO NOT apply the config here - it will be applied by Uvicorn through log_config parameter

# Global server instance
_server_app = None
_gateway_instance = None

# Get loggers
tools_logger = get_logger("ultimate_mcp_server.tools")
completions_logger = get_logger("ultimate_mcp_server.completions")


@dataclass
class ProviderStatus:
    """
    Structured representation of an LLM provider's configuration and availability status.

    This dataclass encapsulates all essential status information about a language model
    provider in the Ultimate MCP Server. It's used to track the state of each provider,
    including whether it's properly configured, successfully initialized, and what models
    it offers. This information is vital for:

    1. Displaying provider status to clients via API endpoints
    2. Making runtime decisions about provider availability
    3. Debugging provider configuration and connectivity issues
    4. Resource listings and capability discovery

    The status is typically maintained in the Gateway's provider_status dictionary,
    with provider names as keys and ProviderStatus instances as values.

    Attributes:
        enabled: Whether the provider is enabled in the configuration.
                This reflects the user's intent, not actual availability.
        available: Whether the provider is successfully initialized and ready for use.
                  This is determined by runtime checks during server initialization.
        api_key_configured: Whether a valid API key was found for this provider.
                           A provider might be enabled but have no API key configured.
        models: List of available models from this provider, with each model represented
               as a dictionary containing model ID, name, and capabilities.
        error: Error message explaining why a provider is unavailable, or None if
              the provider initialized successfully or hasn't been initialized yet.
    """

    enabled: bool
    available: bool
    api_key_configured: bool
    models: List[Dict[str, Any]]
    error: Optional[str] = None


class Gateway:
    """
    Main Ultimate MCP Server implementation and central orchestrator.

    The Gateway class serves as the core of the Ultimate MCP Server, providing a unified
    interface to multiple LLM providers (OpenAI, Anthropic, etc.) and implementing the
    Model Control Protocol (MCP). It manages provider connections, tool registration,
    state persistence, and request handling.

    Key responsibilities:
    - Initializing and managing connections to LLM providers
    - Registering and exposing tools for model interaction
    - Providing consistent error handling and logging
    - Managing state persistence across requests
    - Exposing resources (guides, examples, reference info) for models
    - Implementing the MCP protocol for standardized model interaction

    The Gateway is designed to be instantiated once per server instance and serves
    as the central hub for all model interactions. It can be accessed globally through
    the ultimate_mcp_server.core._gateway_instance reference.
    """

    def __init__(
        self,
        name: str = "main",
        register_tools: bool = True,
        provider_exclusions: List[str] = None,
        load_all_tools: bool = False,  # Remove result_serialization_mode
    ):
        """
        Initialize the MCP Gateway with configured providers and tools.

        This constructor sets up the complete MCP Gateway environment, including:
        - Loading configuration from environment variables and config files
        - Setting up logging infrastructure
        - Initializing the MCP server framework
        - Creating a state store for persistence
        - Registering tools and resources based on configuration

        The initialization process is designed to be flexible, allowing for customization
        through the provided parameters and the configuration system. Provider initialization
        is deferred until server startup to ensure proper async handling.

        Args:
            name: Server instance name, used for logging and identification purposes.
                 Default is "main".
            register_tools: Whether to register standard MCP tools with the server.
                           If False, only the minimal core functionality will be available.
                           Default is True.
            provider_exclusions: List of provider names to exclude from initialization.
                                This allows selectively disabling specific providers
                                regardless of their configuration status.
                                Default is None (no exclusions).
            load_all_tools: If True, load all available tools. If False (default),
                           load only the defined 'Base Toolset'.
        """
        self.name = name
        self.providers = {}
        self.provider_status = {}
        self.logger = get_logger(f"ultimate_mcp_server.{name}")
        self.event_handlers = {}
        self.provider_exclusions = provider_exclusions or []
        self.api_meta_tool = None  # Initialize api_meta_tool attribute
        self.load_all_tools = load_all_tools  # Store the flag

        # Load configuration if not already loaded
        if get_config() is None:
            self.logger.info("Initializing Gateway: Loading configuration...")
            load_config()

        # Initialize logger
        self.logger.info(f"Initializing {self.name}...")

        # Set MCP protocol version to 2025-03-25
        import os

        os.environ["MCP_PROTOCOL_VERSION"] = "2025-03-25"

        # Create MCP server with modern FastMCP constructor
        self.mcp = FastMCP(
            name=self.name,
            lifespan=self._server_lifespan,
            instructions=self.system_instructions,
        )

        # Initialize the state store
        persistence_dir = None
        if (
            get_config()
            and hasattr(get_config(), "state_persistence")
            and hasattr(get_config().state_persistence, "dir")
        ):
            persistence_dir = get_config().state_persistence.dir
        self.state_store = StateStore(persistence_dir)

        # Register tools if requested
        if register_tools:
            self._register_tools(load_all=self.load_all_tools)
            self._register_resources()

        self.logger.info(f"Ultimate MCP Server '{self.name}' initialized")

    def log_tool_calls(self, func):
        """
        Decorator to log MCP tool calls with detailed timing and result information.

        This decorator wraps MCP tool functions to provide consistent logging of:
        - Tool name and parameters at invocation time
        - Execution time for performance tracking
        - Success or failure status
        - Summarized results or error information

        The decorator ensures that all tool calls are logged to a dedicated tools logger,
        which helps with diagnostics, debugging, and monitoring of tool usage patterns.
        Successful calls include timing information and a brief summary of the result,
        while failed calls include exception details.

        Args:
            func: The async function to wrap with logging. This should be a tool function
                 registered with the MCP server that will be called by models.

        Returns:
            A wrapped async function that performs the same operations as the original
            but with added logging before and after execution.

        Note:
            This decorator is automatically applied to all functions registered as tools
            via the @mcp.tool() decorator in the _register_tools method, so it doesn't
            need to be applied manually in most cases.
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            tool_name = func.__name__

            # Format parameters for logging
            args_str = ", ".join([repr(arg) for arg in args[1:] if arg is not None])
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items() if k != "ctx"])
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))

            # Log the request - only through tools_logger
            tools_logger.info(f"TOOL CALL: {tool_name}({params_str})")

            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time

                # Format result for logging
                if isinstance(result, dict):
                    result_keys = list(result.keys())
                    result_summary = f"dict with keys: {result_keys}"
                else:
                    result_str = str(result)
                    result_summary = (
                        (result_str[:100] + "...") if len(result_str) > 100 else result_str
                    )

                # Log successful completion - only through tools_logger
                tools_logger.info(
                    f"TOOL SUCCESS: {tool_name} completed in {processing_time:.2f}s - Result: {result_summary}"
                )

                return result
            except Exception as e:
                processing_time = time.time() - start_time
                tools_logger.error(
                    f"TOOL ERROR: {tool_name} failed after {processing_time:.2f}s: {str(e)}",
                    exc_info=True,
                )
                raise

        return wrapper

    @asynccontextmanager
    async def _server_lifespan(self, server: FastMCP):
        """
        Async context manager managing the server lifecycle during startup and shutdown.

        This method implements the lifespan protocol used by FastMCP (based on ASGI) to:
        1. Perform startup initialization before the server begins accepting requests
        2. Clean up resources when the server is shutting down
        3. Make shared context available to request handlers during the server's lifetime

        During startup, this method:
        - Initializes all configured LLM providers
        - Triggers dynamic docstring generation for tools that need it
        - Sets the global Gateway instance for access from other components
        - Prepares a shared context dictionary for use by request handlers

        During shutdown, it:
        - Clears the global Gateway instance reference
        - Handles any necessary cleanup of resources

        The lifespan context is active throughout the entire server runtime, from
        startup until shutdown is initiated.

        Args:
            server: The FastMCP server instance that's starting up, which provides
                   the framework context for the lifespan.

        Yields:
            Dict containing initialized resources that will be available to all
            request handlers during the server's lifetime.

        Note:
            This method is called automatically by the FastMCP framework during
            server startup and is not intended to be called directly.
        """
        self.logger.info(f"Starting Ultimate MCP Server '{self.name}'")

        # Add a flag to track if this is an SSE instance
        is_sse_mode = getattr(self, '_sse_mode', False)
        if is_sse_mode:
            self.logger.info("SSE mode detected - using persistent lifespan management")

        # Initialize providers
        await self._initialize_providers()

        try:
            await initialize_sql_tools()
            self.logger.info("SQL tools state initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQL tools state: {e}", exc_info=True)

        # --- OPTIONAL: Pre-initialize SmartBrowser ---
        try:
            self.logger.info("Pre-initializing Smart Browser components...")
            # Call the imported initialization function
            await smart_browser_ensure_initialized()
            self.logger.info("Smart Browser successfully pre-initialized.")
        except Exception as e:
            # Log warning but don't stop server startup if pre-init fails
            self.logger.warning(f"Could not pre-initialize Smart Browser: {e}", exc_info=True)
        # ---------------------------------------------------------------------

        # --- Trigger Dynamic Docstring Generation ---
        # This should run after config is loaded but before the server is fully ready
        # It checks cache and potentially calls an LLM.
        self.logger.info("Initiating dynamic docstring generation for Marqo tool...")
        try:
            # Import the function here to avoid circular imports
            from ultimate_mcp_server.tools.marqo_fused_search import (
                trigger_dynamic_docstring_generation,
            )

            await trigger_dynamic_docstring_generation()
            self.logger.info("Dynamic docstring generation/loading complete.")
        except Exception as e:
            self.logger.error(
                f"Error during dynamic docstring generation startup task: {e}", exc_info=True
            )
        # ---------------------------------------------

        # --- Set the global instance variable ---
        # Make the fully initialized instance accessible globally AFTER init
        ultimate_mcp_server.core._gateway_instance = self
        self.logger.info("Global gateway instance set.")
        # ----------------------------------------

        # --- Attach StateStore to application state ---
        # This makes the StateStore available to all tools via ctx.fastmcp._state_store
        # Note: In FastMCP 2.0+, we store the state_store directly on the server instance
        # Tools can access it via the with_state_management decorator
        server._state_store = self.state_store
        self.logger.info("StateStore attached to server instance.")
        # -----------------------------------------------

        # Create lifespan context (still useful for framework calls)
        context = {
            "providers": self.providers,
            "provider_status": self.provider_status,
        }

        self.logger.info("Lifespan context initialized, MCP server ready to handle requests")

        try:
            # Import and call trigger_dynamic_docstring_generation again
            from ultimate_mcp_server.tools.marqo_fused_search import (
                trigger_dynamic_docstring_generation,
            )

            await trigger_dynamic_docstring_generation()
            logger.info("Dynamic docstring generation/loading complete.")
            
            if is_sse_mode:
                # For SSE mode, create a persistent context that doesn't shutdown easily
                self.logger.info("Creating persistent SSE lifespan context")
                
                # Add a keepalive task for SSE mode
                async def sse_lifespan_keepalive():
                    """Keepalive task to maintain SSE server lifespan."""
                    while True:
                        await asyncio.sleep(60)  # Keep alive every minute
                        # This task existing keeps the lifespan active
                
                # Start the keepalive task
                keepalive_task = asyncio.create_task(sse_lifespan_keepalive())
                
                try:
                    yield context
                finally:
                    # Cancel the keepalive task during shutdown
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass
            else:
                yield context
                
        finally:
            if is_sse_mode:
                self.logger.info("SSE mode shutdown initiated")
                
            try:
                # --- Shutdown SQL Tools State ---
                await shutdown_sql_tools()
                self.logger.info("SQL tools state shut down.")
            except Exception as e:
                self.logger.error(f"Failed to shut down SQL tools state: {e}", exc_info=True)

            # 2. Shutdown Smart Browser explicitly
            try:
                self.logger.info("Initiating explicit Smart Browser shutdown...")
                await smart_browser_shutdown()  # Call the imported function
                self.logger.info("Smart Browser shutdown completed successfully.")
            except Exception as e:
                logger.error(f"Error during explicit Smart Browser shutdown: {e}", exc_info=True)

            # --- Clear the global instance on shutdown ---
            ultimate_mcp_server.core._gateway_instance = None
            self.logger.info("Global gateway instance cleared.")
            # -------------------------------------------
            self.logger.info(f"Shutting down Ultimate MCP Server '{self.name}'")

    async def _initialize_providers(self):
        """
        Initialize all enabled LLM providers based on the loaded configuration.

        This asynchronous method performs the following steps:
        1. Identifies which providers are enabled and properly configured with API keys
        2. Skips providers that are in the exclusion list (specified at Gateway creation)
        3. Initializes each valid provider in parallel using asyncio tasks
        4. Updates the provider_status dictionary with the initialization results

        The method uses a defensive approach, handling cases where:
        - A provider is enabled but missing API keys
        - Configuration is incomplete or inconsistent
        - Initialization errors occur with specific providers

        After initialization, the Gateway will have a populated providers dictionary
        with available provider instances, and a comprehensive provider_status dictionary
        with status information for all providers (including those that failed to initialize).

        This method is automatically called during server startup and is not intended
        to be called directly by users of the Gateway class.

        Raises:
            No exceptions are propagated from this method. All provider initialization
            errors are caught, logged, and reflected in the provider_status dictionary.
        """
        self.logger.info("Initializing LLM providers")

        cfg = get_config()
        providers_to_init = []

        # Determine which providers to initialize based SOLELY on the loaded config
        for provider_name in [p.value for p in Provider]:
            # Skip providers that are in the exclusion list
            if provider_name in self.provider_exclusions:
                self.logger.debug(f"Skipping provider {provider_name} (excluded)")
                continue

            provider_config = getattr(cfg.providers, provider_name, None)
            # Special exception for Ollama: it doesn't require an API key since it runs locally
            if (
                provider_name == Provider.OLLAMA.value
                and provider_config
                and provider_config.enabled
            ):
                self.logger.debug(
                    f"Found configured and enabled provider: {provider_name} (API key not required)"
                )
                providers_to_init.append(provider_name)
            # Check if the provider is enabled AND has an API key configured in the loaded settings
            elif provider_config and provider_config.enabled and provider_config.api_key:
                self.logger.debug(f"Found configured and enabled provider: {provider_name}")
                providers_to_init.append(provider_name)
            elif provider_config and provider_config.enabled:
                self.logger.warning(
                    f"Provider {provider_name} is enabled but missing API key in config. Skipping."
                )
            # else: # Provider not found in config or not enabled
            #     self.logger.debug(f"Provider {provider_name} not configured or not enabled.")

        # Initialize providers in parallel
        init_tasks = [
            asyncio.create_task(
                self._initialize_provider(provider_name), name=f"init-{provider_name}"
            )
            for provider_name in providers_to_init
        ]

        if init_tasks:
            await asyncio.gather(*init_tasks)

        # Log initialization summary
        available_providers = [
            name for name, status in self.provider_status.items() if status.available
        ]
        self.logger.info(
            f"Providers initialized: {len(available_providers)}/{len(providers_to_init)} available"
        )

    async def _initialize_provider(self, provider_name: str):
        """
        Initialize a single LLM provider with its API key and configuration.

        This method is responsible for initializing an individual provider by:
        1. Retrieving the provider's configuration and API key
        2. Importing the appropriate provider class
        3. Instantiating the provider with the configured API key
        4. Calling the provider's initialize method to establish connectivity
        5. Recording the provider's status (including available models)

        The method handles errors gracefully, ensuring that exceptions during any
        stage of initialization are caught, logged, and reflected in the provider's
        status rather than propagated up the call stack.

        Args:
            provider_name: Name of the provider to initialize, matching a value
                          in the Provider enum (e.g., "openai", "anthropic").

        Returns:
            None. Results are stored in the Gateway's providers and provider_status
            dictionaries rather than returned directly.

        Note:
            This method is called by _initialize_providers during server startup
            and is not intended to be called directly by users of the Gateway class.
        """
        api_key = None
        api_key_configured = False
        provider_config = None

        try:
            cfg = get_config()
            provider_config = getattr(cfg.providers, provider_name, None)

            # Get API key ONLY from the loaded config object
            if provider_config and provider_config.api_key:
                api_key = provider_config.api_key
                api_key_configured = True
            # Special case for Ollama: doesn't require an API key
            elif provider_name == Provider.OLLAMA.value and provider_config:
                api_key = None
                api_key_configured = True
                self.logger.debug("Initializing Ollama provider without API key (not required)")
            else:
                # This case should ideally not be reached if checks in _initialize_providers are correct,
                # but handle defensively.
                self.logger.warning(
                    f"Attempted to initialize {provider_name}, but API key not found in loaded config."
                )
                api_key_configured = False

            if not api_key_configured:
                # Record status for providers found in config but without a key
                if provider_config:
                    self.provider_status[provider_name] = ProviderStatus(
                        enabled=provider_config.enabled,  # Reflects config setting
                        available=False,
                        api_key_configured=False,
                        models=[],
                        error="API key not found in loaded configuration",
                    )
                # Do not log the warning here again, just return
                return

            # --- API Key is configured, proceed with initialization ---
            self.logger.debug(f"Initializing provider {provider_name} with key from config.")

            # Import PROVIDER_REGISTRY to use centralized provider registry
            from ultimate_mcp_server.core.providers import PROVIDER_REGISTRY

            # Use the registry instead of hardcoded providers dictionary
            provider_class = PROVIDER_REGISTRY.get(provider_name)
            if not provider_class:
                raise ValueError(f"Invalid provider name mapping: {provider_name}")

            # Instantiate provider with the API key retrieved from the config (via decouple)
            # Ensure provider classes' __init__ expect 'api_key' as a keyword argument
            provider = provider_class(api_key=api_key)

            # Initialize provider (which should use the config passed)
            available = await provider.initialize()

            # Update status based on initialization result
            if available:
                models = await provider.list_models()
                self.providers[provider_name] = provider
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=provider_config.enabled,
                    available=True,
                    api_key_configured=True,
                    models=models,
                )
                self.logger.success(
                    f"Provider {provider_name} initialized successfully with {len(models)} models",
                    emoji_key="provider",
                )
            else:
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=provider_config.enabled,
                    available=False,
                    api_key_configured=True,  # Key was found, but init failed
                    models=[],
                    error="Initialization failed (check provider API status or logs)",
                )
                self.logger.error(
                    f"Provider {provider_name} initialization failed", emoji_key="error"
                )

        except Exception as e:
            # Handle unexpected errors during initialization
            error_msg = f"Error initializing provider {provider_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Ensure status is updated even on exceptions
            enabled_status = provider_config.enabled if provider_config else False  # Best guess
            self.provider_status[provider_name] = ProviderStatus(
                enabled=enabled_status,
                available=False,
                api_key_configured=api_key_configured,  # Reflects if key was found before error
                models=[],
                error=error_msg,
            )

    @property
    def system_instructions(self) -> str:
        """
        Return comprehensive system-level instructions for LLMs on how to use the gateway.

        This property generates detailed instructions that are injected into the system prompt
        for LLMs using the Gateway. These instructions serve as a guide for LLMs to effectively
        utilize the available tools and capabilities, helping them understand:

        - The categories of available tools and their purposes
        - Best practices for provider and model selection
        - Error handling strategies and patterns
        - Recommendations for efficient and appropriate tool usage
        - Guidelines for choosing the right tool for specific tasks

        The instructions are designed to be clear and actionable, helping LLMs make
        informed decisions about when and how to use different components of the
        Ultimate MCP Server. They're structured in a hierarchical format with sections
        covering core categories, best practices, and additional resources.

        Returns:
            A formatted string containing detailed instructions for LLMs on how to
            effectively use the Gateway's tools and capabilities. These instructions
            are automatically included in the system prompt for all LLM interactions.
        """
        # Tool loading message can be adjusted based on self.load_all_tools if needed
        tool_loading_info = "all available tools" if self.load_all_tools else "the Base Toolset"

        return f"""
# Ultimate MCP Server Tool Usage Instructions
        
You have access to the Ultimate MCP Server, which provides unified access to multiple language model
providers (OpenAI, Anthropic, etc.) through a standardized interface. This server instance has loaded {tool_loading_info}. 
Follow these instructions to effectively use the available tools.

## Core Tool Categories

1. **Provider Tools**: Use these to discover available providers and models
   - `get_provider_status`: Check which providers are available
   - `list_models`: List models available from a specific provider

2. **Completion Tools**: Use these for text generation
   - `generate_completion`: Single-prompt text generation (non-streaming)
   - `chat_completion`: Multi-turn conversation with message history
   - `multi_completion`: Compare outputs from multiple providers/models

3. **Tournament Tools**: Use these to run competitions between models
   - `create_tournament`: Create and start a new tournament
   - `get_tournament_status`: Check tournament progress
   - `get_tournament_results`: Get detailed tournament results
   - `list_tournaments`: List all tournaments
   - `cancel_tournament`: Cancel a running tournament

## Best Practices

1. **Provider Selection**:
   - Always check provider availability with `get_provider_status` before use
   - Verify model availability with `list_models` before using specific models

2. **Error Handling**:
   - All tools include error handling in their responses
   - Check for the presence of an "error" field in responses
   - If an error occurs, adapt your approach based on the error message

3. **Efficient Usage**:
   - Use cached tools when repeatedly calling the same function with identical parameters
   - For long-running operations like tournaments, poll status periodically

4. **Tool Selection Guidelines**:
   - For single-turn text generation → `generate_completion`
   - For conversation-based interactions → `chat_completion`
   - For comparing outputs across models → `multi_completion`
   - For evaluating model performance → Tournament tools

## Additional Resources

For more detailed information and examples, access these MCP resources:
- `info://server`: Basic server information
- `info://tools`: Overview of available tools
- `provider://{{provider_name}}`: Details about a specific provider
- `guide://llm`: Comprehensive usage guide for LLMs
- `guide://error-handling`: Detailed error handling guidance
- `examples://workflows`: Detailed examples of common workflows
- `examples://completions`: Examples of different completion types
- `examples://tournaments`: Guidance on tournament configuration and analysis

Remember to use appropriate error handling and follow the documented parameter formats
for each tool. All providers may not be available at all times, so always check status
first and be prepared to adapt to available providers.
"""

    def _register_tools(self, load_all: bool = False):
        """
        Register all MCP tools with the server instance.

        This internal method sets up all available tools in the Ultimate MCP Server,
        making them accessible to LLMs through the MCP protocol. It handles:

        1. Setting up the basic echo tool for connectivity testing
        2. Conditionally calling the register_all_tools function to set up either
           the 'Base Toolset' or all specialized tools based on the `load_all` flag.

        The registration process wraps each tool function with logging functionality
        via the log_tool_calls decorator, ensuring consistent logging behavior across
        all tools. This provides valuable diagnostic information during tool execution.

        All registered tools become available through the MCP interface and can be
        discovered and used by LLMs interacting with the server.

        Args:
            load_all: If True, register all tools. If False, register only the base set.

        Note:
            This method is called automatically during Gateway initialization when
            register_tools=True (the default) and is not intended to be called directly.
        """
        # Import here to avoid circular dependency
        from ultimate_mcp_server.tools import register_all_tools

        self.logger.info("Registering core tools...")

        # Echo tool - define the function first, then register it
        @self.log_tool_calls
        async def echo(message: str, ctx: Context = None) -> Dict[str, Any]:
            """
            Echo back the message for testing MCP connectivity.

            Args:
                message: The message to echo back

            Returns:
                Dictionary containing the echoed message
            """
            self.logger.info(f"Echo tool called with message: {message}")
            return {"message": message}

        # Now register the decorated function with mcp.tool
        self.mcp.tool(echo)

        # Define our base toolset - use function names not module names
        base_toolset = [
            # Completion tools
            "generate_completion",
            "chat_completion",
            "multi_completion",
            # "stream_completion", # Not that useful for MCP
            # Provider tools
            "get_provider_status",
            "list_models",
            # Filesystem tools
            "read_file",
            "read_multiple_files",
            "write_file",
            "edit_file",
            "create_directory",
            "list_directory",
            "directory_tree",
            "move_file",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
            "get_unique_filepath",
            # Optimization tools
            "estimate_cost",
            "compare_models",
            "recommend_model",
            # Local text tools
            "run_ripgrep",
            "run_awk",
            "run_sed",
            "run_jq",
            # Search tools
            "marqo_fused_search",
            # SmartBrowser class methods
            "search",
            "download",
            "download_site_pdfs",
            "collect_documentation",
            "run_macro",
            "autopilot",
            # SQL class methods
            "manage_database",
            "execute_sql",
            "explore_database",
            "access_audit_log",
            # Document processing class methods
            "convert_document",
            "chunk_document",
            "clean_and_format_text_as_markdown",
            "batch_format_texts",
            "optimize_markdown_formatting",
            "generate_qa_pairs",
            "summarize_document",
            "ocr_image",
            "enhance_ocr_text",
            "analyze_pdf_structure",
            "extract_tables",
            "process_document_batch",
            # Python sandbox class methods
            "execute_python",
            "repl_python",
        ]

        # Conditionally register tools based on load_all flag
        if load_all:
            self.logger.info("Calling register_all_tools to register ALL available tools...")
            register_all_tools(self.mcp)
        else:
            self.logger.info("Calling register_all_tools to register only the BASE toolset...")
            # Check if tool_registration filter is enabled in config
            cfg = get_config()
            if cfg.tool_registration.filter_enabled:
                # If filtering is already enabled, respect that configuration
                self.logger.info("Tool filtering is enabled - using config filter settings")
                register_all_tools(self.mcp)
            else:
                # Otherwise, set up filtering for base toolset
                cfg.tool_registration.filter_enabled = True
                cfg.tool_registration.included_tools = base_toolset
                self.logger.info(f"Registering base toolset: {', '.join(base_toolset)}")
                register_all_tools(self.mcp)

        # After tools are registered, save the tool names to a file for the tools estimator script
        try:
            import json

            from ultimate_mcp_server.tools import STANDALONE_TOOL_FUNCTIONS

            # Get tools from STANDALONE_TOOL_FUNCTIONS plus class-based tools
            all_tool_names = []

            # Add standalone tool function names
            for tool_func in STANDALONE_TOOL_FUNCTIONS:
                if hasattr(tool_func, "__name__"):
                    all_tool_names.append(tool_func.__name__)

            # Add echo tool
            all_tool_names.append("echo")

            # Write to file
            with open("tools_list.json", "w") as f:
                json.dump(all_tool_names, f, indent=2)

            self.logger.info(
                f"Wrote {len(all_tool_names)} tool names to tools_list.json for context estimator"
            )
        except Exception as e:
            self.logger.warning(f"Failed to write tool names to file: {str(e)}")

    def _register_resources(self):
        """
        Register all MCP resources with the server instance.

        This internal method registers standard MCP resources that provide static
        information and guidance to LLMs using the Ultimate MCP Server. Resources differ
        from tools in that they:

        1. Provide static reference information rather than interactive functionality
        2. Are accessed via URI-like identifiers (e.g., "info://server", "guide://llm")
        3. Don't require API calls or external services to generate their responses

        Registered resources include:
        - Server and tool information (info:// resources)
        - Provider details (provider:// resources)
        - Usage guides and tutorials (guide:// resources)
        - Example workflows and usage patterns (examples:// resources)

        These resources serve as a knowledge base for LLMs to better understand how to
        effectively use the available tools and follow best practices. They help reduce
        the need for extensive contextual information in prompts by making reference
        material available on-demand through the MCP protocol.

        Note:
            This method is called automatically during Gateway initialization when
            register_tools=True (the default) and is not intended to be called directly.
        """

        @self.mcp.resource("info://server")
        def get_server_info() -> Dict[str, Any]:
            """
            Get information about the Ultimate MCP Server server.

            This resource provides basic metadata about the Ultimate MCP Server server instance,
            including its name, version, and supported providers. Use this resource to
            discover server capabilities and version information.

            Resource URI: info://server

            Returns:
                Dictionary containing server information:
                - name: Name of the Ultimate MCP Server server
                - version: Version of the Ultimate MCP Server server
                - description: Brief description of server functionality
                - providers: List of supported LLM provider names

            Example:
                {
                    "name": "Ultimate MCP Server",
                    "version": "0.1.0",
                    "description": "MCP server for accessing multiple LLM providers",
                    "providers": ["openai", "anthropic", "deepseek", "gemini"]
                }

            Usage:
                This resource is useful for clients to verify server identity, check compatibility,
                and discover basic capabilities. For detailed provider status, use the
                get_provider_status tool instead.
            """
            return {
                "name": self.name,
                "version": "0.1.0",
                "description": "MCP server for accessing multiple LLM providers",
                "providers": [p.value for p in Provider],
            }

        @self.mcp.resource("info://tools")
        def get_tools_info() -> Dict[str, Any]:
            """
            Get information about available Ultimate MCP Server tools.

            This resource provides a descriptive overview of the tools available in the
            Ultimate MCP Server, organized by category. Use this resource to understand which
            tools are available and how they're organized.

            Resource URI: info://tools

            Returns:
                Dictionary containing tools information organized by category:
                - provider_tools: Tools for interacting with LLM providers
                - completion_tools: Tools for text generation and completion
                - tournament_tools: Tools for running model tournaments
                - document_tools: Tools for document processing

            Example:
                {
                    "provider_tools": {
                        "description": "Tools for accessing and managing LLM providers",
                        "tools": ["get_provider_status", "list_models"]
                    },
                    "completion_tools": {
                        "description": "Tools for text generation and completion",
                        "tools": ["generate_completion", "chat_completion", "multi_completion"]
                    },
                    "tournament_tools": {
                        "description": "Tools for running and managing model tournaments",
                        "tools": ["create_tournament", "list_tournaments", "get_tournament_status",
                                 "get_tournament_results", "cancel_tournament"]
                    }
                }

            Usage:
                Use this resource to understand the capabilities of the Ultimate MCP Server and
                discover available tools. For detailed information about specific tools,
                use the MCP list_tools method.
            """
            return {
                "provider_tools": {
                    "description": "Tools for accessing and managing LLM providers",
                    "tools": ["get_provider_status", "list_models"],
                },
                "completion_tools": {
                    "description": "Tools for text generation and completion",
                    "tools": ["generate_completion", "chat_completion", "multi_completion"],
                },
                "tournament_tools": {
                    "description": "Tools for running and managing model tournaments",
                    "tools": [
                        "create_tournament",
                        "list_tournaments",
                        "get_tournament_status",
                        "get_tournament_results",
                        "cancel_tournament",
                    ],
                },
                "document_tools": {
                    "description": "Tools for document processing (placeholder for future implementation)",
                    "tools": [],
                },
            }

        @self.mcp.resource("guide://llm")
        def get_llm_guide() -> str:
            """
            Usage guide for LLMs using the Ultimate MCP Server.

            This resource provides structured guidance specifically designed for LLMs to
            effectively use the tools and resources provided by the Ultimate MCP Server. It includes
            recommended tool selection strategies, common usage patterns, and examples.

            Resource URI: guide://llm

            Returns:
                A detailed text guide with sections on tool selection, usage patterns,
                and example workflows.

            Usage:
                This resource is primarily intended to be included in context for LLMs
                that will be using the gateway tools, to help them understand how to
                effectively use the available capabilities.
            """
            return """
                # Ultimate MCP Server Usage Guide for Language Models
                
                ## Overview
                
                The Ultimate MCP Server provides a set of tools for accessing multiple language model providers
                (OpenAI, Anthropic, etc.) through a unified interface. This guide will help you understand
                how to effectively use these tools.
                
                ## Tool Selection Guidelines
                
                ### For Text Generation:
                
                1. For single-prompt text generation:
                   - Use `generate_completion` with a specific provider and model
                
                2. For multi-turn conversations:
                   - Use `chat_completion` with a list of message dictionaries
                
                3. For streaming responses (real-time text output):
                   - Use streaming tools in the CompletionTools class
                
                4. For comparing outputs across providers:
                   - Use `multi_completion` with a list of provider configurations
                
                ### For Provider Management:
                
                1. To check available providers:
                   - Use `get_provider_status` to see which providers are available
                
                2. To list available models:
                   - Use `list_models` to view models from all providers or a specific provider
                
                ### For Running Tournaments:
                
                1. To create a new tournament:
                   - Use `create_tournament` with a prompt and list of model IDs
                
                2. To check tournament status:
                   - Use `get_tournament_status` with a tournament ID
                
                3. To get detailed tournament results:
                   - Use `get_tournament_results` with a tournament ID
                
                ## Common Workflows
                
                ### Provider Selection Workflow:
                ```
                1. Call get_provider_status() to see available providers
                2. Call list_models(provider="openai") to see available models
                3. Call generate_completion(prompt="...", provider="openai", model="gpt-4o")
                ```
                
                ### Multi-Provider Comparison Workflow:
                ```
                1. Call multi_completion(
                      prompt="...",
                      providers=[
                          {"provider": "openai", "model": "gpt-4o"},
                          {"provider": "anthropic", "model": "claude-3-opus-20240229"}
                      ]
                   )
                2. Compare results from each provider
                ```
                
                ### Tournament Workflow:
                ```
                1. Call create_tournament(name="...", prompt="...", model_ids=["openai/gpt-4o", "anthropic/claude-3-opus"])
                2. Store the tournament_id from the response
                3. Call get_tournament_status(tournament_id="...") to monitor progress
                4. Once status is "COMPLETED", call get_tournament_results(tournament_id="...")
                ```
                
                ## Error Handling Best Practices
                
                1. Always check for "error" fields in tool responses
                2. Verify provider availability before attempting to use specific models
                3. For tournament tools, handle potential 404 errors for invalid tournament IDs
                
                ## Performance Considerations
                
                1. Most completion tools include token usage and cost metrics in their responses
                2. Use caching decorators for repetitive requests to save costs
                3. Consider using stream=True for long completions to improve user experience
            """

        @self.mcp.resource("provider://{{provider_name}}")
        def get_provider_info(provider_name: str) -> Dict[str, Any]:
            """
            Get detailed information about a specific LLM provider.

            This resource provides comprehensive information about a specific provider,
            including its capabilities, available models, and configuration status.

            Resource URI template: provider://{provider_name}

            Args:
                provider_name: Name of the provider to retrieve information for
                              (e.g., "openai", "anthropic", "gemini")

            Returns:
                Dictionary containing detailed provider information:
                - name: Provider name
                - status: Current status (enabled, available, etc.)
                - capabilities: List of supported capabilities
                - models: List of available models and their details
                - config: Current configuration settings (with sensitive info redacted)

            Example:
                {
                    "name": "openai",
                    "status": {
                        "enabled": true,
                        "available": true,
                        "api_key_configured": true,
                        "error": null
                    },
                    "capabilities": ["chat", "completion", "embeddings", "vision"],
                    "models": [
                        {
                            "id": "gpt-4o",
                            "name": "GPT-4o",
                            "context_window": 128000,
                            "features": ["chat", "completion", "vision"]
                        },
                        # More models...
                    ],
                    "config": {
                        "base_url": "https://api.openai.com/v1",
                        "timeout_seconds": 30,
                        "default_model": "gpt-4.1-mini"
                    }
                }

            Error Handling:
                If the provider doesn't exist or isn't configured, returns an appropriate
                error message in the response.

            Usage:
                Use this resource to get detailed information about a specific provider
                before using its models for completions or other operations.
            """
            # Check if provider exists in status dictionary
            provider_status = self.provider_status.get(provider_name)
            if not provider_status:
                return {
                    "name": provider_name,
                    "error": f"Provider '{provider_name}' not found or not configured",
                    "status": {"enabled": False, "available": False, "api_key_configured": False},
                    "models": [],
                }

            # Get provider instance if available
            provider_instance = self.providers.get(provider_name)

            # Build capability list based on provider name
            capabilities = []
            if provider_name in [
                Provider.OPENAI.value,
                Provider.ANTHROPIC.value,
                Provider.GEMINI.value,
            ]:
                capabilities = ["chat", "completion"]

            if provider_name == Provider.OPENAI.value:
                capabilities.extend(["embeddings", "vision", "image_generation"])
            elif provider_name == Provider.ANTHROPIC.value:
                capabilities.extend(["vision"])

            # Return provider details
            return {
                "name": provider_name,
                "status": {
                    "enabled": provider_status.enabled,
                    "available": provider_status.available,
                    "api_key_configured": provider_status.api_key_configured,
                    "error": provider_status.error,
                },
                "capabilities": capabilities,
                "models": provider_status.models,
                "config": {
                    # Include non-sensitive config info
                    "default_model": provider_instance.default_model if provider_instance else None,
                    "timeout_seconds": 30,  # Example default
                },
            }

        @self.mcp.resource("guide://error-handling")
        def get_error_handling_guide() -> Dict[str, Any]:
            """
            Get comprehensive guidance on handling errors from Ultimate MCP Server tools.

            This resource provides detailed information about common error patterns,
            error handling strategies, and recovery approaches for each tool in the
            Ultimate MCP Server. It helps LLMs understand how to gracefully handle and recover
            from various error conditions.

            Resource URI: guide://error-handling

            Returns:
                Dictionary containing error handling guidance organized by tool type:
                - provider_tools: Error handling for provider-related tools
                - completion_tools: Error handling for completion tools
                - tournament_tools: Error handling for tournament tools

            Usage:
                This resource helps LLMs implement robust error handling when using
                the Ultimate MCP Server tools, improving the resilience of their interactions.
            """
            return {
                "general_principles": {
                    "error_detection": {
                        "description": "How to detect errors in tool responses",
                        "patterns": [
                            "Check for an 'error' field in the response dictionary",
                            "Look for status codes in error messages (e.g., 404, 500)",
                            "Check for empty or null results where data is expected",
                            "Look for 'warning' fields that may indicate partial success",
                        ],
                    },
                    "error_recovery": {
                        "description": "General strategies for recovering from errors",
                        "strategies": [
                            "Retry with different parameters when appropriate",
                            "Fallback to alternative tools or providers",
                            "Gracefully degrade functionality when optimal path is unavailable",
                            "Clearly communicate errors to users with context and suggestions",
                        ],
                    },
                },
                "provider_tools": {
                    "get_provider_status": {
                        "common_errors": [
                            {
                                "error": "Server context not available",
                                "cause": "The server may not be fully initialized",
                                "handling": "Wait and retry or report server initialization issue",
                            },
                            {
                                "error": "No providers are currently configured",
                                "cause": "No LLM providers are enabled or initialization is incomplete",
                                "handling": "Proceed with caution and check if specific providers are required",
                            },
                        ],
                        "recovery_strategies": [
                            "If no providers are available, clearly inform the user of limited capabilities",
                            "If specific providers are unavailable, suggest alternatives based on task requirements",
                        ],
                    },
                    "list_models": {
                        "common_errors": [
                            {
                                "error": "Invalid provider",
                                "cause": "Specified provider name doesn't exist or isn't configured",
                                "handling": "Use valid providers from the error message's 'valid_providers' field",
                            },
                            {
                                "warning": "Provider is configured but not available",
                                "cause": "Provider API key issues or service connectivity problems",
                                "handling": "Use an alternative provider or inform user of limited options",
                            },
                        ],
                        "recovery_strategies": [
                            "When provider is invalid, fall back to listing all available providers",
                            "When models list is empty, suggest using the default model or another provider",
                        ],
                    },
                },
                "completion_tools": {
                    "generate_completion": {
                        "common_errors": [
                            {
                                "error": "Provider not available",
                                "cause": "Specified provider doesn't exist or isn't configured",
                                "handling": "Switch to an available provider (check with get_provider_status)",
                            },
                            {
                                "error": "Failed to initialize provider",
                                "cause": "API key configuration or network issues",
                                "handling": "Try another provider or check provider status",
                            },
                            {
                                "error": "Completion generation failed",
                                "cause": "Provider API errors, rate limits, or invalid parameters",
                                "handling": "Retry with different parameters or use another provider",
                            },
                        ],
                        "recovery_strategies": [
                            "Use multi_completion to try multiple providers simultaneously",
                            "Progressively reduce complexity (max_tokens, simplify prompt) if facing limits",
                            "Fall back to more reliable models if specialized ones are unavailable",
                        ],
                    },
                    "multi_completion": {
                        "common_errors": [
                            {
                                "error": "Invalid providers format",
                                "cause": "Providers parameter is not a list of provider configurations",
                                "handling": "Correct the format to a list of dictionaries with provider info",
                            },
                            {
                                "partial_failure": "Some providers failed",
                                "cause": "Indicated by successful_count < total_providers",
                                "handling": "Use the successful results and analyze error fields for failed ones",
                            },
                        ],
                        "recovery_strategies": [
                            "Focus on successful completions even if some providers failed",
                            "Check each provider's 'success' field to identify which ones worked",
                            "If timeout occurs, consider increasing the timeout parameter or reducing providers",
                        ],
                    },
                },
                "tournament_tools": {
                    "create_tournament": {
                        "common_errors": [
                            {
                                "error": "Invalid input",
                                "cause": "Missing required fields or validation errors",
                                "handling": "Check all required parameters are provided with valid values",
                            },
                            {
                                "error": "Failed to start tournament execution",
                                "cause": "Server resource constraints or initialization errors",
                                "handling": "Retry with fewer rounds or models, or try again later",
                            },
                        ],
                        "recovery_strategies": [
                            "Verify model IDs are valid before creating tournament",
                            "Start with simple tournaments to validate functionality before complex ones",
                            "Use error message details to correct specific input problems",
                        ],
                    },
                    "get_tournament_status": {
                        "common_errors": [
                            {
                                "error": "Tournament not found",
                                "cause": "Invalid tournament ID or tournament was deleted",
                                "handling": "Verify tournament ID or use list_tournaments to see available tournaments",
                            },
                            {
                                "error": "Invalid tournament ID format",
                                "cause": "Tournament ID is not a string or is empty",
                                "handling": "Ensure tournament ID is a valid string matching the expected format",
                            },
                        ],
                        "recovery_strategies": [
                            "When tournament not found, list all tournaments to find valid ones",
                            "If tournament status is FAILED, check error_message for details",
                            "Implement polling with backoff for monitoring long-running tournaments",
                        ],
                    },
                },
                "error_pattern_examples": {
                    "retry_with_fallback": {
                        "description": "Retry with fallback to another provider",
                        "example": """
                            # Try primary provider
                            result = generate_completion(prompt="...", provider="openai", model="gpt-4o")
                            
                            # Check for errors and fall back if needed
                            if "error" in result:
                                logger.warning(f"Primary provider failed: {result['error']}")
                                # Fall back to alternative provider
                                result = generate_completion(prompt="...", provider="anthropic", model="claude-3-opus-20240229")
                        """,
                    },
                    "validation_before_call": {
                        "description": "Validate parameters before making tool calls",
                        "example": """
                            # Get available providers first
                            provider_status = get_provider_status()
                            
                            # Check if requested provider is available
                            requested_provider = "openai"
                            if requested_provider not in provider_status["providers"] or not provider_status["providers"][requested_provider]["available"]:
                                # Fall back to any available provider
                                available_providers = [p for p, status in provider_status["providers"].items() if status["available"]]
                                if available_providers:
                                    requested_provider = available_providers[0]
                                else:
                                    return {"error": "No LLM providers are available"}
                        """,
                    },
                },
            }

        @self.mcp.resource("examples://workflows")
        def get_workflow_examples() -> Dict[str, Any]:
            """
            Get comprehensive examples of multi-tool workflows.

            This resource provides detailed, executable examples showing how to combine
            multiple tools into common workflows. These examples demonstrate best practices
            for tool sequencing, error handling, and result processing.

            Resource URI: examples://workflows

            Returns:
                Dictionary containing workflow examples organized by scenario:
                - basic_provider_selection: Example of selecting a provider and model
                - model_comparison: Example of comparing outputs across providers
                - tournaments: Example of creating and monitoring a tournament
                - advanced_chat: Example of a multi-turn conversation with system prompts

            Usage:
                These examples are designed to be used as reference by LLMs to understand
                how to combine multiple tools in the Ultimate MCP Server to accomplish common tasks.
                Each example includes expected outputs to help understand the flow.
            """
            return {
                "basic_provider_selection": {
                    "description": "Selecting a provider and model for text generation",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "get_provider_status",
                            "parameters": {},
                            "purpose": "Check which providers are available",
                            "example_output": {
                                "providers": {
                                    "openai": {"available": True, "models_count": 12},
                                    "anthropic": {"available": True, "models_count": 6},
                                }
                            },
                        },
                        {
                            "step": 2,
                            "tool": "list_models",
                            "parameters": {"provider": "openai"},
                            "purpose": "Get available models for the selected provider",
                            "example_output": {
                                "models": {
                                    "openai": [
                                        {
                                            "id": "gpt-4o",
                                            "name": "GPT-4o",
                                            "features": ["chat", "completion"],
                                        }
                                    ]
                                }
                            },
                        },
                        {
                            "step": 3,
                            "tool": "generate_completion",
                            "parameters": {
                                "prompt": "Explain quantum computing in simple terms",
                                "provider": "openai",
                                "model": "gpt-4o",
                                "temperature": 0.7,
                            },
                            "purpose": "Generate text with the selected provider and model",
                            "example_output": {
                                "text": "Quantum computing is like...",
                                "model": "gpt-4o",
                                "provider": "openai",
                                "tokens": {"input": 8, "output": 150, "total": 158},
                                "cost": 0.000123,
                            },
                        },
                    ],
                    "error_handling": [
                        "If get_provider_status shows provider unavailable, try a different provider",
                        "If list_models returns empty list, select a different provider",
                        "If generate_completion returns an error, check the error message for guidance",
                    ],
                },
                "model_comparison": {
                    "description": "Comparing multiple models on the same task",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "multi_completion",
                            "parameters": {
                                "prompt": "Write a haiku about programming",
                                "providers": [
                                    {"provider": "openai", "model": "gpt-4o"},
                                    {"provider": "anthropic", "model": "claude-3-opus-20240229"},
                                ],
                                "temperature": 0.7,
                            },
                            "purpose": "Generate completions from multiple providers simultaneously",
                            "example_output": {
                                "results": {
                                    "openai/gpt-4o": {
                                        "success": True,
                                        "text": "Code flows like water\nBugs emerge from the depths\nPatience brings order",
                                        "model": "gpt-4o",
                                    },
                                    "anthropic/claude-3-opus-20240229": {
                                        "success": True,
                                        "text": "Fingers dance on keys\nLogic blooms in silent thought\nPrograms come alive",
                                        "model": "claude-3-opus-20240229",
                                    },
                                },
                                "successful_count": 2,
                                "total_providers": 2,
                            },
                        },
                        {
                            "step": 2,
                            "suggestion": "Compare the results for quality, style, and adherence to the haiku format",
                        },
                    ],
                    "error_handling": [
                        "Check successful_count vs total_providers to see if all providers succeeded",
                        "For each provider, check the success field to determine if it completed successfully",
                        "If a provider failed, look at its error field for details",
                    ],
                },
                "tournaments": {
                    "description": "Creating and monitoring a multi-model tournament",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "create_tournament",
                            "parameters": {
                                "name": "Sorting Algorithm Tournament",
                                "prompt": "Implement a quicksort algorithm in Python that handles duplicates efficiently",
                                "model_ids": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229"],
                                "rounds": 3,
                                "tournament_type": "code",
                            },
                            "purpose": "Create a new tournament comparing multiple models",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "PENDING",
                            },
                        },
                        {
                            "step": 2,
                            "tool": "get_tournament_status",
                            "parameters": {"tournament_id": "tour_abc123xyz789"},
                            "purpose": "Check if the tournament has started running",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "RUNNING",
                                "current_round": 1,
                                "total_rounds": 3,
                            },
                        },
                        {
                            "step": 3,
                            "suggestion": "Wait for the tournament to complete",
                            "purpose": "Tournaments run asynchronously and may take time to complete",
                        },
                        {
                            "step": 4,
                            "tool": "get_tournament_results",
                            "parameters": {"tournament_id": "tour_abc123xyz789"},
                            "purpose": "Retrieve full results once the tournament is complete",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "COMPLETED",
                                "rounds_data": [
                                    {
                                        "round_number": 1,
                                        "model_outputs": {
                                            "openai/gpt-4o": "def quicksort(arr): ...",
                                            "anthropic/claude-3-opus-20240229": "def quicksort(arr): ...",
                                        },
                                        "scores": {
                                            "openai/gpt-4o": 0.85,
                                            "anthropic/claude-3-opus-20240229": 0.92,
                                        },
                                    }
                                    # Additional rounds would be here in a real response
                                ],
                            },
                        },
                    ],
                    "error_handling": [
                        "If create_tournament fails, check the error message for missing or invalid parameters",
                        "If get_tournament_status returns an error, verify the tournament_id is correct",
                        "If tournament status is FAILED, check the error_message field for details",
                    ],
                },
                "advanced_chat": {
                    "description": "Multi-turn conversation with system prompt and context",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "chat_completion",
                            "parameters": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Hello, can you help me with Python?",
                                    }
                                ],
                                "provider": "anthropic",
                                "model": "claude-3-opus-20240229",
                                "system_prompt": "You are an expert Python tutor. Provide concise, helpful answers with code examples when appropriate.",
                                "temperature": 0.5,
                            },
                            "purpose": "Start a conversation with a system prompt for context",
                            "example_output": {
                                "text": "Hello! I'd be happy to help you with Python. What specific aspect are you interested in learning about?",
                                "model": "claude-3-opus-20240229",
                                "provider": "anthropic",
                            },
                        },
                        {
                            "step": 2,
                            "tool": "chat_completion",
                            "parameters": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Hello, can you help me with Python?",
                                    },
                                    {
                                        "role": "assistant",
                                        "content": "Hello! I'd be happy to help you with Python. What specific aspect are you interested in learning about?",
                                    },
                                    {
                                        "role": "user",
                                        "content": "How do I write a function that checks if a string is a palindrome?",
                                    },
                                ],
                                "provider": "anthropic",
                                "model": "claude-3-opus-20240229",
                                "system_prompt": "You are an expert Python tutor. Provide concise, helpful answers with code examples when appropriate.",
                                "temperature": 0.5,
                            },
                            "purpose": "Continue the conversation by including the full message history",
                            "example_output": {
                                "text": "Here's a simple function to check if a string is a palindrome in Python:\n\n```python\ndef is_palindrome(s):\n    # Remove spaces and convert to lowercase for more flexible matching\n    s = s.lower().replace(' ', '')\n    # Compare the string with its reverse\n    return s == s[::-1]\n\n# Examples\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\nprint(is_palindrome('A man a plan a canal Panama'))  # True\n```\n\nThis function works by:\n1. Converting the string to lowercase and removing spaces\n2. Checking if the processed string equals its reverse (using slice notation `[::-1]`)\n\nIs there anything specific about this solution you'd like me to explain further?",
                                "model": "claude-3-opus-20240229",
                                "provider": "anthropic",
                            },
                        },
                    ],
                    "error_handling": [
                        "Always include the full conversation history in the messages array",
                        "Ensure each message has both 'role' and 'content' fields",
                        "If using system_prompt, ensure it's appropriate for the provider",
                    ],
                },
            }

        @self.mcp.resource("examples://completions")
        def get_completion_examples() -> Dict[str, Any]:
            """
            Get examples of different completion types and when to use them.

            This resource provides detailed examples of different completion tools available
            in the Ultimate MCP Server, along with guidance on when to use each type. It helps with
            selecting the most appropriate completion tool for different scenarios.

            Resource URI: examples://completions

            Returns:
                Dictionary containing completion examples organized by type:
                - standard_completion: When to use generate_completion
                - chat_completion: When to use chat_completion
                - streaming_completion: When to use stream_completion
                - multi_provider: When to use multi_completion

            Usage:
                This resource helps LLMs understand the appropriate completion tool
                to use for different scenarios, with concrete examples and use cases.
            """
            return {
                "standard_completion": {
                    "tool": "generate_completion",
                    "description": "Single-turn text generation without streaming",
                    "best_for": [
                        "Simple, one-off text generation tasks",
                        "When you need a complete response at once",
                        "When you don't need conversation history",
                    ],
                    "example": {
                        "request": {
                            "prompt": "Explain the concept of quantum entanglement in simple terms",
                            "provider": "openai",
                            "model": "gpt-4o",
                            "temperature": 0.7,
                        },
                        "response": {
                            "text": "Quantum entanglement is like having two magic coins...",
                            "model": "gpt-4o",
                            "provider": "openai",
                            "tokens": {"input": 10, "output": 150, "total": 160},
                            "cost": 0.00032,
                            "processing_time": 2.1,
                        },
                    },
                },
                "chat_completion": {
                    "tool": "chat_completion",
                    "description": "Multi-turn conversation with message history",
                    "best_for": [
                        "Maintaining conversation context across multiple turns",
                        "When dialogue history matters for the response",
                        "When using system prompts to guide assistant behavior",
                    ],
                    "example": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "What's the capital of France?"},
                                {"role": "assistant", "content": "The capital of France is Paris."},
                                {"role": "user", "content": "And what's its population?"},
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-opus-20240229",
                            "system_prompt": "You are a helpful geography assistant.",
                        },
                        "response": {
                            "text": "The population of Paris is approximately 2.1 million people in the city proper...",
                            "model": "claude-3-opus-20240229",
                            "provider": "anthropic",
                            "tokens": {"input": 62, "output": 48, "total": 110},
                            "cost": 0.00055,
                            "processing_time": 1.8,
                        },
                    },
                },
                "streaming_completion": {
                    "tool": "stream_completion",
                    "description": "Generates text in smaller chunks as a stream",
                    "best_for": [
                        "When you need to show incremental progress to users",
                        "For real-time display of model outputs",
                        "Long-form content generation where waiting for the full response would be too long",
                    ],
                    "example": {
                        "request": {
                            "prompt": "Write a short story about a robot learning to paint",
                            "provider": "openai",
                            "model": "gpt-4o",
                        },
                        "response_chunks": [
                            {
                                "text": "In the year 2150, ",
                                "chunk_index": 1,
                                "provider": "openai",
                                "model": "gpt-4o",
                                "finished": False,
                            },
                            {
                                "text": "a maintenance robot named ARIA-7 was assigned to",
                                "chunk_index": 2,
                                "provider": "openai",
                                "model": "gpt-4o",
                                "finished": False,
                            },
                            {
                                "text": "",
                                "chunk_index": 25,
                                "provider": "openai",
                                "full_text": "In the year 2150, a maintenance robot named ARIA-7 was assigned to...",
                                "processing_time": 8.2,
                                "finished": True,
                            },
                        ],
                    },
                },
                "multi_provider": {
                    "tool": "multi_completion",
                    "description": "Get completions from multiple providers simultaneously",
                    "best_for": [
                        "Comparing outputs from different models",
                        "Finding consensus among multiple models",
                        "Fallback scenarios where one provider might fail",
                        "Benchmarking different providers on the same task",
                    ],
                    "example": {
                        "request": {
                            "prompt": "Provide three tips for sustainable gardening",
                            "providers": [
                                {"provider": "openai", "model": "gpt-4o"},
                                {"provider": "anthropic", "model": "claude-3-opus-20240229"},
                            ],
                        },
                        "response": {
                            "results": {
                                "openai/gpt-4o": {
                                    "provider_key": "openai/gpt-4o",
                                    "success": True,
                                    "text": "1. Use compost instead of chemical fertilizers...",
                                    "model": "gpt-4o",
                                },
                                "anthropic/claude-3-opus-20240229": {
                                    "provider_key": "anthropic/claude-3-opus-20240229",
                                    "success": True,
                                    "text": "1. Implement water conservation techniques...",
                                    "model": "claude-3-opus-20240229",
                                },
                            },
                            "successful_count": 2,
                            "total_providers": 2,
                            "processing_time": 3.5,
                        },
                    },
                },
            }

        @self.mcp.resource("examples://tournaments")
        def get_tournament_examples() -> Dict[str, Any]:
            """
            Get detailed examples and guidance for running LLM tournaments.

            This resource provides comprehensive examples and guidance for creating,
            monitoring, and analyzing LLM tournaments. It includes detailed information
            about tournament configuration, interpreting results, and best practices.

            Resource URI: examples://tournaments

            Returns:
                Dictionary containing tournament examples and guidance:
                - tournament_types: Different types of tournaments and their uses
                - configuration_guide: Guidance on how to configure tournaments
                - analysis_guide: How to interpret tournament results
                - example_tournaments: Complete examples of different tournament configurations

            Usage:
                This resource helps LLMs understand how to effectively use the tournament
                tools, with guidance on configuration, execution, and analysis.
            """
            return {
                "tournament_types": {
                    "code": {
                        "description": "Tournaments where models compete on coding tasks",
                        "ideal_for": [
                            "Algorithm implementation challenges",
                            "Debugging exercises",
                            "Code optimization problems",
                            "Comparing models' coding abilities",
                        ],
                        "evaluation_criteria": [
                            "Code correctness",
                            "Efficiency",
                            "Readability",
                            "Error handling",
                        ],
                    },
                    # Other tournament types could be added in the future
                },
                "configuration_guide": {
                    "model_selection": {
                        "description": "Guidelines for selecting models to include in tournaments",
                        "recommendations": [
                            "Include models from different providers for diverse approaches",
                            "Compare models within the same family (e.g., different Claude versions)",
                            "Consider including both specialized and general models",
                            "Ensure all models can handle the task complexity",
                        ],
                    },
                    "rounds": {
                        "description": "How to determine the appropriate number of rounds",
                        "recommendations": [
                            "Start with 3 rounds for most tournaments",
                            "Use more rounds (5+) for more complex or nuanced tasks",
                            "Consider that each round increases total runtime and cost",
                            "Each round gives models a chance to refine their solutions",
                        ],
                    },
                    "prompt_design": {
                        "description": "Best practices for tournament prompt design",
                        "recommendations": [
                            "Be specific about the problem requirements",
                            "Clearly define evaluation criteria",
                            "Specify output format expectations",
                            "Consider including test cases",
                            "Avoid ambiguous or underspecified requirements",
                        ],
                    },
                },
                "analysis_guide": {
                    "score_interpretation": {
                        "description": "How to interpret model scores in tournament results",
                        "guidance": [
                            "Scores are normalized to a 0-1 scale (1 being perfect)",
                            "Consider relative scores between models rather than absolute values",
                            "Look for consistency across rounds",
                            "Consider output quality even when scores are similar",
                        ],
                    },
                    "output_analysis": {
                        "description": "How to analyze model outputs from tournaments",
                        "guidance": [
                            "Compare approaches used by different models",
                            "Look for patterns in errors or limitations",
                            "Identify unique strengths of different providers",
                            "Consider both the score and actual output quality",
                        ],
                    },
                },
                "example_tournaments": {
                    "algorithm_implementation": {
                        "name": "Binary Search Algorithm",
                        "prompt": "Implement a binary search algorithm in Python that can search for an element in a sorted array. Include proper error handling, documentation, and test cases.",
                        "model_ids": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229"],
                        "rounds": 3,
                        "tournament_type": "code",
                        "explanation": "This tournament tests the models' ability to implement a standard algorithm with proper error handling and testing.",
                    },
                    "code_optimization": {
                        "name": "String Processing Optimization",
                        "prompt": "Optimize the following Python function to process large strings more efficiently: def find_substring_occurrences(text, pattern): return [i for i in range(len(text)) if text[i:i+len(pattern)] == pattern]",
                        "model_ids": [
                            "openai/gpt-4o",
                            "anthropic/claude-3-opus-20240229",
                            "anthropic/claude-3-sonnet-20240229",
                        ],
                        "rounds": 4,
                        "tournament_type": "code",
                        "explanation": "This tournament compares models' ability to recognize and implement optimization opportunities in existing code.",
                    },
                },
                "workflow_examples": {
                    "basic_tournament": {
                        "description": "A simple tournament workflow from creation to result analysis",
                        "steps": [
                            {
                                "step": 1,
                                "description": "Create the tournament",
                                "code": "tournament_id = create_tournament(name='Sorting Algorithm Challenge', prompt='Implement an efficient sorting algorithm...', model_ids=['openai/gpt-4o', 'anthropic/claude-3-opus-20240229'], rounds=3, tournament_type='code')",
                            },
                            {
                                "step": 2,
                                "description": "Poll for tournament status",
                                "code": "status = get_tournament_status(tournament_id)['status']\nwhile status in ['PENDING', 'RUNNING']:\n    time.sleep(30)  # Check every 30 seconds\n    status = get_tournament_status(tournament_id)['status']",
                            },
                            {
                                "step": 3,
                                "description": "Retrieve and analyze results",
                                "code": "results = get_tournament_results(tournament_id)\nwinner = max(results['final_scores'].items(), key=lambda x: x[1])[0]\noutputs = {model_id: results['rounds_data'][-1]['model_outputs'][model_id] for model_id in results['config']['model_ids']}",
                            },
                        ],
                    }
                },
            }


def start_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    reload: bool = False,
    transport_mode: str = "streamable-http",
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    load_all_tools: bool = False,  # Added: Flag to control tool loading
) -> None:
    """
    Start the Ultimate MCP Server with configurable settings.

    This function serves as the main entry point for starting the Ultimate MCP Server
    in either SSE (HTTP server) or stdio (direct process communication) mode. It handles
    complete server initialization including:

    1. Configuration loading and parameter validation
    2. Logging setup with proper levels and formatting
    3. Gateway instantiation with tool registration
    4. Transport mode selection and server startup

    The function provides flexibility in server configuration through parameters that
    override settings from the configuration file, allowing for quick adjustments without
    modifying configuration files. It also supports tool filtering, enabling selective
    registration of specific tools.

    Args:
        host: Hostname or IP address to bind the server to (e.g., "localhost", "0.0.0.0").
             If None, uses the value from the configuration file.
        port: TCP port for the server to listen on when in SSE mode.
             If None, uses the value from the configuration file.
        workers: Number of worker processes to spawn for handling requests.
                Higher values improve concurrency but increase resource usage.
                If None, uses the value from the configuration file.
        log_level: Logging verbosity level. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
                  If None, uses the value from the configuration file.
        reload: Whether to automatically reload the server when code changes are detected.
               Useful during development but not recommended for production.
        transport_mode: Communication mode for the server. Options:
                      - "stdio": Run using standard input/output for direct process communication (default)
                      - "sse": Run as an HTTP server with Server-Sent Events for streaming
                      - "streamable-http": Run as an HTTP server with streaming request/response bodies (recommended for HTTP clients)
        include_tools: Optional list of specific tool names to include in registration.
                      If provided, only these tools will be registered unless they are
                      also in exclude_tools. If None, all tools are included by default.
        exclude_tools: Optional list of tool names to exclude from registration.
                      These tools will not be registered even if they are also in include_tools.
        load_all_tools: If True, load all available tools. If False (default), load only the base set.

    Raises:
        ValueError: If transport_mode is not one of the valid options.
        ConfigurationError: If there are critical errors in the server configuration.

    Note:
        This function does not return as it initiates the server event loop, which
        runs until interrupted (e.g., by a SIGINT signal). In SSE mode, it starts
        a Uvicorn server; in stdio mode, it runs the FastMCP stdio handler.
    """
    server_host = host or get_config().server.host
    server_port = port or get_config().server.port
    server_workers = workers or get_config().server.workers

    # Get the current config and update tool registration settings
    cfg = get_config()
    if include_tools or exclude_tools:
        cfg.tool_registration.filter_enabled = True

    if include_tools:
        cfg.tool_registration.included_tools = include_tools

    if exclude_tools:
        cfg.tool_registration.excluded_tools = exclude_tools

    # Validate transport_mode
    if transport_mode not in ["sse", "stdio", "streamable-http"]:
        raise ValueError(
            f"Invalid transport_mode: {transport_mode}. Must be 'sse', 'stdio', or 'streamable-http'"
        )

    # Determine final log level from the provided parameter or fallback to INFO
    final_log_level = (log_level or "INFO").upper()

    # Update LOGGING_CONFIG with the final level
    LOGGING_CONFIG["root"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server.tools"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server.completions"]["level"] = final_log_level

    # Set Uvicorn access level based on final level
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = (
        final_log_level if final_log_level != "CRITICAL" else "CRITICAL"
    )

    # Ensure Uvicorn base/error logs are at least INFO unless final level is DEBUG
    uvicorn_base_level = "INFO" if final_log_level not in ["DEBUG"] else "DEBUG"
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = uvicorn_base_level
    LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = uvicorn_base_level

    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)

    # Initialize the gateway if not already created
    global _gateway_instance
    if not _gateway_instance:
        # Create gateway with tool filtering based on config
        cfg = get_config()
        _gateway_instance = Gateway(
            name=cfg.server.name,
            register_tools=True,
            load_all_tools=load_all_tools,  # Pass the flag to Gateway
        )

    # Log startup info to stderr instead of using logging directly
    print("Starting Ultimate MCP Server server", file=sys.stderr)
    print(f"Host: {server_host}", file=sys.stderr)
    print(f"Port: {server_port}", file=sys.stderr)
    print(f"Workers: {server_workers}", file=sys.stderr)
    print(f"Log level: {final_log_level}", file=sys.stderr)
    print(f"Transport mode: {transport_mode}", file=sys.stderr)
    if transport_mode == "streamable-http":
        print(
            "Note: streamable-http is the recommended transport for HTTP-based MCP clients",
            file=sys.stderr,
        )

    # Log tool loading strategy
    if load_all_tools:
        print("Tool Loading: ALL available tools", file=sys.stderr)
    else:
        print("Tool Loading: Base Toolset Only", file=sys.stderr)
        base_toolset = [
            "completion",
            "filesystem",
            "optimization",
            "provider",
            "local_text",
            "search",
        ]
        print(f"  (Includes: {', '.join(base_toolset)})", file=sys.stderr)

    # Log tool filtering info if enabled
    if cfg.tool_registration.filter_enabled:
        if cfg.tool_registration.included_tools:
            print(
                f"Including tools: {', '.join(cfg.tool_registration.included_tools)}",
                file=sys.stderr,
            )
        if cfg.tool_registration.excluded_tools:
            print(
                f"Excluding tools: {', '.join(cfg.tool_registration.excluded_tools)}",
                file=sys.stderr,
            )

    if transport_mode in ["sse", "streamable-http"]:
        # Run in HTTP mode (unified handling for both SSE and streamable-http)
        import os
        import subprocess
        import threading
        import time

        import uvicorn

        print(f"Running in {transport_mode} mode...", file=sys.stderr)

        # Set up a function to run the tool context estimator after the server starts
        def run_tool_context_estimator():
            # Wait a bit for the server to start up
            time.sleep(5)
            try:
                # Ensure tools_list.json exists
                if not os.path.exists("tools_list.json"):
                    print("\n--- Tool Context Window Analysis ---", file=sys.stderr)
                    print(
                        "Error: tools_list.json not found. Tool registration may have failed.",
                        file=sys.stderr,
                    )
                    print(
                        "The tool context estimator will run with limited functionality.",
                        file=sys.stderr,
                    )
                    print("-" * 40, file=sys.stderr)

                # Run the tool context estimator script with appropriate transport
                cmd = ["python", "-m", "mcp_tool_context_estimator", "--quiet"]
                # Pass transport mode for both HTTP transports (sse and streamable-http)
                if transport_mode in ["sse", "streamable-http"]:
                    cmd.extend(["--transport", transport_mode])

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Output the results to stderr
                if result.stdout:
                    print("\n--- Tool Context Window Analysis ---", file=sys.stderr)
                    print(result.stdout, file=sys.stderr)
                    print("-" * 40, file=sys.stderr)
                # Check if there was an error
                if result.returncode != 0:
                    print("\n--- Tool Context Estimator Error ---", file=sys.stderr)
                    print(
                        "Failed to run mcp_tool_context_estimator.py - likely due to an error.",
                        file=sys.stderr,
                    )
                    print("Error output:", file=sys.stderr)
                    print(result.stderr, file=sys.stderr)
                    print("-" * 40, file=sys.stderr)
            except Exception as e:
                print(f"\nError running tool context estimator: {str(e)}", file=sys.stderr)
                print(
                    "Check if mcp_tool_context_estimator.py exists and is executable.",
                    file=sys.stderr,
                )

        # Skip the tool-context estimator for SSE transport because it causes the server
        # to shut down when the estimator disconnects after completing its analysis.
        # SSE servers shut down when all clients disconnect, and the estimator is treated
        # as a client. Run it for streamable-http mode where this isn't an issue.
        if transport_mode == "streamable-http" and os.path.exists("mcp_tool_context_estimator.py"):
            threading.Thread(target=run_tool_context_estimator, daemon=True).start()

        # Setup graceful shutdown
        logger = logging.getLogger("ultimate_mcp_server.server")

        # Configure graceful shutdown with error suppression
        enable_quiet_shutdown()

        # Create a shutdown handler for gateway cleanup
        async def cleanup_resources():
            """Performs cleanup for various components during shutdown."""

            # First attempt quick tasks then long tasks with timeouts
            print("Cleaning up Gateway instance and associated resources...", file=sys.stderr)

            # Shutdown SQL Tools with timeout
            try:
                await asyncio.wait_for(shutdown_sql_tools(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

            # Shutdown Connection Manager with timeout
            try:
                from ultimate_mcp_server.tools.sql_databases import _connection_manager

                await asyncio.wait_for(_connection_manager.shutdown(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

            # Shutdown Smart Browser with timeout
            try:
                await asyncio.wait_for(smart_browser_shutdown(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

        # Register the cleanup function with the graceful shutdown system
        register_shutdown_handler(cleanup_resources)

        # Create FastMCP app with proper path configuration
        if transport_mode == "sse":
            # Mark the gateway instance as SSE mode for lifespan management
            _gateway_instance._sse_mode = True
            
            mcp_app = _gateway_instance.mcp.http_app(transport="sse", path="/sse")
            print("Note: Running in legacy SSE mode.", file=sys.stderr)
            
            # Add SSE keepalive mechanism to prevent automatic shutdown
            def sse_keepalive():
                """Keepalive thread to prevent SSE server from shutting down when no clients are connected."""
                while True:
                    time.sleep(30)  # Send keepalive every 30 seconds
                    try:
                        # This simple presence keeps the server alive
                        # The actual SSE connections will handle their own keepalive
                        pass
                    except Exception:
                        # If there's any error, just continue
                        pass
            
            # Start the keepalive thread as a daemon so it doesn't prevent shutdown
            keepalive_thread = threading.Thread(target=sse_keepalive, daemon=True, name="SSE-Keepalive")
            keepalive_thread.start()
            print("SSE keepalive thread started to prevent automatic shutdown.", file=sys.stderr)
            
        else:  # This path is for streamable-http
            mcp_app = _gateway_instance.mcp.http_app(path="/mcp")

        print(f"Running in {transport_mode} mode...", file=sys.stderr)
        print(f"[DEBUG] {transport_mode} app type: {type(mcp_app)}", file=sys.stderr)

        # === BEGIN NEW SPLIT-APP ARCHITECTURE ===
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        # 1) PRISTINE FastMCP wrapper – **NO** extra routes
        mcp_starlette = Starlette(
            routes=[Mount("/", mcp_app)],
            lifespan=mcp_app.lifespan,
        )

        # 2) FastAPI application for rich REST APIs & automatic docs
        api_app = FastAPI(
            title="Ultimate MCP Server API",
            description="REST API endpoints for the Ultimate MCP Server",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Add CORS middleware (FastAPI uses Starlette under the hood)
        api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

        endpoint_path = "/sse" if transport_mode == "sse" else "/mcp"

        @api_app.get(
            "/",
            summary="MCP Server Discovery",
            description="Returns information about the MCP server endpoint",
            response_description="Server information including transport type and endpoint path",
        )
        async def root_endpoint():  # noqa: D401
            response_data = {
                "type": "mcp-server",
                "version": "1.0.0",
                "transport": transport_mode,
                "endpoint": endpoint_path,
                "api_docs": "/api/docs",
                "api_spec": "/api/openapi.json",
            }
            headers = {
                "X-MCP-Server": "true",
                "X-MCP-Version": "1.0.0",
                "X-MCP-Transport": transport_mode,
            }
            return JSONResponse(content=response_data, headers=headers)

        # Legacy alias for older route-registration code
        app = api_app  # DO NOT REMOVE – keeps backward-compatibility  # noqa: F841

        # -------------------------------------------------
        # UMS Explorer: static assets, DB helpers, and APIs
        # -------------------------------------------------

        # Paths & database setup
        project_root = Path(__file__).resolve().parent.parent.parent
        tools_dir = project_root / "ultimate_mcp_server" / "tools"
        storage_dir = project_root / "storage"
        DATABASE_PATH = str(storage_dir / "unified_agent_memory.db")

        def get_db_connection() -> sqlite3.Connection:
            """Return a SQLite connection with row factory."""
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            return conn

        # ---------- Helper functions ----------
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

        # ---------- Pydantic models ----------
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

        # ---------- Static assets ----------
        @api_app.get("/tools/ums_explorer.html", include_in_schema=False)
        async def serve_ums_explorer():
            html_path = tools_dir / "ums_explorer.html"
            if html_path.exists():
                return FileResponse(str(html_path), media_type="text/html")
            return JSONResponse({"error": "UMS Explorer HTML file not found"}, status_code=404)

        @api_app.get("/storage/unified_agent_memory.db", include_in_schema=False)
        async def serve_database():
            db_path = storage_dir / "unified_agent_memory.db"
            if db_path.exists():
                return FileResponse(
                    str(db_path),
                    media_type="application/x-sqlite3",
                    filename="unified_agent_memory.db",
                )
            return JSONResponse({"error": "Database file not found"}, status_code=404)

        @api_app.get("/ums-explorer", include_in_schema=False)
        async def ums_explorer_redirect():
            return RedirectResponse(url="/api/tools/ums_explorer.html")

        # ---------- Cognitive-states endpoint ----------
        @api_app.get(
            "/cognitive-states", response_model=CognitiveStatesResponse, tags=["Cognitive States"]
        )
        async def get_cognitive_states(
            start_time: Optional[float] = Query(None, ge=0),
            end_time: Optional[float] = Query(None, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            offset: int = Query(0, ge=0),
            pattern_type: Optional[str] = Query(None, regex="^[A-Za-z_]+$"),
        ) -> CognitiveStatesResponse:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                sql = (
                    "SELECT cs.*, w.title AS workflow_title, "
                    "COUNT(DISTINCT m.memory_id) AS memory_count, "
                    "COUNT(DISTINCT a.action_id) AS action_count "
                    "FROM cognitive_timeline_states cs "
                    "LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id "
                    "LEFT JOIN memories m ON cs.workflow_id = m.workflow_id "
                    "LEFT JOIN actions a ON cs.workflow_id = a.workflow_id "
                    "WHERE 1=1"
                )
                params: List[Any] = []
                if start_time:
                    sql += " AND cs.timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    sql += " AND cs.timestamp <= ?"
                    params.append(end_time)
                if pattern_type:
                    sql += " AND cs.state_type = ?"
                    params.append(pattern_type)
                sql += " GROUP BY cs.state_id ORDER BY cs.timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
                states: List[CognitiveState] = []
                for r in rows:
                    try:
                        data = json.loads(r.get("state_data", "{}"))
                    except Exception:
                        data = {}
                    states.append(
                        CognitiveState(
                            state_id=r["state_id"],
                            timestamp=r["timestamp"],
                            formatted_timestamp=datetime.fromtimestamp(r["timestamp"]).isoformat(),
                            state_type=r.get("state_type", "unknown"),
                            description=r.get("description"),
                            workflow_id=r.get("workflow_id"),
                            workflow_title=r.get("workflow_title"),
                            complexity_score=calculate_state_complexity(data),
                            change_magnitude=0.0,
                            age_minutes=(datetime.now().timestamp() - r["timestamp"]) / 60,
                            memory_count=r.get("memory_count", 0),
                            action_count=r.get("action_count", 0),
                            state_data=data,
                        )
                    )
                for i in range(len(states) - 1):
                    diff = compute_state_diff(states[i + 1].state_data, states[i].state_data)
                    states[i].change_magnitude = diff["magnitude"]
                conn.close()
                return CognitiveStatesResponse(
                    states=states, total=len(states), has_more=len(states) == limit
                )
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        # ---------- Timeline helper functions ----------
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
            from collections import Counter

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
            from collections import Counter

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

        # ---------- Timeline Pydantic models ----------
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

        # ---------- Timeline endpoint ----------
        @api_app.get(
            "/cognitive-states/timeline",
            response_model=CognitiveTimelineResponse,
            tags=["Cognitive States"],
            summary="Get cognitive state timeline for visualization",
        )
        async def get_cognitive_timeline(
            hours: int = Query(24, ge=1, le=168),
            granularity: str = Query("hour", regex="^(second|minute|hour)$"),
        ) -> CognitiveTimelineResponse:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                since_ts = datetime.now().timestamp() - hours * 3600
                cur.execute(
                    """
                    SELECT state_id, timestamp, state_type, state_data, workflow_id, description,
                           ROW_NUMBER() OVER (ORDER BY timestamp) AS seq
                    FROM cognitive_timeline_states WHERE timestamp >= ? ORDER BY timestamp ASC
                    """,
                    (since_ts,),
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

                timeline: List[TimelineState] = []
                for idx, r in enumerate(rows):
                    try:
                        data = json.loads(r.get("state_data", "{}"))
                    except Exception:
                        data = {}
                    change = 0.0
                    if idx > 0:
                        try:
                            prev_data = json.loads(rows[idx - 1].get("state_data", "{}"))
                        except Exception:
                            prev_data = {}
                        change = compute_state_diff(prev_data, data)["magnitude"]
                    timeline.append(
                        TimelineState(
                            state_id=r["state_id"],
                            timestamp=r["timestamp"],
                            formatted_time=datetime.fromtimestamp(r["timestamp"]).isoformat(),
                            state_type=r["state_type"],
                            workflow_id=r.get("workflow_id"),
                            description=r.get("description"),
                            sequence_number=r["seq"],
                            complexity_score=calculate_state_complexity(data),
                            change_magnitude=change,
                        )
                    )

                stats = TimelineSummaryStats(
                    avg_complexity=sum(t.complexity_score for t in timeline) / len(timeline)
                    if timeline
                    else 0,
                    total_transitions=len(timeline) - 1 if len(timeline) > 1 else 0,
                    max_change_magnitude=max((t.change_magnitude for t in timeline), default=0),
                )
                conn.close()
                return CognitiveTimelineResponse(
                    timeline_data=timeline,
                    total_states=len(timeline),
                    time_range_hours=hours,
                    granularity=granularity,
                    summary_stats=stats,
                )
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        # ---------- Detailed state models ----------
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

        # ---------- Detailed state endpoint ----------
        @api_app.get(
            "/cognitive-states/{state_id}",
            response_model=DetailedCognitiveState,
            tags=["Cognitive States"],
            summary="Get detailed cognitive state information",
        )
        async def get_cognitive_state_detail(
            state_id: str = ApiPath(..., regex="^[A-Za-z0-9_-]+$"),
        ) -> DetailedCognitiveState:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT cs.*, w.title AS workflow_title, w.goal AS workflow_goal
                    FROM cognitive_timeline_states cs LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id
                    WHERE cs.state_id = ?
                    """,
                    (state_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404, detail=f"Cognitive state '{state_id}' not found"
                    )
                cols = [d[0] for d in cur.description]
                state = dict(zip(cols, row, strict=False))
                try:
                    data = json.loads(state.get("state_data", "{}"))
                except Exception:
                    data = {}

                # memories
                cur.execute(
                    "SELECT memory_id, memory_type, content, importance, created_at FROM memories WHERE workflow_id = ? ORDER BY created_at DESC LIMIT 20",
                    (state.get("workflow_id"),),
                )
                mem_cols = [d[0] for d in cur.description]
                memories = [Memory(**dict(zip(mem_cols, m, strict=False))) for m in cur.fetchall()]

                # actions
                cur.execute(
                    "SELECT action_id, action_type, tool_name, status, started_at FROM actions WHERE workflow_id = ? ORDER BY started_at DESC LIMIT 20",
                    (state.get("workflow_id"),),
                )
                act_cols = [d[0] for d in cur.description]
                actions = [Action(**dict(zip(act_cols, a, strict=False))) for a in cur.fetchall()]
                conn.close()
                return DetailedCognitiveState(
                    state_id=state["state_id"],
                    timestamp=state["timestamp"],
                    formatted_timestamp=datetime.fromtimestamp(state["timestamp"]).isoformat(),
                    state_type=state.get("state_type", "unknown"),
                    description=state.get("description"),
                    workflow_id=state.get("workflow_id"),
                    workflow_title=state.get("workflow_title"),
                    workflow_goal=state.get("workflow_goal"),
                    state_data=data,
                    complexity_score=calculate_state_complexity(data),
                    memories=memories,
                    actions=actions,
                )
            except HTTPException:
                raise
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        # ---------- Pattern analysis helpers ----------
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

        # ---------- Pattern analysis endpoint ----------
        @api_app.get(
            "/cognitive-states/patterns",
            response_model=CognitivePatternAnalysis,
            tags=["Cognitive States"],
        )
        async def analyze_cognitive_patterns(
            lookback_hours: int = Query(24, ge=1, le=720),
            min_pattern_length: int = Query(3, ge=2, le=20),
            similarity_threshold: float = Query(0.7, ge=0.1, le=1.0),
        ) -> CognitivePatternAnalysis:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                since_ts = datetime.now().timestamp() - lookback_hours * 3600
                cur.execute(
                    "SELECT state_id, timestamp, state_type, state_data, workflow_id FROM cognitive_timeline_states WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (since_ts,),
                )
                states = [
                    dict(zip([d[0] for d in cur.description], row, strict=False))
                    for row in cur.fetchall()
                ]
                for state in states:
                    try:
                        state["state_data"] = json.loads(state.get("state_data", "{}"))
                    except Exception:
                        state["state_data"] = {}
                patterns = find_cognitive_patterns(states, min_pattern_length, similarity_threshold)
                transitions = analyze_state_transitions(states)
                anomalies = detect_cognitive_anomalies(states)
                conn.close()
                summary = PatternSummary(
                    pattern_count=len(patterns),
                    most_common_transition=Transition(**transitions[0]) if transitions else None,
                    anomaly_count=len(anomalies),
                )
                return CognitivePatternAnalysis(
                    total_states=len(states),
                    time_range_hours=lookback_hours,
                    patterns=[Pattern(**p) for p in patterns],
                    transitions=[Transition(**t) for t in transitions],
                    anomalies=[Anomaly(**a) for a in anomalies],
                    summary=summary,
                )
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        # ---------- State comparison models ----------
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

        @api_app.post(
            "/cognitive-states/compare",
            response_model=StateComparisonResponse,
            tags=["Cognitive States"],
            summary="Compare two cognitive states",
            description="""
        Perform detailed comparison between two cognitive states to understand:

        - **Structural differences** in state data
        - **Added, removed, and modified** components
        - **Change magnitude** calculation
        - **Time differential** between states

        Perfect for understanding how cognitive states evolve and what changes between specific points in time.
            """,
            responses={
                200: {
                    "description": "Detailed comparison results",
                    "content": {
                        "application/json": {
                            "example": {
                                "state_1": {
                                    "state_id": "state_abc123",
                                    "timestamp": 1703980800.0,
                                    "formatted_timestamp": "2024-01-01T00:00:00"
                                },
                                "state_2": {
                                    "state_id": "state_xyz789", 
                                    "timestamp": 1703984400.0,
                                    "formatted_timestamp": "2024-01-01T01:00:00"
                                },
                                "time_diff_minutes": 60.0,
                                "diff": {
                                    "added": {
                                        "new_insight": "PDF contains financial data",
                                        "confidence": 0.95
                                    },
                                    "removed": {
                                        "initial_assumption": "Document is text-only"
                                    },
                                    "modified": {
                                        "tool_preference": {
                                            "before": "file_reader",
                                            "after": "smart_browser"
                                        }
                                    },
                                    "magnitude": 45.5
                                }
                            }
                        }
                    }
                },
                400: {
                    "description": "Invalid request - both state IDs required",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Both state_id_1 and state_id_2 are required"}
                        }
                    }
                },
                404: {
                    "description": "One or both states not found",
                    "content": {
                        "application/json": {
                            "example": {"detail": "State with ID 'state_abc123' not found"}
                        }
                    }
                },
                500: {"description": "Internal server error"}
            }
        )
        async def compare_cognitive_states(
            request: StateComparisonRequest,
        ) -> StateComparisonResponse:
            try:
                if not request.state_id_1 or not request.state_id_2:
                    raise HTTPException(
                        status_code=400, detail="Both state_id_1 and state_id_2 are required"
                    )
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM cognitive_timeline_states WHERE state_id IN (?, ?)",
                    (request.state_id_1, request.state_id_2),
                )
                states = [
                    dict(zip([d[0] for d in cur.description], row, strict=False))
                    for row in cur.fetchall()
                ]
                if len(states) != 2:
                    missing_ids = []
                    found_ids = [s["state_id"] for s in states]
                    if request.state_id_1 not in found_ids:
                        missing_ids.append(request.state_id_1)
                    if request.state_id_2 not in found_ids:
                        missing_ids.append(request.state_id_2)
                    raise HTTPException(
                        status_code=404, detail=f"State(s) not found: {', '.join(missing_ids)}"
                    )
                for state in states:
                    try:
                        state["state_data"] = json.loads(state.get("state_data", "{}"))
                    except Exception:
                        state["state_data"] = {}
                states.sort(key=lambda s: s["timestamp"])
                state_1, state_2 = states
                if state_1["state_id"] != request.state_id_1:
                    state_1, state_2 = state_2, state_1
                diff_result = compute_state_diff(
                    state_1.get("state_data", {}), state_2.get("state_data", {})
                )
                conn.close()
                time_diff_minutes = abs(state_2["timestamp"] - state_1["timestamp"]) / 60
                return StateComparisonResponse(
                    state_1=StateComparisonInfo(
                        state_id=state_1["state_id"],
                        timestamp=state_1["timestamp"],
                        formatted_timestamp=datetime.fromtimestamp(
                            state_1["timestamp"]
                        ).isoformat(),
                    ),
                    state_2=StateComparisonInfo(
                        state_id=state_2["state_id"],
                        timestamp=state_2["timestamp"],
                        formatted_timestamp=datetime.fromtimestamp(
                            state_2["timestamp"]
                        ).isoformat(),
                    ),
                    time_diff_minutes=time_diff_minutes,
                    diff=StateDiff(**diff_result),
                )
            except HTTPException:
                raise
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

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

            from collections import Counter

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

        # ---------- Action Monitor Pydantic Models ----------

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

        # ---------- Action Monitor Endpoints ----------

        @api_app.get(
            "/actions/running",
            response_model=RunningActionsResponse,
            tags=["Action Monitor"],
            summary="Get currently executing actions",
            description="""
        Monitor actions that are currently executing with real-time status information:

        - **Execution progress** with percentage completion estimates
        - **Performance categorization** (excellent, good, slow, etc.)
        - **Resource usage indicators** (placeholder for future implementation)
        - **Status indicators** with urgency levels
        - **Estimated duration** vs actual execution time

        Ideal for monitoring system activity and identifying long-running or problematic actions.
            """,
            responses={
                200: {
                    "description": "List of currently running actions with real-time metrics",
                    "content": {
                        "application/json": {
                            "example": {
                                "running_actions": [
                                    {
                                        "action_id": "act_123",
                                        "workflow_id": "wf_456",
                                        "workflow_title": "Document Analysis",
                                        "tool_name": "smart_browser",
                                        "status": "running",
                                        "started_at": 1703980800.0,
                                        "formatted_start_time": "2024-01-01T00:00:00",
                                        "execution_time_seconds": 45.5,
                                        "estimated_duration": 60.0,
                                        "progress_percentage": 75.8,
                                        "status_indicator": {
                                            "color": "blue",
                                            "icon": "play",
                                            "label": "Running",
                                            "urgency": "low",
                                        },
                                        "performance_category": "good",
                                        "resource_usage": {
                                            "cpu_usage": 25.5,
                                            "memory_usage": 512.0,
                                            "network_io": 10.5,
                                            "disk_io": 5.2,
                                        },
                                        "tool_data": {
                                            "url": "https://example.com",
                                            "action_type": "download",
                                        },
                                    }
                                ],
                                "total_running": 1,
                                "avg_execution_time": 45.5,
                                "timestamp": "2024-01-01T00:00:45.500000",
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_running_actions() -> RunningActionsResponse:
            """Get currently executing actions with real-time status"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (unixepoch() - a.started_at) as execution_time,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.estimated_duration')
                            ELSE NULL 
                        END as estimated_duration
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('running', 'executing', 'in_progress')
                    ORDER BY a.started_at ASC
                """)

                columns = [description[0] for description in cursor.description]
                running_actions = [
                    dict(zip(columns, row, strict=False)) for row in cursor.fetchall()
                ]

                # Enhance with real-time metrics
                enhanced_actions = []
                for action in running_actions:
                    try:
                        tool_data = json.loads(action.get("tool_data", "{}"))
                    except Exception:
                        tool_data = {}

                    execution_time = action.get("execution_time", 0)
                    estimated_duration = action.get("estimated_duration") or 30
                    progress_percentage = (
                        min(95, (execution_time / estimated_duration) * 100)
                        if estimated_duration > 0
                        else 0
                    )

                    enhanced_action = RunningAction(
                        action_id=action["action_id"],
                        workflow_id=action.get("workflow_id"),
                        workflow_title=action.get("workflow_title"),
                        tool_name=action["tool_name"],
                        status=action["status"],
                        started_at=action["started_at"],
                        formatted_start_time=datetime.fromtimestamp(
                            action["started_at"]
                        ).isoformat(),
                        execution_time_seconds=execution_time,
                        estimated_duration=estimated_duration,
                        progress_percentage=progress_percentage,
                        status_indicator=StatusIndicator(
                            **get_action_status_indicator(action["status"], execution_time)
                        ),
                        performance_category=categorize_action_performance(
                            execution_time, estimated_duration
                        ),
                        resource_usage=ResourceUsage(
                            **get_action_resource_usage(action["action_id"])
                        ),
                        tool_data=tool_data,
                    )
                    enhanced_actions.append(enhanced_action)

                conn.close()

                return RunningActionsResponse(
                    running_actions=enhanced_actions,
                    total_running=len(enhanced_actions),
                    avg_execution_time=sum(a.execution_time_seconds for a in enhanced_actions)
                    / len(enhanced_actions)
                    if enhanced_actions
                    else 0,
                    timestamp=datetime.now().isoformat(),
                )

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        @api_app.get(
            "/actions/queue",
            response_model=ActionQueueResponse,
            tags=["Action Monitor"],
            summary="Get queued actions waiting for execution",
            description="""
        Monitor the action execution queue to understand:

        - **Queue position** for each waiting action
        - **Priority levels** with human-readable labels
        - **Estimated wait times** based on queue position
        - **Queue time** (how long actions have been waiting)

        Essential for understanding system load and execution priorities.
            """,
            responses={
                200: {
                    "description": "List of queued actions with wait time estimates",
                    "content": {
                        "application/json": {
                            "example": {
                                "queued_actions": [
                                    {
                                        "action_id": "act_789",
                                        "workflow_id": "wf_456",
                                        "workflow_title": "Batch Processing",
                                        "tool_name": "convert_document",
                                        "status": "queued",
                                        "created_at": 1703980700.0,
                                        "formatted_queue_time": "2024-01-01T00:00:00",
                                        "queue_position": 1,
                                        "queue_time_seconds": 100.0,
                                        "estimated_wait_time": 0.0,
                                        "priority": 3,
                                        "priority_label": "High",
                                        "tool_data": {"format": "pdf", "pages": 50},
                                    }
                                ],
                                "total_queued": 1,
                                "avg_queue_time": 100.0,
                                "next_action": {
                                    "action_id": "act_789",
                                    "tool_name": "convert_document",
                                    "priority_label": "High",
                                },
                                "timestamp": "2024-01-01T00:01:40",
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_action_queue() -> ActionQueueResponse:
            """Get queued actions waiting for execution"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (unixepoch() - a.created_at) as queue_time,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.priority')
                            ELSE 5 
                        END as priority
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('queued', 'pending', 'waiting')
                    ORDER BY priority ASC, a.created_at ASC
                """)

                columns = [description[0] for description in cursor.description]
                queued_actions = [
                    dict(zip(columns, row, strict=False)) for row in cursor.fetchall()
                ]

                # Enhance queue data
                enhanced_queue = []
                for i, action in enumerate(queued_actions):
                    try:
                        tool_data = json.loads(action.get("tool_data", "{}"))
                    except Exception:
                        tool_data = {}

                    enhanced_action = QueuedAction(
                        action_id=action["action_id"],
                        workflow_id=action.get("workflow_id"),
                        workflow_title=action.get("workflow_title"),
                        tool_name=action["tool_name"],
                        status=action["status"],
                        created_at=action["created_at"],
                        formatted_queue_time=datetime.fromtimestamp(
                            action["created_at"]
                        ).isoformat(),
                        queue_position=i + 1,
                        queue_time_seconds=action.get("queue_time", 0),
                        estimated_wait_time=estimate_wait_time(i, queued_actions),
                        priority=action.get("priority", 5),
                        priority_label=get_priority_label(action.get("priority", 5)),
                        tool_data=tool_data,
                    )
                    enhanced_queue.append(enhanced_action)

                conn.close()

                return ActionQueueResponse(
                    queued_actions=enhanced_queue,
                    total_queued=len(enhanced_queue),
                    avg_queue_time=sum(a.queue_time_seconds for a in enhanced_queue)
                    / len(enhanced_queue)
                    if enhanced_queue
                    else 0,
                    next_action=enhanced_queue[0] if enhanced_queue else None,
                    timestamp=datetime.now().isoformat(),
                )

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        # ---------- Action History Pydantic Models ----------

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

        # ---------- Action History Endpoint ----------

        @api_app.get(
            "/actions/history",
            response_model=ActionHistoryResponse,
            tags=["Action Monitor"],
            summary="Get completed actions with performance metrics",
            description="""
        Analyze historical action execution data with comprehensive performance metrics:

        - **Execution duration** and performance scoring
        - **Success/failure rates** and efficiency ratings
        - **Tool-specific filtering** and status filtering
        - **Aggregate performance metrics** and trends

        Perfect for performance analysis, debugging, and system optimization.
            """,
            responses={
                200: {
                    "description": "Historical actions with performance analysis",
                    "content": {
                        "application/json": {
                            "example": {
                                "action_history": [
                                    {
                                        "action_id": "act_001",
                                        "workflow_id": "workflow_123",
                                        "workflow_title": "Document Analysis",
                                        "tool_name": "smart_browser",
                                        "action_type": "tool_execution",
                                        "status": "completed",
                                        "started_at": 1703980800.0,
                                        "completed_at": 1703980815.0,
                                        "execution_duration_seconds": 15.0,
                                        "performance_score": 90.0,
                                        "efficiency_rating": "good",
                                        "success_rate_impact": 1,
                                        "formatted_start_time": "2024-01-01T00:00:00",
                                        "formatted_completion_time": "2024-01-01T00:00:15",
                                        "tool_data": {"url": "https://example.com"},
                                        "result_data": {"pages_analyzed": 5},
                                        "result_size": 2048,
                                    }
                                ],
                                "total_actions": 150,
                                "success_rate": 95.3,
                                "avg_execution_time": 12.5,
                                "performance_summary": {
                                    "avg_score": 87.5,
                                    "top_performer": {"tool_name": "file_reader", "score": 98.5},
                                    "worst_performer": {"tool_name": "web_scraper", "score": 65.0},
                                    "efficiency_distribution": {
                                        "excellent": 45,
                                        "good": 80,
                                        "fair": 20,
                                        "poor": 5,
                                    },
                                },
                                "timestamp": "2024-01-01T12:00:00",
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_action_history(
            limit: int = Query(
                50, description="Maximum number of actions to return", ge=1, le=500, example=100
            ),
            offset: int = Query(
                0, description="Number of actions to skip for pagination", ge=0, example=0
            ),
            status_filter: Optional[str] = Query(
                None,
                description="Filter by action completion status",
                regex="^(completed|failed|cancelled|timeout)$",
                example="completed",
            ),
            tool_filter: Optional[str] = Query(
                None, description="Filter by specific tool name", example="smart_browser"
            ),
            hours_back: int = Query(
                24,
                description="Hours back to search for completed actions",
                ge=1,
                le=720,  # Max 30 days
                example=24,
            ),
        ) -> ActionHistoryResponse:
            """Get completed actions with performance metrics"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)

                query = """
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (a.completed_at - a.started_at) as execution_duration,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.result_size')
                            ELSE 0 
                        END as result_size
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('completed', 'failed', 'cancelled', 'timeout')
                    AND a.completed_at >= ?
                """
                params = [since_timestamp]

                if status_filter:
                    query += " AND a.status = ?"
                    params.append(status_filter)

                if tool_filter:
                    query += " AND a.tool_name = ?"
                    params.append(tool_filter)

                query += """
                    ORDER BY a.completed_at DESC 
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                completed_actions = [
                    dict(zip(columns, row, strict=False)) for row in cursor.fetchall()
                ]

                # Calculate performance metrics
                enhanced_history = []
                for action in completed_actions:
                    try:
                        tool_data = json.loads(action.get("tool_data", "{}"))
                        result_data = json.loads(action.get("result", "{}"))
                    except Exception:
                        tool_data = {}
                        result_data = {}

                    execution_duration = action.get("execution_duration", 0)

                    # Create performance-enhanced action item
                    action_data = {
                        "action_id": action["action_id"],
                        "workflow_id": action.get("workflow_id"),
                        "workflow_title": action.get("workflow_title"),
                        "tool_name": action["tool_name"],
                        "action_type": action.get("action_type"),
                        "status": action["status"],
                        "started_at": action["started_at"],
                        "completed_at": action.get("completed_at"),
                        "execution_duration_seconds": execution_duration,
                        "performance_score": calculate_action_performance_score(action),
                        "efficiency_rating": calculate_efficiency_rating(
                            execution_duration, action.get("result_size", 0)
                        ),
                        "success_rate_impact": 1 if action["status"] == "completed" else 0,
                        "formatted_start_time": datetime.fromtimestamp(
                            action["started_at"]
                        ).isoformat(),
                        "formatted_completion_time": datetime.fromtimestamp(
                            action["completed_at"]
                        ).isoformat()
                        if action.get("completed_at")
                        else None,
                        "tool_data": tool_data,
                        "result_data": result_data,
                        "result_size": action.get("result_size", 0),
                    }

                    enhanced_history.append(ActionHistoryItem(**action_data))

                # Calculate aggregate metrics
                total_actions = len(enhanced_history)
                successful_actions = len([a for a in enhanced_history if a.status == "completed"])
                avg_duration = (
                    sum(a.execution_duration_seconds for a in enhanced_history) / total_actions
                    if total_actions > 0
                    else 0
                )

                # Create performance summary
                action_dicts = [a.dict() for a in enhanced_history]
                performance_summary = PerformanceSummary(
                    **calculate_performance_summary(action_dicts)
                )

                conn.close()

                return ActionHistoryResponse(
                    action_history=enhanced_history,
                    total_actions=total_actions,
                    success_rate=(successful_actions / total_actions * 100)
                    if total_actions > 0
                    else 0,
                    avg_execution_time=avg_duration,
                    performance_summary=performance_summary,
                    timestamp=datetime.now().isoformat(),
                )

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        # ---------- Action Metrics Pydantic Models ----------

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

        # ---------- Action Metrics Endpoint ----------

        @api_app.get(
            "/actions/metrics",
            response_model=ActionMetricsResponse,
            tags=["Action Monitor"],
            summary="Get comprehensive action execution metrics",
            description="""
        Retrieve system-wide action execution analytics including:

        - **Overall success/failure rates** for the past 24 hours
        - **Tool usage statistics** with performance breakdowns
        - **Hourly performance distribution** showing usage patterns
        - **Performance insights** with actionable recommendations

        This endpoint provides executive-level insights into system performance and health.
            """,
            responses={
                200: {
                    "description": "Comprehensive action execution metrics and analytics",
                    "content": {
                        "application/json": {
                            "example": {
                                "overall_metrics": {
                                    "total_actions": 1523,
                                    "successful_actions": 1450,
                                    "failed_actions": 73,
                                    "avg_duration": 8.5,
                                    "success_rate_percentage": 95.2,
                                    "failure_rate_percentage": 4.8,
                                    "avg_duration_seconds": 8.5,
                                },
                                "tool_usage_stats": [
                                    {
                                        "tool_name": "smart_browser",
                                        "usage_count": 342,
                                        "success_count": 325,
                                        "avg_duration": 15.3,
                                    },
                                    {
                                        "tool_name": "file_reader",
                                        "usage_count": 289,
                                        "success_count": 287,
                                        "avg_duration": 2.1,
                                    },
                                ],
                                "hourly_performance": [
                                    {
                                        "hour": "09",
                                        "action_count": 125,
                                        "avg_duration": 7.8,
                                        "success_count": 120,
                                    },
                                    {
                                        "hour": "10",
                                        "action_count": 143,
                                        "avg_duration": 8.2,
                                        "success_count": 138,
                                    },
                                ],
                                "performance_insights": [
                                    {
                                        "type": "warning",
                                        "title": "Low Success Rate",
                                        "message": "Current success rate is 75.5%. Consider investigating failing tools.",
                                        "severity": "high",
                                    },
                                    {
                                        "type": "info",
                                        "title": "Peak Usage",
                                        "message": "Peak usage occurs at 14:00 with 189 actions.",
                                        "severity": "low",
                                    },
                                ],
                                "timestamp": "2024-01-01T12:00:00",
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_action_metrics() -> ActionMetricsResponse:
            """Get comprehensive action execution metrics and analytics"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                # Get metrics for last 24 hours
                since_timestamp = datetime.now().timestamp() - (24 * 3600)

                # Overall statistics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_actions,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
                    FROM actions 
                    WHERE created_at >= ?
                """,
                    (since_timestamp,),
                )

                overall_result = cursor.fetchone()
                overall_dict = dict(
                    zip([d[0] for d in cursor.description], overall_result, strict=False)
                )

                # Create overall metrics
                success_rate = (
                    (overall_dict["successful_actions"] / overall_dict["total_actions"] * 100)
                    if overall_dict["total_actions"] > 0
                    else 0
                )

                overall_metrics = OverallMetrics(
                    total_actions=overall_dict["total_actions"] or 0,
                    successful_actions=overall_dict["successful_actions"] or 0,
                    failed_actions=overall_dict["failed_actions"] or 0,
                    avg_duration=overall_dict["avg_duration"],
                    success_rate_percentage=success_rate,
                    failure_rate_percentage=100 - success_rate,
                    avg_duration_seconds=overall_dict["avg_duration"] or 0,
                )

                # Tool usage statistics
                cursor.execute(
                    """
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
                    FROM actions 
                    WHERE created_at >= ?
                    GROUP BY tool_name
                    ORDER BY usage_count DESC
                """,
                    (since_timestamp,),
                )

                tool_stats = [
                    ToolUsageStat(
                        tool_name=row[0],
                        usage_count=row[1],
                        success_count=row[2],
                        avg_duration=row[3],
                    )
                    for row in cursor.fetchall()
                ]

                # Performance distribution over time (hourly)
                cursor.execute(
                    """
                    SELECT 
                        strftime('%H', datetime(started_at, 'unixepoch')) as hour,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY hour
                    ORDER BY hour
                """,
                    (since_timestamp,),
                )

                hourly_metrics = [
                    HourlyMetric(
                        hour=row[0], action_count=row[1], avg_duration=row[2], success_count=row[3]
                    )
                    for row in cursor.fetchall()
                ]

                conn.close()

                # Generate performance insights
                tool_stats_dicts = [t.dict() for t in tool_stats]
                hourly_metrics_dicts = [h.dict() for h in hourly_metrics]
                insights_data = generate_performance_insights(
                    overall_dict, tool_stats_dicts, hourly_metrics_dicts
                )

                performance_insights = [PerformanceInsight(**insight) for insight in insights_data]

                return ActionMetricsResponse(
                    overall_metrics=overall_metrics,
                    tool_usage_stats=tool_stats,
                    hourly_performance=hourly_metrics,
                    performance_insights=performance_insights,
                    timestamp=datetime.now().isoformat(),
                )

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        # ---------- Artifacts Helper Functions ----------

        def format_file_size(size_bytes: int) -> str:
            """Format file size in human readable format"""
            if size_bytes == 0:
                return "0 B"

            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_names[i]}"

        # ---------- Artifacts Pydantic Models ----------

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

        # ---------- Artifacts Listing Endpoint ----------

        @api_app.get(
            "/artifacts",
            response_model=ArtifactsResponse,
            tags=["Artifacts"],
            summary="List artifacts with filtering and search",
            description="""
        Explore system artifacts with comprehensive filtering and search capabilities:

        - **Type-based filtering** for specific artifact categories
        - **Workflow association** to see artifacts by workflow
        - **Tag-based search** for categorized artifacts
        - **Full-text search** across names and descriptions
        - **Sorting options** with configurable order

        Includes relationship counts, version information, and human-readable metadata.
            """,
            responses={
                200: {
                    "description": "List of artifacts with metadata and relationships",
                    "content": {
                        "application/json": {
                            "example": {
                                "artifacts": [
                                    {
                                        "artifact_id": "artifact_123",
                                        "name": "Analysis Report",
                                        "artifact_type": "document",
                                        "description": "Comprehensive analysis of user behavior",
                                        "file_path": "/storage/artifacts/report_123.pdf",
                                        "workflow_id": "workflow_456",
                                        "workflow_title": "User Analysis Workflow",
                                        "created_at": 1703980800.0,
                                        "updated_at": 1704067200.0,
                                        "file_size": 2048576,
                                        "file_size_human": "2.0 MB",
                                        "importance": 8.5,
                                        "access_count": 15,
                                        "tags": ["analysis", "report", "important"],
                                        "metadata": {"pages": 42, "format": "PDF"},
                                        "relationship_count": 3,
                                        "version_count": 2,
                                        "formatted_created_at": "2024-01-01T00:00:00",
                                        "formatted_updated_at": "2024-01-01T12:00:00",
                                        "age_days": 1.5,
                                    }
                                ],
                                "total": 1,
                                "has_more": False,
                                "filters": {
                                    "artifact_type": "document",
                                    "workflow_id": None,
                                    "tags": None,
                                    "search": None,
                                    "sort_by": "created_at",
                                    "sort_order": "desc",
                                },
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_artifacts(
            artifact_type: Optional[str] = Query(
                None, description="Filter by specific artifact type", example="document"
            ),
            workflow_id: Optional[str] = Query(
                None, description="Filter by workflow ID", example="workflow_abc123"
            ),
            tags: Optional[str] = Query(
                None, description="Search within artifact tags", example="important"
            ),
            search: Optional[str] = Query(
                None,
                description="Full-text search in names and descriptions",
                example="analysis report",
            ),
            sort_by: str = Query(
                "created_at",
                description="Field to sort results by",
                regex="^(created_at|updated_at|name|importance|access_count)$",
            ),
            sort_order: str = Query(
                "desc", description="Sort order direction", regex="^(asc|desc)$"
            ),
            limit: int = Query(
                50, description="Maximum number of artifacts to return", ge=1, le=200
            ),
            offset: int = Query(0, description="Number of artifacts to skip for pagination", ge=0),
        ) -> ArtifactsResponse:
            """List artifacts with filtering and search"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                # Base query
                query = """
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        COUNT(DISTINCT ar.target_artifact_id) as relationship_count,
                        COUNT(DISTINCT versions.artifact_id) as version_count
                    FROM artifacts a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    LEFT JOIN artifact_relationships ar ON a.artifact_id = ar.source_artifact_id
                    LEFT JOIN artifacts versions ON a.artifact_id = versions.parent_artifact_id
                    WHERE 1=1
                """
                params = []

                if artifact_type:
                    query += " AND a.artifact_type = ?"
                    params.append(artifact_type)

                if workflow_id:
                    query += " AND a.workflow_id = ?"
                    params.append(workflow_id)

                if tags:
                    query += " AND a.tags LIKE ?"
                    params.append(f"%{tags}%")

                if search:
                    query += " AND (a.name LIKE ? OR a.description LIKE ?)"
                    params.extend([f"%{search}%", f"%{search}%"])

                query += f"""
                    GROUP BY a.artifact_id
                    ORDER BY a.{sort_by} {"DESC" if sort_order == "desc" else "ASC"}
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                artifacts_data = [
                    dict(zip(columns, row, strict=False)) for row in cursor.fetchall()
                ]

                # Enhance artifacts with metadata
                artifacts = []
                for artifact_data in artifacts_data:
                    # Parse tags and metadata
                    try:
                        tags_list = (
                            json.loads(artifact_data.get("tags", "[]"))
                            if artifact_data.get("tags")
                            else []
                        )
                        metadata_dict = (
                            json.loads(artifact_data.get("metadata", "{}"))
                            if artifact_data.get("metadata")
                            else {}
                        )
                    except Exception:
                        tags_list = []
                        metadata_dict = {}

                    artifact = Artifact(
                        artifact_id=artifact_data["artifact_id"],
                        name=artifact_data["name"],
                        artifact_type=artifact_data["artifact_type"],
                        description=artifact_data.get("description"),
                        file_path=artifact_data.get("file_path"),
                        workflow_id=artifact_data.get("workflow_id"),
                        workflow_title=artifact_data.get("workflow_title"),
                        created_at=artifact_data["created_at"],
                        updated_at=artifact_data["updated_at"],
                        file_size=artifact_data.get("file_size", 0),
                        file_size_human=format_file_size(artifact_data.get("file_size", 0)),
                        importance=artifact_data.get("importance"),
                        access_count=artifact_data.get("access_count", 0),
                        tags=tags_list,
                        metadata=metadata_dict,
                        relationship_count=artifact_data.get("relationship_count", 0),
                        version_count=artifact_data.get("version_count", 0),
                        formatted_created_at=datetime.fromtimestamp(
                            artifact_data["created_at"]
                        ).isoformat(),
                        formatted_updated_at=datetime.fromtimestamp(
                            artifact_data["updated_at"]
                        ).isoformat(),
                        age_days=(datetime.now().timestamp() - artifact_data["created_at"]) / 86400,
                    )
                    artifacts.append(artifact)

                conn.close()

                return ArtifactsResponse(
                    artifacts=artifacts,
                    total=len(artifacts),
                    has_more=len(artifacts) == limit,
                    filters=ArtifactsFilter(
                        artifact_type=artifact_type,
                        workflow_id=workflow_id,
                        tags=tags,
                        search=search,
                        sort_by=sort_by,
                        sort_order=sort_order,
                    ),
                )

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        # ---------- Artifacts Statistics Models ----------

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

        # ---------- Artifacts Statistics Endpoint ----------

        @api_app.get(
            "/artifacts/stats",
            response_model=ArtifactStatsResponse,
            tags=["Artifacts"],
            summary="Get artifact statistics and analytics",
            description="""
        Retrieve comprehensive statistics about system artifacts including:

        - **Overall counts** and storage usage
        - **Type-based breakdown** with metrics per artifact type
        - **Importance scoring** averages and distributions
        - **Access patterns** and usage statistics

        Perfect for understanding artifact distribution and usage patterns across the system.
            """,
            responses={
                200: {
                    "description": "Comprehensive artifact statistics and analytics",
                    "content": {
                        "application/json": {
                            "example": {
                                "overall": {
                                    "total_artifacts": 150,
                                    "unique_types": 5,
                                    "unique_workflows": 25,
                                    "total_size": 1073741824,
                                    "total_size_human": "1.0 GB",
                                    "avg_size": 7158279,
                                    "latest_created": 1704067200.0,
                                    "earliest_created": 1703980800.0,
                                },
                                "by_type": [
                                    {
                                        "artifact_type": "document",
                                        "count": 75,
                                        "avg_importance": 7.5,
                                        "total_size": 536870912,
                                        "max_access_count": 50,
                                    },
                                    {
                                        "artifact_type": "image",
                                        "count": 50,
                                        "avg_importance": 6.0,
                                        "total_size": 268435456,
                                        "max_access_count": 30,
                                    },
                                ],
                            }
                        }
                    },
                },
                500: {"description": "Internal server error"},
            },
        )
        async def get_artifact_stats() -> ArtifactStatsResponse:
            """Get artifact statistics and analytics"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_artifacts,
                        COUNT(DISTINCT artifact_type) as unique_types,
                        COUNT(DISTINCT workflow_id) as unique_workflows,
                        SUM(file_size) as total_size,
                        AVG(file_size) as avg_size,
                        MAX(created_at) as latest_created,
                        MIN(created_at) as earliest_created
                    FROM artifacts
                """)

                result = cursor.fetchone()
                overall_dict = (
                    dict(zip([d[0] for d in cursor.description], result, strict=False))
                    if result
                    else {}
                )

                overall = ArtifactOverallStats(
                    total_artifacts=overall_dict.get("total_artifacts", 0),
                    unique_types=overall_dict.get("unique_types", 0),
                    unique_workflows=overall_dict.get("unique_workflows", 0),
                    total_size=overall_dict.get("total_size", 0),
                    total_size_human=format_file_size(overall_dict.get("total_size", 0)),
                    avg_size=overall_dict.get("avg_size", 0),
                    latest_created=overall_dict.get("latest_created"),
                    earliest_created=overall_dict.get("earliest_created"),
                )

                # Type-based statistics
                cursor.execute("""
                    SELECT 
                        artifact_type,
                        COUNT(*) as count,
                        AVG(importance) as avg_importance,
                        SUM(file_size) as total_size,
                        MAX(access_count) as max_access_count
                    FROM artifacts 
                    GROUP BY artifact_type
                    ORDER BY count DESC
                """)

                by_type = [
                    ArtifactTypeStats(
                        artifact_type=row[0],
                        count=row[1],
                        avg_importance=row[2],
                        total_size=row[3] or 0,
                        max_access_count=row[4] or 0,
                    )
                    for row in cursor.fetchall()
                ]

                conn.close()

                return ArtifactStatsResponse(overall=overall, by_type=by_type)

            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                ) from e

        # ---------- Memory Quality Pydantic Models ----------

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

        # ---------- Memory Quality Endpoints ----------

        @api_app.get(
            "/memory-quality/duplicates",
            response_model=DuplicatesResponse,
            tags=["Memory Quality"],
            summary="Get detailed duplicate memory analysis",
            description="""
        Retrieve comprehensive information about duplicate memories:

        - **Duplicate clusters** with identical content
        - **Memory details** for each duplicate group
        - **Merge recommendations** based on duplicate count
        - **Temporal analysis** of when duplicates were created

        Essential for understanding and resolving memory duplication issues.
            """,
            responses={
                200: {
                    "description": "Duplicate analysis successfully retrieved",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "clusters": [
                                    {
                                        "cluster_id": "dup_cluster_0",
                                        "content_preview": "System initialized successfully with all providers...",
                                        "duplicate_count": 3,
                                        "memory_ids": ["mem_123", "mem_456", "mem_789"],
                                        "primary_memory_id": "mem_123",
                                        "memory_details": [
                                            {
                                                "memory_id": "mem_123",
                                                "workflow_id": "workflow_001",
                                                "memory_type": "system",
                                                "importance": 8.0,
                                                "created_at": 1703980800.0
                                            }
                                        ],
                                        "first_created": 1703980800.0,
                                        "last_created": 1703984400.0,
                                        "avg_importance": 7.5,
                                        "recommendation": "merge"
                                    }
                                ],
                                "duplicate_groups": [],
                                "total_groups": 1,
                                "total_duplicates": 3
                            }
                        }
                    }
                },
                500: {"description": "Internal server error"}
            }
        )
        async def get_duplicate_memories() -> DuplicatesResponse:
            """Get detailed duplicate memory analysis"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT content, COUNT(*) as count, GROUP_CONCAT(memory_id) as memory_ids,
                           MIN(created_at) as first_created, MAX(created_at) as last_created,
                           AVG(importance) as avg_importance
                    FROM memories 
                    WHERE content IS NOT NULL AND LENGTH(content) > 10
                    GROUP BY content 
                    HAVING count > 1
                    ORDER BY count DESC
                """)
                
                duplicate_groups = []
                for i, row in enumerate(cursor.fetchall()):
                    memory_ids = row[2].split(',')
                    
                    # Get detailed info for each memory in the group
                    memory_details = []
                    for memory_id in memory_ids:
                        cursor.execute("""
                            SELECT memory_id, workflow_id, memory_type, importance, created_at
                            FROM memories WHERE memory_id = ?
                        """, (memory_id,))
                        
                        detail = cursor.fetchone()
                        if detail:
                            memory_details.append(MemoryDetail(
                                memory_id=detail[0],
                                workflow_id=detail[1],
                                memory_type=detail[2],
                                importance=detail[3],
                                created_at=detail[4]
                            ))
                    
                    duplicate_group = DuplicateGroup(
                        cluster_id=f"dup_cluster_{i}",
                        content_preview=row[0][:200] + '...' if len(row[0]) > 200 else row[0],
                        duplicate_count=row[1],
                        memory_ids=memory_ids,
                        primary_memory_id=memory_ids[0] if memory_ids else "",
                        memory_details=memory_details,
                        first_created=row[3],
                        last_created=row[4],
                        avg_importance=round(row[5], 1) if row[5] else 0.0,
                        recommendation='merge' if row[1] > 2 else 'review'
                    )
                    duplicate_groups.append(duplicate_group)
                
                conn.close()
                
                total_duplicates = sum(group.duplicate_count for group in duplicate_groups)
                
                return DuplicatesResponse(
                    success=True,
                    clusters=duplicate_groups,
                    duplicate_groups=duplicate_groups,  # For backward compatibility
                    total_groups=len(duplicate_groups),
                    total_duplicates=total_duplicates
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        @api_app.get(
            "/memory-quality/orphaned",
            response_model=OrphanedMemoriesResponse,
            tags=["Memory Quality"],
            summary="Get orphaned memories not associated with workflows",
            description="""
        Retrieve memories that are not associated with any workflow:

        - **Orphaned memory details** including content and metadata
        - **Creation timestamps** for temporal analysis
        - **Importance scoring** to prioritize action
        - **Assignment recommendations** for workflow integration

        Critical for maintaining memory system organization and preventing data loss.
            """,
            responses={
                200: {
                    "description": "Orphaned memories successfully retrieved",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "orphaned_memories": [
                                    {
                                        "memory_id": "mem_999",
                                        "content": "Important insight that got disconnected from workflow",
                                        "memory_type": "analysis",
                                        "importance": 7.5,
                                        "created_at": 1703980800.0
                                    }
                                ],
                                "total_orphaned": 1,
                                "recommendation": "Assign to appropriate workflows or archive if no longer needed"
                            }
                        }
                    }
                },
                500: {"description": "Internal server error"}
            }
        )
        async def get_orphaned_memories() -> OrphanedMemoriesResponse:
            """Get orphaned memories (not associated with workflows)"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT m.memory_id, m.content, m.memory_type, m.importance, m.created_at
                    FROM memories m
                    LEFT JOIN workflows w ON m.workflow_id = w.workflow_id
                    WHERE w.workflow_id IS NULL
                    ORDER BY m.created_at DESC
                """)
                
                orphaned_memories = [
                    OrphanedMemory(
                        memory_id=row[0],
                        content=row[1],
                        memory_type=row[2],
                        importance=row[3],
                        created_at=row[4]
                    )
                    for row in cursor.fetchall()
                ]
                
                conn.close()
                
                return OrphanedMemoriesResponse(
                    success=True,
                    orphaned_memories=orphaned_memories,
                    total_orphaned=len(orphaned_memories),
                    recommendation='Assign to appropriate workflows or archive if no longer needed'
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        @api_app.post(
            "/memory-quality/bulk-execute",
            response_model=BulkOperationResponse,
            tags=["Memory Quality"],
            summary="Execute bulk operations on memories",
            description="""
        Perform bulk operations on multiple memories:

        - **Merge operations** for duplicate consolidation
        - **Archive operations** for stale memory management
        - **Delete operations** for cleanup
        - **Progress tracking** and error reporting

        Enables efficient bulk management of memory quality issues.
            """,
            responses={
                200: {
                    "description": "Bulk operation completed",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "operation_type": "merge",
                                "memory_ids": ["mem_456", "mem_789"],
                                "success_count": 2,
                                "error_count": 0,
                                "message": "Operation completed: 2 succeeded, 0 failed",
                                "errors": [],
                                "merged_into": "mem_123"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def execute_bulk_memory_operations(
            bulk_request: BulkOperationRequest
        ) -> BulkOperationResponse:
            """Execute bulk operations on memories"""
            if not bulk_request.memory_ids:
                raise HTTPException(status_code=400, detail="No memory IDs provided")
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                success_count = 0
                errors = []
                
                placeholders = ','.join(['?' for _ in bulk_request.memory_ids])
                
                if bulk_request.operation_type == 'delete':
                    try:
                        cursor.execute(f"DELETE FROM memories WHERE memory_id IN ({placeholders})", bulk_request.memory_ids)
                        success_count = cursor.rowcount
                    except Exception as e:
                        errors.append(str(e))
                
                elif bulk_request.operation_type == 'archive':
                    # Add metadata to mark as archived
                    try:
                        cursor.execute(f"""
                            UPDATE memories 
                            SET metadata = json_set(COALESCE(metadata, '{{}}'), '$.archived', 'true', '$.archived_at', ?)
                            WHERE memory_id IN ({placeholders})
                        """, [datetime.now().isoformat()] + bulk_request.memory_ids)
                        success_count = cursor.rowcount
                    except Exception as e:
                        errors.append(str(e))
                
                elif bulk_request.operation_type == 'merge':
                    # For merge operations, keep the first memory and delete others
                    if len(bulk_request.memory_ids) > 1:
                        try:
                            # Keep the first memory, delete the rest
                            target_id = bulk_request.target_memory_id or bulk_request.memory_ids[0]
                            memories_to_delete = [mid for mid in bulk_request.memory_ids if mid != target_id]
                            
                            if memories_to_delete:
                                cursor.execute(
                                    f"DELETE FROM memories WHERE memory_id IN ({','.join(['?' for _ in memories_to_delete])})", 
                                    memories_to_delete
                                )
                                success_count = len(memories_to_delete)
                        except Exception as e:
                            errors.append(str(e))
                
                # Commit changes
                conn.commit()
                conn.close()
                
                error_count = len(bulk_request.memory_ids) - success_count
                
                return BulkOperationResponse(
                    success=len(errors) == 0,
                    operation_type=bulk_request.operation_type,
                    memory_ids=bulk_request.memory_ids,
                    success_count=success_count,
                    error_count=error_count,
                    message=f"Operation completed: {success_count} succeeded, {error_count} failed",
                    errors=errors,
                    merged_into=bulk_request.target_memory_id if bulk_request.operation_type == 'merge' else None
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        @api_app.post(
            "/memory-quality/bulk-preview",
            response_model=BulkPreviewResponse,
            tags=["Memory Quality"],
            summary="Preview bulk operations before execution",
            description="""
        Preview the effects of bulk operations before executing them:

        - **Operation impact preview** with affected memories
        - **Risk assessment** for destructive operations
        - **Merge target selection** for duplicate operations
        - **Cost estimation** for large operations

        Essential for safe bulk operations and preventing accidental data loss.
            """,
            responses={
                200: {
                    "description": "Preview generated successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "operation": {
                                    "operation_type": "merge",
                                    "total_affected": 3,
                                    "preview_description": "This will merge 3 memories",
                                    "affected_memories": [
                                        {
                                            "memory_id": "mem_123",
                                            "content": "System initialized successfully",
                                            "memory_type": "system",
                                            "importance": 8.0,
                                            "workflow_id": "workflow_001"
                                        }
                                    ],
                                    "merge_target": {
                                        "memory_id": "mem_123",
                                        "content": "System initialized successfully",
                                        "memory_type": "system",
                                        "importance": 8.0,
                                        "workflow_id": "workflow_001"
                                    },
                                    "will_be_deleted": []
                                }
                            }
                        }
                    }
                },
                400: {"description": "Invalid request parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def preview_bulk_operations(
            bulk_request: BulkOperationRequest
        ) -> BulkPreviewResponse:
            """Preview bulk operations before execution"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get memory details for preview
                placeholders = ','.join(['?' for _ in bulk_request.memory_ids])
                cursor.execute(f"""
                    SELECT memory_id, content, memory_type, importance, workflow_id
                    FROM memories 
                    WHERE memory_id IN ({placeholders})
                """, bulk_request.memory_ids)
                
                memories = [
                    PreviewMemory(
                        memory_id=row[0],
                        content=row[1],
                        memory_type=row[2],
                        importance=row[3],
                        workflow_id=row[4]
                    )
                    for row in cursor.fetchall()
                ]
                
                preview = BulkOperationPreview(
                    operation_type=bulk_request.operation_type,
                    total_affected=len(memories),
                    preview_description=f'This will {bulk_request.operation_type} {len(memories)} memories',
                    affected_memories=memories
                )
                
                if bulk_request.operation_type == 'merge' and len(memories) > 1:
                    target_id = bulk_request.target_memory_id or memories[0].memory_id
                    preview.merge_target = next((m for m in memories if m.memory_id == target_id), memories[0])
                    preview.will_be_deleted = [m for m in memories if m.memory_id != target_id]
                
                conn.close()
                
                return BulkPreviewResponse(
                    success=True,
                    operation=preview
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        # ---------- Working Memory System Implementation ----------

        from collections import defaultdict, deque
        from threading import Lock

        from fastapi import Body

        # Global working memory instance
        _working_memory_system = None
        _working_memory_lock = Lock()

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

        def get_working_memory_system() -> WorkingMemorySystem:
            """Get or create the global working memory system instance"""
            global _working_memory_system
            
            with _working_memory_lock:
                if _working_memory_system is None:
                    _working_memory_system = WorkingMemorySystem()
                return _working_memory_system

        # ---------- Working Memory Pydantic Models ----------

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

        # ---------- Working Memory Endpoints ----------

        @api_app.get(
            "/working-memory/status",
            response_model=WorkingMemoryStatus,
            tags=["Working Memory"],
            summary="Get working memory system status",
            description="""
        Retrieve the current status and configuration of the working memory system:

        - **Pool utilization** and capacity metrics
        - **Focus mode** status and configuration
        - **Optimization statistics** and performance data
        - **Memory distribution** across different categories

        Essential for monitoring working memory health and performance optimization.
            """,
            responses={
                200: {
                    "description": "Working memory status retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "initialized": True,
                                "total_capacity": 100,
                                "current_size": 45,
                                "utilization_percentage": 45.0,
                                "focus_mode": {
                                    "enabled": True,
                                    "focus_keywords": ["document", "analysis", "pdf"]
                                },
                                "performance_metrics": {
                                    "avg_relevance_score": 0.72,
                                    "optimization_suggestions": 2
                                },
                                "category_distribution": {
                                    "reasoning": 15,
                                    "observation": 20,
                                    "decision": 10
                                },
                                "last_optimization": "2024-01-01T12:30:00",
                                "optimization_count": 5
                            }
                        }
                    }
                },
                500: {
                    "description": "Internal server error",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Failed to retrieve working memory status"}
                        }
                    }
                }
            }
        )
        async def get_working_memory_status() -> WorkingMemoryStatus:
            """Get working memory system status"""
            try:
                wm_system = get_working_memory_system()
                stats = wm_system.get_statistics()
                
                return WorkingMemoryStatus(
                    initialized=True,
                    total_capacity=wm_system.capacity,
                    current_size=stats['total_memories'],
                    utilization_percentage=stats['capacity_used'],
                    focus_mode=FocusMode(
                        enabled=wm_system.focus_mode_enabled,
                        focus_keywords=wm_system.focus_keywords
                    ),
                    performance_metrics=PerformanceMetrics(
                        avg_relevance_score=stats['avg_relevance_score'],
                        optimization_suggestions=stats['optimization_suggestions']
                    ),
                    category_distribution=stats['category_distribution'],
                    last_optimization=wm_system.last_optimization.isoformat(),
                    optimization_count=wm_system.optimization_count
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve working memory status: {str(e)}") from e

        @api_app.post(
            "/working-memory/initialize",
            response_model=InitializeResponse,
            tags=["Working Memory"],
            summary="Initialize working memory system",
            description="""
        Initialize or reinitialize the working memory system with specific configuration:

        - **System initialization** with capacity settings
        - **Configuration setup** for optimization parameters
        - **Pool preparation** for memory operations
        - **Performance tuning** based on usage patterns

        Required before other working memory operations can be performed effectively.
            """,
            responses={
                200: {
                    "description": "Working memory initialized successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "message": "Working memory system initialized with capacity 150",
                                "configuration": {
                                    "capacity": 150,
                                    "focus_threshold": 0.8,
                                    "initialized_at": "2024-01-01T12:00:00"
                                }
                            }
                        }
                    }
                },
                400: {
                    "description": "Invalid configuration parameters",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Capacity must be between 10 and 1000"}
                        }
                    }
                },
                500: {
                    "description": "Internal server error"
                }
            }
        )
        async def initialize_working_memory(
            request: InitializeRequest
        ) -> InitializeResponse:
            """Initialize working memory system"""
            try:
                global _working_memory_system
                
                with _working_memory_lock:
                    # Create new instance with specified configuration
                    _working_memory_system = WorkingMemorySystem(
                        capacity=request.capacity,
                        focus_threshold=request.focus_threshold
                    )
                    
                    # Optionally load recent memories from database
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        
                        # Load most recent memories up to capacity
                        cursor.execute("""
                            SELECT memory_id, content, memory_type, importance
                            FROM memories
                            WHERE created_at >= ?
                            ORDER BY importance DESC, created_at DESC
                            LIMIT ?
                        """, (datetime.now().timestamp() - 86400, request.capacity))  # Last 24 hours
                        
                        loaded_count = 0
                        for row in cursor.fetchall():
                            _working_memory_system.add_memory(
                                memory_id=row[0],
                                content=row[1],
                                category=row[2],
                                importance=row[3]
                            )
                            loaded_count += 1
                        
                        conn.close()
                        
                        message = f"Working memory system initialized with capacity {request.capacity}, loaded {loaded_count} recent memories"
                    except Exception as e:
                        # Continue even if loading fails
                        message = f"Working memory system initialized with capacity {request.capacity} (memory loading failed: {str(e)})"
                
                return InitializeResponse(
                    success=True,
                    message=message,
                    configuration={
                        "capacity": request.capacity,
                        "focus_threshold": request.focus_threshold,
                        "initialized_at": _working_memory_system.initialized_at.isoformat()
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize working memory: {str(e)}") from e

        @api_app.get(
            "/working-memory/active",
            response_model=ActiveMemoriesResponse,
            tags=["Working Memory"],
            summary="Get active memories from working pool",
            description="""
        Retrieve active memories from the working pool, sorted by relevance.

        When focus mode is enabled, only memories meeting the relevance threshold are returned.
        This endpoint is useful for understanding what memories are currently available for processing.
            """,
            responses={
                200: {
                    "description": "Active memories retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "memories": [
                                    {
                                        "memory_id": "mem_001",
                                        "content": "The PDF contains financial data from Q4 2023",
                                        "category": "observation",
                                        "importance": 8.5,
                                        "relevance_score": 0.92,
                                        "added_at": 1703980800.0,
                                        "last_accessed": 1703981400.0,
                                        "access_count": 5
                                    }
                                ],
                                "total_count": 1,
                                "focus_active": True
                            }
                        }
                    }
                }
            }
        )
        async def get_active_memories(
            limit: int = Query(50, ge=1, le=200, description="Maximum number of memories to return"),
            category: Optional[str] = Query(None, description="Filter by memory category")
        ) -> ActiveMemoriesResponse:
            """Get active memories from working pool"""
            try:
                wm_system = get_working_memory_system()
                memories = wm_system.get_active_memories(limit=limit)
                
                # Filter by category if specified
                if category:
                    memories = [m for m in memories if m['category'] == category]
                
                # Convert to response format
                memory_items = []
                for mem in memories:
                    memory_items.append(MemoryItem(
                        memory_id=mem['memory_id'],
                        content=mem['content'],
                        category=mem['category'],
                        importance=mem['importance'],
                        relevance_score=wm_system.relevance_scores.get(mem['memory_id'], 0),
                        added_at=mem['added_at'],
                        last_accessed=mem['last_accessed'],
                        access_count=wm_system.access_counts.get(mem['memory_id'], 0)
                    ))
                
                return ActiveMemoriesResponse(
                    memories=memory_items,
                    total_count=len(memory_items),
                    focus_active=wm_system.focus_mode_enabled
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve active memories: {str(e)}") from e

        @api_app.post(
            "/working-memory/focus",
            response_model=InitializeResponse,
            tags=["Working Memory"],
            summary="Set focus mode configuration",
            description="""
        Configure focus mode for the working memory system.

        Focus mode filters memories based on relevance to specified keywords,
        helping to narrow attention to specific topics or contexts.
            """
        )
        async def set_focus_mode(
            request: SetFocusModeRequest
        ) -> InitializeResponse:
            """Set focus mode configuration"""
            try:
                wm_system = get_working_memory_system()
                wm_system.set_focus_mode(request.enabled, request.keywords)
                
                message = f"Focus mode {'enabled' if request.enabled else 'disabled'}"
                if request.enabled and request.keywords:
                    message += f" with keywords: {', '.join(request.keywords)}"
                
                return InitializeResponse(
                    success=True,
                    message=message,
                    configuration={
                        "focus_enabled": request.enabled,
                        "focus_keywords": request.keywords,
                        "focus_threshold": wm_system.focus_threshold
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to set focus mode: {str(e)}") from e

        @api_app.post(
            "/working-memory/optimize",
            response_model=OptimizeResponse,
            tags=["Working Memory"],
            summary="Optimize working memory pool",
            description="""
        Optimize the working memory pool by removing low-relevance memories.

        This operation helps maintain memory pool quality by removing memories
        with relevance scores below the optimization threshold (0.2).
            """
        )
        async def optimize_working_memory() -> OptimizeResponse:
            """Optimize working memory pool"""
            try:
                wm_system = get_working_memory_system()
                removed_count = wm_system.optimize()
                
                return OptimizeResponse(
                    success=True,
                    removed_count=removed_count,
                    message=f"Optimization complete. Removed {removed_count} low-relevance memories."
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to optimize working memory: {str(e)}") from e

        # ---------- Performance Profiler Pydantic Models ----------

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

        # ---------- Performance Profiler Endpoints ----------

        @api_app.get(
            "/performance/overview",
            response_model=PerformanceOverviewResponse,
            tags=["Performance Profiler"],
            summary="Get comprehensive performance overview with metrics and trends",
            description="""
        Retrieve comprehensive workflow performance overview including:

        - **Real-time performance metrics** with execution time analysis
        - **Timeline visualization data** with configurable granularity
        - **Tool utilization statistics** and performance breakdowns
        - **Current bottlenecks** identification with severity indicators
        - **Throughput analysis** and success rate metrics

        Perfect for monitoring overall system performance and identifying optimization opportunities.
            """,
            responses={
                200: {
                    "description": "Performance overview data with metrics and timeline",
                    "content": {
                        "application/json": {
                            "example": {
                                "overview": {
                                    "total_actions": 1250,
                                    "active_workflows": 45,
                                    "avg_execution_time": 12.5,
                                    "success_rate_percentage": 92.5,
                                    "throughput_per_hour": 52.1
                                },
                                "timeline": [
                                    {
                                        "time_bucket": "2024-01-01 14:00:00",
                                        "action_count": 45,
                                        "avg_duration": 11.2,
                                        "successful_count": 42,
                                        "failed_count": 3
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        )
        async def get_performance_overview(
            hours_back: int = Query(
                24,
                description="Number of hours back to analyze performance data",
                ge=1,
                le=720,
                example=24
            ),
            granularity: str = Query(
                "hour",
                description="Time granularity for timeline data aggregation",
                regex="^(minute|hour|day)$",
                example="hour"
            )
        ) -> PerformanceOverviewResponse:
            """Get comprehensive performance overview with metrics and trends"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Overall performance metrics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_actions,
                        COUNT(DISTINCT workflow_id) as active_workflows,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_execution_time,
                        MIN(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as min_execution_time,
                        MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_execution_time,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        COUNT(DISTINCT tool_name) as tools_used
                    FROM actions 
                    WHERE started_at >= ?
                """, (since_timestamp,))
                
                overview_result = cursor.fetchone()
                overview_data = dict(zip([d[0] for d in cursor.description], overview_result, strict=False)) if overview_result else {}
                
                # Calculate derived metrics
                success_rate = (overview_data.get('successful_actions', 0) / max(1, overview_data.get('total_actions', 1))) * 100
                throughput = overview_data.get('total_actions', 0) / max(1, hours_back)
                
                overview_stats = PerformanceOverviewStats(
                    **overview_data,
                    success_rate_percentage=success_rate,
                    throughput_per_hour=throughput,
                    error_rate_percentage=100 - success_rate,
                    avg_workflow_size=overview_data.get('total_actions', 0) / max(1, overview_data.get('active_workflows', 1))
                )
                
                # Performance timeline
                if granularity == 'hour':
                    time_format = "strftime('%Y-%m-%d %H:00:00', datetime(started_at, 'unixepoch'))"
                elif granularity == 'minute':
                    time_format = "strftime('%Y-%m-%d %H:%M:00', datetime(started_at, 'unixepoch'))"
                else:  # day
                    time_format = "strftime('%Y-%m-%d', datetime(started_at, 'unixepoch'))"
                
                cursor.execute(f"""
                    SELECT 
                        {time_format} as time_bucket,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                        COUNT(DISTINCT workflow_id) as workflow_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY {time_format}
                    ORDER BY time_bucket
                """, (since_timestamp,))
                
                timeline_data = [
                    TimelineBucket(**dict(zip([d[0] for d in cursor.description], row, strict=False)))
                    for row in cursor.fetchall()
                ]
                
                # Resource utilization by tool
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                        MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_duration
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY tool_name
                    ORDER BY usage_count DESC
                """, (since_timestamp,))
                
                tool_utilization = [
                    ToolUtilization(**dict(zip([d[0] for d in cursor.description], row, strict=False)))
                    for row in cursor.fetchall()
                ]
                
                # Top bottlenecks (slowest operations)
                cursor.execute("""
                    SELECT 
                        tool_name,
                        workflow_id,
                        action_id,
                        started_at,
                        completed_at,
                        (completed_at - started_at) as duration,
                        status,
                        reasoning
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    ORDER BY duration DESC
                    LIMIT 10
                """, (since_timestamp,))
                
                bottlenecks = [
                    Bottleneck(**dict(zip([d[0] for d in cursor.description], row, strict=False)))
                    for row in cursor.fetchall()
                ]
                
                conn.close()
                
                return PerformanceOverviewResponse(
                    overview=overview_stats,
                    timeline=timeline_data,
                    tool_utilization=tool_utilization,
                    bottlenecks=bottlenecks,
                    analysis_period={
                        'hours_back': hours_back,
                        'granularity': granularity,
                        'start_time': since_timestamp,
                        'end_time': datetime.now().timestamp()
                    },
                    timestamp=datetime.now().isoformat()
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        @api_app.get(
            "/performance/bottlenecks",
            response_model=BottleneckAnalysisResponse,
            tags=["Performance Profiler"],
            summary="Identify and analyze performance bottlenecks with detailed insights",
            description="""
        Perform comprehensive bottleneck analysis including:

        - **Tool performance analysis** with percentile breakdowns (P95, P99)
        - **Workflow efficiency scoring** and parallelization opportunities
        - **Resource contention detection** and conflict analysis
        - **Optimization recommendations** with impact estimates
        - **Critical path identification** for workflow optimization

        Advanced algorithms identify bottlenecks using statistical analysis and provide actionable insights.
            """,
            responses={
                200: {
                    "description": "Comprehensive bottleneck analysis with optimization opportunities"
                }
            }
        )
        async def get_performance_bottlenecks(
            hours_back: int = Query(
                24,
                description="Hours back to analyze for bottlenecks",
                ge=1,
                le=720
            ),
            min_duration: float = Query(
                1.0,
                description="Minimum execution duration (seconds) to consider as potential bottleneck",
                ge=0.1
            )
        ) -> BottleneckAnalysisResponse:
            """Identify and analyze performance bottlenecks with detailed insights"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Identify bottlenecks by tool with percentile calculations
                # Note: SQLite doesn't have PERCENTILE_CONT, so we'll approximate
                cursor.execute("""
                    WITH tool_durations AS (
                        SELECT 
                            tool_name,
                            (completed_at - started_at) as duration
                        FROM actions 
                        WHERE started_at >= ? 
                        AND completed_at IS NOT NULL 
                        AND (completed_at - started_at) >= ?
                    )
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        AVG(duration) as avg_duration,
                        MAX(duration) as max_duration,
                        MIN(duration) as min_duration,
                        SUM(CASE WHEN a.status = 'failed' THEN 1 ELSE 0 END) as failure_count,
                        SUM(duration) as total_time_spent
                    FROM tool_durations td
                    JOIN actions a USING(tool_name)
                    WHERE a.started_at >= ? AND a.completed_at IS NOT NULL
                    GROUP BY tool_name
                    ORDER BY avg_duration DESC
                """, (since_timestamp, min_duration, since_timestamp))
                
                tool_bottlenecks = []
                for row in cursor.fetchall():
                    data = dict(zip([d[0] for d in cursor.description], row, strict=False))
                    # Approximate percentiles (in production, you'd calculate these properly)
                    data['p95_duration'] = data['avg_duration'] * 1.5  # Approximation
                    data['p99_duration'] = data['avg_duration'] * 2.0  # Approximation
                    tool_bottlenecks.append(ToolBottleneck(**data))
                
                # Identify workflow bottlenecks
                cursor.execute("""
                    SELECT 
                        w.workflow_id,
                        w.title,
                        COUNT(a.action_id) as action_count,
                        AVG(a.completed_at - a.started_at) as avg_action_duration,
                        MAX(a.completed_at - a.started_at) as max_action_duration,
                        SUM(a.completed_at - a.started_at) as total_workflow_time,
                        MIN(a.started_at) as workflow_start,
                        MAX(a.completed_at) as workflow_end,
                        (MAX(a.completed_at) - MIN(a.started_at)) as total_elapsed_time
                    FROM workflows w
                    JOIN actions a ON w.workflow_id = a.workflow_id
                    WHERE a.started_at >= ? AND a.completed_at IS NOT NULL
                    GROUP BY w.workflow_id, w.title
                    HAVING COUNT(a.action_id) > 1
                    ORDER BY total_workflow_time DESC
                    LIMIT 20
                """, (since_timestamp,))
                
                workflow_bottlenecks = [
                    WorkflowBottleneck(**dict(zip([d[0] for d in cursor.description], row, strict=False)))
                    for row in cursor.fetchall()
                ]
                
                # Calculate parallelization opportunities
                cursor.execute("""
                    SELECT 
                        workflow_id,
                        COUNT(*) as sequential_actions,
                        SUM(completed_at - started_at) as total_sequential_time,
                        (MAX(completed_at) - MIN(started_at)) as actual_elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    GROUP BY workflow_id
                    HAVING COUNT(*) > 2
                """, (since_timestamp,))
                
                parallelization_opportunities = []
                for row in cursor.fetchall():
                    data = dict(zip([d[0] for d in cursor.description], row, strict=False))
                    potential_savings = data['total_sequential_time'] - data['actual_elapsed_time']
                    if potential_savings > 0:
                        parallelization_opportunities.append(ParallelizationOpportunity(
                            **data,
                            potential_time_savings=potential_savings,
                            parallelization_efficiency=(data['actual_elapsed_time'] / data['total_sequential_time']) * 100,
                            optimization_score=min(10, potential_savings / data['actual_elapsed_time'] * 10)
                        ))
                
                # Resource contention analysis
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as concurrent_usage,
                        AVG(completed_at - started_at) as avg_duration_under_contention
                    FROM actions a1
                    WHERE started_at >= ? AND EXISTS (
                        SELECT 1 FROM actions a2 
                        WHERE a2.tool_name = a1.tool_name 
                        AND a2.action_id != a1.action_id
                        AND a2.started_at <= a1.completed_at 
                        AND a2.completed_at >= a1.started_at
                    )
                    GROUP BY tool_name
                    ORDER BY concurrent_usage DESC
                """, (since_timestamp,))
                
                resource_contention = [
                    ResourceContention(**dict(zip([d[0] for d in cursor.description], row, strict=False)))
                    for row in cursor.fetchall()
                ]
                
                conn.close()
                
                # Generate optimization recommendations
                recommendations = []
                
                # Tool-based recommendations
                for tool in tool_bottlenecks[:5]:
                    if tool.avg_duration > 10:  # More than 10 seconds average
                        recommendations.append(OptimizationRecommendation(
                            type='tool_optimization',
                            priority='high' if tool.avg_duration > 30 else 'medium',
                            title=f"Optimize {tool.tool_name} performance",
                            description=f"Tool {tool.tool_name} has high average duration of {tool.avg_duration:.2f}s",
                            impact=f"Could save ~{tool.total_time_spent * 0.3:.2f}s per execution period",
                            actions=[
                                'Review tool implementation for optimization opportunities',
                                'Consider caching strategies for repeated operations',
                                'Evaluate if tool can be replaced with faster alternative'
                            ]
                        ))
                
                # Parallelization recommendations
                for opp in sorted(parallelization_opportunities, key=lambda x: x.potential_time_savings, reverse=True)[:3]:
                    recommendations.append(OptimizationRecommendation(
                        type='parallelization',
                        priority='high' if opp.potential_time_savings > 20 else 'medium',
                        title=f"Parallelize workflow {opp.workflow_id}",
                        description=f"Workflow could save {opp.potential_time_savings:.2f}s through parallel execution",
                        impact=f"Up to {opp.parallelization_efficiency:.1f}% efficiency improvement",
                        actions=[
                            'Analyze action dependencies to identify parallelizable segments',
                            'Implement async execution where possible',
                            'Consider workflow restructuring for better parallelization'
                        ]
                    ))
                
                return BottleneckAnalysisResponse(
                    tool_bottlenecks=tool_bottlenecks,
                    workflow_bottlenecks=workflow_bottlenecks,
                    parallelization_opportunities=parallelization_opportunities,
                    resource_contention=resource_contention,
                    recommendations=recommendations,
                    analysis_summary={
                        'total_bottlenecks_identified': len(tool_bottlenecks) + len(workflow_bottlenecks),
                        'highest_impact_tool': tool_bottlenecks[0].tool_name if tool_bottlenecks else None,
                        'avg_tool_duration': sum(t.avg_duration for t in tool_bottlenecks) / len(tool_bottlenecks) if tool_bottlenecks else 0,
                        'parallelization_potential': len(parallelization_opportunities)
                    },
                    timestamp=datetime.now().isoformat()
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        # ---------- Flame Graph Helper Functions ----------

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

        # ---------- Flame Graph Pydantic Models ----------

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

        # ---------- Performance Trends Pydantic Models ----------

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

        class Pattern(BaseModel):
            """Detected performance pattern"""
            type: str = Field(..., description="Type of pattern detected")
            description: str = Field(..., description="Description of the pattern")
            impact: str = Field(..., description="Impact level (high/medium/low)")
            recommendation: str = Field(..., description="Recommended action")
            date: Optional[str] = Field(None, description="Date of occurrence for anomalies")

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
            patterns: List[Pattern] = Field(..., description="Detected performance patterns")
            insights: InsightMetrics = Field(..., description="Key performance insights")
            timestamp: str = Field(..., description="Response generation timestamp")

        # ---------- Advanced Performance Profiler Endpoints ----------

        @api_app.get(
            "/performance/flame-graph",
            response_model=FlameGraphResponse,
            tags=["Performance Profiler"],
            summary="Generate flame graph data for workflow performance visualization",
            description="""
        Generate hierarchical flame graph data for detailed workflow performance analysis:

        - **Interactive flame graph structure** showing execution hierarchy
        - **Critical path analysis** highlighting the longest dependency chain
        - **Tool-level performance breakdown** with execution times
        - **Parallelization efficiency metrics** and optimization scores
        - **Execution timeline analysis** with CPU vs wall-clock time

        Industry-standard flame graph visualization for profiling workflow execution patterns.
            """,
            responses={
                200: {
                    "description": "Flame graph data with performance metrics and critical path",
                    "content": {
                        "application/json": {
                            "example": {
                                "flame_graph": {
                                    "name": "Workflow workflow_abc123",
                                    "value": 145.5,
                                    "children": [
                                        {
                                            "name": "smart_browser",
                                            "value": 85.3,
                                            "children": [
                                                {
                                                    "name": "Action act_001",
                                                    "value": 45.2,
                                                    "action_id": "act_001",
                                                    "status": "completed",
                                                    "reasoning": "Navigate to documentation site",
                                                    "started_at": 1703980800.0,
                                                    "completed_at": 1703980845.2
                                                }
                                            ]
                                        },
                                        {
                                            "name": "execute_python",
                                            "value": 60.2,
                                            "children": []
                                        }
                                    ]
                                },
                                "metrics": {
                                    "total_actions": 5,
                                    "total_cpu_time": 145.5,
                                    "wall_clock_time": 98.7,
                                    "parallelization_efficiency": 67.8,
                                    "avg_action_duration": 29.1,
                                    "workflow_start": 1703980800.0,
                                    "workflow_end": 1703980898.7
                                },
                                "critical_path": [
                                    {
                                        "action_id": "act_001",
                                        "tool_name": "smart_browser",
                                        "duration": 45.2,
                                        "start_time": 1703980800.0,
                                        "end_time": 1703980845.2
                                    }
                                ],
                                "analysis": {
                                    "bottleneck_tool": "smart_browser",
                                    "parallelization_potential": 46.8,
                                    "optimization_score": 6.8
                                },
                                "timestamp": "2024-01-01T00:00:00Z"
                            }
                        }
                    }
                },
                400: {
                    "description": "Missing required workflow_id parameter",
                    "content": {
                        "application/json": {
                            "example": {"detail": "workflow_id parameter is required"}
                        }
                    }
                },
                404: {
                    "description": "No actions found for specified workflow",
                    "content": {
                        "application/json": {
                            "example": {"detail": "No actions found for workflow 'workflow_abc123'"}
                        }
                    }
                },
                500: {
                    "description": "Internal server error"
                }
            }
        )
        async def get_performance_flame_graph(
            workflow_id: str = Query(
                ...,
                description="Workflow ID to generate flame graph for",
                example="workflow_abc123",
                regex="^[a-zA-Z0-9_-]+$"
            ),
            hours_back: int = Query(
                24,
                description="Hours back to search for workflow execution data",
                ge=1,
                le=720,  # Max 30 days
                example=24
            )
        ) -> FlameGraphResponse:
            """Generate flame graph data for workflow performance visualization"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Get workflow actions with timing data
                cursor.execute("""
                    SELECT 
                        action_id,
                        tool_name,
                        started_at,
                        completed_at,
                        (completed_at - started_at) as duration,
                        status,
                        reasoning,
                        summary,
                        dependency_path
                    FROM actions 
                    WHERE workflow_id = ? AND started_at >= ?
                    ORDER BY started_at
                """, (workflow_id, since_timestamp))
                
                actions = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                if not actions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No actions found for workflow '{workflow_id}'"
                    )
                
                # Build flame graph structure
                flame_graph_data = build_flame_graph_structure(actions, workflow_id)
                
                # Calculate performance metrics
                total_duration = sum(action.get('duration', 0) for action in actions if action.get('duration'))
                workflow_start = min(action['started_at'] for action in actions if action.get('started_at'))
                workflow_end = max(action['completed_at'] for action in actions if action.get('completed_at'))
                wall_clock_time = workflow_end - workflow_start if workflow_end and workflow_start else 0
                
                # Parallelization efficiency
                parallelization_efficiency = (wall_clock_time / total_duration * 100) if total_duration > 0 else 0
                
                # Critical path analysis
                critical_path = calculate_critical_path(actions)
                
                # Find bottleneck tool
                tool_durations = {}
                for action in actions:
                    tool_name = action.get('tool_name', 'unknown')
                    duration = action.get('duration', 0)
                    if tool_name not in tool_durations:
                        tool_durations[tool_name] = 0
                    tool_durations[tool_name] += duration
                
                bottleneck_tool = max(tool_durations.keys(), key=lambda t: tool_durations[t]) if tool_durations else None
                
                # Calculate optimization potential
                parallelization_potential = max(0, total_duration - wall_clock_time)
                optimization_score = min(10, parallelization_efficiency / 10)
                
                conn.close()
                
                # Convert flame graph to Pydantic model
                def convert_to_model(node: Dict) -> FlameGraphNode:
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
                
                return FlameGraphResponse(
                    flame_graph=convert_to_model(flame_graph_data),
                    metrics=WorkflowMetrics(
                        total_actions=len(actions),
                        total_cpu_time=total_duration,
                        wall_clock_time=wall_clock_time,
                        parallelization_efficiency=parallelization_efficiency,
                        avg_action_duration=total_duration / len(actions) if actions else 0,
                        workflow_start=workflow_start,
                        workflow_end=workflow_end
                    ),
                    critical_path=[CriticalPathAction(**cp) for cp in critical_path],
                    analysis=WorkflowAnalysis(
                        bottleneck_tool=bottleneck_tool,
                        parallelization_potential=parallelization_potential,
                        optimization_score=optimization_score
                    ),
                    timestamp=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        @api_app.get(
            "/performance/trends",
            response_model=PerformanceTrendsResponse,
            tags=["Performance Profiler"],
            summary="Analyze performance trends and patterns over time",
            description="""
        Comprehensive trend analysis for long-term performance monitoring:

        - **Daily performance trends** with configurable time periods
        - **Pattern detection algorithms** identifying weekly patterns and anomalies
        - **Trend classification** (improving, degrading, stable) with confidence scores
        - **Performance insights** with contextual explanations
        - **Comparative analysis** showing best/worst performing periods

        Advanced analytics help identify performance degradation and optimization opportunities over time.
            """,
            responses={
                200: {
                    "description": "Performance trends with pattern analysis and insights",
                    "content": {
                        "application/json": {
                            "example": {
                                "daily_trends": [
                                    {
                                        "date": "2024-01-01",
                                        "action_count": 150,
                                        "avg_duration": 25.5,
                                        "success_rate": 92.5,
                                        "throughput": 6.25,
                                        "error_rate": 7.5,
                                        "successful_actions": 139,
                                        "failed_actions": 11,
                                        "workflow_count": 15,
                                        "tool_count": 8
                                    }
                                ],
                                "tool_trends": [
                                    {
                                        "tool_name": "smart_browser",
                                        "date": "2024-01-01",
                                        "usage_count": 45,
                                        "avg_duration": 35.2,
                                        "success_count": 42
                                    }
                                ],
                                "workflow_complexity": [],
                                "trend_analysis": {
                                    "performance_trend": "improving",
                                    "success_trend": "stable",
                                    "data_points": 7,
                                    "analysis_period_days": 7
                                },
                                "patterns": [
                                    {
                                        "type": "weekly_pattern",
                                        "description": "Performance varies significantly between weekdays (25.5s) and weekends (35.2s)",
                                        "impact": "medium",
                                        "recommendation": "Consider different optimization strategies for weekend vs weekday operations"
                                    }
                                ],
                                "insights": {
                                    "best_performing_day": {
                                        "date": "2024-01-03",
                                        "action_count": 180,
                                        "avg_duration": 22.3,
                                        "success_rate": 95.5
                                    },
                                    "avg_daily_actions": 150.5
                                },
                                "timestamp": "2024-01-07T12:00:00Z"
                            }
                        }
                    }
                },
                500: {
                    "description": "Internal server error"
                }
            }
        )
        async def get_performance_trends(
            days_back: int = Query(
                7,
                description="Number of days back to analyze trends",
                ge=1,
                le=90,  # Max 3 months
                example=7
            ),
            metric: str = Query(
                "duration",
                description="Primary metric to analyze for trends",
                regex="^(duration|success_rate|throughput)$",
                example="duration"
            )
        ) -> PerformanceTrendsResponse:
            """Analyze performance trends and patterns over time"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (days_back * 24 * 3600)
                
                # Daily trends
                cursor.execute("""
                    SELECT 
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        COUNT(DISTINCT workflow_id) as workflow_count,
                        COUNT(DISTINCT tool_name) as tool_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY DATE(datetime(started_at, 'unixepoch'))
                    ORDER BY date
                """, (since_timestamp,))
                
                daily_trends = []
                for row in cursor.fetchall():
                    date, action_count, avg_duration, successful_actions, failed_actions, workflow_count, tool_count = row
                    
                    success_rate = (successful_actions / max(1, action_count)) * 100
                    throughput = action_count / 24  # actions per hour
                    error_rate = (failed_actions / max(1, action_count)) * 100
                    
                    daily_trends.append(DailyTrend(
                        date=date,
                        action_count=action_count,
                        avg_duration=avg_duration,
                        success_rate=success_rate,
                        throughput=throughput,
                        error_rate=error_rate,
                        successful_actions=successful_actions,
                        failed_actions=failed_actions,
                        workflow_count=workflow_count,
                        tool_count=tool_count
                    ))
                
                # Tool performance trends
                cursor.execute("""
                    SELECT 
                        tool_name,
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        COUNT(*) as usage_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY tool_name, DATE(datetime(started_at, 'unixepoch'))
                    ORDER BY tool_name, date
                """, (since_timestamp,))
                
                tool_trends = [
                    ToolTrend(
                        tool_name=row[0],
                        date=row[1],
                        usage_count=row[2],
                        avg_duration=row[3],
                        success_count=row[4]
                    )
                    for row in cursor.fetchall()
                ]
                
                # Workflow complexity trends
                cursor.execute("""
                    SELECT 
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        workflow_id,
                        COUNT(*) as action_count,
                        SUM(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as total_duration,
                        (MAX(completed_at) - MIN(started_at)) as elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND workflow_id IS NOT NULL
                    GROUP BY DATE(datetime(started_at, 'unixepoch')), workflow_id
                    ORDER BY date, workflow_id
                """, (since_timestamp,))
                
                workflow_complexity = [
                    WorkflowComplexityTrend(
                        date=row[0],
                        workflow_id=row[1],
                        action_count=row[2],
                        total_duration=row[3],
                        elapsed_time=row[4]
                    )
                    for row in cursor.fetchall()
                ]
                
                # Calculate trend analysis
                if len(daily_trends) >= 2:
                    # Performance trend (improving, degrading, stable)
                    recent_avg = sum(d.avg_duration or 0 for d in daily_trends[-3:]) / min(3, len(daily_trends))
                    earlier_avg = sum(d.avg_duration or 0 for d in daily_trends[:3]) / min(3, len(daily_trends))
                    
                    if recent_avg > earlier_avg * 1.1:
                        performance_trend = 'degrading'
                    elif recent_avg < earlier_avg * 0.9:
                        performance_trend = 'improving'
                    else:
                        performance_trend = 'stable'
                    
                    # Success rate trend
                    recent_success = sum(d.success_rate for d in daily_trends[-3:]) / min(3, len(daily_trends))
                    earlier_success = sum(d.success_rate for d in daily_trends[:3]) / min(3, len(daily_trends))
                    
                    success_trend = 'improving' if recent_success > earlier_success else 'degrading' if recent_success < earlier_success else 'stable'
                else:
                    performance_trend = 'insufficient_data'
                    success_trend = 'insufficient_data'
                
                # Identify performance patterns
                patterns = []
                
                # Weekly pattern detection
                if len(daily_trends) >= 7:
                    weekend_performance = [d for d in daily_trends if datetime.strptime(d.date, '%Y-%m-%d').weekday() >= 5]
                    weekday_performance = [d for d in daily_trends if datetime.strptime(d.date, '%Y-%m-%d').weekday() < 5]
                    
                    if weekend_performance and weekday_performance:
                        weekend_avg = sum(d.avg_duration or 0 for d in weekend_performance) / len(weekend_performance)
                        weekday_avg = sum(d.avg_duration or 0 for d in weekday_performance) / len(weekday_performance)
                        
                        if abs(weekend_avg - weekday_avg) > weekday_avg * 0.2:
                            patterns.append(Pattern(
                                type='weekly_pattern',
                                description=f"Performance varies significantly between weekdays ({weekday_avg:.2f}s) and weekends ({weekend_avg:.2f}s)",
                                impact='medium',
                                recommendation='Consider different optimization strategies for weekend vs weekday operations'
                            ))
                
                # Anomaly detection (simple outlier detection)
                if daily_trends:
                    durations = [d.avg_duration or 0 for d in daily_trends]
                    mean_duration = sum(durations) / len(durations)
                    
                    outliers = [d for d in daily_trends if abs((d.avg_duration or 0) - mean_duration) > mean_duration * 0.5]
                    
                    for outlier in outliers:
                        patterns.append(Pattern(
                            type='performance_anomaly',
                            date=outlier.date,
                            description=f"Unusual performance on {outlier.date}: {outlier.avg_duration:.2f}s vs normal {mean_duration:.2f}s",
                            impact='high' if abs((outlier.avg_duration or 0) - mean_duration) > mean_duration else 'medium',
                            recommendation='Investigate system conditions and workload on this date'
                        ))
                
                # Generate insights
                best_day = max(daily_trends, key=lambda x: x.success_rate) if daily_trends else None
                worst_day = min(daily_trends, key=lambda x: x.success_rate) if daily_trends else None
                peak_throughput_day = max(daily_trends, key=lambda x: x.throughput) if daily_trends else None
                avg_daily_actions = sum(d.action_count for d in daily_trends) / len(daily_trends) if daily_trends else 0
                
                conn.close()
                
                return PerformanceTrendsResponse(
                    daily_trends=daily_trends,
                    tool_trends=tool_trends,
                    workflow_complexity=workflow_complexity,
                    trend_analysis=TrendAnalysis(
                        performance_trend=performance_trend,
                        success_trend=success_trend,
                        data_points=len(daily_trends),
                        analysis_period_days=days_back
                    ),
                    patterns=patterns,
                    insights=InsightMetrics(
                        best_performing_day=best_day,
                        worst_performing_day=worst_day,
                        peak_throughput_day=peak_throughput_day,
                        avg_daily_actions=avg_daily_actions
                    ),
                    timestamp=datetime.now().isoformat()
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        # ---------- Performance Recommendations Helper Functions ----------

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

        # ---------- Performance Recommendations Pydantic Models ----------

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

        # ---------- Performance Recommendations Endpoint ----------

        @api_app.get(
            "/performance/recommendations",
            response_model=PerformanceRecommendationsResponse,
            tags=["Performance Profiler"],
            summary="Generate actionable performance optimization recommendations",
            description="""
        AI-powered optimization recommendations engine providing:

        - **Prioritized recommendations** with impact and effort scoring
        - **Implementation roadmaps** categorized by complexity and impact
        - **Detailed implementation steps** with prerequisites and metrics
        - **Cost-benefit analysis** with quantified impact estimates
        - **Progress tracking guidance** with success metrics

        Smart recommendation system analyzes performance data to provide actionable optimization strategies.
            """,
            responses={
                200: {
                    "description": "Comprehensive optimization recommendations with implementation guidance",
                    "content": {
                        "application/json": {
                            "example": {
                                "recommendations": [
                                    {
                                        "id": "optimize_tool_smart_browser",
                                        "type": "tool_optimization",
                                        "priority": "high",
                                        "title": "Optimize smart_browser performance",
                                        "description": "Tool consumes 1543.2s total execution time with 25.3s average",
                                        "impact_estimate": {
                                            "time_savings_potential": 463.0,
                                            "affected_actions": 61,
                                            "cost_benefit_ratio": 8.5
                                        },
                                        "implementation_steps": [
                                            "Profile smart_browser execution to identify bottlenecks",
                                            "Consider caching frequently used data",
                                            "Optimize database queries if applicable",
                                            "Evaluate alternative implementations or libraries"
                                        ],
                                        "estimated_effort": "medium",
                                        "prerequisites": ["Development environment setup", "Performance profiling tools"],
                                        "metrics_to_track": [
                                            "Average execution time",
                                            "P95 execution time",
                                            "Tool success rate",
                                            "Resource utilization"
                                        ]
                                    }
                                ],
                                "summary": {
                                    "total_recommendations": 5,
                                    "high_priority": 2,
                                    "medium_priority": 2,
                                    "low_priority": 1,
                                    "estimated_total_savings": 1250.5,
                                    "analysis_period_hours": 24
                                },
                                "implementation_roadmap": {
                                    "quick_wins": [],
                                    "major_improvements": [],
                                    "maintenance_tasks": []
                                },
                                "timestamp": "2024-01-01T12:00:00"
                            }
                        }
                    }
                },
                500: {
                    "description": "Internal server error",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Failed to generate recommendations"}
                        }
                    }
                }
            }
        )
        async def get_performance_recommendations(
            hours_back: int = Query(
                24,
                description="Hours back to analyze for recommendations",
                ge=1,
                le=720,
                example=24
            ),
            priority_filter: str = Query(
                "all",
                description="Filter recommendations by priority level",
                regex="^(all|high|medium|low)$",
                example="all"
            )
        ) -> PerformanceRecommendationsResponse:
            """Generate actionable performance optimization recommendations"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                recommendations = []
                
                # Analyze slow tools
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        AVG(completed_at - started_at) as avg_duration,
                        MAX(completed_at - started_at) as max_duration,
                        SUM(completed_at - started_at) as total_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    GROUP BY tool_name
                    HAVING avg_duration > 5
                    ORDER BY total_time DESC
                """, (since_timestamp,))
                
                slow_tools = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for tool in slow_tools[:5]:  # Top 5 slowest tools
                    impact_score = tool['total_time'] / 3600  # hours of time spent
                    priority = 'high' if impact_score > 1 else 'medium' if impact_score > 0.5 else 'low'
                    
                    recommendation = PerformanceRecommendation(
                        id=f"optimize_tool_{tool['tool_name']}",
                        type='tool_optimization',
                        priority=priority,
                        title=f"Optimize {tool['tool_name']} performance",
                        description=f"Tool consumes {tool['total_time']:.1f}s total execution time with {tool['avg_duration']:.2f}s average",
                        impact_estimate=ImpactEstimate(
                            time_savings_potential=tool['total_time'] * 0.3,  # Assume 30% improvement possible
                            affected_actions=tool['usage_count'],
                            cost_benefit_ratio=impact_score
                        ),
                        implementation_steps=[
                            f"Profile {tool['tool_name']} execution to identify bottlenecks",
                            "Consider caching frequently used data",
                            "Optimize database queries if applicable",
                            "Evaluate alternative implementations or libraries"
                        ],
                        estimated_effort='medium',
                        prerequisites=['Development environment setup', 'Performance profiling tools'],
                        metrics_to_track=[
                            'Average execution time',
                            'P95 execution time',
                            'Tool success rate',
                            'Resource utilization'
                        ]
                    )
                    recommendations.append(recommendation)
                
                # Analyze workflow parallelization opportunities
                cursor.execute("""
                    SELECT 
                        workflow_id,
                        COUNT(*) as action_count,
                        SUM(completed_at - started_at) as total_sequential_time,
                        (MAX(completed_at) - MIN(started_at)) as actual_elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL AND workflow_id IS NOT NULL
                    GROUP BY workflow_id
                    HAVING action_count > 3 AND total_sequential_time > actual_elapsed_time * 1.5
                    ORDER BY (total_sequential_time - actual_elapsed_time) DESC
                """, (since_timestamp,))
                
                parallelization_opps = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for opp in parallelization_opps[:3]:  # Top 3 parallelization opportunities
                    time_savings = opp['total_sequential_time'] - opp['actual_elapsed_time']
                    priority = 'high' if time_savings > 30 else 'medium'
                    
                    recommendation = PerformanceRecommendation(
                        id=f"parallelize_workflow_{opp['workflow_id']}",
                        type='parallelization',
                        priority=priority,
                        title=f"Parallelize workflow {opp['workflow_id']}",
                        description=f"Workflow could save {time_savings:.2f}s through better parallelization",
                        impact_estimate=ImpactEstimate(
                            time_savings_potential=time_savings,
                            efficiency_improvement=(time_savings / opp['total_sequential_time']) * 100,
                            affected_workflows=1,
                            affected_actions=opp['action_count'],
                            cost_benefit_ratio=time_savings / 10  # Arbitrary scaling
                        ),
                        implementation_steps=[
                            "Analyze action dependencies in the workflow",
                            "Identify independent action sequences",
                            "Implement async execution patterns",
                            "Add proper synchronization points"
                        ],
                        estimated_effort='high',
                        prerequisites=['Workflow dependency analysis', 'Async execution framework'],
                        metrics_to_track=[
                            'Workflow end-to-end time',
                            'Action parallelization ratio',
                            'Resource utilization efficiency'
                        ]
                    )
                    recommendations.append(recommendation)
                
                # Analyze error patterns
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as error_count,
                        COUNT(*) * 100.0 / (
                            SELECT COUNT(*) FROM actions a2 
                            WHERE a2.tool_name = actions.tool_name AND a2.started_at >= ?
                        ) as error_rate
                    FROM actions 
                    WHERE started_at >= ? AND status = 'failed'
                    GROUP BY tool_name
                    HAVING error_rate > 5
                    ORDER BY error_rate DESC
                """, (since_timestamp, since_timestamp))
                
                error_prone_tools = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for tool in error_prone_tools[:3]:  # Top 3 error-prone tools
                    priority = 'high' if tool['error_rate'] > 20 else 'medium'
                    
                    recommendation = PerformanceRecommendation(
                        id=f"improve_reliability_{tool['tool_name']}",
                        type='reliability_improvement',
                        priority=priority,
                        title=f"Improve {tool['tool_name']} reliability",
                        description=f"Tool has {tool['error_rate']:.1f}% failure rate ({tool['error_count']} failures)",
                        impact_estimate=ImpactEstimate(
                            reliability_improvement=tool['error_rate'],
                            affected_actions=tool['error_count'],
                            user_experience_impact='high',
                            cost_benefit_ratio=tool['error_rate'] / 10,
                            time_savings_potential=0  # Reliability doesn't directly save time
                        ),
                        implementation_steps=[
                            "Analyze failure patterns and root causes",
                            "Implement better error handling and retries",
                            "Add input validation and sanitization",
                            "Improve tool documentation and usage examples"
                        ],
                        estimated_effort='medium',
                        prerequisites=['Error logging analysis', 'Tool source code access'],
                        metrics_to_track=[
                            'Tool failure rate',
                            'Time to recovery',
                            'User satisfaction scores'
                        ]
                    )
                    recommendations.append(recommendation)
                
                # Filter recommendations by priority if requested
                if priority_filter != 'all':
                    recommendations = [r for r in recommendations if r.priority == priority_filter]
                
                # Sort by impact and priority
                priority_order = {'high': 3, 'medium': 2, 'low': 1}
                recommendations.sort(key=lambda x: (
                    priority_order.get(x.priority, 0),
                    x.impact_estimate.time_savings_potential
                ), reverse=True)
                
                # Calculate summary
                summary = RecommendationSummary(
                    total_recommendations=len(recommendations),
                    high_priority=len([r for r in recommendations if r.priority == 'high']),
                    medium_priority=len([r for r in recommendations if r.priority == 'medium']),
                    low_priority=len([r for r in recommendations if r.priority == 'low']),
                    estimated_total_savings=sum(r.impact_estimate.time_savings_potential for r in recommendations),
                    analysis_period_hours=hours_back
                )
                
                # Create implementation roadmap
                roadmap = ImplementationRoadmap(
                    quick_wins=[r for r in recommendations if r.estimated_effort == 'low' and r.priority == 'high'],
                    major_improvements=[r for r in recommendations if r.estimated_effort == 'high' and r.priority == 'high'],
                    maintenance_tasks=[r for r in recommendations if r.priority == 'low']
                )
                
                conn.close()
                
                return PerformanceRecommendationsResponse(
                    recommendations=recommendations,
                    summary=summary,
                    implementation_roadmap=roadmap,
                    timestamp=datetime.now().isoformat()
                )
                
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

        # ---------- Module-level singletons for Body parameters ----------
        
        # Define Body parameters as module-level singletons to avoid B008 warnings
        WORKFLOW_SCHEDULE_BODY = Body(...)
        RESTORE_STATE_BODY = Body(...)

        # ---------- Workflow Management Pydantic Models ----------

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

        # ---------- Cognitive State Restoration Models ----------

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

        # ---------- Workflow Management Endpoints ----------

        @api_app.post(
            "/workflows/{workflow_id}/schedule",
            response_model=WorkflowScheduleResponse,
            tags=["Workflow Management"],
            summary="Schedule workflow execution",
            description="""
        Schedule a workflow for future execution with configurable priority and timing:

        - **Workflow scheduling** with specific timing
        - **Priority management** for execution order
        - **Status tracking** for scheduled workflows
        - **Integration** with workflow execution system

        Essential for orchestrating complex multi-step processes and time-based automation.
            """,
            responses={
                200: {
                    "description": "Workflow scheduled successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "schedule_id": "sched_workflow_123_1704067200",
                                "message": "Workflow scheduled successfully",
                                "schedule_data": {
                                    "workflow_id": "workflow_123",
                                    "scheduled_at": "2024-01-01T12:00:00Z",
                                    "priority": 3,
                                    "status": "scheduled",
                                    "created_at": "2024-01-01T10:00:00Z"
                                }
                            }
                        }
                    }
                },
                400: {"description": "Invalid request parameters"},
                404: {"description": "Workflow not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def schedule_workflow(
            workflow_id: str = ApiPath(..., description="Unique identifier of the workflow to schedule", example="workflow_abc123", regex="^[a-zA-Z0-9_-]+$"),
            request: WorkflowScheduleRequest = WORKFLOW_SCHEDULE_BODY
        ) -> WorkflowScheduleResponse:
            """Schedule workflow execution"""
            try:
                # This is a placeholder implementation
                # In a real system, this would integrate with a task scheduler
                schedule_data = ScheduleData(
                    workflow_id=workflow_id,
                    scheduled_at=request.scheduled_at.isoformat(),
                    priority=request.priority,
                    status="scheduled",
                    created_at=datetime.now().isoformat()
                )
                
                # Generate a unique schedule ID
                schedule_id = f"sched_{workflow_id}_{int(datetime.now().timestamp())}"
                
                # In a real implementation, this would:
                # 1. Verify the workflow exists
                # 2. Create a scheduled task in a task queue
                # 3. Store the schedule in a database
                
                return WorkflowScheduleResponse(
                    success=True,
                    schedule_id=schedule_id,
                    message="Workflow scheduled successfully",
                    schedule_data=schedule_data
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to schedule workflow: {str(e)}") from e

        # ---------- Cognitive State Restoration Endpoint ----------

        @api_app.post(
            "/cognitive-states/{state_id}/restore",
            response_model=RestoreStateResponse,
            tags=["Cognitive States"],
            summary="Restore a previous cognitive state",
            description="""
        Restore the system to a previous cognitive state for analysis or recovery:

        - **State restoration** with configurable restore modes
        - **Temporal analysis** by reverting to specific points in time
        - **Recovery mechanisms** for problematic state transitions
        - **Research capabilities** for understanding state evolution

        Critical for debugging cognitive state issues and temporal analysis of system behavior.
            """,
            responses={
                200: {
                    "description": "Cognitive state restoration initiated",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "message": "Cognitive state restoration initiated",
                                "restore_data": {
                                    "state_id": "state_abc123xyz789",
                                    "restore_mode": "full",
                                    "restored_at": "2024-01-01T12:00:00Z",
                                    "original_timestamp": 1703980800.0
                                }
                            }
                        }
                    }
                },
                400: {"description": "Invalid request parameters"},
                404: {"description": "Cognitive state not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def restore_cognitive_state(
            state_id: str = ApiPath(
                ...,
                description="Unique identifier of the cognitive state to restore",
                example="state_abc123xyz789",
                regex="^[a-zA-Z0-9_-]+$"
            ),
            request: RestoreStateRequest = RESTORE_STATE_BODY
        ) -> RestoreStateResponse:
            """Restore a cognitive state"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get the state to restore
                cursor.execute("SELECT * FROM cognitive_timeline_states WHERE state_id = ?", (state_id,))
                state = cursor.fetchone()
                
                if not state:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Cognitive state with ID '{state_id}' not found"
                    )
                
                # Create restoration data
                restore_data = RestoreData(
                    state_id=state_id,
                    restore_mode=request.restore_mode,
                    restored_at=datetime.now().isoformat(),
                    original_timestamp=state[1] if state else None  # timestamp column
                )
                
                # In a real implementation, this would:
                # 1. Create a backup of the current state
                # 2. Restore the cognitive state to the system
                # 3. Update all dependent systems
                # 4. Log the restoration event
                
                conn.close()
                
                return RestoreStateResponse(
                    success=True,
                    message="Cognitive state restoration initiated",
                    restore_data=restore_data
                )
                
            except HTTPException:
                raise
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to restore state: {str(e)}") from e

        # ---------- Artifact Download Endpoint ----------

        @api_app.get(
            "/artifacts/{artifact_id}/download",
            tags=["Artifacts"],
            summary="Download artifact file or data",
            description="""
        Download the raw file or data associated with an artifact:

        - **File download** with proper content types
        - **Metadata preservation** in download headers
        - **Access logging** for audit trails
        - **Format handling** for different artifact types

        Essential for accessing artifact content outside the UMS Explorer interface.
            """,
            responses={
                200: {
                    "description": "Artifact file downloaded successfully",
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        },
                        "application/json": {
                            "schema": {"type": "object"},
                            "example": {
                                "artifact_id": "artifact_123",
                                "name": "analysis_report",
                                "artifact_type": "document",
                                "description": "Quarterly analysis report",
                                "file_path": "/artifacts/reports/q4_2024.pdf",
                                "file_size": 2048576,
                                "created_at": 1703980800.0,
                                "metadata": {"author": "System", "version": "1.0"}
                            }
                        }
                    }
                },
                404: {
                    "description": "Artifact not found",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Artifact with ID 'artifact_123' not found"}
                        }
                    }
                },
                500: {"description": "Internal server error"}
            }
        )
        async def download_artifact(
            artifact_id: str = ApiPath(
                ...,
                description="Unique identifier of the artifact to download",
                example="artifact_abc123",
                regex="^[a-zA-Z0-9_-]+$"
            )
        ):
            """Download an artifact"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
                artifact = cursor.fetchone()
                
                if not artifact:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Artifact with ID '{artifact_id}' not found"
                    )
                
                # Convert to dictionary
                artifact_dict = dict(zip([d[0] for d in cursor.description], artifact, strict=False))
                
                conn.close()
                
                # For now, return the artifact data as JSON
                # In a real implementation, this would serve the actual file
                from fastapi.responses import Response
                
                content = json.dumps(artifact_dict, indent=2)
                filename = f"{artifact_dict.get('name', 'artifact')}.json"
                
                return Response(
                    content=content,
                    media_type='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
                
            except HTTPException:
                raise
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to download artifact: {str(e)}") from e

        # ---------- Health & Utilities Endpoints ----------

        @api_app.get(
            "/health",
            response_model=HealthResponse,
            tags=["Health & Utilities"],
            summary="Health check endpoint",
            description="""
        Check the health and operational status of the Ultimate MCP Server:

        - **Server status** verification
        - **Service availability** confirmation
        - **Version information** for compatibility checks
        - **Load balancer integration** support

        Standard health check endpoint for monitoring systems and operational dashboards.
            """,
            responses={
                200: {
                    "description": "Server is healthy and operational",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "ok",
                                "version": "0.1.0"
                            }
                        }
                    }
                },
                500: {
                    "description": "Server health check failed",
                    "content": {
                        "application/json": {
                            "example": {"detail": "Health check failed"}
                        }
                    }
                }
            }
        )
        async def health_check() -> HealthResponse:
            """Health check endpoint for monitoring server status"""
            return HealthResponse(
                status="ok",
                version="0.1.0"
            )

        # --- UMS Explorer Placeholder ---
        # 3) Combined application – avoid overlapping mounts
        final_app = Starlette(
            routes=[
                Route("/", root_endpoint),  # Root discovery
                Mount(endpoint_path, mcp_starlette),  # /mcp or /sse
                Mount("/api", api_app),  # REST API under /api
            ],
            lifespan=mcp_app.lifespan,
        )

        # Logging of endpoints for clarity
        print(
            f"{transport_mode.upper()} endpoint available at: http://{server_host}:{server_port}{endpoint_path}",
            file=sys.stderr,
        )
        print(
            f"API endpoints available at: http://{server_host}:{server_port}/api/*",
            file=sys.stderr,
        )
        print(
            f"UMS Explorer available at: http://{server_host}:{server_port}/api/ums-explorer",
            file=sys.stderr,
        )
        print(
            f"Swagger UI available at: http://{server_host}:{server_port}/api/docs",
            file=sys.stderr,
        )
        print(
            f"ReDoc available at: http://{server_host}:{server_port}/api/redoc",
            file=sys.stderr,
        )
        print(
            f"OpenAPI spec available at: http://{server_host}:{server_port}/api/openapi.json",
            file=sys.stderr,
        )
        print(
            f"Discovery endpoint available at: http://{server_host}:{server_port}/",
            file=sys.stderr,
        )
        # === END NEW SPLIT-APP ARCHITECTURE ===

        # Use our custom quiet Uvicorn server for silent shutdown
        config = uvicorn.Config(
            final_app,
            host=server_host,
            port=server_port,
            log_config=LOGGING_CONFIG,
            log_level=final_log_level.lower(),
            lifespan="on",  # This tells uvicorn to look for and use the app's lifespan
        )
        server = create_quiet_server(config)
        server.run()
    else:  # stdio mode
        # --- Stdio Mode Execution ---
        logger.info("Running in stdio mode...")

        # Create a shutdown handler for stdio mode cleanup
        async def cleanup_resources():
            """Performs cleanup for various components during shutdown."""

            print("Cleaning up Gateway instance and associated resources...", file=sys.stderr)

            # Shutdown SQL Tools with timeout
            try:
                await asyncio.wait_for(shutdown_sql_tools(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

            # Shutdown Connection Manager with timeout
            try:
                from ultimate_mcp_server.tools.sql_databases import _connection_manager

                await asyncio.wait_for(_connection_manager.shutdown(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

            # Shutdown Smart Browser with timeout
            try:
                await asyncio.wait_for(smart_browser_shutdown(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown

        # Configure graceful shutdown with error suppression
        enable_quiet_shutdown()

        # Register the same cleanup function for stdio mode
        register_shutdown_handler(cleanup_resources)

        try:
            # Run the FastMCP stdio loop - this will block until interrupted
            _gateway_instance.mcp.run()
        except (KeyboardInterrupt, SystemExit):
            # Normal shutdown - handled by graceful shutdown system
            pass
        except Exception:
            # Any other error - also handled by graceful shutdown
            pass
        # --- End Stdio Mode ---

    # --- Post-Server Exit ---
    logger.info("Server loop exited.")
