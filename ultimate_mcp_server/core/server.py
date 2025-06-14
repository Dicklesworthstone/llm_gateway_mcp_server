"""Main server implementation for Ultimate MCP Server."""

import asyncio
import logging
import logging.config
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Context, FastMCP

import ultimate_mcp_server

# Import core specifically to set the global instance
import ultimate_mcp_server.core
from ultimate_mcp_server.config import get_config, load_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.state_store import StateStore

# Import UMS API utilities and database functions
from ultimate_mcp_server.core.ums_api import (
    setup_ums_api,
)
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
        from starlette.routing import Mount

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

        # Setup all UMS API endpoints
        setup_ums_api(api_app)

        # --- UMS Explorer Placeholder ---
        # 3) Combined application – avoid overlapping mounts
        final_app = Starlette(
            routes=[
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
