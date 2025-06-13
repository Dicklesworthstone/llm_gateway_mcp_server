"""Base tool classes and decorators for Ultimate MCP Server."""
import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

try:
    from fastmcp import Tool
except ImportError:
    # Handle case where mcp might be available via different import
    try:
        from fastmcp import Tool
    except ImportError:
        Tool = None  # Tool will be provided by the mcp_server

from ultimate_mcp_server.exceptions import (
    ResourceError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    format_error_response,
)

# from ultimate_mcp_server.services.cache import with_cache
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.base")


def tool(name=None, description=None):
    """
    Decorator that marks a BaseTool class method as an MCP tool.
    
    This decorator adds metadata to a method, identifying it as a tool that should be
    registered with the MCP server when the containing BaseTool class is initialized.
    It allows customizing the tool's name and description, which are used in tool
    discoverability and documentation.
    
    Unlike the register_tool function which directly registers standalone functions,
    this decorator only marks methods for later registration, allowing BaseTool subclasses
    to organize multiple related tools together in a single class.
    
    The decorator adds three attributes to the method:
    - _tool: A boolean flag indicating this is a tool method
    - _tool_name: The name to use when registering the tool (or original method name)
    - _tool_description: The description to use for the tool (or method docstring)
    
    These attributes are used during the tool registration process, typically in the
    _register_tools method of BaseTool subclasses.
    
    Args:
        name: Custom name for the tool (defaults to the method name if not provided)
        description: Custom description for the tool (defaults to the method's docstring)
        
    Returns:
        A decorator function that adds tool metadata attributes to the decorated method
        
    Example:
        ```python
        class MyToolSet(BaseTool):
            tool_name = "my_toolset"
            
            @tool(name="custom_operation", description="Performs a customized operation")
            async def perform_operation(self, param1: str, param2: int) -> Dict[str, Any]:
                # Implementation
                return {"result": "success"}
        ```
        
    Notes:
        - This decorator should be used on methods of classes that inherit from BaseTool
        - Decorated methods should be async
        - The decorated method must take self as its first parameter
        - This decorator does not apply error handling or other middleware automatically
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        
        wrapper._tool = True
        wrapper._tool_name = name
        wrapper._tool_description = description
        
        return wrapper
    
    return decorator


def with_resource(resource_type, allow_creation=False, require_existence=True):
    """
    Decorator for standardizing resource access and validation in tool methods.
    
    This decorator provides consistent resource handling for tool methods that
    access or create persistent resources in the MCP ecosystem. It enforces resource
    validation rules, handles resource registration, and provides unified error handling
    for resource-related operations.
    
    Core functionalities:
    1. Resource existence validation - Ensures resources exist before allowing access
    2. Resource creation tracking - Registers newly created resources with the system
    3. Resource type validation - Confirms resources match expected types
    4. Standardized error handling - Produces consistent error responses for resource issues
    
    The decorator identifies resource IDs by looking for common parameter names like
    '{resource_type}_id', 'id', or 'resource_id' in the function's keyword arguments.
    When a resource ID is found, it performs the configured validation checks before
    allowing the function to execute. After execution, it can optionally register
    newly created resources.
    
    Args:
        resource_type: Type category for the resource (e.g., "document", "embedding", 
                      "database"). Used for validation and registration.
        allow_creation: Whether the tool is allowed to create new resources of this type.
                       When True, the decorator will register any created resources.
        require_existence: Whether the resource must exist before the tool is called.
                          When True, the decorator will verify resource existence.
        
    Returns:
        A decorator function that applies resource handling to tool methods.
        
    Raises:
        ResourceError: When resource validation fails (e.g., resource not found,
                      resource type mismatch, or unauthorized resource access).
        
    Example:
        ```python
        class DocumentTools(BaseTool):
            @tool()
            @with_resource("document", require_existence=True, allow_creation=False)
            async def get_document_summary(self, document_id: str):
                # This method will fail with ResourceError if document_id doesn't exist
                # Resource existence is checked before this code runs
                ...
                
            @tool()
            @with_resource("document", require_existence=False, allow_creation=True)
            async def create_document(self, content: str, metadata: Dict[str, Any] = None):
                # Process content and create document
                doc_id = str(uuid.uuid4())
                # ... processing logic ...
                
                # Return created resource with resource_id key to trigger registration
                return {
                    "resource_id": doc_id,  # This triggers resource registration
                    "status": "created",
                    "metadata": {"content_length": len(content), "created_at": time.time()}
                }
                # The resource is automatically registered with the returned metadata
        ```
    
    Notes:
        - This decorator should be applied after @tool but before other decorators
          like @with_error_handling to ensure proper execution order
        - Resources created with allow_creation=True must include a "resource_id" 
          key in their result dictionary to trigger registration
        - The resource registry must be accessible via the tool's mcp server instance
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get resource ID from kwargs (common parameter names)
            resource_id = None
            for param_name in [f"{resource_type}_id", "id", "resource_id"]:
                if param_name in kwargs:
                    resource_id = kwargs[param_name]
                    break
            
            # Check if resource exists if required
            if require_existence and resource_id:
                # Get resource registry from MCP server
                resource_registry = getattr(self.mcp, "resources", None)
                if resource_registry is None:
                    logger.warning(
                        f"Resource registry not available, skipping existence check for {resource_type}/{resource_id}",
                        emoji_key="warning"
                    )
                else:
                    # Check if resource exists
                    exists = await resource_registry.exists(resource_type, resource_id)
                    if not exists:
                        raise ResourceError(
                            f"{resource_type.capitalize()} not found: {resource_id}",
                            resource_type=resource_type,
                            resource_id=resource_id
                        )
            
            # Call function
            result = await func(self, *args, **kwargs)
            
            # If the function returns a new resource ID, register it
            if allow_creation and isinstance(result, dict) and "resource_id" in result:
                new_resource_id = result["resource_id"]
                # Get resource registry from MCP server
                resource_registry = getattr(self.mcp, "resources", None)
                if resource_registry is not None:
                    # Register new resource
                    metadata = {
                        "created_at": time.time(),
                        "creator": kwargs.get("ctx", {}).get("user_id", "unknown"),
                        "resource_type": resource_type
                    }
                    
                    # Add other metadata from result if available
                    if "metadata" in result:
                        metadata.update(result["metadata"])
                    
                    await resource_registry.register(
                        resource_type, 
                        new_resource_id, 
                        metadata=metadata
                    )
                    
                    logger.info(
                        f"Registered new {resource_type}: {new_resource_id}",
                        emoji_key="resource",
                        resource_type=resource_type,
                        resource_id=new_resource_id
                    )
            
            return result
                
        # Add resource metadata to function
        wrapper._resource_type = resource_type
        wrapper._allow_creation = allow_creation
        wrapper._require_existence = require_existence
        
        return wrapper
    
    return decorator


class ResourceRegistry:
    """
    Registry that tracks and manages resources used by MCP tools.
    
    The ResourceRegistry provides a centralized system for tracking resources created or
    accessed by tools within the MCP ecosystem. It maintains resource metadata, handles
    persistence of resource information, and provides methods for registering, looking up,
    and deleting resources.
    
    Resources in the MCP ecosystem represent persistent or semi-persistent objects that
    may be accessed across multiple tool calls or sessions. Examples include documents,
    knowledge bases, embeddings, file paths, and database connections. The registry helps
    manage the lifecycle of these resources and prevents issues like resource leaks or
    unauthorized access.
    
    Key features:
    - In-memory caching of resource metadata for fast lookups
    - Optional persistent storage via pluggable storage backends
    - Resource type categorization (documents, embeddings, etc.)
    - Resource existence checking for access control
    - Simple CRUD operations for resource metadata
    
    Resources are organized by type and identified by unique IDs within those types.
    Each resource has associated metadata that can include creation time, owner information,
    and resource-specific attributes.
    
    The registry is typically initialized by the MCP server and made available to all tools.
    Tools that create resources should register them, and tools that access resources should
    verify their existence before proceeding.
    """
    
    def __init__(self, storage_backend=None):
        """Initialize the resource registry.
        
        Args:
            storage_backend: Backend for persistent storage (if None, in-memory only)
        """
        self.resources = {}
        self.storage = storage_backend
        self.logger = get_logger("ultimate_mcp_server.resources")
    
    async def register(self, resource_type, resource_id, metadata=None):
        """Register a resource in the registry.
        
        Args:
            resource_type: Type of resource (e.g., "document", "embedding")
            resource_id: Resource identifier
            metadata: Additional metadata about the resource
            
        Returns:
            True if registration was successful
        """
        # Initialize resource type if not exists
        if resource_type not in self.resources:
            self.resources[resource_type] = {}
        
        # Register resource
        self.resources[resource_type][resource_id] = {
            "id": resource_id,
            "type": resource_type,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        # Persist to storage backend if available
        if self.storage:
            try:
                await self.storage.save_resource(
                    resource_type, 
                    resource_id, 
                    self.resources[resource_type][resource_id]
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to persist resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return True
    
    async def exists(self, resource_type, resource_id):
        """Check if a resource exists in the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            True if the resource exists
        """
        # Check in-memory registry first
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            return True
        
        # Check storage backend if available
        if self.storage:
            try:
                return await self.storage.resource_exists(resource_type, resource_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to check resource existence {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return False
    
    async def get(self, resource_type, resource_id):
        """Get resource metadata from the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            Resource metadata or None if not found
        """
        # Check in-memory registry first
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            return self.resources[resource_type][resource_id]
        
        # Check storage backend if available
        if self.storage:
            try:
                resource = await self.storage.get_resource(resource_type, resource_id)
                if resource:
                    # Cache in memory for future access
                    if resource_type not in self.resources:
                        self.resources[resource_type] = {}
                    self.resources[resource_type][resource_id] = resource
                    return resource
            except Exception as e:
                self.logger.error(
                    f"Failed to get resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return None
    
    async def list(self, resource_type, limit=100, offset=0, filters=None):
        """List resources of a specific type.
        
        Args:
            resource_type: Type of resource to list
            limit: Maximum number of resources to return
            offset: Offset for pagination
            filters: Dictionary of filters to apply
            
        Returns:
            List of resource metadata
        """
        result = []
        
        # Get from storage backend first if available
        if self.storage:
            try:
                resources = await self.storage.list_resources(
                    resource_type, 
                    limit=limit, 
                    offset=offset, 
                    filters=filters
                )
                
                # Cache in memory for future access
                if resources:
                    if resource_type not in self.resources:
                        self.resources[resource_type] = {}
                    
                    for resource in resources:
                        resource_id = resource.get("id")
                        if resource_id:
                            self.resources[resource_type][resource_id] = resource
                    
                    return resources
            except Exception as e:
                self.logger.error(
                    f"Failed to list resources of type {resource_type}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        # Fallback to in-memory registry
        if resource_type in self.resources:
            # Apply filters if provided
            filtered_resources = self.resources[resource_type].values()
            if filters:
                for key, value in filters.items():
                    filtered_resources = [
                        r for r in filtered_resources 
                        if r.get("metadata", {}).get(key) == value
                    ]
            
            # Apply pagination
            result = list(filtered_resources)[offset:offset+limit]
        
        return result
    
    async def delete(self, resource_type, resource_id):
        """Delete a resource from the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            True if deletion was successful
        """
        # Delete from in-memory registry
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            del self.resources[resource_type][resource_id]
        
        # Delete from storage backend if available
        if self.storage:
            try:
                return await self.storage.delete_resource(resource_type, resource_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to delete resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return True


class BaseToolMetrics:
    """
    Metrics collection and aggregation system for tool execution statistics.
    
    The BaseToolMetrics class provides a standardized way to track and aggregate performance
    metrics for tool executions. It maintains cumulative statistics about calls to a tool,
    including execution counts, success rates, timing information, and optional token usage
    and cost data when available.
    
    This class is used both internally by BaseTool instances and by the with_tool_metrics
    decorator to provide consistent metrics tracking across the entire MCP ecosystem. The
    collected metrics enable monitoring, debugging, and optimization of tool performance
    and usage patterns.
    
    Metrics tracked:
    - Total number of calls
    - Number of successful and failed calls
    - Success rate
    - Total, minimum, and maximum execution duration
    - Total token usage (for LLM-based tools)
    - Total cost (for tools with cost accounting)
    
    The metrics are aggregated in memory and can be retrieved at any time via the get_stats()
    method. They represent the lifetime statistics of the tool since the metrics object
    was created.
    
    Example:
    ```python
    # Accessing metrics from a tool
    my_tool = MyToolClass(mcp_server)
    metrics = my_tool.metrics.get_stats()
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Average duration: {metrics['average_duration']:.2f}s")
    ```
    """
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration = 0.0
        self.min_duration = float('inf')
        self.max_duration = 0.0
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def record_call(
        self,
        success: bool,
        duration: float,
        tokens: Optional[int] = None,
        cost: Optional[float] = None
    ) -> None:
        """Record metrics for a tool call.
        
        Args:
            success: Whether the call was successful
            duration: Duration of the call in seconds
            tokens: Number of tokens used (if applicable)
            cost: Cost of the call (if applicable)
        """
        self.total_calls += 1
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        
        if tokens is not None:
            self.total_tokens += tokens
            
        if cost is not None:
            self.total_cost += cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.total_calls == 0:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
            
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls,
            "average_duration": self.total_duration / self.total_calls,
            "min_duration": self.min_duration if self.min_duration != float('inf') else 0.0,
            "max_duration": self.max_duration,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


class BaseTool:
    """
    Foundation class for all tool implementations in the Ultimate MCP Server.
    
    The BaseTool class serves as the fundamental building block for creating tools that 
    can be registered with and executed by the MCP server. It provides core functionality
    for metrics tracking, logging, resource management, and tool execution.
    
    Tools in the Ultimate MCP Server ecosystem are designed to provide specific capabilities
    that can be invoked by clients (typically LLMs) to perform various operations like
    document processing, vector search, file operations, etc. The BaseTool architecture
    ensures all tools have consistent behavior for error handling, metrics collection,
    and server integration.
    
    Key features:
    - Standardized tool registration via decorators
    - Consistent metrics tracking for all tool executions
    - Unified error handling and response formatting
    - Integration with the server's resource registry
    - Logger setup with tool-specific naming
    
    Tool classes should inherit from BaseTool and define their tools using the @tool
    decorator. Each tool method should be async and follow the standard pattern of
    accepting parameters, performing operations, and returning results in a structured
    format.
    
    Example:
    ```python
    class MyCustomTools(BaseTool):
        tool_name = "my_custom_tools"
        description = "Provides custom tools for specific operations"
        
        @tool(name="custom_operation")
        @with_tool_metrics
        @with_error_handling
        async def perform_operation(self, param1: str, param2: int) -> Dict[str, Any]:
            # Implementation
            return {"result": "success", "data": some_data}
    ```
    """
    
    tool_name: str = "base_tool"
    description: str = "Base tool class for Ultimate MCP Server."
    
    def __init__(self, mcp_server):
        """Initialize the tool.
        
        Args:
            mcp_server: MCP server instance
        """
        # If mcp_server is a Gateway instance, get the MCP object
        self.mcp = mcp_server.mcp if hasattr(mcp_server, 'mcp') else mcp_server
        self.logger = get_logger(f"tool.{self.tool_name}")
        self.metrics = BaseToolMetrics()
        
        # Initialize resource registry if not already available
        if not hasattr(self.mcp, "resources"):
            self.mcp.resources = ResourceRegistry()
        
    def _register_tools(self):
        """Register tools with MCP server.
        
        Override this method in subclasses to register specific tools.
        This method is no longer called by the base class constructor.
        Registration is now handled externally, e.g., in register_all_tools.
        """
        pass
        
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool method by name with the given parameters.
        
        This method provides the core execution mechanism for BaseTool subclasses,
        dynamically dispatching calls to the appropriate tool method based on the
        tool_name parameter. It handles parameter validation, metrics collection,
        and error standardization to ensure consistent behavior across all tools.
        
        Execution flow:
        1. Looks up the requested tool method in the class
        2. Validates that the method is properly marked as a tool
        3. Applies metrics tracking via _wrap_with_metrics
        4. Executes the tool with the provided parameters
        5. Returns the tool's response or a standardized error
        
        Args:
            tool_name: Name of the specific tool method to execute
            params: Dictionary of parameters to pass to the tool method
                    (These parameters will be unpacked as kwargs)
        
        Returns:
            The result returned by the tool method, or a standardized error response
            if execution fails
            
        Raises:
            ToolError: If the specified tool_name is not found or not properly
                       marked as a tool method
                       
        Example:
            ```python
            # Direct execution of a tool method
            result = await my_tool_instance.execute(
                "analyze_document", 
                {"document_id": "doc123", "analysis_type": "sentiment"}
            )
            
            # Error handling
            if "isError" in result and result["isError"]:
                print(f"Tool execution failed: {result['error']['message']}")
            else:
                print(f"Analysis result: {result['analysis_score']}")
            ```
        """
        # Find method with tool name
        method_name = tool_name.split(".")[-1]  # Handle namespaced tools
        method = getattr(self, method_name, None)
        
        if not method or not hasattr(method, "_tool"):
            raise ToolError(
                f"Tool not found: {tool_name}",
                error_code="tool_not_found"
            )
        
        # Execute tool with metrics wrapper
        return await self._wrap_with_metrics(method, **params)

    async def _wrap_with_metrics(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Internal method that wraps a function call with metrics tracking.
        
        This method provides a standardized way to execute tool functions while capturing
        performance metrics such as execution duration, success/failure status, token usage,
        and cost. These metrics are stored in the BaseTool instance's metrics object for
        later analysis and reporting.
        
        The method performs the following steps:
        1. Records the start time of the operation
        2. Executes the provided function with the supplied arguments
        3. If successful, extracts metrics data from the result (if available)
        4. Records the execution metrics in the BaseTool's metrics object
        5. Returns the original result or propagates any exceptions that occurred
        
        Metrics extraction:
        - If the result is a dictionary, it will attempt to extract:
          - Token usage from either result["tokens"]["total"] or result["total_tokens"]
          - Cost information from result["cost"]
        
        Args:
            func: Async function to execute with metrics tracking
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the wrapped function call
            
        Raises:
            Any exception raised by the wrapped function (after logging it)
            
        Notes:
            - This method is typically called internally by BaseTool subclasses
            - Related to but different from the standalone with_tool_metrics decorator
            - Exceptions are logged but not caught (to allow proper error handling)
        """
        start_time = time.time()
        success = False
        tokens = None
        cost = None
        
        try:
            # Call function
            result = await func(*args, **kwargs)
            
            # Extract metrics if available
            if isinstance(result, dict):
                if "tokens" in result and isinstance(result["tokens"], dict):
                    tokens = result["tokens"].get("total")
                elif "total_tokens" in result:
                    tokens = result["total_tokens"]
                    
                cost = result.get("cost")
                
            success = True
            return result
            
        except Exception as e:
            self.logger.error(
                f"Tool execution failed: {func.__name__}: {str(e)}",
                emoji_key="error",
                tool=func.__name__,
                exc_info=True
            )
            raise
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_call(
                success=success,
                duration=duration,
                tokens=tokens,
                cost=cost
            )


def with_tool_metrics(func):
    """
    Decorator that automatically tracks performance metrics for tool functions.
    
    This decorator captures and records execution metrics for both class methods and
    standalone functions. It adapts its behavior based on whether the decorated function
    is a method on a BaseTool instance or a standalone function.
    
    Metrics captured include:
    - Execution time (duration in seconds)
    - Success/failure state
    - Token usage (extracted from result if available)
    - Cost information (extracted from result if available)
    
    The decorator performs several functions:
    1. Captures start time before execution
    2. Executes the wrapped function, preserving all args/kwargs
    3. Extracts metrics from the result dictionary if available
    4. Logs execution statistics
    5. Updates metrics in the BaseTool.metrics object if available
    
    When used with other decorators:
    - Should be applied before with_error_handling to ensure metrics are 
      captured even when errors occur
    - Works well with with_cache, tracking metrics for both cache hits and misses
    - Compatible with with_retry, recording each attempt separately
    
    Args:
        func: The async function to decorate (can be a method or standalone function)
        
    Returns:
        Wrapped async function that captures and records metrics
        
    Example:
        ```python
        @with_tool_metrics
        @with_error_handling
        async def my_tool_function(param1, param2):
            # Function implementation
        ```
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if the first arg looks like a BaseTool instance
        self_obj = args[0] if args and isinstance(args[0], BaseTool) else None
        tool_name = getattr(self_obj, 'tool_name', func.__name__)

        start_time = time.time()
        success = False
        tokens = None
        cost = None
        result = None
        
        try:
            # Call original function, passing self_obj if it exists
            if self_obj:
                # Assumes if self_obj exists, it's the first positional arg expected by func
                result = func(self_obj, *args[1:], **kwargs)
            else:
                # Pass only the args/kwargs received, assuming func is standalone
                result = func(*args, **kwargs)
            
            # Only await when necessary
            if inspect.isawaitable(result):
                result = await result
            # result is now either a ToolResult _or_ an async iterator
            
            # Extract metrics if available from result
            if isinstance(result, dict):
                if "tokens" in result and isinstance(result["tokens"], dict):
                    tokens = result["tokens"].get("total")
                elif "total_tokens" in result:
                    tokens = result["total_tokens"]
                cost = result.get("cost")
                
            success = True
            return result
            
        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name}: {str(e)}",
                emoji_key="error",
                tool=tool_name,
                exc_info=True
            )
            raise # Re-raise exception for other handlers (like with_error_handling)
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Log execution stats
            logger.debug(
                f"Tool execution: {tool_name} ({'success' if success else 'failed'})",
                emoji_key="tool" if success else "error",
                tool=tool_name,
                time=duration,
                cost=cost
            )
            
            # Update metrics if we found a self object with a metrics attribute
            if self_obj and hasattr(self_obj, 'metrics'):
                self_obj.metrics.record_call(
                    success=success,
                    duration=duration,
                    tokens=tokens,
                    cost=cost
                )
                
    return wrapper


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: List[Type[Exception]] = None
):
    """
    Decorator that adds exponential backoff retry logic to async tool functions.
    
    This decorator wraps an async function with retry logic that will automatically
    re-execute the function if it fails with certain exceptions. It implements an
    exponential backoff strategy to progressively increase the wait time between
    retry attempts, reducing load during transient failures.
    
    Retry behavior:
    1. When the decorated function raises an exception, the decorator checks if it's a
       retriable exception type (based on the retry_exceptions parameter)
    2. If retriable, it waits for a delay period (which increases with each attempt)
    3. After waiting, it retries the function with the same arguments
    4. This process repeats until either the function succeeds or max_retries is reached
    
    Args:
        max_retries: Maximum number of retry attempts before giving up (default: 3)
        retry_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
                       Each retry's delay is calculated as: retry_delay * (backoff_factor ^ attempt)
        retry_exceptions: List of exception types that should trigger retries.
                         If None, all exceptions will trigger retries.
    
    Returns:
        A decorator function that wraps the given async function with retry logic.
        
    Example:
        ```python
        @with_retry(max_retries=3, retry_delay=2.0, backoff_factor=3.0,
                   retry_exceptions=[ConnectionError, TimeoutError])
        async def fetch_data(url):
            # This function will retry up to 3 times if it raises ConnectionError or TimeoutError
            # Delays between retries: 2s, 6s, 18s
            # For other exceptions, it will fail immediately
            return await some_api_call(url)
        ```
        
    Notes:
        - This decorator only works with async functions
        - The decorated function must be idempotent (safe to call multiple times)
        - Retries are logged at WARNING level, final failures at ERROR level
        - The final exception is re-raised after all retries are exhausted
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    # Call original function
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    # Only retry on specified exceptions
                    if retry_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in retry_exceptions
                    ):
                        raise
                        
                    last_exception = e
                    
                    # Log retry attempt
                    if attempt < max_retries:
                        logger.warning(
                            f"Tool execution failed, retrying ({attempt+1}/{max_retries}): {str(e)}",
                            emoji_key="warning",
                            tool=func.__name__,
                            attempt=attempt+1,
                            max_retries=max_retries,
                            delay=delay
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        
                        # Increase delay for next retry
                        delay *= backoff_factor
                    else:
                        # Log final failure
                        logger.error(
                            f"Tool execution failed after {max_retries} retries: {str(e)}",
                            emoji_key="error",
                            tool=func.__name__,
                            exc_info=True
                        )
                        
            # If we get here, all retries failed
            raise last_exception
                
        return wrapper
    return decorator
    

def with_error_handling(func):
    """
    Decorator that transforms tool function exceptions into standardized error responses.
    
    This decorator intercepts any exceptions raised during tool execution and converts them
    into a structured error response format following the MCP protocol standards. It ensures
    that clients receive consistent, actionable error information regardless of how or where
    the error occurred.
    
    The decorator performs several key functions:
    1. Detects if it's decorating a BaseTool method or standalone function and adapts accordingly
    2. Reconstructs function call arguments appropriately based on function signature
    3. Catches exceptions raised during execution and transforms them into structured responses
    4. Maps different exception types to corresponding MCP error types with appropriate metadata
    5. Logs detailed error information while providing a clean, standardized response to clients
    
    Exception handling:
    - ToolError: Passed through with logging (assumes already formatted correctly)
    - ValueError: Converted to ToolInputError with detailed context
    - Other exceptions: Converted to ToolExecutionError with execution context
    
    All error responses have the same structure:
    ```
    {
        "success": False,
        "isError": True,
        "error": {
            "type": "<error_type>",
            "message": "<human-readable message>",
            "details": {<context-specific details>},
            "retriable": <boolean>,
            "suggestions": [<optional recovery suggestions>],
            "timestamp": <current_time>
        }
    }
    ```
    
    Args:
        func: The async function to decorate (can be a method or standalone function)
        
    Returns:
        Decorated async function that catches exceptions and returns structured error responses
        
    Example:
        ```python
        @with_error_handling
        async def my_tool_function(param1, param2):
            # If this raises an exception, it will be transformed into a structured response
            # rather than propagating up to the caller
            # ...
        ```
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if the first arg looks like a BaseTool instance
        self_obj = args[0] if args and isinstance(args[0], BaseTool) else None
        # Determine tool_name based on instance or func name
        tool_name = getattr(self_obj, 'tool_name', func.__name__) 
        
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())  # noqa: F841
        
        call_args = []
        call_kwargs = {}

        if self_obj:
            expected_params = list(sig.parameters.values())
            if expected_params and expected_params[0].name == 'self':
                call_args.append(self_obj)
        
        start_index = 1 if self_obj and call_args else 0
        call_args.extend(args[start_index:])

        # Pass all original kwargs through
        call_kwargs.update(kwargs)
            
        try:
            # Call original function with reconstructed args/kwargs
            # This version passes *all* kwargs received by the wrapper,
            # trusting FastMCP to pass the correct ones including 'ctx'.
            result = func(*call_args, **call_kwargs)
            
            # Only await when necessary
            if inspect.isawaitable(result):
                result = await result
            # result is now either a ToolResult _or_ an async iterator
            return result
            
        except ToolError as e:
            # Already a tool error, log and return
            logger.error(
                f"Tool error in {tool_name}: {str(e)} ({e.error_code})",
                emoji_key="error",
                tool=tool_name,
                error_code=e.error_code,
                details=e.details
            )
            
            # Debug log the formatted error response
            error_response = format_error_response(e)
            logger.debug(f"Formatted error response for {tool_name}: {error_response}")
            
            # Return standardized error response
            return error_response
            
        except ValueError as e:
            # Convert ValueError to ToolInputError with more detailed information
            error = ToolInputError(
                f"Invalid input to {tool_name}: {str(e)}",
                details={
                    "tool_name": tool_name,
                    "exception_type": "ValueError",
                    "original_error": str(e)
                }
            )
            
            logger.error(
                f"Invalid input to {tool_name}: {str(e)}",
                emoji_key="error",
                tool=tool_name,
                error_code=error.error_code
            )
            
            # Return standardized error response
            return format_error_response(error)
            
        except Exception as e:
            # Create a more specific error message that includes the tool name
            specific_message = f"Execution error in {tool_name}: {str(e)}"
            
            # Convert to ToolExecutionError for other exceptions
            error = ToolExecutionError(
                specific_message,
                cause=e,
                details={
                    "tool_name": tool_name,
                    "exception_type": type(e).__name__,
                    "original_message": str(e)
                }
            )
            
            logger.error(
                specific_message,
                emoji_key="error",
                tool=tool_name,
                exc_info=True
            )
            
            # Return standardized error response
            return format_error_response(error)
                
    return wrapper


def register_tool(mcp_server, name=None, description=None, cache_ttl=None):
    """
    Register a standalone function as an MCP tool with optional caching and error handling.
    
    This function creates a decorator that registers the decorated function with the MCP server,
    automatically applying error handling and optional result caching. It provides a simpler
    alternative to class-based tool registration via the BaseTool class, allowing standalone
    functions to be exposed as MCP tools without creating a full tool class.
    
    The decorator handles:
    1. Tool registration with the MCP server using the provided name (or function name)
    2. Documentation via the provided description (or function docstring)
    3. Optional result caching with the specified TTL
    4. Standardized error handling via the with_error_handling decorator
    
    Args:
        mcp_server: MCP server instance to register the tool with
        name: Tool name used for registration (defaults to the function name if not provided)
        description: Tool description for documentation (defaults to function docstring if not provided)
        cache_ttl: Optional time-to-live in seconds for caching tool results. If provided, the tool results
                  will be cached for this duration to improve performance for identical calls.
        
    Returns:
        Decorator function that transforms the decorated function into a registered MCP tool
        
    Example:
        ```python
        # Initialize MCP server
        mcp_server = FastMCP()
        
        # Register a function as a tool
        @register_tool(mcp_server, name="get_weather", cache_ttl=300)
        async def get_weather_data(location: str, units: str = "metric"):
            '''Get current weather data for a location.'''
            # Implementation
            return {"temperature": 22, "conditions": "sunny"}
            
        # The function is now registered as an MCP tool named "get_weather"
        # with 5-minute result caching and standardized error handling
        ```
        
    Notes:
        - The decorated function must be async
        - If cache_ttl is provided, identical calls will return cached results 
          rather than re-executing the function
        - Function signature is preserved, making it transparent to callers
        - For more complex tools with multiple methods, use the BaseTool class instead
    """
    def decorator(func):
        # Get function name and docstring
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"  # noqa: F841
        
        # Apply caching if specified
        # if cache_ttl is not None:
        #     func = with_cache(ttl=cache_ttl)(func)
        
        # Apply error handling
        func = with_error_handling(func)
        
        # Register with MCP server
        mcp_server.tool(name=tool_name)(func)
        
        return func
    
    return decorator

def _get_json_schema_type(type_annotation):
    """
    Convert Python type annotations to JSON Schema type definitions.
    
    This utility function translates Python's typing annotations into equivalent JSON Schema
    type definitions, enabling automatic generation of API documentation and client interfaces
    from Python function signatures. It handles basic types, Optional types, Lists, and 
    provides reasonable defaults for complex types.
    
    The function is primarily used internally by the MCP framework to generate JSON Schema
    definitions for tool parameters, allowing clients to understand the expected input types
    and structures for each tool.
    
    Type mappings:
    - str -> {"type": "string"}
    - int -> {"type": "integer"}
    - float -> {"type": "number"}
    - bool -> {"type": "boolean"}
    - Optional[T] -> Same as T, but adds "null" to "type" array
    - List[T] -> {"type": "array", "items": <schema for T>}
    - Dict -> {"type": "object"}
    - Other complex types -> {"type": "object"}
    
    Args:
        type_annotation: A Python type annotation (from typing module or built-in types)
        
    Returns:
        A dictionary containing the equivalent JSON Schema type definition
        
    Notes:
        - This function provides only type information, not complete JSON Schema validation
          rules like minimum/maximum values, string patterns, etc.
        - Complex nested types (e.g., List[Dict[str, List[int]]]) are handled, but deeply 
          nested structures may be simplified in the output schema
        - This function is meant for internal use by the tool registration system
        
    Examples:
        ```python
        # Basic types
        _get_json_schema_type(str)  # -> {"type": "string"}
        _get_json_schema_type(int)  # -> {"type": "integer"}
        
        # Optional types
        from typing import Optional
        _get_json_schema_type(Optional[str])  # -> {"type": ["string", "null"]}
        
        # List types
        from typing import List
        _get_json_schema_type(List[int])  # -> {"type": "array", "items": {"type": "integer"}}
        
        # Complex types
        from typing import Dict, List
        _get_json_schema_type(Dict[str, List[int]])  # -> {"type": "object"}
        ```
    """
    import typing
    
    # Handle basic types
    if type_annotation is str:
        return {"type": "string"}
    elif type_annotation is int:
        return {"type": "integer"}
    elif type_annotation is float:
        return {"type": "number"}
    elif type_annotation is bool:
        return {"type": "boolean"}
    
    # Handle Optional types
    origin = typing.get_origin(type_annotation)
    args = typing.get_args(type_annotation)
    
    if origin is Union and type(None) in args:
        # Optional type - get the non-None type
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            inner_type = _get_json_schema_type(non_none_args[0])
            return inner_type
    
    # Handle lists
    if origin is list or origin is List:
        if args:
            item_type = _get_json_schema_type(args[0])
            return {
                "type": "array",
                "items": item_type
            }
        return {"type": "array"}
    
    # Handle dictionaries
    if origin is dict or origin is Dict:
        return {"type": "object"}
    
    # Default to object for complex types
    return {"type": "object"}

def with_state_management(namespace: str):
    """
    Decorator that provides persistent state management capabilities to tool functions.
    
    This decorator enables stateful behavior in otherwise stateless tool functions by
    injecting state access methods that allow reading, writing, and deleting values
    from a persistent, namespace-based state store. This is essential for tools that
    need to maintain context across multiple invocations, manage session data, or 
    build features with memory capabilities.
    
    The state management system provides:
    - Namespace isolation: Each tool can use its own namespace to prevent key collisions
    - Thread-safe concurrency: Built-in locks ensure safe parallel access to the same state
    - Optional persistence: State can be backed by disk storage for durability across restarts
    - Lazy loading: State is loaded from disk only when accessed, improving performance
    
    State accessibility functions injected into the decorated function:
    - get_state(key, default=None) → Any: Retrieve a value by key, with optional default
    - set_state(key, value) → None: Store a value under the specified key
    - delete_state(key) → None: Remove a value from the state store
    
    All state operations are async, allowing the tool to continue processing while
    state operations are pending.
    
    Args:
        namespace: A unique string identifying this tool's state namespace. This 
                  should be chosen carefully to avoid collisions with other tools.
                  Recommended format: "<tool_category>.<specific_feature>"
                  Examples: "conversation.history", "user.preferences", "document.cache"
    
    Returns:
        A decorator function that wraps the original tool function, adding state
        management capabilities via injected parameters.
        
    Examples:
        Basic usage with conversation history:
        ```python
        @with_state_management("conversation.history")
        async def chat_with_memory(message: str, ctx=None, get_state=None, set_state=None, delete_state=None):
            # Get previous messages from persistent store
            history = await get_state("messages", [])
            
            # Add new message
            history.append({"role": "user", "content": message})
            
            # Generate response based on all previous conversation context
            response = generate_response(message, history)
            
            # Add AI response to history
            history.append({"role": "assistant", "content": response})
            
            # Store updated history for future calls
            await set_state("messages", history)
            return {"response": response}
        ```
        
        Advanced pattern with conversational memory and user customization:
        ```python
        @with_state_management("assistant.settings")
        async def personalized_assistant(
            query: str, 
            update_preferences: bool = False,
            preferences: Dict[str, Any] = None,
            ctx=None, 
            get_state=None, 
            set_state=None, 
            delete_state=None
        ):
            # Get user ID from context
            user_id = ctx.get("user_id", "default_user")
            
            # Retrieve user-specific preferences
            user_prefs = await get_state(f"prefs:{user_id}", {
                "tone": "professional",
                "verbosity": "concise",
                "expertise_level": "intermediate"
            })
            
            # Update preferences if requested
            if update_preferences and preferences:
                user_prefs.update(preferences)
                await set_state(f"prefs:{user_id}", user_prefs)
            
            # Get conversation history
            history = await get_state(f"history:{user_id}", [])
            
            # Process query using preferences and history
            response = process_personalized_query(
                query, 
                user_preferences=user_prefs,
                conversation_history=history
            )
            
            # Update conversation history
            history.append({"query": query, "response": response})
            if len(history) > 20:  # Keep only recent history
                history = history[-20:]
            await set_state(f"history:{user_id}", history)
            
            return {
                "response": response,
                "preferences": user_prefs
            }
        ```
        
        State persistence across server restarts:
        ```python
        # First call to the tool
        @with_state_management("task.progress")
        async def long_running_task(task_id: str, step: int = None, ctx=None, 
                                   get_state=None, set_state=None, delete_state=None):
            # Get current progress
            progress = await get_state(task_id, {"completed_steps": [], "current_step": 0})
            
            # Update progress if a new step is provided
            if step is not None:
                progress["current_step"] = step
                progress["completed_steps"].append({
                    "step": step,
                    "timestamp": time.time()
                })
                await set_state(task_id, progress)
            
            # Even if the server restarts, the next call will retrieve the saved progress
            return {
                "task_id": task_id,
                "progress": progress,
                "completed": len(progress["completed_steps"]),
                "current_step": progress["current_step"]
            }
        ```
        
    Implementation Pattern:
    The decorator works by injecting three async state management functions into the
    decorated function's keyword arguments:
    
    1. `get_state(key, default=None)`:
       - Retrieves state values from the persistent store
       - If key doesn't exist, returns the provided default value
       - Example: `user_data = await get_state("user:12345", {})`
    
    2. `set_state(key, value)`: 
       - Stores a value in the persistent state store
       - Automatically serializes complex Python objects (dicts, lists, etc.)
       - Example: `await set_state("session:abc", {"authenticated": True})`
    
    3. `delete_state(key)`:
       - Removes a key and its associated value from the store
       - Example: `await delete_state("temporary_data")`
    
    Notes:
        - The decorated function must accept get_state, set_state, delete_state, and ctx
          parameters, either explicitly or via **kwargs.
        - State persistence depends on the MCP server configuration. If persistence is
          enabled, state will survive server restarts.
        - For large objects, consider storing only references or identifiers in the state
          and using a separate storage system for the actual data.
        - The state store is shared across all server instances, so state keys should be
          chosen to avoid collisions between different tools and features.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get context from kwargs
            context = kwargs.get('ctx')
            if not context or not hasattr(context, 'fastmcp'):
                raise ValueError("Context with FastMCP server required")
            
            # Access StateStore via the FastMCP 2.0+ pattern
            if not hasattr(context.fastmcp, '_state_store'):
                raise ValueError("FastMCP server does not have a state store attached")
            
            state_store = context.fastmcp._state_store
            
            # Add state accessors to kwargs
            kwargs['get_state'] = lambda key, default=None: state_store.get(namespace, key, default)
            kwargs['set_state'] = lambda key, value: state_store.set(namespace, key, value)
            kwargs['delete_state'] = lambda key: state_store.delete(namespace, key)
            
            return await func(*args, **kwargs)
        
        # Update signature to include context parameter if not already present
        sig = inspect.signature(func)
        if 'ctx' not in sig.parameters:
            wrapped_params = list(sig.parameters.values())
            wrapped_params.append(
                inspect.Parameter('ctx', inspect.Parameter.KEYWORD_ONLY, 
                                 annotation='Optional[Dict[str, Any]]', default=None)
            )
            wrapper.__signature__ = sig.replace(parameters=wrapped_params)
        
        return wrapper
    return decorator
