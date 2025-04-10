"""MCP Tools for LLM Gateway."""

import inspect
from typing import Any, Dict, Type

# Import base decorators/classes that might be used by other tool modules
from llm_gateway.tools.base import (
    BaseTool,  # Keep BaseTool in case other modules use it
    register_tool,  # Keep if used elsewhere, otherwise remove
    with_error_handling,  # Make sure this is available if used directly
    with_retry,
    with_tool_metrics,
)

# Import other tool modules/classes if they exist and follow the class pattern
# Removed: from llm_gateway.tools.optimization import OptimizationTools
from llm_gateway.utils import get_logger

# Import standalone tool functions
from .completion import chat_completion, generate_completion, multi_completion, stream_completion

# Import new standalone functions from document.py
from .document import (
    chunk_document,
    extract_entities,
    generate_qa_pairs,
    process_document_batch,
    summarize_document,
)

# Import new standalone functions from extraction.py
from .extraction import (
    extract_code_from_response,
    extract_json,
    extract_key_value_pairs,
    extract_semantic_schema,
    extract_table,
)
from .meta import (
    get_llm_instructions,
    get_tool_info,
    get_tool_recommendations,
)

# Import standalone functions from optimization.py
from .optimization import (
    compare_models,
    estimate_cost,
    execute_optimized_workflow,
    recommend_model,
)
from .provider import get_provider_status, list_models

# Import new standalone functions from rag.py
from .rag import (
    add_documents,
    create_knowledge_base,
    delete_knowledge_base,
    generate_with_rag,
    list_knowledge_bases,
    retrieve_context,
)

# Import standalone functions from tournament.py
from .tournament import (
    cancel_tournament,
    create_tournament,
    get_tournament_results,
    get_tournament_status,
    list_tournaments,
)

# Update __all__ - remove classes/functions that are no longer relevant or defined
__all__ = [
    # Base decorators/classes
    "BaseTool",
    "with_tool_metrics",
    "with_retry",
    "with_error_handling",
    "register_tool", 
    
    # Standalone tool functions (explicitly list them)
    "generate_completion",
    "stream_completion",
    "chat_completion",
    "multi_completion",
    "get_provider_status",
    "list_models",
    "get_tool_info",
    "get_llm_instructions",
    "get_tool_recommendations",
    "chunk_document",
    "summarize_document",
    "extract_entities",
    "generate_qa_pairs",
    "process_document_batch",
    "extract_json",
    "extract_table",
    "extract_key_value_pairs",
    "extract_semantic_schema",
    "create_knowledge_base",
    "list_knowledge_bases",
    "delete_knowledge_base",
    "add_documents",
    "retrieve_context",
    "generate_with_rag",
    "create_tournament",
    "get_tournament_status",
    "list_tournaments",
    "get_tournament_results",
    "cancel_tournament",
    "estimate_cost", # Added Optimization functions
    "compare_models",
    "recommend_model",
    "execute_optimized_workflow",
    
    # Other tool classes (to be refactored) - Should be empty now
    # Removed: "OptimizationTools",
    
    # Utility functions
    "extract_code_from_response",
]

logger = get_logger("llm_gateway.tools")

# Removed TOOL_REGISTRY

# --- Tool Registration --- 

# List of standalone functions to register
STANDALONE_TOOL_FUNCTIONS = [
    generate_completion,
    stream_completion,
    chat_completion,
    multi_completion,
    get_provider_status,
    list_models,
    get_tool_info,
    get_llm_instructions,
    get_tool_recommendations,
    chunk_document,
    summarize_document,
    extract_entities,
    generate_qa_pairs,
    process_document_batch,
    extract_json,
    extract_table,
    extract_key_value_pairs,
    extract_semantic_schema,
    create_knowledge_base,
    list_knowledge_bases,
    delete_knowledge_base,
    add_documents,
    retrieve_context,
    generate_with_rag,
    create_tournament,
    get_tournament_status,
    list_tournaments,
    get_tournament_results,
    cancel_tournament,
    estimate_cost, # Added Optimization functions
    compare_models,
    recommend_model,
    execute_optimized_workflow,
]

# Registry of tool classes (for tools still using the class pattern) - Should be empty
CLASS_BASED_TOOL_REGISTRY: Dict[str, Type[BaseTool]] = {
    # Removed: "optimization": OptimizationTools,
}


def register_all_tools(mcp_server) -> Dict[str, Any]:
    """Registers all tools (standalone and class-based) with the MCP server.

    Args:
        mcp_server: The MCP server instance.

    Returns:
        Dictionary containing information about registered tools.
    """
    logger.info("Calling register_all_tools to register standalone and class-based tools...")
    registered_tools: Dict[str, Any] = {}
    
    # --- Register Standalone Functions ---
    standalone_count = 0
    for tool_func in STANDALONE_TOOL_FUNCTIONS:
        if callable(tool_func) and inspect.iscoroutinefunction(tool_func):
            tool_name = tool_func.__name__
            # Use the function itself for the tool call, apply MCP decorator
            # The decorators like @with_tool_metrics are already applied
            mcp_server.tool(name=tool_name)(tool_func)
            registered_tools[tool_name] = {
                "description": inspect.getdoc(tool_func) or "",
                "type": "standalone_function"
                # We could potentially inspect signature for parameters later
            }
            logger.info(f"Registered standalone tool function: {tool_name}", emoji_key="⚙️")
            standalone_count += 1
        else:
            logger.warning(f"Item {getattr(tool_func, '__name__', repr(tool_func))} in STANDALONE_TOOL_FUNCTIONS is not a callable async function.")

    # --- Register Class-Based Tools (Refactoring Needed Here) ---
    # This section should ideally do nothing now if all tools are refactored.
    # Keeping the discovery logic commented out or removed might be cleaner.
    class_based_modules_to_skip = ["meta", "document", "extraction", "rag", "tournament", "optimization"] # Added optimization  # noqa: F841
    # No need to iterate if CLASS_BASED_TOOL_REGISTRY is empty and all modules are skipped.
    # class_based_modules = []
    # package = importlib.import_module(__name__.split('.')[0] + ".tools")
    
    # for _, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
    #     if not ispkg and not name.endswith(".base"):
    #         module_short_name = name.split('.')[-1]
    #         if module_short_name not in class_based_modules_to_skip:
    #              try:
    #                 module = importlib.import_module(name)
    #                 for attribute_name in dir(module):
    #                     attribute = getattr(module, attribute_name)
    #                     # Check if it's a class inheriting from BaseTool (but not BaseTool itself)
    #                     if inspect.isclass(attribute) and issubclass(attribute, BaseTool) and attribute is not BaseTool:
    #                         try:
    #                             logger.warning(f"Found unrefactored class-based tool: {attribute_name} in {module_short_name}. Skipping initialization.", emoji_key="⚠️")
    #                             # tool_instance = attribute(mcp_server=mcp_server)  # noqa: F841
    #                             # class_based_modules.append(module_short_name)
    #                         except Exception as e:
    #                             logger.error(f"Error processing potential class {attribute_name} in {name}: {e}", exc_info=True)
    #              except ImportError as e:
    #                  logger.error(f"Failed to import potential tool module {name}: {e}", exc_info=True)

    # Update summary log
    logger.info(
        f"Completed tool registration. Registered {standalone_count} standalone functions.", 
        emoji_key="✅"
        # No class-based modules should be initialized now
    )
    
    # Return info about standalone tools
    return registered_tools

# Removed get_tool_instance function
# Removed get_all_tools function