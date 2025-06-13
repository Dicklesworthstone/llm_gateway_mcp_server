"""Provider tools for Ultimate MCP Server."""
from typing import Any, Dict, Optional

# Import ToolError explicitly
from ultimate_mcp_server.exceptions import ToolError

# REMOVE global instance logic
from ultimate_mcp_server.utils import get_logger

from .base import with_error_handling, with_tool_metrics

logger = get_logger("ultimate_mcp_server.tools.provider")

def _get_provider_status_dict() -> Dict[str, Any]:
    """Reliably gets the provider_status dictionary from the gateway instance."""
    provider_status = {}
    # Import here to avoid circular dependency at module load time
    try:
        from ultimate_mcp_server.core import get_gateway_instance
        gateway = get_gateway_instance()
        if gateway and hasattr(gateway, 'provider_status'):
            provider_status = gateway.provider_status
            if provider_status:
                logger.debug("Retrieved provider status via global instance.")
                return provider_status
    except ImportError as e:
        logger.error(f"Failed to import get_gateway_instance: {e}")
    except Exception as e:
        logger.error(f"Error accessing global gateway instance: {e}")
        
    if not provider_status:
        logger.warning("Could not retrieve provider status from global gateway instance.")
        
    return provider_status

# --- Tool Functions (Standalone, Decorated) ---

@with_tool_metrics
@with_error_handling
async def get_provider_status() -> Dict[str, Any]:
    """Checks the status and availability of all configured LLM providers.

    Use this tool to determine which LLM providers (e.g., OpenAI, Anthropic, Gemini)
    are currently enabled, configured correctly (e.g., API keys), and ready to accept requests.
    This helps in deciding which provider to use for a task or for troubleshooting.

    Returns:
        A dictionary mapping provider names to their status details:
        {
            "providers": {
                "openai": {                      # Example for one provider
                    "enabled": true,             # Is the provider enabled in the server config?
                    "available": true,             # Is the provider initialized and ready for requests?
                    "api_key_configured": true,  # Is the necessary API key set?
                    "error": null,               # Error message if initialization failed, null otherwise.
                    "models_count": 38           # Number of models detected for this provider.
                },
                "anthropic": {                   # Example for another provider
                    "enabled": true,
                    "available": false,
                    "api_key_configured": true,
                    "error": "Initialization failed: Connection timeout",
                    "models_count": 0
                },
                ...
            }
        }
        Returns an empty "providers" dict and a message if status info is unavailable.

    Usage:
        - Call this tool before attempting complex tasks to ensure required providers are available.
        - Use the output to inform the user about available options or diagnose issues.
        - If a provider shows "available: false", check the "error" field for clues.
    """
    provider_status = _get_provider_status_dict()

    if not provider_status:
        # Raise ToolError if status cannot be retrieved
        raise ToolError(status_code=503, detail="Provider status information is currently unavailable. The server might be initializing or an internal error occurred.")

    return {
        "providers": {
            name: {
                "enabled": status.enabled,
                "available": status.available,
                "api_key_configured": status.api_key_configured,
                "error": status.error,
                "models_count": len(status.models)
            }
            for name, status in provider_status.items()
        }
    }

@with_tool_metrics
@with_error_handling
async def list_models(
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """Lists available LLM models, optionally filtered by provider.

    Use this tool to discover specific models offered by the configured and available LLM providers.
    The returned model IDs (e.g., 'openai/gpt-4.1-mini') are needed for other tools like
    `chat_completion`, `generate_completion`, `estimate_cost`, or `create_tournament`.

    Args:
        provider: (Optional) The specific provider name (e.g., "openai", "anthropic", "gemini")
                  to list models for. If omitted, models from *all available* providers are listed.

    Returns:
        A dictionary mapping provider names to a list of their available models:
        {
            "models": {
                "openai": [                       # Example for one provider
                    {
                        "id": "openai/gpt-4.1-mini", # Unique ID used in other tools
                        "name": "GPT-4o Mini",     # Human-friendly name
                        "context_window": 128000,
                        "features": ["chat", "completion", "vision"],
                        "input_cost_pmt": 0.15,  # Cost per Million Tokens (Input)
                        "output_cost_pmt": 0.60  # Cost per Million Tokens (Output)
                    },
                    ...
                ],
                "gemini": [                      # Example for another provider
                    {
                        "id": "gemini/gemini-2.5-pro-preview-03-25",
                        "name": "Gemini 2.5 Pro Experimental",
                        "context_window": 8192,
                        "features": ["chat", "completion"],
                        "input_cost_pmt": null, # Cost info might be null
                        "output_cost_pmt": null
                    },
                    ...
                ],
                ...
            }
        }
        Returns an empty "models" dict or includes warnings/errors if providers/models are unavailable.

    Usage Flow:
        1. (Optional) Call `get_provider_status` to see which providers are generally available.
        2. Call `list_models` (optionally specifying a provider) to get usable model IDs.
        3. Use a specific model ID (like "openai/gpt-4.1-mini") as the 'model' parameter in other tools.

    Raises:
        ToolError: If the specified provider name is invalid or provider status is unavailable.
    """
    provider_status = _get_provider_status_dict()

    if not provider_status:
        raise ToolError(status_code=503, detail="Provider status information is currently unavailable. Cannot list models.")

    models = {}
    if provider:
        if provider not in provider_status:
            valid_providers = list(provider_status.keys())
            raise ToolError(status_code=404, detail=f"Invalid provider specified: '{provider}'. Valid options: {valid_providers}")

        status = provider_status[provider]
        if not status.available:
            # Return empty list for the provider but include a warning message
            return {
                "models": {provider: []},
                "warning": f"Provider '{provider}' is configured but currently unavailable. Reason: {status.error or 'Unknown error'}"
            }
        # Use model details directly from the ProviderStatus object
        models[provider] = [m for m in status.models] if status.models else []
    else:
        # List models for all *available* providers
        any_available = False
        for name, status in provider_status.items():
            if status.available:
                any_available = True
                 # Use model details directly from the ProviderStatus object
                models[name] = [m for m in status.models] if status.models else []
            # else: Provider not available, don't include it unless specifically requested

        if not any_available:
            return {
                "models": {},
                "warning": "No providers are currently available. Check provider status using get_provider_status."
            }
        elif all(len(model_list) == 0 for model_list in models.values()):
             return {
                "models": models,
                "warning": "No models listed for any available provider. Check provider status or configuration."
            }

    return {"models": models} 