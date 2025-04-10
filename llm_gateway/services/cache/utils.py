"""Cache utility functions for LLM Gateway.

This module provides utility functions for working with the cache service
that were previously defined in example scripts but are now part of the library.
"""

import hashlib

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.cache import get_cache_service
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("llm_gateway.services.cache.utils")

async def run_completion_with_cache(
    prompt: str,
    provider_name: str = Provider.OPENAI.value,
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = None,
    use_cache: bool = True,
    ttl: int = 3600,  # Default 1 hour cache TTL
    api_key: str = None
):
    """Run a completion with automatic caching.
    
    This utility function handles provider initialization, cache key generation,
    cache lookups, and caching results automatically.
    
    Args:
        prompt: Text prompt for completion
        provider_name: Provider to use (default: OpenAI)
        model: Model name (optional, uses provider default if not specified)
        temperature: Temperature for generation (default: 0.1)
        max_tokens: Maximum tokens to generate (optional)
        use_cache: Whether to use cache (default: True)
        ttl: Cache TTL in seconds (default: 3600/1 hour)
        api_key: Provider API key (optional, falls back to env vars)
        
    Returns:
        Completion result with additional processing_time attribute
    """
    # Get provider with API key from parameter or env
    if not api_key:
        # Simplify key retrieval
        key_map = {
            Provider.OPENAI.value: "OPENAI_API_KEY",
            Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
            Provider.GEMINI.value: "GEMINI_API_KEY",
            Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY"
        }
        api_key_name = key_map.get(provider_name)
        if api_key_name:
            api_key = decouple_config(api_key_name, default=None)
    
    if not api_key:
        # Log warning but allow fallback if provider supports keyless (unlikely for these)
        logger.warning(f"API key for {provider_name} not found. Request may fail.", emoji_key="warning")

    try:
        provider = get_provider(provider_name, api_key=api_key)
        await provider.initialize()
    except Exception as e:
         logger.error(f"Failed to initialize provider {provider_name}: {e}", emoji_key="error")
         raise # Re-raise exception to stop execution if provider fails
    
    cache_service = get_cache_service()
    
    # Create a more robust cache key using all relevant parameters
    model_id = model or provider.get_default_model() # Ensure we have a model id
    
    # Create consistent hash of parameters that affect the result
    params_str = f"{prompt}:{temperature}:{max_tokens if max_tokens else 'default'}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    cache_key = f"completion:{provider_name}:{model_id}:{params_hash}"
    
    if use_cache and cache_service.enabled:
        cached_result = await cache_service.get(cache_key)
        if cached_result is not None:
            logger.success("Cache hit! Using cached result", emoji_key="cache")
            # Set processing time for cache retrieval (negligible)
            cached_result.processing_time = 0.001 
            return cached_result
    
    # Generate completion if not cached or cache disabled
    if use_cache:
        logger.info("Cache miss. Generating new completion...", emoji_key="processing")
    else:
        logger.info("Cache disabled by request. Generating new completion...", emoji_key="processing")
        
    # Use the determined model_id and pass through other parameters
    result = await provider.generate_completion(
        prompt=prompt,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Save to cache if enabled
    if use_cache and cache_service.enabled:
        await cache_service.set(
            key=cache_key,
            value=result,
            ttl=ttl
        )
        logger.info(f"Result saved to cache (key: ...{cache_key[-10:]})", emoji_key="cache")
        
    return result 