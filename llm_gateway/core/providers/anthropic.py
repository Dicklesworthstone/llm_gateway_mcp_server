"""Anthropic (Claude) provider implementation."""
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from anthropic import AsyncAnthropic
from anthropic.types import Message

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic (Claude) API."""
    
    provider_name = Provider.ANTHROPIC.value
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url")
        self.models_cache = None
        
    async def initialize(self) -> bool:
        """Initialize the Anthropic client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            self.client = AsyncAnthropic(
                api_key=self.api_key, 
                base_url=self.base_url,
            )
            
            self.logger.success(
                "Anthropic provider initialized successfully", 
                emoji_key="provider"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Anthropic provider: {str(e)}", 
                emoji_key="error"
            )
            return False
        
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a completion using Anthropic Claude.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "claude-3-opus-20240229")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse: Standardized response
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Prepare system prompt if provided
        system = kwargs.pop("system", None)
        
        # Create messages
        messages = kwargs.pop("messages", None)
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Add system if specified
        if system:
            params["system"] = system
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Log request
        self.logger.info(
            f"Generating completion with Anthropic model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        try:
            # Make API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.messages.create, **params
            )
            
            # Extract response text
            completion_text = response.content[0].text
            
            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                processing_time=processing_time,
                raw_response=response,
            )
            
            # Log success
            self.logger.success(
                f"Anthropic completion successful",
                emoji_key="success",
                model=model,
                tokens={
                    "input": result.input_tokens,
                    "output": result.output_tokens
                },
                cost=result.cost,
                time=result.processing_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Anthropic completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using Anthropic Claude.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "claude-3-opus-20240229")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Yields:
            Tuple of (text_chunk, metadata)
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Prepare system prompt if provided
        system = kwargs.pop("system", None)
        
        # Create messages
        messages = kwargs.pop("messages", None)
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Add system if specified
        if system:
            params["system"] = system
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Log request
        self.logger.info(
            f"Generating streaming completion with Anthropic model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        start_time = time.time()
        total_chunks = 0
        
        try:
            # Make streaming API call
            async with self.client.messages.stream(**params) as stream:
                # Process the stream
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        total_chunks += 1
                        
                        # Extract content from the chunk
                        content = chunk.delta.text
                        
                        # Metadata for this chunk
                        metadata = {
                            "model": model,
                            "provider": self.provider_name,
                            "chunk_index": total_chunks,
                        }
                        
                        yield content, metadata
                        
                # Get final message for token counts
                final_message = await stream.get_final_message()
                
                # Log success
                processing_time = time.time() - start_time
                self.logger.success(
                    f"Anthropic streaming completion successful",
                    emoji_key="success",
                    model=model,
                    chunks=total_chunks,
                    tokens={
                        "input": final_message.usage.input_tokens,
                        "output": final_message.usage.output_tokens
                    },
                    time=processing_time
                )
                
        except Exception as e:
            self.logger.error(
                f"Anthropic streaming completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Anthropic Claude models.
        
        Returns:
            List of model information dictionaries
        """
        # Anthropic doesn't have a list models endpoint, so we return a static list
        if self.models_cache:
            return self.models_cache
            
        models = [
            {
                "id": "claude-3-opus-20240229",
                "provider": self.provider_name,
                "description": "Most powerful Claude model",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "provider": self.provider_name,
                "description": "Balanced performance Claude model",
            },
            {
                "id": "claude-3-haiku-20240307",
                "provider": self.provider_name,
                "description": "Fast, efficient Claude model",
            },
            {
                "id": "claude-3-5-sonnet-20240620",
                "provider": self.provider_name,
                "description": "Updated, improved sonnet model",
            },
            {
                "id": "claude-3-5-haiku-latest",
                "provider": self.provider_name,
                "description": "Latest and fastest Claude model",
            },
        ]
        
        # Cache results
        self.models_cache = models
        
        return models
            
    def get_default_model(self) -> str:
        """Get the default Anthropic Claude model.
        
        Returns:
            Default model name
        """
        from llm_gateway.config import config
        
        # Get from config if available
        provider_config = getattr(config.providers, self.provider_name, None)
        if provider_config and provider_config.default_model:
            return provider_config.default_model
            
        # Otherwise return hard-coded default
        return "claude-3-5-haiku-latest"
        
    async def check_api_key(self) -> bool:
        """Check if the Anthropic API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            # Try to create a simple message to validate the API key
            await self.client.messages.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False