"""
LLM Provider Factory
Factory function to create and configure LLM providers
"""

import logging
import os
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def get_llm_provider(provider_type: str, 
                    model_name: Optional[str] = None,
                    **kwargs) -> 'BaseLLMProvider':
    """
    Factory function to create and configure an LLM provider
    
    Args:
        provider_type: Type of provider (openai, anthropic, huggingface, local)
        model_name: Name of the model to use
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured LLM provider
    """
    provider_type = provider_type.lower()
    
    if provider_type == "openai":
        from .openai_provider import OpenAIProvider
        # Default model for OpenAI
        model = model_name or "gpt-3.5-turbo"
        return OpenAIProvider(model_name=model, **kwargs)
    
    elif provider_type == "anthropic":
        try:
            from .anthropic_provider import AnthropicProvider
            # Default model for Anthropic
            model = model_name or "claude-3-opus-20240229"
            return AnthropicProvider(model_name=model, **kwargs)
        except ImportError:
            logger.error("Anthropic provider not available. Install with 'pip install anthropic'")
            raise
    
    elif provider_type == "huggingface":
        try:
            from .huggingface_provider import HuggingFaceProvider
            # Default model for HuggingFace
            model = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
            return HuggingFaceProvider(model_name=model, **kwargs)
        except ImportError:
            logger.error("HuggingFace Inference API client not available. Install with 'pip install requests'")
            raise
    
    elif provider_type == "local":
        try:
            from .local_provider import LocalModelProvider
            # For local models, we need a model_path
            model_path = kwargs.pop("model_path", None)
            if not model_path:
                raise ValueError("For local models, 'model_path' must be provided")
            
            backend = kwargs.pop("backend", "llamacpp")
            return LocalModelProvider(model_path=model_path, backend=backend, model_name=model_name, **kwargs)
        except ImportError:
            logger.error("Local model provider not available. Install with 'pip install llama-cpp-python' or 'pip install vllm'")
            raise
    
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}. Use 'openai', 'anthropic', 'huggingface', or 'local'")
