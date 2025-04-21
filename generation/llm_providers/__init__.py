"""
LLM Providers Module
Collection of different large language model providers for response generation
"""

from typing import Dict, List, Any, Optional, Union, Tuple

from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider

# Conditionally import providers based on available dependencies
try:
    from .anthropic_provider import AnthropicProvider
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from .huggingface_provider import HuggingFaceProvider
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from .local_provider import LocalModelProvider
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

# Import provider factory
from .provider_factory import get_llm_provider

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "get_llm_provider"
]

if ANTHROPIC_AVAILABLE:
    __all__.append("AnthropicProvider")

if HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceProvider")

if LOCAL_MODELS_AVAILABLE:
    __all__.append("LocalModelProvider")
