"""
Anthropic Provider
Implementation of LLM provider for Anthropic Claude models
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models"""
    
    def __init__(self, 
                model_name: str = "claude-3-opus-20240229", 
                api_key: Optional[str] = None,
                **kwargs):
        """
        Initialize Anthropic provider
        
        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Please set the ANTHROPIC_API_KEY environment variable.")
        
        # Dict mapping Claude model names to context lengths
        self.model_context_lengths = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000,
            "claude-instant-1.1": 100000,
            "claude-instant-1.0": 100000,
            "claude-1.0": 100000,
            "claude-1.2": 100000,
            "claude-1.3": 100000
        }
        
        # Set max context length based on model name
        self.max_context_length = self.model_context_lengths.get(
            self.model_name, 
            100000  # Default to 100k if model not found
        )
        
        # Import Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.anthropic_constants = anthropic
            logger.info(f"Initialized Anthropic provider with model: {model_name}")
        except ImportError:
            logger.error("Anthropic package not installed. Install with 'pip install anthropic'")
            raise
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1024,
                stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from Anthropic Claude
        
        Args:
            prompt: User prompt or query
            system_prompt: Optional system prompt to define behavior
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Prepare the parameters
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
            
            # Add stop sequences if provided
            if stop_sequences:
                params["stop_sequences"] = stop_sequences
            
            # Get token count estimate for prompt
            prompt_tokens = self.get_token_count(prompt)
            system_tokens = self.get_token_count(system_prompt) if system_prompt else 0
            
            # Call the Anthropic API
            response = self.client.messages.create(**params)
            
            # Extract response text
            response_text = response.content[0].text
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": "anthropic",
                "tokens": {
                    "prompt": prompt_tokens + system_tokens,  # Estimate
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "response_metadata": {
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                    "temperature": temperature,
                    "id": response.id,
                    "type": response.type
                }
            }
            
            logger.debug(f"Generated Claude response with {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "anthropic",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def generate_chat(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 1024,
                     stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from Anthropic Claude using chat format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Process messages - extract system message if present
            system_content = None
            anthropic_messages = []
            
            for msg in messages:
                role = msg["role"].lower()
                content = msg["content"]
                
                if role == "system":
                    system_content = content
                elif role in ["user", "assistant"]:
                    anthropic_messages.append({"role": role, "content": content})
            
            # Prepare the parameters
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages
            }
            
            # Add system instruction if provided
            if system_content:
                params["system"] = system_content
            
            # Add stop sequences if provided
            if stop_sequences:
                params["stop_sequences"] = stop_sequences
            
            # Estimate token count for input
            prompt_tokens = sum(self.get_token_count(msg["content"]) for msg in messages)
            
            # Call the Anthropic API
            response = self.client.messages.create(**params)
            
            # Extract response text
            response_text = response.content[0].text
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": "anthropic",
                "tokens": {
                    "prompt": prompt_tokens,  # Estimate
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "response_metadata": {
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                    "temperature": temperature,
                    "id": response.id,
                    "type": response.type
                }
            }
            
            logger.debug(f"Generated Claude chat response with {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat response from Anthropic: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "anthropic",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        
        This is an approximation since Anthropic's tokenizer isn't publicly available
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens (approximated)
        """
        if not text:
            return 0
            
        try:
            # Anthropic uses approximately 4 characters per token
            # This is a rough estimate
            return len(text) // 4 + 1
        except Exception as e:
            logger.warning(f"Error counting tokens, returning approximate: {e}")
            # Approximate token count (rough estimate)
            return len(text.split()) * 1.3
