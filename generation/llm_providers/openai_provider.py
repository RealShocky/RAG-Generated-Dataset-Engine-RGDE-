"""
OpenAI Provider
Implementation of LLM provider for OpenAI models
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import tiktoken

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models (GPT-3.5-turbo, GPT-4)"""
    
    def __init__(self, 
                model_name: str = "gpt-3.5-turbo", 
                api_key: Optional[str] = None,
                organization_id: Optional[str] = None,
                **kwargs):
        """
        Initialize OpenAI provider
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            organization_id: Optional OpenAI organization ID
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
        
        self.organization_id = organization_id or os.environ.get("OPENAI_ORGANIZATION_ID")
        
        # Load the appropriate tokenizer for the model
        try:
            model_prefix = self.model_name.split("-")[0]
            if model_prefix == "gpt4":
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            elif model_prefix in ["gpt3", "text"]:
                self.tokenizer = tiktoken.encoding_for_model("text-davinci-003")
            else:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load specific tokenizer for {self.model_name}, using default: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default for recent OpenAI models
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                organization=self.organization_id
            )
            logger.info(f"Initialized OpenAI provider with model: {model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            raise
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1024,
                stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from OpenAI
        
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
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": "openai",
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "response_metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "object": response.object,
                    "temperature": temperature
                }
            }
            
            logger.debug(f"Generated response with {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "openai",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def generate_chat(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 1024,
                     stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from OpenAI using chat format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": "openai",
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "response_metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "object": response.object,
                    "temperature": temperature
                }
            }
            
            logger.debug(f"Generated chat response with {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat response from OpenAI: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "openai",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens, returning approximate: {e}")
            # Approximate token count (rough estimate)
            return len(text.split()) * 1.3
