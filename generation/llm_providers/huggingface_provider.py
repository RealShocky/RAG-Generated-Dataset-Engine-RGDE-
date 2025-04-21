"""
HuggingFace Provider
Implementation of LLM provider for HuggingFace Inference API
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseLLMProvider):
    """Provider for HuggingFace Inference API"""
    
    def __init__(self, 
                model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                api_key: Optional[str] = None,
                api_url: Optional[str] = None,
                **kwargs):
        """
        Initialize HuggingFace provider
        
        Args:
            model_name: Name of the model to use (repository_id)
            api_key: HuggingFace API key (falls back to HF_API_TOKEN env var)
            api_url: Custom API URL (optional)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.environ.get("HF_API_TOKEN")
        if not self.api_key:
            logger.warning("No HuggingFace API key provided. Please set the HF_API_TOKEN environment variable.")
        
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Import requests for API calls
        try:
            import requests
            self.requests = requests
            logger.info(f"Initialized HuggingFace provider with model: {model_name}")
        except ImportError:
            logger.error("Requests package not installed. Install with 'pip install requests'")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1024,
                stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from HuggingFace Inference API
        
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
            # Format prompt with system instruction if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Prepare payload with parameters
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                payload["parameters"]["stop"] = stop_sequences
            
            # Set headers with auth token
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            response = self.requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            result_json = response.json()
            
            # Handle different response formats from HF API
            if isinstance(result_json, list) and len(result_json) > 0:
                if "generated_text" in result_json[0]:
                    response_text = result_json[0]["generated_text"]
                else:
                    response_text = result_json[0]
            elif isinstance(result_json, dict) and "generated_text" in result_json:
                response_text = result_json["generated_text"]
            else:
                response_text = str(result_json)
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": "huggingface",
                "tokens": {
                    "prompt": self.get_token_count(full_prompt),  # Estimate
                    "completion": self.get_token_count(response_text),  # Estimate
                    "total": self.get_token_count(full_prompt) + self.get_token_count(response_text)  # Estimate
                },
                "response_metadata": {
                    "status_code": response.status_code,
                    "model": self.model_name,
                    "temperature": temperature
                }
            }
            
            logger.debug(f"Generated HuggingFace response with estimated {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from HuggingFace: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "huggingface",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def generate_chat(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 1024,
                     stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from HuggingFace Inference API using chat format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Convert messages to a format suitable for the model
            # This varies by model, but we'll use a common format
            system_content = None
            conversation = []
            
            for msg in messages:
                role = msg["role"].lower()
                content = msg["content"]
                
                if role == "system":
                    system_content = content
                else:
                    conversation.append({"role": role, "content": content})
            
            # Format conversation based on common templates
            # Mistral-like format:
            # <s>[INST] {system_prompt} [/INST]</s>[INST] {user_prompt} [/INST]
            formatted_prompt = ""
            
            if "mistral" in self.model_name.lower():
                if system_content:
                    formatted_prompt = f"<s>[INST] {system_content} [/INST]</s>"
                
                for i, msg in enumerate(conversation):
                    if msg["role"] == "user":
                        formatted_prompt += f"[INST] {msg['content']} [/INST]"
                    else:
                        formatted_prompt += f" {msg['content']} "
            
            # Llama-like format:
            elif "llama" in self.model_name.lower():
                if system_content:
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n"
                else:
                    formatted_prompt = "<s>[INST] "
                
                for i, msg in enumerate(conversation):
                    if msg["role"] == "user":
                        formatted_prompt += f"{msg['content']} [/INST]"
                    else:
                        formatted_prompt += f" {msg['content']} </s><s>[INST] "
            
            # Generic format (default fallback)
            else:
                if system_content:
                    formatted_prompt = f"System: {system_content}\n\n"
                
                for msg in conversation:
                    role_name = msg["role"].capitalize()
                    formatted_prompt += f"{role_name}: {msg['content']}\n"
                
                formatted_prompt += "Assistant: "
            
            # Now use the standard generate method with our formatted prompt
            return self.generate(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
            
        except Exception as e:
            logger.error(f"Error generating chat response from HuggingFace: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": "huggingface",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string (estimation)
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens (approximated)
        """
        if not text:
            return 0
            
        try:
            # Simple approximation: 4 characters per token on average
            return len(text) // 4 + 1
        except Exception as e:
            logger.warning(f"Error counting tokens, using word count approximation: {e}")
            # Fallback approximation
            return len(text.split()) * 1.3
