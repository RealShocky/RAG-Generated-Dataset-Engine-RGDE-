"""
Local Model Provider
Implementation of LLM provider for local model inference using llama.cpp or vLLM
"""

import os
import logging
import subprocess
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class LocalModelProvider(BaseLLMProvider):
    """Provider for local LLM inference using llama.cpp or vLLM"""
    
    def __init__(self, 
                model_path: str, 
                backend: str = "llamacpp",
                model_name: Optional[str] = None,
                context_length: int = 4096,
                **kwargs):
        """
        Initialize local model provider
        
        Args:
            model_path: Path to the model file or directory
            backend: Inference backend ('llamacpp' or 'vllm')
            model_name: Optional model name for identification
            context_length: Context window size for the model
            **kwargs: Additional configuration parameters
        """
        self.model_path = os.path.abspath(model_path)
        self.backend = backend.lower()
        self.context_length = context_length
        
        # Use the model filename as the model_name if not provided
        if not model_name:
            model_name = os.path.basename(model_path)
        
        super().__init__(model_name, **kwargs)
        
        # Initialize backend-specific attributes
        self.client = None
        self.tokenizer = None
        
        if self.backend == "llamacpp":
            self._init_llamacpp(**kwargs)
        elif self.backend == "vllm":
            self._init_vllm(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'llamacpp' or 'vllm'")
        
        logger.info(f"Initialized {self.backend} provider for local model: {model_name}")
    
    def _init_llamacpp(self, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs):
        """Initialize llama.cpp backend"""
        try:
            from llama_cpp import Llama
            
            # Load the model with llama.cpp
            self.client = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx or self.context_length,
                n_gpu_layers=n_gpu_layers,
                **kwargs
            )
            
            # Use the built-in tokenizer for token counting
            self.tokenizer = self.client
            
            logger.info(f"Loaded model with llama.cpp: {self.model_name}")
        except ImportError:
            logger.error("llama_cpp package not installed. Install with 'pip install llama-cpp-python'")
            raise
        except Exception as e:
            logger.error(f"Error initializing llama.cpp model: {e}")
            raise
    
    def _init_vllm(self, **kwargs):
        """Initialize vLLM backend"""
        try:
            from vllm import LLM, SamplingParams
            
            # Load the model with vLLM
            self.client = LLM(
                model=self.model_path,
                dtype="half",  # Use float16 by default for efficiency
                **kwargs
            )
            
            # Store SamplingParams class for later use
            self.SamplingParams = SamplingParams
            
            logger.info(f"Loaded model with vLLM: {self.model_name}")
        except ImportError:
            logger.error("vllm package not installed. Install with 'pip install vllm'")
            raise
        except Exception as e:
            logger.error(f"Error initializing vLLM model: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1024,
                stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from a local model
        
        Args:
            prompt: User prompt or query
            system_prompt: Optional system prompt to define behavior
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Format prompt with system instruction if provided
        if system_prompt:
            full_prompt = self._format_system_prompt(system_prompt, prompt)
        else:
            full_prompt = prompt
        
        # Generate a response based on the backend
        if self.backend == "llamacpp":
            return self._generate_llamacpp(full_prompt, temperature, max_tokens, stop_sequences)
        elif self.backend == "vllm":
            return self._generate_vllm(full_prompt, temperature, max_tokens, stop_sequences)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _generate_llamacpp(self, prompt: str, temperature: float, max_tokens: int, stop_sequences: Optional[List[str]]) -> Dict[str, Any]:
        """Generate response using llama.cpp backend"""
        try:
            start_time = time.time()
            
            # Generate completion
            output = self.client.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences or []
            )
            
            # Extract the generated text
            response_text = output["choices"][0]["text"]
            
            # Track token usage (llama_cpp provides this)
            prompt_tokens = self.client.n_tokens(prompt)
            completion_tokens = len(self.client.tokenize(bytes(response_text, "utf-8")))
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": f"local_{self.backend}",
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                },
                "response_metadata": {
                    "finish_reason": output["choices"][0].get("finish_reason", "unknown"),
                    "model": self.model_name,
                    "temperature": temperature,
                    "generation_time": time.time() - start_time
                }
            }
            
            logger.debug(f"Generated local {self.backend} response with {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with llama.cpp: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": f"local_{self.backend}",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def _generate_vllm(self, prompt: str, temperature: float, max_tokens: int, stop_sequences: Optional[List[str]]) -> Dict[str, Any]:
        """Generate response using vLLM backend"""
        try:
            start_time = time.time()
            
            # Set up sampling parameters
            sampling_params = self.SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences or []
            )
            
            # Generate completion
            outputs = self.client.generate(
                prompts=[prompt],
                sampling_params=sampling_params
            )
            
            # Extract the generated text
            output = outputs[0]
            response_text = output.outputs[0].text
            
            # Estimate token usage (vLLM doesn't provide this directly)
            prompt_tokens = self.get_token_count(prompt)
            completion_tokens = self.get_token_count(response_text)
            
            # Prepare the result
            result = {
                "text": response_text,
                "model": self.model_name,
                "provider": f"local_{self.backend}",
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                },
                "response_metadata": {
                    "finish_reason": "stop" if outputs[0].finished else "length",
                    "model": self.model_name,
                    "temperature": temperature,
                    "generation_time": time.time() - start_time
                }
            }
            
            logger.debug(f"Generated local {self.backend} response with est. {result['tokens']['total']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with vLLM: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": f"local_{self.backend}",
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }
    
    def generate_chat(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 1024,
                     stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from a local model using chat format
        
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
            # This varies by model, but we'll format based on common templates
            
            # Extract system message if present
            system_content = None
            for msg in messages:
                if msg["role"].lower() == "system":
                    system_content = msg["content"]
                    break
            
            # Format the rest of the messages based on common templates
            # This is a simplified version that works for many local models
            formatted_prompt = ""
            
            # Check if this looks like a Llama model
            if "llama" in self.model_name.lower():
                formatted_prompt = self._format_llama_chat(messages)
            # Check if this looks like a Mistral model
            elif "mistral" in self.model_name.lower():
                formatted_prompt = self._format_mistral_chat(messages)
            # Default generic chat format
            else:
                formatted_prompt = self._format_generic_chat(messages)
            
            # Now use the standard generate method with our formatted prompt
            return self.generate(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
            
        except Exception as e:
            logger.error(f"Error generating chat response from local model: {e}")
            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_name,
                "provider": f"local_{self.backend}",
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
        if not text:
            return 0
            
        try:
            if self.backend == "llamacpp" and hasattr(self.client, "n_tokens"):
                return self.client.n_tokens(text)
            else:
                # Approximation for when tokenizer not available
                return len(text) // 4 + 1  # Rough approximation: ~4 chars per token
        except Exception as e:
            logger.warning(f"Error counting tokens, using approximation: {e}")
            # Approximate token count (rough estimate)
            return len(text.split()) * 1.3
    
    def _format_system_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format a prompt with system instructions based on the model type"""
        if "llama" in self.model_name.lower():
            # Llama-style format
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        elif "mistral" in self.model_name.lower():
            # Mistral-style format
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        else:
            # Generic format
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def _format_llama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for Llama models"""
        system_content = None
        formatted_messages = []
        
        for msg in messages:
            role = msg["role"].lower()
            content = msg["content"]
            
            if role == "system":
                system_content = content
            else:
                formatted_messages.append({"role": role, "content": content})
        
        # Build the prompt string in Llama chat format
        prompt = ""
        
        # First message with system prompt if available
        if formatted_messages and formatted_messages[0]["role"] == "user":
            if system_content:
                prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{formatted_messages[0]['content']} [/INST]"
            else:
                prompt = f"<s>[INST] {formatted_messages[0]['content']} [/INST]"
        
        # Add remaining messages
        for i in range(1, len(formatted_messages)):
            msg = formatted_messages[i]
            prev_msg = formatted_messages[i-1]
            
            if msg["role"] == "user" and prev_msg["role"] == "assistant":
                prompt += f"</s><s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant" and prev_msg["role"] == "user":
                prompt += f" {msg['content']}"
        
        # Add final token to indicate where the model should continue
        if formatted_messages and formatted_messages[-1]["role"] == "user":
            # If the last message is from the user, we've already added [/INST]
            # Don't add anything else
            pass
        elif formatted_messages and formatted_messages[-1]["role"] == "assistant":
            # If the last message is from the assistant, add a new user instruction
            prompt += f"</s><s>[INST] "
        
        return prompt
    
    def _format_mistral_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for Mistral models"""
        formatted_prompt = ""
        for i, msg in enumerate(messages):
            role = msg["role"].lower()
            content = msg["content"]
            
            if role == "user":
                formatted_prompt += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                formatted_prompt += f" {content} </s>"
            elif role == "system" and i == 0:
                # Only use system at the beginning
                formatted_prompt = f"<s>[INST] {content}\n\n"
        
        return formatted_prompt
    
    def _format_generic_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for generic models"""
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            
            formatted_prompt += f"{role}: {content}\n\n"
        
        # Add final assistant prompt
        formatted_prompt += "Assistant:"
        
        return formatted_prompt
