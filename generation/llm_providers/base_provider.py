"""
Base LLM Provider
Abstract base class defining interface for all LLM providers
"""

import abc
from typing import Dict, List, Any, Optional, Union, Callable

class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM provider
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abc.abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1024,
                stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User prompt or query
            system_prompt: Optional system prompt to define behavior
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @abc.abstractmethod
    def generate_chat(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 1024,
                     stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM using chat format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @abc.abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is valid
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Try a simple request to verify API key
            self.generate("Test API key", max_tokens=5)
            return True
        except Exception:
            return False
    
    @staticmethod
    def format_prompt_with_context(query: str, 
                                  context: List[Dict[str, Any]], 
                                  prompt_template: Optional[str] = None) -> str:
        """
        Format a prompt with retrieved context
        
        Args:
            query: User query
            context: List of retrieved documents
            prompt_template: Optional template to use
            
        Returns:
            Formatted prompt string
        """
        if not prompt_template:
            prompt_template = """Answer the question based on the following context:

Context:
{context}

Question: {query}

Answer:"""
        
        # Format context
        formatted_context = ""
        for i, doc in enumerate(context):
            doc_text = doc.get("text", "")
            source = doc.get("source", f"Document {i+1}")
            score = doc.get("retrieval_score", "")
            score_str = f" (score: {score:.4f})" if score else ""
            
            formatted_context += f"[{source}{score_str}]\n{doc_text}\n\n"
        
        # Return the formatted prompt
        return prompt_template.format(context=formatted_context, query=query)
    
    @staticmethod
    def construct_chat_with_context(query: str, 
                                  context: List[Dict[str, Any]], 
                                  system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Construct chat messages with retrieved context
        
        Args:
            query: User query
            context: List of retrieved documents
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question based on the provided context. "
                           "If the context doesn't contain the answer, say so."
            })
        
        # Format context
        formatted_context = ""
        for i, doc in enumerate(context):
            doc_text = doc.get("text", "")
            source = doc.get("source", f"Document {i+1}")
            score = doc.get("retrieval_score", "")
            score_str = f" (score: {score:.4f})" if score else ""
            
            formatted_context += f"[{source}{score_str}]\n{doc_text}\n\n"
        
        # Add context as assistant message
        messages.append({
            "role": "assistant",
            "content": f"Here is the context to help answer your question:\n\n{formatted_context}"
        })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
