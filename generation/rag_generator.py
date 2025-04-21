"""
RAG Generator Module
Uses retrieved documents to generate grounded answers with a language model.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config
from config import PROMPT_TEMPLATES
from retrieval.retriever import get_retriever

# Import provider factory
from generation.llm_providers import get_llm_provider
from generation.llm_providers.base_provider import BaseLLMProvider

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class BaseGenerator:
    """Base class for RAG generators."""
    
    def __init__(self, model_name: str = config.RAG_MODEL_NAME):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the language model to use
        """
        self.model_name = model_name
        logger.info(f"Initialized BaseGenerator with model: {model_name}")
    
    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Extract document text and metadata
            title = doc.get("title", f"Document {i}")
            text = doc.get("text", "")
            source = doc.get("source", "Unknown source")
            
            # Format document as context entry
            context_parts.append(
                f"[{i}] {title}\n"
                f"Source: {source}\n"
                f"Content: {text}\n"
            )
        
        # Join all documents with separators
        return "\n---\n".join(context_parts)
    
    def format_prompt(self, question: str, context: str) -> str:
        """
        Format the prompt with question and context.
        
        Args:
            question: Input question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        # Use the RAG prompt template from config
        return PROMPT_TEMPLATES["rag_prompt"].format(
            context=context,
            question=question
        )
    
    def generate(
        self, 
        question: str, 
        documents: List[Dict],
        temperature: float = config.RAG_TEMPERATURE,
        max_tokens: int = config.RAG_MAX_TOKENS
    ) -> Dict:
        """
        Generate a response using RAG.
        
        Args:
            question: Input question
            documents: Retrieved documents
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response with metadata
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def generate_responses(
        self,
        questions: List[Dict],
        retrieved_docs: Optional[List[List[Dict]]] = None,
        use_batch_processing: bool = True,
        batch_size: int = 5,
        max_concurrency: int = 3
    ) -> List[Dict]:
        """
        Generate responses for a list of questions using RAG.
        
        Args:
            questions: List of question dictionaries with 'text' field
            retrieved_docs: Optional list of retrieved documents per question
            use_batch_processing: Whether to use batch processing for efficiency
            batch_size: Size of each batch when using batch processing
            max_concurrency: Maximum number of concurrent generations
            
        Returns:
            List of response dictionaries with question, response, and metadata
        """
        if not questions:
            logger.warning("No questions provided for generation")
            return []
        
        logger.info(f"Generating responses for {len(questions)} questions")
        
        if use_batch_processing and len(questions) > 1:
            # Import batch processor here to avoid circular imports
            from generation.batch_processor import BatchProcessor, BatchConfig
            
            # Prepare input data for batch processing
            batch_inputs = []
            for i, question in enumerate(questions):
                question_text = question.get("text", "") if isinstance(question, dict) else question
                
                # Get retrieved docs for this question if provided
                docs_for_question = None
                if retrieved_docs and i < len(retrieved_docs):
                    docs_for_question = retrieved_docs[i]
                
                batch_inputs.append({
                    "question": question_text,
                    "docs": docs_for_question,
                    "question_obj": question
                })
            
            # Create batch processor with configuration
            batch_config = BatchConfig(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                show_progress=True
            )
            processor = BatchProcessor(batch_config)
            
            # Define processing function
            def process_question(item):
                response = self.generate(
                    question=item["question"], 
                    documents=item["docs"] or []
                )
                
                # Add question info to response
                response["question"] = item["question"]
                if isinstance(item["question_obj"], dict) and "id" in item["question_obj"]:
                    response["question_id"] = item["question_obj"]["id"]
                
                return response
            
            # Process all questions in batches
            responses = processor.process(batch_inputs, process_question)
            
            # Log metrics
            metrics = processor.get_metrics()
            logger.info(f"Batch processing completed in {metrics['total_time']:.2f} seconds")
            logger.info(f"Processed {metrics['successful_items']}/{metrics['total_items']} questions successfully")
            logger.info(f"Processing rate: {metrics['items_per_second']:.2f} items/second")
            
            return responses
        else:
            # Fall back to sequential processing for a single question or when batch processing is disabled
            responses = []
            
            for i, question in enumerate(questions):
                question_text = question.get("text", "") if isinstance(question, dict) else question
                
                # Get retrieved docs for this question if provided
                docs_for_question = None
                if retrieved_docs and i < len(retrieved_docs):
                    docs_for_question = retrieved_docs[i]
                
                # Generate response
                response = self.generate(question_text, docs_for_question or [])
                
                # Add question info to response
                response["question"] = question_text
                if isinstance(question, dict) and "id" in question:
                    response["question_id"] = question["id"]
                
                responses.append(response)
                
                logger.info(f"Generated response for question {i+1}/{len(questions)}")
            
            return responses


class ProviderGenerator(BaseGenerator):
    """RAG generator using LLM provider system."""
    
    def __init__(self, 
                model_name: str = config.RAG_MODEL_NAME,
                provider_type: str = config.RAG_MODEL_PROVIDER,
                **provider_kwargs):
        """
        Initialize the provider generator.
        
        Args:
            model_name: Name of the model to use
            provider_type: Type of provider (openai, anthropic, huggingface, local)
            provider_kwargs: Additional provider-specific parameters
        """
        super().__init__(model_name)
        self.provider_type = provider_type
        
        # Initialize the provider
        try:
            self.provider = get_llm_provider(
                provider_type=provider_type, 
                model_name=model_name,
                **provider_kwargs
            )
            logger.info(f"Initialized {provider_type} provider with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing {provider_type} provider: {e}")
            raise
    
    def generate(
        self, 
        question: str, 
        documents: List[Dict],
        temperature: float = config.RAG_TEMPERATURE,
        max_tokens: int = config.RAG_MAX_TOKENS
    ) -> Dict:
        """
        Generate a response using the configured LLM provider with retrieved context.
        
        Args:
            question: Input question
            documents: Retrieved documents
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response with metadata
        """
        logger.info(f"Generating response for: {question[:50]}...")
        
        try:
            # Format context from retrieved documents
            context = self.format_context(documents)
            
            # Format prompt with question and context
            try:
                prompt = self.format_prompt(question, context)
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}")
                # Fallback to direct formatting if method is not available
                prompt = PROMPT_TEMPLATES["rag_prompt"].format(
                    context=context,
                    question=question
                )
            
            # Generate system prompt
            system_prompt = "You are a helpful assistant that provides accurate answers based on the given context."
            
            # Decide which method to use based on the provider capabilities
            if self.provider_type in ["openai", "anthropic"]:
                # These providers work best with chat format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response_data = self.provider.generate_chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use standard generate method with system prompt
                response_data = self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Extract response text
            response_text = response_data.get("text", "Error: No response text generated")
            
            # Extract token information when available
            token_info = response_data.get("tokens", {})
            prompt_tokens = token_info.get("prompt", 0)
            completion_tokens = token_info.get("completion", 0)
            total_tokens = token_info.get("total", 0)
            
            # Prepare the result
            result = {
                "question": question,
                "generated_response": response_text,
                "model": self.model_name,
                "provider": self.provider_type,
                "retrieved_documents": [self.get_document_metadata(doc) for doc in documents],
                "metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "temperature": temperature,
                    "provider_metadata": response_data.get("response_metadata", {})
                }
            }
            
            logger.info(f"Generated response with {result['metadata']['total_tokens']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return basic error response
            return {
                "question": question,
                "generated_response": f"Error generating response: {str(e)}",
                "model": self.model_name,
                "provider": self.provider_type,
                "retrieved_documents": [self.get_document_metadata(doc) for doc in documents],
                "metadata": {"error": str(e)}
            }   


class HuggingFaceGenerator(BaseGenerator):
    """RAG generator using Hugging Face models."""
    
    def __init__(
        self, 
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu"
    ):
        """
        Initialize HuggingFace generator.
        
        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda or cpu)
        """
        super().__init__(model_name)
        self.device = device
        self.model = None
        self.tokenizer = None
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate settings based on device
            if device == "cuda" and torch.cuda.is_available():
                logger.info(f"Using GPU for inference")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                logger.info(f"Using CPU for inference")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": "cpu"}
                )
                
        except ImportError:
            logger.error("Transformers library not installed. Install with 'pip install transformers'")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
    
    def generate(
        self, 
        question: str, 
        documents: List[Dict],
        temperature: float = config.RAG_TEMPERATURE,
        max_tokens: int = config.RAG_MAX_TOKENS
    ) -> Dict:
        """
        Generate a response using HuggingFace model with retrieved context.
        
        Args:
            question: Input question
            documents: Retrieved documents
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response with metadata
        """
        if self.model is None or self.tokenizer is None:
            logger.error("HuggingFace model not initialized")
            return {
                "question": question,
                "retrieved_documents": documents,
                "generated_response": "Error: Model not initialized",
                "generation_metadata": {
                    "model": self.model_name,
                    "error": "Model not initialized"
                }
            }
        
        try:
            import torch
            
            # Format context from documents
            context = self.format_context(documents)
            
            # Prepare RAG prompt
            prompt = PROMPT_TEMPLATES["rag_prompt"].format(
                context=context,
                question=question
            )
            
            # Format for model
            input_text = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Create response object with metadata
            result = {
                "question": question,
                "retrieved_documents": documents,
                "generated_response": generated_text,
                "generation_metadata": {
                    "model": self.model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "device": self.device
                }
            }
            
            logger.info(f"Generated response for question: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with HuggingFace: {e}")
            return {
                "question": question,
                "retrieved_documents": documents,
                "generated_response": f"Error generating response: {str(e)}",
                "generation_metadata": {
                    "model": self.model_name,
                    "error": str(e)
                }
            }


def get_generator(model_provider: str = config.RAG_MODEL_PROVIDER, **kwargs) -> BaseGenerator:
    """
    Factory function to get the appropriate generator based on configuration.
    
    Args:
        model_provider: Model provider to use (openai, anthropic, huggingface, local)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Initialized generator instance
    """
    if model_provider.lower() in ["openai", "anthropic", "huggingface"]:
        # Extract model_name from kwargs to avoid duplicate parameters
        provider_kwargs = kwargs.copy()
        model_name = provider_kwargs.pop("model_name", config.RAG_MODEL_NAME)
        
        # Use our new unified provider generator
        return ProviderGenerator(
            model_name=model_name,
            provider_type=model_provider.lower(),
            **provider_kwargs
        )
    elif model_provider.lower() == "local":
        # Local models need additional configuration
        provider_kwargs = kwargs.copy()
        model_path = provider_kwargs.pop("model_path", config.LOCAL_MODEL_PATH)
        
        if not model_path:
            logger.error("model_path must be provided for local models")
            raise ValueError("model_path must be provided for local models")
        
        # Extract model_name and backend to avoid duplicate parameters
        model_name = provider_kwargs.pop("model_name", os.path.basename(model_path))
        backend = provider_kwargs.pop("backend", config.LOCAL_MODEL_BACKEND)
            
        return ProviderGenerator(
            model_name=model_name,
            provider_type="local",
            model_path=model_path,
            backend=backend,
            **provider_kwargs
        )
    elif model_provider.lower() == "huggingface_local":
        # For backward compatibility with old HuggingFaceGenerator
        model_name = kwargs.pop("model_name", config.RAG_MODEL_NAME) if "model_name" in kwargs else config.RAG_MODEL_NAME
        return HuggingFaceGenerator(model_name=model_name)
    else:
        logger.warning(f"Unknown model provider: {model_provider}, using OpenAI")
        # Extract model_name from kwargs to avoid duplicate parameters
        provider_kwargs = kwargs.copy()
        provider_kwargs.pop("model_name", None)  # Remove if exists to avoid duplication
        
        return ProviderGenerator(
            model_name=config.RAG_MODEL_NAME,
            provider_type="openai",
            **provider_kwargs
        )


def generate_rag_response(
    question: str,
    top_k: int = config.RAG_TOP_K,
    temperature: float = config.RAG_TEMPERATURE,
    max_tokens: int = config.RAG_MAX_TOKENS,
    provider: str = config.RAG_MODEL_PROVIDER,
    **kwargs
) -> Dict:
    """
    End-to-end function to generate RAG responses.
    
    Args:
        question: Input question
        top_k: Number of documents to retrieve
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response with metadata
    """
    # Get retriever and generator
    retriever = get_retriever()
    
    # Configure generator with provider and model settings
    generator_kwargs = {
        "model_name": kwargs.get("model_name", config.RAG_MODEL_NAME),
        "temperature": temperature
    }
    
    # Add provider-specific parameters
    if provider == "local":
        # For local models, ensure we have path and backend
        model_path = kwargs.get("model_path", config.LOCAL_MODEL_PATH)
        if model_path:
            generator_kwargs["model_path"] = model_path
            generator_kwargs["backend"] = kwargs.get("backend", config.LOCAL_MODEL_BACKEND)
    
    # Create generator with provider settings
    generator = get_generator(provider, **generator_kwargs)
    
    # Retrieve documents
    documents = retriever.retrieve(question, top_k)
    
    # Generate response
    result = generator.generate(
        question, 
        documents,
        temperature,
        max_tokens
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RAG responses")
    parser.add_argument("--question", type=str, required=True, help="Input question")
    parser.add_argument("--output", type=str, help="Output file for the response")
    parser.add_argument("--top_k", type=int, default=config.RAG_TOP_K, help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=config.RAG_TEMPERATURE, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=config.RAG_MAX_TOKENS, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Generate RAG response
    result = generate_rag_response(
        args.question,
        args.top_k,
        args.temperature,
        args.max_tokens
    )
    
    # Print or save result
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Response saved to {args.output}")
    else:
        print("\n--- Question ---")
        print(result["question"])
        
        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\n[{i+1}] {doc.get('title', f'Document {i+1}')}")
            print(f"Score: {doc.get('retrieval_score', 'N/A')}")
            print(f"Text: {doc.get('text', 'N/A')[:200]}...")
            
        print("\n--- Generated Response ---")
        print(result["generated_response"])
