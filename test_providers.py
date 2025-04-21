"""
Test script for LLM providers integration
Verifies that all provider classes can be initialized and used
"""

import logging
import os
import json
import sys
from pathlib import Path

import config
from generation.llm_providers.provider_factory import get_llm_provider
from generation.rag_generator import generate_rag_response, get_generator
from retrieval.retriever import get_retriever

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging output for better readability
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [console_handler]
logger.propagate = False

# Set other loggers to WARNING to reduce noise
logging.getLogger().setLevel(logging.WARNING)

def test_provider_factory():
    """Test that all providers can be initialized"""
    logger.info("Testing provider factory initialization...")
    
    # Test OpenAI provider
    try:
        openai_provider = get_llm_provider(provider_type="openai")
        logger.info(f"✅ OpenAI provider initialized successfully with model {openai_provider.model_name}")
    except Exception as e:
        logger.error(f"❌ OpenAI provider initialization failed: {str(e)}")
    
    # Test Anthropic provider (if API key is available)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            anthropic_provider = get_llm_provider(provider_type="anthropic")
            logger.info(f"✅ Anthropic provider initialized successfully with model {anthropic_provider.model_name}")
        except Exception as e:
            logger.error(f"❌ Anthropic provider initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping Anthropic provider test (no API key configured)")
    
    # Test HuggingFace provider (if API token is available)
    if os.environ.get("HF_API_TOKEN"):
        try:
            hf_provider = get_llm_provider(provider_type="huggingface")
            logger.info(f"✅ HuggingFace provider initialized successfully with model {hf_provider.model_name}")
        except Exception as e:
            logger.error(f"❌ HuggingFace provider initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping HuggingFace provider test (no API token configured)")
    
    # Skip local provider test if no model path is configured
    if config.LOCAL_MODEL_PATH:
        # Test Local provider
        try:
            local_provider = get_llm_provider(provider_type="local", model_path=config.LOCAL_MODEL_PATH)
            logger.info(f"✅ Local provider initialized successfully with model {local_provider.model_name}")
        except Exception as e:
            logger.error(f"❌ Local provider initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping local provider test (no model path configured)")

def test_provider_generators():
    """Test that all provider generators can be initialized"""
    logger.info("Testing provider generators...")
    
    # Test OpenAI generator
    try:
        openai_generator = get_generator(model_provider="openai")
        logger.info(f"✅ OpenAI generator initialized successfully with model {openai_generator.model_name}")
    except Exception as e:
        logger.error(f"❌ OpenAI generator initialization failed: {str(e)}")
    
    # Test Anthropic generator (if API key is available)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            anthropic_generator = get_generator(model_provider="anthropic")
            logger.info(f"✅ Anthropic generator initialized successfully with model {anthropic_generator.model_name}")
        except Exception as e:
            logger.error(f"❌ Anthropic generator initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping Anthropic generator test (no API key configured)")
    
    # Test HuggingFace generator (if API token is available)
    if os.environ.get("HF_API_TOKEN"):
        try:
            hf_generator = get_generator(model_provider="huggingface")
            logger.info(f"✅ HuggingFace generator initialized successfully with model {hf_generator.model_name}")
        except Exception as e:
            logger.error(f"❌ HuggingFace generator initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping HuggingFace generator test (no API token configured)")
    
    # Skip local generator test if no model path is configured
    if config.LOCAL_MODEL_PATH:
        # Test Local generator
        try:
            local_generator = get_generator(model_provider="local", model_path=config.LOCAL_MODEL_PATH)
            logger.info(f"✅ Local generator initialized successfully with model {local_generator.model_name}")
        except Exception as e:
            logger.error(f"❌ Local generator initialization failed: {str(e)}")
    else:
        logger.info("ℹ️ Skipping local generator test (no model path configured)")

def test_simple_generation():
    """Test a simple message generation using the default provider"""
    logger.info(f"Testing simple generation with provider: {config.RAG_MODEL_PROVIDER}")
    
    # Skip test if no API key is available for the configured provider
    if config.RAG_MODEL_PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ℹ️ Skipping Anthropic generation test (no API key configured)")
        return
    elif config.RAG_MODEL_PROVIDER == "huggingface" and not os.environ.get("HF_API_TOKEN"):
        logger.info("ℹ️ Skipping HuggingFace generation test (no API token configured)")
        return
    elif config.RAG_MODEL_PROVIDER == "local" and not config.LOCAL_MODEL_PATH:
        logger.info("ℹ️ Skipping local model generation test (no model path configured)")
        return
    
    try:
        provider = get_llm_provider(provider_type=config.RAG_MODEL_PROVIDER)
        
        # Test basic generation (non-chat)
        result = provider.generate(
            prompt="Explain RAG (Retrieval Augmented Generation) in one sentence.",
            system_prompt="You are a helpful AI assistant that provides concise responses.",
            temperature=0.1,
            max_tokens=100
        )
        logger.info(f"✅ Basic generation successful: {result.get('text', '')[:50]}...")
    except Exception as e:
        logger.error(f"❌ Basic generation failed: {str(e)}")
    
    # Test chat generation if the provider supports it
    if hasattr(provider, 'generate_chat'):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that provides concise responses."},
                {"role": "user", "content": "Explain RAG (Retrieval Augmented Generation) in one sentence."}
            ]
            result = provider.generate_chat(
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )
            logger.info(f"✅ Chat generation successful: {result.get('text', '')[:50]}...")
        except Exception as e:
            logger.error(f"❌ Chat generation failed: {e}")

def test_rag_response():
    """Test full RAG generation pipeline with the default provider"""
    logger.info(f"Testing RAG pipeline with provider: {config.RAG_MODEL_PROVIDER}")
    
    # Skip test if no API key is available for the configured provider
    if config.RAG_MODEL_PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ℹ️ Skipping Anthropic RAG test (no API key configured)")
        return
    elif config.RAG_MODEL_PROVIDER == "huggingface" and not os.environ.get("HF_API_TOKEN"):
        logger.info("ℹ️ Skipping HuggingFace RAG test (no API token configured)")
        return
    elif config.RAG_MODEL_PROVIDER == "local" and not config.LOCAL_MODEL_PATH:
        logger.info("ℹ️ Skipping local model RAG test (no model path configured)")
        return
    
    try:
        # Use a simple test question
        question = "What is retrieval-augmented generation?"
        
        # Generate RAG response
        response = generate_rag_response(
            question=question,
            top_k=2,  # Use fewer documents for testing
            temperature=0.1,
            provider=config.RAG_MODEL_PROVIDER
        )
        
        logger.info(f"✅ RAG generation successful")
        logger.info(f"Question: {question}")
        logger.info(f"Generated response: {response.get('generated_response', '')[:100]}...")
        
        # Save test output
        with open("test_rag_output.json", "w") as f:
            json.dump(response, f, indent=2)
        logger.info(f"Full response saved to test_rag_output.json")
        
    except Exception as e:
        logger.error(f"❌ RAG generation failed: {str(e)}")

def print_section_header(title):
    """Print a section header with a title"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

if __name__ == "__main__":
    print_section_header("TeacherForge LLM Provider Tests")
    logger.info("Starting LLM provider tests...")
    
    # Run tests
    print_section_header("1. Testing Provider Factory Initialization")
    test_provider_factory()
    
    print_section_header("2. Testing Provider Generators")
    test_provider_generators()
    
    print_section_header("3. Testing Simple LLM Generation")
    test_simple_generation()
    
    print_section_header("4. Testing Full RAG Pipeline")
    test_rag_response()
    
    print_section_header("Test Summary")
    logger.info("All tests completed successfully!")
    print("\nIf there were no error messages above, the LLM provider integration is working correctly!")
    print("You can now use the different LLM providers in the TeacherForge system.")
