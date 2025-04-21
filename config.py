"""
Configuration settings for TeacherForge.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.absolute()
PROMPTS_DIR = ROOT_DIR / "prompts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Ensure all directories exist
for dir_path in [PROMPTS_DIR, OUTPUTS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# API Keys (load from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# Vector DB settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # Options: faiss, qdrant, weaviate
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "")
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "teacherforge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# FAISS settings
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
FAISS_DOCUMENTS_PATH = os.getenv("FAISS_DOCUMENTS_PATH", "data/sample_documents.json")

# Weaviate settings
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# Chroma settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# RAG model settings
RAG_MODEL_PROVIDER = os.getenv("RAG_MODEL_PROVIDER", "openai")  # Options: openai, anthropic, huggingface, local
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "gpt-4")
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))  # Number of documents to retrieve
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1000"))

# Local model settings
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")  # Path to local model file
LOCAL_MODEL_BACKEND = os.getenv("LOCAL_MODEL_BACKEND", "llamacpp")  # Options: llamacpp, vllm
LOCAL_MODEL_N_GPU_LAYERS = int(os.getenv("LOCAL_MODEL_N_GPU_LAYERS", "-1"))  # -1 means use all GPU layers
LOCAL_MODEL_CONTEXT_LENGTH = int(os.getenv("LOCAL_MODEL_CONTEXT_LENGTH", "4096"))

# Student model settings
STUDENT_MODEL_NAME = os.getenv("STUDENT_MODEL_NAME", "mistralai/Mistral-7B-v0.1")
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "8"))
TRAINING_GRADIENT_ACCUMULATION_STEPS = int(os.getenv("TRAINING_GRADIENT_ACCUMULATION_STEPS", "4"))
TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", "2e-4"))
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "3"))

# Default prompt templates
PROMPT_TEMPLATES = {
    "rag_prompt": """You are a helpful AI assistant that generates accurate, factual responses based on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a comprehensive, detailed answer to the question using only the information from the context. 
If the context doesn't contain enough information to answer the question, state "I don't have enough information to answer this question."
""",
    
    "prompt_generator": """Generate {num_prompts} diverse, {domain_specific} questions that would be useful for training an AI assistant.
The questions should be varied in complexity, format, and subject matter within the domain.
Make sure the questions are clear, specific, and would require detailed responses.

Output each question on a new line."""
}

# Default logging configuration
import logging
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
}
