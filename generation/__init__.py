"""
TeacherForge Generation Module
Handles the generation of answers from retrieved documents using language models.
"""

from generation.rag_generator import generate_rag_response, get_generator
from generation.llm_providers import get_llm_provider

__all__ = [
    "generate_rag_response",
    "get_generator",
    "get_llm_provider"
]
