"""
Demo script to add sample documents to Chroma and test retrieval.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.absolute()))
import config

from sentence_transformers import SentenceTransformer
from retrieval.vector_stores.chroma_store import ChromaVectorStore
from retrieval.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Load sample documents
    sample_docs_path = os.getenv("SAMPLE_DOCS_PATH", "data/sample_documents.json")
    logger.info(f"Loading documents from {sample_docs_path}")
    documents = load_documents(sample_docs_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Process documents with chunking
    logger.info("Processing documents with chunking")
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50,
        add_metadata_to_text=True
    )
    processed_docs = processor.process_documents(documents)
    logger.info(f"Created {len(processed_docs)} chunks from {len(documents)} documents")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info(f"Initialized embedding model: {config.EMBEDDING_MODEL}")
    
    # Initialize Chroma store
    collection_name = "teacherforge_demo"
    persist_dir = "chroma_db"
    logger.info(f"Initializing Chroma store with collection: {collection_name}")
    chroma_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_model=embedding_model
    )
    
    # Add documents to Chroma
    logger.info("Adding documents to Chroma")
    chroma_store.add_documents(processed_docs)
    logger.info(f"Added {len(processed_docs)} documents to Chroma")
    
    # Test retrieval
    test_queries = [
        "What is TeacherForge?",
        "How does the retrieval system work?",
        "Explain RAG-based generation",
    ]
    
    logger.info("Testing retrieval")
    for query in test_queries:
        logger.info(f"Query: {query}")
        results = chroma_store.search(query, top_k=3)
        logger.info(f"Found {len(results)} results")
        
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: Score: {doc['retrieval_score']:.4f}")
            logger.info(f"Title: {doc.get('title', 'Untitled')}")
            logger.info(f"Text: {doc.get('text', '')[:100]}...")
            logger.info("-" * 50)
        
        print("\n")
    
    logger.info("Chroma demo completed successfully")

if __name__ == "__main__":
    main()
