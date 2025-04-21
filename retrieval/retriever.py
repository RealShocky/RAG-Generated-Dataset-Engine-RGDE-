"""
Retriever Module
Retrieves relevant documents from a vector database based on input questions.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import caching system
from retrieval.cache import CacheManager, CacheConfig

from config import VECTOR_DB_TYPE, FAISS_INDEX_PATH, FAISS_DOCUMENTS_PATH

# Conditionally import vector stores to avoid requiring all dependencies
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class BaseRetriever:
    """Base class for all retrievers."""
    
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL, use_cache: bool = True):
        """
        Initialize the retriever with an embedding model.
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
            use_cache: Whether to use caching for embeddings and retrieval results
        """
        self.embedding_model_name = embedding_model
        self.use_cache = use_cache
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = None
        if self.use_cache:
            cache_config = CacheConfig()
            self.cache_manager = CacheManager(cache_config)
            logger.info("Initialized caching system for retriever")
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text. Uses cache if available.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first if enabled
        if self.use_cache and self.cache_manager:
            cached_embedding = self.cache_manager.embedding_cache.get(text, self.embedding_model_name)
            if cached_embedding is not None:
                logger.debug(f"Using cached embedding for text: {text[:50]}...")
                return cached_embedding
        
        # Generate new embedding
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        
        # Store in cache if enabled
        if self.use_cache and self.cache_manager:
            self.cache_manager.embedding_cache.put(text, self.embedding_model_name, embedding)
            
        return embedding
        
    def _get_cached_retrieval(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results if available"""
        if self.use_cache and self.cache_manager:
            return self.cache_manager.retrieval_cache.get(
                query, 
                getattr(self, 'collection_name', 'default'),
                top_k
            )
        return None
    
    def _cache_retrieval_results(self, query: str, top_k: int, results: List[Dict[str, Any]]) -> None:
        """Cache retrieval results"""
        if self.use_cache and self.cache_manager:
            self.cache_manager.retrieval_cache.put(
                query, 
                getattr(self, 'collection_name', 'default'),
                top_k,
                results
            )
            
    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Question or query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        raise NotImplementedError("Subclasses must implement retrieve()")


class FaissRetriever(BaseRetriever):
    """Retriever using FAISS vector store."""
    
    def __init__(
        self, 
        index_path: Optional[str] = None,
        documents_path: Optional[str] = None,
        embedding_model: str = config.EMBEDDING_MODEL,
        use_cache: bool = True
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file
            documents_path: Path to documents JSON file
            embedding_model: Name of the sentence transformer model for embeddings
            use_cache: Whether to use caching for embeddings and retrieval results
        """
        super().__init__(embedding_model, use_cache)
        
        # Set collection name for cache
        self.collection_name = "faiss_" + (os.path.basename(documents_path) if documents_path else "default")
        
        self.index = None
        self.documents = []
        
        if index_path and documents_path:
            self.load_index(index_path, documents_path)
    
    def load_index(self, index_path: str, documents_path: str) -> None:
        """
        Load FAISS index and documents.
        
        Args:
            index_path: Path to FAISS index file
            documents_path: Path to documents JSON file
        """
        try:
            import faiss
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
            
            # Load documents
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} documents from {documents_path}")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> List[Dict]:
        """
        Retrieve relevant documents for a query using FAISS.
        
        Args:
            query: Question or query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.index is None:
            logger.error("FAISS index not loaded")
            return []
        
        # Check cache first
        cached_results = self._get_cached_retrieval(query, top_k)
        if cached_results is not None:
            logger.info(f"Using cached retrieval results for query: {query[:50]}...")
            return cached_results
        
        try:
            # Embed query using our cached embedding method
            query_embedding = self._get_embedding(query)
            
            # Search FAISS index
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), top_k
            )
            
            # Retrieve documents
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    doc = self.documents[idx].copy()
                    doc["retrieval_score"] = float(distances[0][i])
                    retrieved_docs.append(doc)
            
            # Cache results
            self._cache_retrieval_results(query, top_k, retrieved_docs)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


class QdrantRetriever(BaseRetriever):
    """Retriever using Qdrant vector store."""
    
    def __init__(
        self,
        collection_name: str = config.VECTOR_DB_COLLECTION,
        url: Optional[str] = config.VECTOR_DB_URL,
        api_key: Optional[str] = None,
        embedding_model: str = config.EMBEDDING_MODEL,
        use_cache: bool = True
    ):
        """
        Initialize Qdrant retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: URL of the Qdrant server
            api_key: API key for Qdrant
            embedding_model: Name of the sentence transformer model for embeddings
            use_cache: Whether to use caching for embeddings and retrieval results
        """
        super().__init__(embedding_model, use_cache)
        
        self.collection_name = collection_name
        self.client = None
        
        try:
            from qdrant_client import QdrantClient
            
            # Initialize Qdrant client
            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
                logger.info(f"Connected to Qdrant at {url}")
            else:
                logger.warning("No Qdrant URL provided, using local instance")
                self.client = QdrantClient(":memory:")
                
        except ImportError:
            logger.error("Qdrant client not installed. Install with 'pip install qdrant-client'")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
    
    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> List[Dict]:
        """
        Retrieve relevant documents for a query using Qdrant.
        
        Args:
            query: Question or query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return []
        
        # Check cache first
        cached_results = self._get_cached_retrieval(query, top_k)
        if cached_results is not None:
            logger.info(f"Using cached Qdrant retrieval results for query: {query[:50]}...")
            return cached_results
        
        try:
            from qdrant_client.http import models
            
            # Embed query using our cached embedding method
            query_embedding = self._get_embedding(query)
            
            # Search Qdrant collection
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            # Process results
            retrieved_docs = []
            for scored_point in search_result:
                # Extract document
                doc = scored_point.payload.copy()
                
                # Add metadata if not present
                if "metadata" not in doc:
                    doc["metadata"] = {}
                
                # Add retrieval score
                doc["retrieval_score"] = float(scored_point.score)
                
                retrieved_docs.append(doc)
            
            # Cache results
            self._cache_retrieval_results(query, top_k, retrieved_docs)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents from Qdrant for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents from Qdrant: {e}")
            return []


class WeaviateRetriever(BaseRetriever):
    """Retriever using Weaviate vector store."""
    
    def __init__(
        self,
        collection_name: str = config.VECTOR_DB_COLLECTION,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        embedding_model: str = config.EMBEDDING_MODEL
    ):
        """
        Initialize Weaviate retriever.
        
        Args:
            collection_name: Name of the Weaviate collection/class
            url: URL of the Weaviate instance
            api_key: API key for Weaviate authentication
            embedding_model: Name of the sentence transformer model for embeddings
        """
        super().__init__(embedding_model)
        
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "Weaviate is not installed. "
                "Please install it with 'pip install weaviate-client'"
            )
        
        self.collection_name = collection_name
        self.client = None
        
        try:
            # Initialize Weaviate client
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config,
                additional_headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")  # For optional OpenAI modules
                }
            )
            logger.info(f"Connected to Weaviate at {url}")
        
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> List[Dict]:
        """
        Retrieve relevant documents for a query using Weaviate.
        
        Args:
            query: Question or query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.client is None:
            logger.error("Weaviate client not initialized")
            return []
        
        try:
            # Embed query
            query_embedding = self.embed_query(query)
            
            # Build Weaviate query
            query_builder = (
                self.client.query
                .get(self.collection_name, ["text", "title", "source", "doc_id", "chunk_id", "metadata"])
                .with_near_vector({"vector": query_embedding.tolist(), "certainty": 0.7})
                .with_limit(top_k)
            )
            
            # Execute the query
            result = query_builder.do()
            hits = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
            
            # Format the results
            retrieved_docs = []
            for hit in hits:
                # Parse metadata from string to dict
                metadata = {}
                try:
                    metadata_str = hit.get("metadata", "{}")
                    metadata = json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                # Add document to results
                doc = {
                    "text": hit.get("text", ""),
                    "title": hit.get("title", ""),
                    "source": hit.get("source", ""),
                    "id": hit.get("doc_id", ""),
                    "chunk_id": hit.get("chunk_id", 0),
                    "metadata": metadata,
                    "retrieval_score": hit.get("_additional", {}).get("certainty", 0.0)
                }
                retrieved_docs.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents from Weaviate for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents from Weaviate: {e}")
            return []


class ChromaRetriever(BaseRetriever):
    """Retriever using Chroma vector store."""
    
    def __init__(
        self,
        collection_name: str = config.VECTOR_DB_COLLECTION,
        persist_directory: str = "./chroma_db",
        embedding_model: str = config.EMBEDDING_MODEL
    ):
        """
        Initialize Chroma retriever.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the Chroma database
            embedding_model: Name of the sentence transformer model for embeddings
        """
        super().__init__(embedding_model)
        
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "Chroma is not installed. "
                "Please install it with 'pip install chromadb'"
            )
        
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        try:
            from chromadb.config import Settings
            
            # Initialize ChromaDB client with persistance
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create embedding function using our SentenceTransformer model
            from chromadb.utils import embedding_functions
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"description": "TeacherForge document collection"}
            )
            
            logger.info(f"Connected to Chroma collection: {collection_name}")
        
        except Exception as e:
            logger.error(f"Error connecting to Chroma: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> List[Dict]:
        """
        Retrieve relevant documents for a query using Chroma.
        
        Args:
            query: Question or query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.collection is None:
            logger.error("Chroma collection not initialized")
            return []
        
        try:
            # Perform the search using the embedded query function
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results and "documents" in results and results["documents"]:
                for i, doc_text in enumerate(results["documents"][0]):
                    if i >= len(results["metadatas"][0]) or i >= len(results["distances"][0]):
                        continue
                    
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Extract standard metadata fields
                    title = metadata.get("title", "")
                    source = metadata.get("source", "")
                    doc_id = metadata.get("doc_id", "")
                    chunk_id = metadata.get("chunk_id", "0")
                    
                    # Extract custom metadata fields
                    custom_metadata = {}
                    for k, v in metadata.items():
                        if k.startswith("metadata_"):
                            # Try to parse JSON values
                            try:
                                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                                    custom_metadata[k[9:]] = json.loads(v)
                                else:
                                    custom_metadata[k[9:]] = v
                            except json.JSONDecodeError:
                                custom_metadata[k[9:]] = v
                    
                    # Convert distance to similarity score (1 - distance for cosine)
                    similarity = 1.0 - distance
                    
                    # Create document dictionary
                    document = {
                        "text": doc_text,
                        "title": title,
                        "source": source,
                        "id": doc_id,
                        "chunk_id": chunk_id,
                        "metadata": custom_metadata,
                        "retrieval_score": similarity
                    }
                    
                    retrieved_docs.append(document)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents from Chroma for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents from Chroma: {e}")
            return []


def get_retriever(vector_db_type: str = config.VECTOR_DB_TYPE) -> BaseRetriever:
    """
    Factory function to get the appropriate retriever based on configuration.
    
    Args:
        vector_db_type: Type of vector database to use
        
    Returns:
        Initialized retriever instance
    """
    vector_db_type = vector_db_type.lower()
    
    if vector_db_type == "faiss":
        # Create FAISS retriever
        index_path = os.getenv("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
        documents_path = os.getenv("FAISS_DOCUMENTS_PATH", FAISS_DOCUMENTS_PATH)
        
        return FaissRetriever(
            index_path=index_path, 
            documents_path=documents_path,
            embedding_model=config.EMBEDDING_MODEL
        )
    
    elif vector_db_type == "qdrant":
        # Create Qdrant retriever
        return QdrantRetriever(
            collection_name=config.VECTOR_DB_COLLECTION,
            url=config.VECTOR_DB_URL,
            embedding_model=config.EMBEDDING_MODEL
        )
    
    elif vector_db_type == "weaviate":
        # Create Weaviate retriever if available
        if not WEAVIATE_AVAILABLE:
            logger.error("Weaviate is not installed. Please install with 'pip install weaviate-client'")
            raise ImportError("Weaviate is not installed. Please install with 'pip install weaviate-client'")
        
        return WeaviateRetriever(
            collection_name=config.VECTOR_DB_COLLECTION,
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            embedding_model=config.EMBEDDING_MODEL
        )
    
    elif vector_db_type == "chroma":
        # Create Chroma retriever if available
        if not CHROMA_AVAILABLE:
            logger.error("Chroma is not installed. Please install with 'pip install chromadb'")
            raise ImportError("Chroma is not installed. Please install with 'pip install chromadb'")
        
        return ChromaRetriever(
            collection_name=config.VECTOR_DB_COLLECTION,
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            embedding_model=config.EMBEDDING_MODEL
        )
    
    else:
        # Default to FAISS
        logger.warning(f"Unknown vector database type: {vector_db_type}, defaulting to FAISS")
        index_path = os.getenv("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
        documents_path = os.getenv("FAISS_DOCUMENTS_PATH", FAISS_DOCUMENTS_PATH)
        
        return FaissRetriever(
            index_path=index_path, 
            documents_path=documents_path,
            embedding_model=config.EMBEDDING_MODEL
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the retriever")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--db", type=str, default=config.VECTOR_DB_TYPE, help="Vector DB type")
    parser.add_argument("--top_k", type=int, default=config.RAG_TOP_K, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = get_retriever(args.db)
    
    # Retrieve documents
    documents = retriever.retrieve(args.query, args.top_k)
    
    # Print results
    print(f"Retrieved {len(documents)} documents for query: {args.query}")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} (Score: {doc.get('retrieval_score', 'N/A')}) ---")
        print(f"Title: {doc.get('title', 'N/A')}")
        print(f"Text: {doc.get('text', 'N/A')[:200]}...")
