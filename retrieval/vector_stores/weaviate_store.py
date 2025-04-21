"""
Weaviate vector store implementation for TeacherForge.
This module provides integration with Weaviate for vector search and document retrieval.
"""
import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Union

import weaviate
from weaviate.util import get_valid_uuid
from weaviate.exceptions import UnexpectedStatusCodeException
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class WeaviateVectorStore:
    """
    Weaviate vector store implementation for TeacherForge.
    
    This class provides methods to store documents in Weaviate, search for similar documents,
    and manage the Weaviate index.
    """
    
    def __init__(
        self,
        collection_name: str = "TeacherForge",
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        embedding_model: Optional[SentenceTransformer] = None,
        embedding_dim: int = 768,
    ):
        """
        Initialize the Weaviate vector store.
        
        Args:
            collection_name: Name of the Weaviate collection to use
            url: URL of the Weaviate instance
            api_key: API key for Weaviate authentication (optional)
            embedding_model: SentenceTransformer model for generating embeddings
            embedding_dim: Dimension of the embedding vectors
        """
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # Initialize Weaviate client
        auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")  # For optional OpenAI modules
            }
        )
        
        # Ensure collection exists
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create the collection in Weaviate if it doesn't already exist."""
        try:
            if not self.client.schema.exists(self.collection_name):
                logger.info(f"Creating collection {self.collection_name} in Weaviate")
                
                # Define collection schema
                class_obj = {
                    "class": self.collection_name,
                    "description": "TeacherForge document collection",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {
                        "skip": False,
                        "ef": 128,
                        "efConstruction": 128,
                        "maxConnections": 64,
                        "distance": "cosine"
                    },
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "The document text",
                            "indexInverted": True
                        },
                        {
                            "name": "title",
                            "dataType": ["text"],
                            "description": "The document title",
                            "indexInverted": True
                        },
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "The document source",
                            "indexInverted": True
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "JSON string containing document metadata",
                            "indexInverted": True
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["int"],
                            "description": "Chunk identifier for document chunks",
                            "indexInverted": True
                        },
                        {
                            "name": "doc_id",
                            "dataType": ["text"],
                            "description": "Original document identifier",
                            "indexInverted": True
                        }
                    ]
                }
                
                # Create the class/collection
                self.client.schema.create_class(class_obj)
                logger.info(f"Created collection {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists in Weaviate")
        
        except Exception as e:
            logger.error(f"Error creating Weaviate collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the Weaviate store.
        
        Args:
            documents: List of document dictionaries. Each document should have
                     'text', 'title', and optionally 'metadata', 'source', etc.
        """
        logger.info(f"Adding {len(documents)} documents to Weaviate collection {self.collection_name}")
        
        # Batch import configuration
        batch_size = 100
        with self.client.batch.configure(batch_size=batch_size, dynamic=True) as batch:
            for i, doc in enumerate(documents):
                # Generate embedding if model is provided
                vector = None
                if self.embedding_model is not None and "text" in doc:
                    vector = self.embedding_model.encode(doc["text"]).tolist()
                
                # Prepare document properties
                properties = {
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "doc_id": doc.get("id", str(uuid.uuid4())),
                    "chunk_id": doc.get("chunk_id", i),
                }
                
                # Handle metadata - convert to string if it's a dict
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    properties["metadata"] = json.dumps(metadata)
                else:
                    properties["metadata"] = str(metadata)
                
                # Generate a deterministic UUID based on doc_id
                uuid_str = get_valid_uuid(properties["doc_id"])
                
                # Add object to batch
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.collection_name,
                    uuid=uuid_str,
                    vector=vector
                )
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1} documents")
        
        logger.info(f"Added {len(documents)} documents to Weaviate")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filters: Dictionary of filters to apply to the search
        
        Returns:
            List of document dictionaries with similarity scores
        """
        # Generate embedding for query
        if self.embedding_model is None:
            raise ValueError("Embedding model not provided for search operation")
        
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Build Weaviate query
        query_builder = (
            self.client.query
            .get(self.collection_name, ["text", "title", "source", "doc_id", "chunk_id", "metadata"])
            .with_near_vector({"vector": query_vector, "certainty": 0.7})
            .with_limit(top_k)
        )
        
        # Add filters if provided
        if filters:
            where_filter = self._build_where_filter(filters)
            if where_filter:
                query_builder = query_builder.with_where(where_filter)
        
        # Execute the query
        try:
            result = query_builder.do()
            hits = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
        except Exception as e:
            logger.error(f"Error executing Weaviate query: {e}")
            return []
        
        # Format the results
        results = []
        for hit in hits:
            # Parse metadata from string to dict
            metadata = {}
            try:
                metadata_str = hit.get("metadata", "{}")
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            
            # Add document to results
            results.append({
                "text": hit.get("text", ""),
                "title": hit.get("title", ""),
                "source": hit.get("source", ""),
                "id": hit.get("doc_id", ""),
                "chunk_id": hit.get("chunk_id", 0),
                "metadata": metadata,
                "retrieval_score": hit.get("_additional", {}).get("certainty", 0.0)
            })
        
        return results
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert filters dictionary to Weaviate where filter.
        
        Args:
            filters: Dictionary of filters
        
        Returns:
            Weaviate where filter object
        """
        where_filter = {}
        operators = {
            "eq": "Equal",
            "ne": "NotEqual",
            "gt": "GreaterThan",
            "gte": "GreaterThanEqual",
            "lt": "LessThan",
            "lte": "LessThanEqual",
            "contains": "ContainsAny"
        }
        
        for field, condition in filters.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op in operators:
                        where_filter["path"] = [field]
                        where_filter["operator"] = operators[op]
                        where_filter["valueText"] = value
                        break
            else:
                # Simple equality filter
                where_filter["path"] = [field]
                where_filter["operator"] = "Equal"
                where_filter["valueText"] = condition
        
        return where_filter
    
    def delete_collection(self) -> None:
        """Delete the entire collection and all documents in it."""
        try:
            if self.client.schema.exists(self.collection_name):
                self.client.schema.delete_class(self.collection_name)
                logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Weaviate collection: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            result = (
                self.client.query
                .aggregate(self.collection_name)
                .with_meta_count()
                .do()
            )
            count = result.get("data", {}).get("Aggregate", {}).get(self.collection_name, [{}])[0].get("meta", {}).get("count", 0)
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    @classmethod
    def from_documents(
        cls,
        documents: List[Dict[str, Any]],
        embedding_model: SentenceTransformer,
        collection_name: str = "TeacherForge",
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ) -> "WeaviateVectorStore":
        """
        Create a WeaviateVectorStore from a list of documents.
        
        Args:
            documents: List of document dictionaries
            embedding_model: SentenceTransformer model for generating embeddings
            collection_name: Name of the collection to create
            url: URL of the Weaviate instance
            api_key: API key for Weaviate authentication
        
        Returns:
            WeaviateVectorStore instance with documents added
        """
        store = cls(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            embedding_model=embedding_model
        )
        store.add_documents(documents)
        return store
