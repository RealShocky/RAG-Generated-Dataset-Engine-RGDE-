"""
Chroma vector store implementation for TeacherForge.
This module provides integration with Chroma for vector search and document retrieval.
"""
import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Chroma vector store implementation for TeacherForge.
    
    This class provides methods to store documents in Chroma, search for similar documents,
    and manage the Chroma collection.
    """
    
    def __init__(
        self,
        collection_name: str = "teacherforge",
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[SentenceTransformer] = None,
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name: Name of the Chroma collection to use
            persist_directory: Directory to persist the Chroma database
            embedding_model: SentenceTransformer model for generating embeddings
            embedding_function: Custom embedding function to use instead of embedding_model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Setup embeddings
        if embedding_function:
            self.embedding_function = embedding_function
        elif embedding_model:
            # If we're provided with an actual SentenceTransformer model instance,
            # create a custom embedding function that follows Chroma's interface
            class CustomSentenceTransformerFunction(embedding_functions.EmbeddingFunction):
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    return embeddings
                
            self.embedding_function = CustomSentenceTransformerFunction(embedding_model)
        else:
            # Default to a standard Sentence Transformer embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Initialize ChromaDB client with persistance
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "TeacherForge document collection"}
            )
            logger.info(f"Connected to Chroma collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error connecting to Chroma collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the Chroma store.
        
        Args:
            documents: List of document dictionaries. Each document should have
                      'text', 'title', and optionally 'metadata', 'source', etc.
        """
        logger.info(f"Adding {len(documents)} documents to Chroma collection {self.collection_name}")
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            # Generate a unique ID
            doc_id = doc.get("id", str(uuid.uuid4()))
            ids.append(doc_id)
            
            # Get the document text
            texts.append(doc.get("text", ""))
            
            # Prepare metadata - must be a flat dict of string keys and string/int/float values
            metadata = {
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "doc_id": doc.get("id", ""),
                "chunk_id": str(doc.get("chunk_id", "0")),
            }
            
            # Handle nested metadata - flatten or convert to string
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                for k, v in doc["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[f"metadata_{k}"] = v
                    else:
                        metadata[f"metadata_{k}"] = json.dumps(v)
            
            metadatas.append(metadata)
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(ids):
                logger.info(f"Added {min(i + batch_size, len(ids))} documents")
        
        logger.info(f"Added {len(documents)} documents to Chroma")
    
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
        # Convert filters to Chroma format if provided
        where_clause = None
        if filters:
            where_clause = self._build_where_filter(filters)
        
        # Perform the search
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            return []
        
        # Process results
        documents = []
        
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
                
                documents.append(document)
        
        return documents
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert filters dictionary to Chroma where filter.
        
        Args:
            filters: Dictionary of filters
        
        Returns:
            Chroma where filter object
        """
        where_filter = {}
        
        # Handle simple equality filters
        for field, value in filters.items():
            if isinstance(value, dict):
                # Handle operators like $eq, $gt, etc.
                for op, op_value in value.items():
                    if op == "eq":
                        where_filter[field] = op_value
                    elif op in ["gt", "gte", "lt", "lte"]:
                        # Chroma uses specific operator syntax
                        where_filter[field] = {f"${op}": op_value}
            else:
                # Simple equality
                where_filter[field] = value
        
        return where_filter
    
    def delete_collection(self) -> None:
        """Delete the entire collection and all documents in it."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Chroma collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Dict[str, Any]],
        embedding_model: Optional[SentenceTransformer] = None,
        collection_name: str = "teacherforge",
        persist_directory: str = "./chroma_db",
    ) -> "ChromaVectorStore":
        """
        Create a ChromaVectorStore from a list of documents.
        
        Args:
            documents: List of document dictionaries
            embedding_model: SentenceTransformer model for generating embeddings
            collection_name: Name of the collection to create
            persist_directory: Directory to persist the Chroma database
        
        Returns:
            ChromaVectorStore instance with documents added
        """
        store = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        store.add_documents(documents)
        return store
