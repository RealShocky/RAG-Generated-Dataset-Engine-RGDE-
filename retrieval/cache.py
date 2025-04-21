"""
Cache Module for TeacherForge
Provides caching mechanisms for embeddings and responses to improve performance
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for the cache system"""
    # General settings
    enabled: bool = True
    cache_dir: str = "./cache"
    max_age_days: int = 30  # Maximum age of cache entries
    
    # Type-specific settings
    embeddings_enabled: bool = True
    responses_enabled: bool = True
    retrieval_enabled: bool = True
    
    # Size limits
    max_embeddings_cache_size_mb: int = 500  # 500 MB
    max_responses_cache_size_mb: int = 200   # 200 MB


class BaseCache:
    """Base cache implementation with common functionality"""
    
    def __init__(self, config: CacheConfig, cache_type: str):
        self.config = config
        self.cache_type = cache_type
        self.cache_path = Path(config.cache_dir) / cache_type
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)
        
        # Load cache index
        self.index_path = self.cache_path / "index.json"
        self.cache_index = self._load_index()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _compute_key(self, data: Any) -> str:
        """Compute a unique key for the given data"""
        if isinstance(data, str):
            hash_input = data.encode('utf-8')
        elif isinstance(data, dict):
            hash_input = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, (list, tuple)):
            hash_input = json.dumps([str(item) for item in data], sort_keys=True).encode('utf-8')
        else:
            hash_input = str(data).encode('utf-8')
        
        return hashlib.md5(hash_input).hexdigest()
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get the path to a cache file for a given key"""
        return self.cache_path / f"{key}.pkl"
    
    def clear(self) -> None:
        """Clear the cache"""
        for file in self.cache_path.glob("*.pkl"):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Error removing cache file {file}: {e}")
        
        self.cache_index = {}
        self._save_index()
        logger.info(f"Cleared {self.cache_type} cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(os.path.getsize(f) for f in self.cache_path.glob("*.pkl") if os.path.isfile(f))
        
        return {
            "type": self.cache_type,
            "entries": len(self.cache_index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
    
    def prune_old_entries(self) -> int:
        """Remove entries older than the configured max age"""
        now = time.time()
        max_age_seconds = self.config.max_age_days * 24 * 60 * 60
        pruned_count = 0
        
        for key, metadata in list(self.cache_index.items()):
            if now - metadata.get("timestamp", 0) > max_age_seconds:
                cache_file = self._get_cache_file_path(key)
                try:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    del self.cache_index[key]
                    pruned_count += 1
                except Exception as e:
                    logger.warning(f"Error pruning cache entry {key}: {e}")
        
        if pruned_count > 0:
            self._save_index()
            logger.info(f"Pruned {pruned_count} old entries from {self.cache_type} cache")
        
        return pruned_count


class EmbeddingCache(BaseCache):
    """Cache for document embeddings"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config, "embeddings")
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get embedding from cache if it exists"""
        if not self.config.enabled or not self.config.embeddings_enabled:
            return None
        
        # Create a key using both text and model name
        data = {"text": text, "model": model_name}
        key = self._compute_key(data)
        
        if key in self.cache_index:
            try:
                cache_file = self._get_cache_file_path(key)
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    
                    # Update timestamp to mark as recently used
                    self.cache_index[key]["last_accessed"] = time.time()
                    self._save_index()
                    
                    self.hits += 1
                    return embedding
            except Exception as e:
                logger.warning(f"Error retrieving embedding from cache: {e}")
        
        self.misses += 1
        return None
    
    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        if not self.config.enabled or not self.config.embeddings_enabled:
            return
        
        # Create a key using both text and model name
        data = {"text": text, "model": model_name}
        key = self._compute_key(data)
        
        try:
            # Serialize embedding to disk
            cache_file = self._get_cache_file_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update index
            self.cache_index[key] = {
                "text_prefix": text[:100] + "..." if len(text) > 100 else text,
                "model": model_name,
                "shape": embedding.shape,
                "timestamp": time.time(),
                "last_accessed": time.time()
            }
            self._save_index()
            
            # Check if cache is too large and prune if necessary
            self._enforce_size_limits()
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    def _enforce_size_limits(self) -> None:
        """Enforce cache size limits by removing least recently used entries"""
        total_size = sum(os.path.getsize(f) for f in self.cache_path.glob("*.pkl") if os.path.isfile(f))
        max_size_bytes = self.config.max_embeddings_cache_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort entries by last accessed time
            entries = [(k, v.get("last_accessed", 0)) for k, v in self.cache_index.items()]
            entries.sort(key=lambda x: x[1])  # Sort by last accessed time (oldest first)
            
            # Remove oldest entries until we're under the limit
            removed = 0
            for key, _ in entries:
                cache_file = self._get_cache_file_path(key)
                if os.path.exists(cache_file):
                    file_size = os.path.getsize(cache_file)
                    os.remove(cache_file)
                    total_size -= file_size
                    del self.cache_index[key]
                    removed += 1
                
                if total_size <= max_size_bytes * 0.9:  # Target 90% of limit to avoid frequent pruning
                    break
            
            self._save_index()
            logger.info(f"Pruned {removed} entries from embedding cache to enforce size limits")


class ResponseCache(BaseCache):
    """Cache for generated responses"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config, "responses")
    
    def get(self, question: str, context: List[Dict[str, Any]], model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached response if it exists"""
        if not self.config.enabled or not self.config.responses_enabled:
            return None
        
        # Create a key using question, context, and model name
        data = {
            "question": question,
            "context": self._normalize_context(context),
            "model": model_name
        }
        key = self._compute_key(data)
        
        if key in self.cache_index:
            try:
                cache_file = self._get_cache_file_path(key)
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        response = pickle.load(f)
                    
                    # Update timestamp to mark as recently used
                    self.cache_index[key]["last_accessed"] = time.time()
                    self._save_index()
                    
                    self.hits += 1
                    return response
            except Exception as e:
                logger.warning(f"Error retrieving response from cache: {e}")
        
        self.misses += 1
        return None
    
    def put(self, question: str, context: List[Dict[str, Any]], model_name: str, response: Dict[str, Any]) -> None:
        """Store response in cache"""
        if not self.config.enabled or not self.config.responses_enabled:
            return
        
        # Create a key using question, context, and model name
        data = {
            "question": question,
            "context": self._normalize_context(context),
            "model": model_name
        }
        key = self._compute_key(data)
        
        try:
            # Serialize response to disk
            cache_file = self._get_cache_file_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
            
            # Update index
            self.cache_index[key] = {
                "question_prefix": question[:100] + "..." if len(question) > 100 else question,
                "model": model_name,
                "timestamp": time.time(),
                "last_accessed": time.time()
            }
            self._save_index()
            
            # Check if cache is too large and prune if necessary
            self._enforce_size_limits()
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
    
    def _normalize_context(self, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize context for consistent hashing by extracting only text and source"""
        normalized = []
        for doc in context:
            normalized.append({
                "text": doc.get("text", ""),
                "source": doc.get("source", "")
            })
        return normalized
    
    def _enforce_size_limits(self) -> None:
        """Enforce cache size limits by removing least recently used entries"""
        total_size = sum(os.path.getsize(f) for f in self.cache_path.glob("*.pkl") if os.path.isfile(f))
        max_size_bytes = self.config.max_responses_cache_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort entries by last accessed time
            entries = [(k, v.get("last_accessed", 0)) for k, v in self.cache_index.items()]
            entries.sort(key=lambda x: x[1])  # Sort by last accessed time (oldest first)
            
            # Remove oldest entries until we're under the limit
            removed = 0
            for key, _ in entries:
                cache_file = self._get_cache_file_path(key)
                if os.path.exists(cache_file):
                    file_size = os.path.getsize(cache_file)
                    os.remove(cache_file)
                    total_size -= file_size
                    del self.cache_index[key]
                    removed += 1
                
                if total_size <= max_size_bytes * 0.9:  # Target 90% of limit to avoid frequent pruning
                    break
            
            self._save_index()
            logger.info(f"Pruned {removed} entries from response cache to enforce size limits")


class RetrievalCache(BaseCache):
    """Cache for document retrieval results"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config, "retrieval")
    
    def get(self, query: str, collection_name: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results if they exist"""
        if not self.config.enabled or not self.config.retrieval_enabled:
            return None
        
        # Create a key using query, collection, and top_k
        data = {
            "query": query, 
            "collection": collection_name,
            "top_k": top_k
        }
        key = self._compute_key(data)
        
        if key in self.cache_index:
            try:
                cache_file = self._get_cache_file_path(key)
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Update timestamp to mark as recently used
                    self.cache_index[key]["last_accessed"] = time.time()
                    self._save_index()
                    
                    self.hits += 1
                    return results
            except Exception as e:
                logger.warning(f"Error retrieving retrieval results from cache: {e}")
        
        self.misses += 1
        return None
    
    def put(self, query: str, collection_name: str, top_k: int, results: List[Dict[str, Any]]) -> None:
        """Store retrieval results in cache"""
        if not self.config.enabled or not self.config.retrieval_enabled:
            return
        
        # Create a key using query, collection, and top_k
        data = {
            "query": query, 
            "collection": collection_name,
            "top_k": top_k
        }
        key = self._compute_key(data)
        
        try:
            # Serialize results to disk
            cache_file = self._get_cache_file_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Update index
            self.cache_index[key] = {
                "query_prefix": query[:100] + "..." if len(query) > 100 else query,
                "collection": collection_name,
                "top_k": top_k,
                "num_results": len(results),
                "timestamp": time.time(),
                "last_accessed": time.time()
            }
            self._save_index()
        except Exception as e:
            logger.warning(f"Error caching retrieval results: {e}")


class CacheManager:
    """Manages all cache types"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.embedding_cache = EmbeddingCache(self.config)
        self.response_cache = ResponseCache(self.config)
        self.retrieval_cache = RetrievalCache(self.config)
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.retrieval_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        embedding_stats = self.embedding_cache.get_stats()
        response_stats = self.response_cache.get_stats()
        retrieval_stats = self.retrieval_cache.get_stats()
        
        total_size = embedding_stats["total_size_bytes"] + response_stats["total_size_bytes"] + retrieval_stats["total_size_bytes"]
        total_entries = embedding_stats["entries"] + response_stats["entries"] + retrieval_stats["entries"]
        total_hits = embedding_stats["hits"] + response_stats["hits"] + retrieval_stats["hits"]
        total_misses = embedding_stats["misses"] + response_stats["misses"] + retrieval_stats["misses"]
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_entries": total_entries,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_ratio": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
            "caches": {
                "embeddings": embedding_stats,
                "responses": response_stats,
                "retrieval": retrieval_stats
            }
        }
    
    def prune_old_entries(self) -> Dict[str, int]:
        """Prune old entries from all caches"""
        return {
            "embeddings": self.embedding_cache.prune_old_entries(),
            "responses": self.response_cache.prune_old_entries(),
            "retrieval": self.retrieval_cache.prune_old_entries()
        }
