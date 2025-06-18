import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import hashlib
import json
import pickle
import os
import functools
import logging

def timing_decorator(func):
    """Decorator to measure execution time of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

class PerformanceOptimizer:
    """
    Drop-in performance optimizer for existing RAG pipeline.
    Adds caching, parallel processing, and response optimization.
    """
    
    def __init__(self, cache_dir: str = "cache", max_workers: int = 4, max_cache_size: int = 100):
        """
        Initialize performance optimizer.
        
        Args:
            cache_dir: Directory to store cache files
            max_workers: Maximum number of concurrent workers for parallel processing
            max_cache_size: Maximum number of queries to cache
        """
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.max_cache_size = max_cache_size
        self.logger = logging.getLogger("optimizer")
        self.response_cache = {}
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load persistent caches
        self._load_caches()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _load_caches(self):
        """Load cached data from disk"""
        try:
            response_cache_path = os.path.join(self.cache_dir, "response_cache.pkl")
            if os.path.exists(response_cache_path):
                with open(response_cache_path, 'rb') as f:
                    self.response_cache = pickle.load(f)
            
            embedding_cache_path = os.path.join(self.cache_dir, "embedding_cache.pkl")
            if os.path.exists(embedding_cache_path):
                with open(embedding_cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
        except Exception as e:
            print(f"Cache loading failed, starting fresh: {e}")
            self.response_cache = {}
            self.embedding_cache = {}
    
    def _save_caches(self):
        """Save caches to disk"""
        try:
            with open(os.path.join(self.cache_dir, "response_cache.pkl"), 'wb') as f:
                pickle.dump(self.response_cache, f)
            
            with open(os.path.join(self.cache_dir, "embedding_cache.pkl"), 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Cache saving failed: {e}")
    
    def _get_query_hash(self, query: str, context: str = "") -> str:
        """Generate hash for query caching"""
        combined = f"{query.lower().strip()}_{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def cached_response(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Check if response is cached"""
        query_hash = self._get_query_hash(query, context)
        cached_data = self.response_cache.get(query_hash)
        
        if cached_data:
            # Check if cache is still valid (1 hour by default)
            if time.time() - cached_data["timestamp"] < 3600:
                return cached_data["response"]
            else:
                # Remove expired cache
                del self.response_cache[query_hash]
        
        return None
    
    def cache_response(self, query: str, response: Dict[str, Any], context: str = ""):
        """Cache a response"""
        query_hash = self._get_query_hash(query, context)
        self.response_cache[query_hash] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Auto-save every 10 responses
        if len(self.response_cache) % 10 == 0:
            self._save_caches()
    
    def is_simple_query(self, query: str) -> bool:
        """Detect simple queries that can be answered quickly"""
        simple_patterns = [
            "qu'est-ce que", "définition", "c'est quoi", "qui est", 
            "combien", "quand", "où", "comment", "quelle est", "quel est"
        ]
        query_lower = query.lower()
        
        # Simple if short and contains common patterns
        return (len(query.split()) <= 8 and 
                any(pattern in query_lower for pattern in simple_patterns))
    
    def parallel_search(self, rag_pipeline, query_variations: List[str], top_k: int = 5) -> List[Any]:
        """Perform parallel searches for multiple query variations"""
        futures = []
        
        for variation in query_variations[:3]:  # Limit to 3 variations for speed
            future = self.executor.submit(rag_pipeline.search, variation, top_k)
            futures.append(future)
        
        all_results = []
        for future in as_completed(futures, timeout=5):  # 5 second timeout
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Parallel search failed: {e}")
                continue
        
        return all_results
    
    def optimize_chunk_selection(self, chunks: List[Any], max_chunks: int = 3) -> List[Any]:
        """Select best chunks quickly for faster processing"""
        if not chunks:
            return []
        
        # Sort by score and take top chunks
        sorted_chunks = sorted(chunks, key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # For speed, limit to max_chunks
        selected = sorted_chunks[:max_chunks]
        
        # Quick deduplication
        unique_chunks = []
        seen_content = set()
        
        for chunk in selected:
            content_hash = hashlib.md5(chunk.chunk.content[:200].encode()).hexdigest()
            if content_hash not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content_hash)
        
        return unique_chunks
    
    def create_fast_prompt(self, query: str, chunks: List[Any], max_context_length: int = 1500) -> str:
        """Create optimized prompt for faster processing"""
        if not chunks:
            return f"Répondez brièvement à: {query}"
        
        # Build context efficiently
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks for speed
            chunk_text = chunk.chunk.content
            if current_length + len(chunk_text) > max_context_length:
                # Truncate to fit
                remaining = max_context_length - current_length
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(f"[{i+1}] {chunk_text}")
                break
            else:
                context_parts.append(f"[{i+1}] {chunk_text}")
                current_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        # Simplified prompt for speed
        prompt = f"""Contexte:
{context}

Question: {query}
Réponse (soyez concis et précis):"""
        
        return prompt
    
    def detect_intent_quickly(self, query: str) -> str:
        """Quick intent detection for routing"""
        query_lower = query.lower()
        
        # Fast pattern matching
        if any(word in query_lower for word in ["image", "génère", "créer", "dessiner", "generate", "draw"]):
            return "image_generation"
        elif any(word in query_lower for word in ["recherche", "web", "internet", "actualité", "search"]):
            return "web_search"
        elif any(word in query_lower for word in ["système", "ordinateur", "mémoire", "processeur", "system"]):
            return "system_info"
        else:
            return "rag_query"
    
    def cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if a similar query is in the cache and return the cached response"""
        self.total_queries += 1
        
        query_hash = self._get_query_hash(query)
        normalized_query = query.lower().strip()
        
        # Check exact match
        if query_hash in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[query_hash]["response"]
        
        # Check for similar queries (simple similarity check)
        for cached_hash, entry in self.response_cache.items():
            cached_query = entry["query"].lower().strip()
            
            # If queries are similar enough, use the cached response
            if self._are_queries_similar(normalized_query, cached_query):
                self.cache_hits += 1
                return entry["response"]
        
        self.cache_misses += 1
        return None
    
    def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """
        Check if two queries are similar enough to use the same cached response.
        Uses a simple word overlap ratio approach.
        """
        # Split into words
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        # Calculate overlap
        overlap = words1.intersection(words2)
        
        # Calculate Jaccard similarity
        union = words1.union(words2)
        if not union:
            return False
        
        similarity = len(overlap) / len(union)
        
        # Queries are similar if they share at least 80% of words
        return similarity >= 0.8
    
    def cache_response(self, query: str, response: Dict[str, Any]) -> None:
        """Cache a response for future use"""
        query_hash = self._get_query_hash(query)
        
        # Add to cache
        self.response_cache[query_hash] = {
            "query": query,
            "response": response,
            "timestamp": time.time()
        }
        
        # Limit cache size
        if len(self.response_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.response_cache.items(), key=lambda x: x[1]["timestamp"])
            self.response_cache = dict(sorted_items[-(self.max_cache_size):])
        
        # Save cache to disk periodically (every 10 new entries)
        if self.total_queries % 10 == 0:
            self._save_caches()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": self.total_queries,
            "hit_rate": hit_rate,
            "response_cache_size": len(self.response_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "cache_directory": self.cache_dir
        }
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self.response_cache.clear()
        self.embedding_cache.clear()
        self._save_caches()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self._save_caches()
        self.executor.shutdown(wait=True)