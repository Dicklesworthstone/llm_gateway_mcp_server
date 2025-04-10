"""Cache service for LLM and RAG results."""
import asyncio
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

from llm_gateway.utils import get_logger

logger = get_logger(__name__)

# Singleton instance
_cache_service = None


def get_cache_service():
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


class CacheService:
    """Service for caching LLM and RAG results."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache service.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or os.environ.get(
            "CACHE_DIR", 
            str(Path.home() / ".llm_gateway" / "cache")
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # In-memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load cache from disk
        self._load_cache()
        
        # Schedule cache maintenance
        self._schedule_maintenance()
        
        logger.info(f"Cache service initialized with directory: {self.cache_dir}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            cache_file = Path(self.cache_dir) / "cache.pickle"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    loaded_cache = pickle.load(f)
                    
                    # Filter out expired items
                    current_time = time.time()
                    filtered_cache = {
                        key: value for key, value in loaded_cache.items()
                        if "expiry" not in value or value["expiry"] > current_time
                    }
                    
                    self.memory_cache = filtered_cache
                    logger.info(f"Loaded {len(self.memory_cache)} items from cache")
            else:
                logger.info("No cache file found, starting with empty cache")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            # Start with empty cache
            self.memory_cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_file = Path(self.cache_dir) / "cache.pickle"
            
            with open(cache_file, "wb") as f:
                pickle.dump(self.memory_cache, f)
                
            logger.info(f"Saved {len(self.memory_cache)} items to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _schedule_maintenance(self) -> None:
        """Schedule periodic cache maintenance."""
        asyncio.create_task(self._periodic_maintenance())
    
    async def _periodic_maintenance(self) -> None:
        """Perform periodic cache maintenance."""
        while True:
            try:
                # Clean expired items
                self._clean_expired()
                
                # Save cache to disk
                self._save_cache()
                
                # Wait for next maintenance cycle (every hour)
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in cache maintenance: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _clean_expired(self) -> None:
        """Clean expired items from cache."""
        current_time = time.time()
        initial_count = len(self.memory_cache)
        
        self.memory_cache = {
            key: value for key, value in self.memory_cache.items()
            if "expiry" not in value or value["expiry"] > current_time
        }
        
        removed = initial_count - len(self.memory_cache)
        if removed > 0:
            logger.info(f"Cleaned {removed} expired items from cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.memory_cache:
            return None
            
        cache_item = self.memory_cache[key]
        
        # Check expiry
        if "expiry" in cache_item and cache_item["expiry"] < time.time():
            # Remove expired item
            del self.memory_cache[key]
            return None
        
        # Update access time
        cache_item["last_access"] = time.time()
        cache_item["access_count"] = cache_item.get("access_count", 0) + 1
        
        return cache_item["value"]
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiry)
            
        Returns:
            True if successful
        """
        try:
            expiry = time.time() + ttl if ttl is not None else None
            
            self.memory_cache[key] = {
                "value": value,
                "created": time.time(),
                "last_access": time.time(),
                "access_count": 0,
                "expiry": expiry
            }
            
            # Schedule save if more than 10 items have been added since last save
            if len(self.memory_cache) % 10 == 0:
                asyncio.create_task(self._async_save_cache())
                
            return True
        except Exception as e:
            logger.error(f"Error setting cache item: {str(e)}")
            return False
    
    async def _async_save_cache(self) -> None:
        """Save cache asynchronously."""
        self._save_cache()
    
    async def delete(self, key: str) -> bool:
        """Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        if key in self.memory_cache:
            del self.memory_cache[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        self.memory_cache.clear()
        self._save_cache()
        logger.info("Cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total_items = len(self.memory_cache)
        
        # Count expired items
        current_time = time.time()
        expired_items = sum(
            1 for item in self.memory_cache.values()
            if "expiry" in item and item["expiry"] < current_time
        )
        
        # Calculate average access count
        access_counts = [
            item.get("access_count", 0) 
            for item in self.memory_cache.values()
        ]
        avg_access = sum(access_counts) / max(1, len(access_counts))
        
        return {
            "total_items": total_items,
            "expired_items": expired_items,
            "active_items": total_items - expired_items,
            "avg_access_count": avg_access,
            "cache_dir": self.cache_dir
        } 