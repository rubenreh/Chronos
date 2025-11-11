"""Caching utilities for API responses."""
import hashlib
import json
import time
from typing import Any, Optional, Dict
from functools import wraps
from datetime import datetime, timedelta


class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=self.ttl),
            'created_at': datetime.now()
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Remove expired entries
        now = datetime.now()
        expired = [k for k, v in self.cache.items() if now > v['expires_at']]
        for k in expired:
            del self.cache[k]
        
        return {
            'size': len(self.cache),
            'ttl_seconds': self.ttl
        }


# Global cache instance
_cache = SimpleCache(ttl_seconds=300)


def cached(ttl_seconds: int = 300):
    """Decorator for caching function results."""
    def decorator(func):
        cache = SimpleCache(ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._make_key(*args, **kwargs)
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


def get_cache() -> SimpleCache:
    """Get global cache instance."""
    return _cache

