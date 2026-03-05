"""
Caching utilities for API responses.

This module provides a lightweight, in-memory caching layer used by every API
endpoint to avoid redundant model inference and expensive computations. When
the same request (identified by a composite string key) arrives within the TTL
window, the cached result is returned instantly instead of re-running the model.

Design choices:
    - In-memory dict (no external dependency like Redis) — keeps deployment simple
    - TTL-based expiration (default 300 s / 5 min) — stale predictions are purged
    - MD5-hashed keys — deterministic, fixed-length keys regardless of input size
    - Thread-safe enough for single-worker Uvicorn; for multi-worker deploys an
      external cache (Redis, Memcached) would be substituted

Exports:
    - SimpleCache     : the cache class itself
    - cached()        : decorator that transparently caches any function's return value
    - get_cache()     : accessor for the module-level singleton instance
"""
import hashlib                                  # MD5 hashing for deterministic cache key generation
import json                                     # Serialize args/kwargs into a canonical string for hashing
import time                                     # (Available for future timing; not directly used currently)
from typing import Any, Optional, Dict          # Type annotations for cache entries
from functools import wraps                     # Preserves the wrapped function's name/docstring in the decorator
from datetime import datetime, timedelta        # Used for TTL expiration calculations


class SimpleCache:
    """Simple in-memory cache with TTL (Time-To-Live) expiration.

    Each entry stores a value alongside an ``expires_at`` timestamp. On every
    ``get``, the timestamp is checked and expired entries are lazily evicted.
    This avoids a background reaper thread while keeping memory bounded over time.
    """

    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
                         Defaults to 300 (5 minutes), balancing freshness
                         against avoiding redundant model inference.
        """
        # Internal storage: maps MD5 hex key → {"value": …, "expires_at": …, "created_at": …}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds  # Store the TTL so it can be applied to every new entry

    def _make_key(self, *args, **kwargs) -> str:
        """Generate a deterministic cache key from arbitrary arguments.

        Serializes all positional and keyword arguments into a sorted JSON
        string, then hashes it with MD5 to produce a fixed-length hex digest.
        Sorting ensures that {'a':1,'b':2} and {'b':2,'a':1} yield the same key.
        """
        key_data = json.dumps(                   # Convert args+kwargs to a canonical JSON string
            {'args': args, 'kwargs': kwargs},
            sort_keys=True                       # Deterministic ordering regardless of dict insertion order
        )
        return hashlib.md5(                      # MD5 is fast and collision-resistant enough for cache keys
            key_data.encode()                    # Hash operates on bytes, not str
        ).hexdigest()                            # Return the 32-char hex string

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache if it exists and has not expired.

        Returns None on cache miss or expiration (lazy eviction).
        """
        if key not in self.cache:                # Cache miss — key was never stored
            return None

        entry = self.cache[key]                  # Retrieve the stored entry dict
        if datetime.now() > entry['expires_at']: # Check if the entry's TTL has elapsed
            del self.cache[key]                  # Evict the stale entry to free memory
            return None                          # Treat as a miss

        return entry['value']                    # Cache hit — return the stored value

    def set(self, key: str, value: Any):
        """Store a value in the cache with a TTL-based expiration time.

        Overwrites any existing entry for the same key.
        """
        self.cache[key] = {
            'value': value,                                              # The actual cached payload
            'expires_at': datetime.now() + timedelta(seconds=self.ttl),  # Absolute expiration timestamp
            'created_at': datetime.now()                                 # Diagnostic: when the entry was created
        }

    def clear(self):
        """Remove all entries from the cache (e.g. after a model retrain)."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics after purging expired entries.

        Used by the /health endpoint to surface cache size in monitoring.
        """
        now = datetime.now()
        # Collect keys whose expiration time has passed
        expired = [k for k, v in self.cache.items() if now > v['expires_at']]
        for k in expired:                        # Eagerly evict all expired entries
            del self.cache[k]

        return {
            'size': len(self.cache),             # Number of live (non-expired) entries
            'ttl_seconds': self.ttl              # Configured TTL for reference
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single cache instance shared across all route modules. Using a singleton
# ensures that a forecast cached by one request can be reused by the next.
_cache = SimpleCache(ttl_seconds=300)


def cached(ttl_seconds: int = 300):
    """Decorator factory that caches the return value of the wrapped function.

    Usage::

        @cached(ttl_seconds=120)
        def expensive_computation(x, y):
            ...

    Each unique combination of arguments produces a distinct cache key.
    Subsequent calls with the same arguments return the cached result
    until the TTL expires.
    """
    def decorator(func):
        cache = SimpleCache(ttl_seconds=ttl_seconds)  # Each decorated function gets its own cache namespace

        @wraps(func)                                   # Preserve original function metadata (__name__, __doc__)
        def wrapper(*args, **kwargs):
            key = cache._make_key(*args, **kwargs)     # Hash the call arguments into a cache key
            cached_value = cache.get(key)              # Attempt a cache lookup
            if cached_value is not None:               # Cache hit — skip recomputation
                return cached_value

            result = func(*args, **kwargs)             # Cache miss — execute the original function
            cache.set(key, result)                     # Store the result for future calls
            return result

        return wrapper
    return decorator


def get_cache() -> SimpleCache:
    """Return the global SimpleCache singleton.

    All route modules call this to share a single cache, so a prediction
    cached by /forecast can also be detected as cached by /health stats.
    """
    return _cache
