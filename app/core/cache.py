# app/core/cache.py
"""Caching utilities for the business finance application."""

import streamlit as st
import hashlib
import json
import pandas as pd
from typing import Any, Callable, Optional
from functools import wraps
from .logger import get_logger
from .settings import settings

logger = get_logger("cache")

class CacheManager:
    """Enhanced caching manager with multiple strategies."""
    
    @staticmethod
    def generate_cache_key(*args, **kwargs) -> str:
        """Generate a unique cache key from function arguments."""
        # Create a consistent string representation of arguments
        key_data = {
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        
        # Create hash from the key data
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def cache_data(ttl: Optional[int] = None):
        """Decorator for caching function results with Streamlit cache."""
        if ttl is None:
            ttl = settings.CACHE_TTL
        
        def decorator(func: Callable) -> Callable:
            @st.cache_data(ttl=ttl, show_spinner=False)
            @wraps(func)
            def wrapper(*args, **kwargs):
                logger.debug(f"Cache miss for {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"Cached result for {func.__name__}")
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def cache_resource():
        """Decorator for caching expensive resources (models, connections)."""
        def decorator(func: Callable) -> Callable:
            @st.cache_resource(show_spinner=False)
            @wraps(func)
            def wrapper(*args, **kwargs):
                logger.debug(f"Loading resource: {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"Resource loaded: {func.__name__}")
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def clear_cache():
        """Clear all Streamlit caches."""
        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("All caches cleared")

# Session state utilities
class SessionStateManager:
    """Manage Streamlit session state efficiently."""
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state with default."""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state."""
        st.session_state[key] = value
    
    @staticmethod
    def delete(key: str) -> None:
        """Delete key from session state."""
        if key in st.session_state:
            del st.session_state[key]
    
    @staticmethod
    def has(key: str) -> bool:
        """Check if key exists in session state."""
        return key in st.session_state
    
    @staticmethod
    def clear_session() -> None:
        """Clear all session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        logger.info("Session state cleared")
    
    @staticmethod
    def get_or_create(key: str, factory: Callable) -> Any:
        """Get value from session state or create it using factory function."""
        if not SessionStateManager.has(key):
            SessionStateManager.set(key, factory())
        return SessionStateManager.get(key)

# Memory usage monitoring
class MemoryMonitor:
    """Monitor memory usage for caching decisions."""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def should_cache(data_size_mb: float) -> bool:
        """Determine if data should be cached based on size and available memory."""
        memory_stats = MemoryMonitor.get_memory_usage()
        
        # Don't cache if data is too large (>100MB) or memory usage is high
        if data_size_mb > 100:
            return False
        
        if memory_stats['percent'] > 80:  # Over 80% memory usage
            return False
        
        return True

# Data-specific caching utilities
class DataCache:
    """Specialized caching for different data types."""
    
    @staticmethod
    @CacheManager.cache_data(ttl=1800)  # 30 minutes
    def cache_transaction_data(data_hash: str, data: pd.DataFrame) -> pd.DataFrame:
        """Cache processed transaction data."""
        logger.debug(f"Caching transaction data with hash: {data_hash}")
        return data.copy()
    
    @staticmethod
    @CacheManager.cache_data(ttl=3600)  # 1 hour
    def cache_financial_metrics(company_id: str, metrics: dict) -> dict:
        """Cache calculated financial metrics."""
        logger.debug(f"Caching financial metrics for company: {company_id}")
        return metrics.copy()
    
    @staticmethod
    @CacheManager.cache_data(ttl=7200)  # 2 hours
    def cache_industry_data(industry: str, data: dict) -> dict:
        """Cache industry-specific data."""
        logger.debug(f"Caching industry data for: {industry}")
        return data.copy()
    
    @staticmethod
    def get_data_hash(data: pd.DataFrame) -> str:
        """Generate hash for DataFrame to use as cache key."""
        # Use a subset of columns and row count for hash
        hash_data = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
        }
        
        # Add sample of data if not too large
        if len(data) <= 1000:
            hash_data['sample'] = data.head(10).to_dict()
        
        return hashlib.md5(
            json.dumps(hash_data, sort_keys=True, default=str).encode()
        ).hexdigest()

# Global cache manager instance
cache_manager = CacheManager()
session_manager = SessionStateManager()
data_cache = DataCache()