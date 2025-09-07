"""
Redis cache utilities for AI backend services.

Provides caching functionality for ML models, notification scheduling,
and other performance-critical operations.
"""

import redis
import json
import pickle
import logging
from typing import Any, Optional, Union, List
from datetime import timedelta, datetime
import os

logger = logging.getLogger(__name__)

class RedisCache:
    """
    Redis cache client with JSON and pickle serialization support.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self._client = None
    
    @property
    def redis_client(self):
        """Get the Redis client instance"""
        return self.client
    
    @property
    def client(self):
        """
        Lazy initialization of Redis client.
        """
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                # Fallback to in-memory cache for development
                self._client = InMemoryCache()
        
        return self._client
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.
        """
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: str, ex: int = None) -> bool:
        """
        Set value in cache with optional expiration.
        """
        try:
            return self.client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    async def setex(self, key: str, time: int, value: str) -> bool:
        """
        Set value with expiration time.
        """
        try:
            return self.client.setex(key, time, value)
        except Exception as e:
            logger.error(f"Error setting cache key {key} with expiration: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        """
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        """
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {str(e)}")
            return False
    
    async def incr(self, key: str) -> int:
        """
        Increment counter.
        """
        try:
            return self.client.incr(key)
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {str(e)}")
            return 0
    
    async def expire(self, key: str, time: int) -> bool:
        """
        Set expiration time for key.
        """
        try:
            return self.client.expire(key, time)
        except Exception as e:
            logger.error(f"Error setting expiration for cache key {key}: {str(e)}")
            return False
    
    async def keys(self, pattern: str) -> list:
        """
        Get keys matching pattern.
        """
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {str(e)}")
            return []
    
    async def get_json(self, key: str) -> Optional[dict]:
        """
        Get JSON value from cache.
        """
        try:
            value = await self.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting JSON from cache key {key}: {str(e)}")
            return None
    
    async def set_json(self, key: str, value: dict, ex: int = None) -> bool:
        """
        Set JSON value in cache.
        """
        try:
            json_value = json.dumps(value)
            return await self.set(key, json_value, ex=ex)
        except Exception as e:
            logger.error(f"Error setting JSON to cache key {key}: {str(e)}")
            return False
    
    async def get_pickle(self, key: str) -> Optional[Any]:
        """
        Get pickled object from cache.
        """
        try:
            # Use binary Redis client for pickle operations
            binary_client = redis.from_url(self.redis_url, decode_responses=False)
            value = binary_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting pickle from cache key {key}: {str(e)}")
            return None
    
    async def set_pickle(self, key: str, value: Any, ex: int = None) -> bool:
        """
        Set pickled object in cache.
        """
        try:
            # Use binary Redis client for pickle operations
            binary_client = redis.from_url(self.redis_url, decode_responses=False)
            pickled_value = pickle.dumps(value)
            return binary_client.set(key, pickled_value, ex=ex)
        except Exception as e:
            logger.error(f"Error setting pickle to cache key {key}: {str(e)}")
            return False

class InMemoryCache:
    """
    Fallback in-memory cache for development when Redis is not available.
    """
    
    def __init__(self):
        self._cache = {}
        self._expiry = {}
        logger.warning("Using in-memory cache fallback - not suitable for production")
    
    def get(self, key: str) -> Optional[str]:
        """
        Get value from in-memory cache.
        """
        import time
        
        # Check if key has expired
        if key in self._expiry and time.time() > self._expiry[key]:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            return None
        
        return self._cache.get(key)
    
    def set(self, key: str, value: str, ex: int = None) -> bool:
        """
        Set value in in-memory cache.
        """
        import time
        
        self._cache[key] = value
        
        if ex:
            self._expiry[key] = time.time() + ex
        
        return True
    
    def setex(self, key: str, time: int, value: str) -> bool:
        """
        Set value with expiration time.
        """
        return self.set(key, value, ex=time)
    
    def delete(self, key: str) -> int:
        """
        Delete key from in-memory cache.
        """
        deleted = 0
        if key in self._cache:
            self._cache.pop(key)
            deleted += 1
        if key in self._expiry:
            self._expiry.pop(key)
        return deleted
    
    def exists(self, key: str) -> int:
        """
        Check if key exists in in-memory cache.
        """
        return 1 if key in self._cache else 0
    
    def incr(self, key: str) -> int:
        """
        Increment counter in in-memory cache.
        """
        current = int(self._cache.get(key, 0))
        current += 1
        self._cache[key] = str(current)
        return current
    
    def expire(self, key: str, time: int) -> bool:
        """
        Set expiration time for key.
        """
        import time as time_module
        
        if key in self._cache:
            self._expiry[key] = time_module.time() + time
            return True
        return False
    
    def keys(self, pattern: str) -> list:
        """
        Get keys matching pattern (simple implementation).
        """
        import fnmatch
        return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def ping(self) -> bool:
        """
        Ping method for compatibility.
        """
        return True

# Global cache instance
_cache_instance = None

def get_redis_client() -> RedisCache:
    """
    Get global Redis cache instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

# Cache decorators for common use cases

def cache_result(key_prefix: str, expiration: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        key_prefix: Prefix for cache key
        expiration: Cache expiration time in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            import hashlib
            key_data = f"{key_prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cache = get_redis_client()
            cached_result = await cache.get_json(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set_json(cache_key, result, ex=expiration)
            logger.debug(f"Cache miss, stored result for key: {cache_key}")
            
            return result
        
        return wrapper
    return decorator

def cache_ml_model(model_name: str, version: str = "latest"):
    """
    Decorator to cache ML model loading.
    
    Args:
        model_name: Name of the ML model
        version: Model version
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"ml_model:{model_name}:{version}"
            
            # Try to get from cache
            cache = get_redis_client()
            cached_model = await cache.get_pickle(cache_key)
            
            if cached_model is not None:
                logger.debug(f"ML model cache hit: {model_name}")
                return cached_model
            
            # Load model and cache
            model = await func(*args, **kwargs)
            await cache.set_pickle(cache_key, model, ex=86400)  # Cache for 24 hours
            logger.debug(f"ML model cached: {model_name}")
            
            return model
        
        return wrapper
    return decorator

# ML Model Caching Utilities

class MLModelCache:
    """
    Specialized cache for ML models and predictions with resource management.
    """
    
    def __init__(self, redis_client: RedisCache = None):
        self.cache = redis_client or get_redis_client()
        self.model_memory_limit = 500 * 1024 * 1024  # 500MB limit for cached models
        self.prediction_cache_ttl = 3600  # 1 hour for predictions
        self.model_cache_ttl = 86400  # 24 hours for models
        self.performance_cache_ttl = 1800  # 30 minutes for performance metrics
        
    async def cache_model(self, model_name: str, model_version: str, model_obj: Any, 
                         metadata: dict = None) -> bool:
        """
        Cache ML model with memory management.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_obj: The actual model object
            metadata: Optional metadata about the model
        """
        try:
            # Check memory usage before caching
            await self._cleanup_old_models_if_needed()
            
            cache_key = f"ml_model:{model_name}:{model_version}"
            
            # Cache model object
            model_cached = await self.cache.set_pickle(cache_key, model_obj, ex=self.model_cache_ttl)
            
            # Cache metadata separately for quick access
            if metadata:
                metadata_key = f"ml_model_meta:{model_name}:{model_version}"
                await self.cache.set_json(metadata_key, metadata, ex=self.model_cache_ttl)
            
            # Track model in registry
            await self._register_cached_model(model_name, model_version)
            
            logger.info(f"ML model cached: {model_name} v{model_version}")
            return model_cached
            
        except Exception as e:
            logger.error(f"Failed to cache ML model {model_name}: {str(e)}")
            return False
    
    async def get_cached_model(self, model_name: str, model_version: str = "latest") -> Optional[Any]:
        """
        Retrieve cached ML model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (default: latest)
        """
        try:
            if model_version == "latest":
                model_version = await self._get_latest_model_version(model_name)
                if not model_version:
                    return None
            
            cache_key = f"ml_model:{model_name}:{model_version}"
            model = await self.cache.get_pickle(cache_key)
            
            if model:
                # Update access time for LRU eviction
                await self._update_model_access_time(model_name, model_version)
                logger.debug(f"ML model cache hit: {model_name} v{model_version}")
            else:
                logger.debug(f"ML model cache miss: {model_name} v{model_version}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to get cached ML model {model_name}: {str(e)}")
            return None
    
    async def cache_prediction_result(self, model_name: str, input_hash: str, 
                                    prediction: dict, confidence: float = None) -> bool:
        """
        Cache prediction results for duplicate request optimization.
        
        Args:
            model_name: Name of the model that made the prediction
            input_hash: Hash of the input data
            prediction: The prediction result
            confidence: Optional confidence score
        """
        try:
            cache_key = f"prediction:{model_name}:{input_hash}"
            
            result_data = {
                "prediction": prediction,
                "confidence": confidence,
                "cached_at": datetime.utcnow().isoformat(),
                "model_name": model_name
            }
            
            return await self.cache.set_json(cache_key, result_data, ex=self.prediction_cache_ttl)
            
        except Exception as e:
            logger.error(f"Failed to cache prediction result: {str(e)}")
            return False
    
    async def get_cached_prediction(self, model_name: str, input_hash: str) -> Optional[dict]:
        """
        Retrieve cached prediction result.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of the input data
        """
        try:
            cache_key = f"prediction:{model_name}:{input_hash}"
            return await self.cache.get_json(cache_key)
            
        except Exception as e:
            logger.error(f"Failed to get cached prediction: {str(e)}")
            return None
    
    async def cache_performance_metrics(self, model_name: str, metrics: dict) -> bool:
        """
        Cache model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
        """
        try:
            cache_key = f"performance:{model_name}"
            
            metrics_data = {
                **metrics,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            return await self.cache.set_json(cache_key, metrics_data, ex=self.performance_cache_ttl)
            
        except Exception as e:
            logger.error(f"Failed to cache performance metrics: {str(e)}")
            return False
    
    async def get_cached_performance_metrics(self, model_name: str) -> Optional[dict]:
        """
        Retrieve cached performance metrics.
        
        Args:
            model_name: Name of the model
        """
        try:
            cache_key = f"performance:{model_name}"
            return await self.cache.get_json(cache_key)
            
        except Exception as e:
            logger.error(f"Failed to get cached performance metrics: {str(e)}")
            return None
    
    async def invalidate_model_cache(self, model_name: str, model_version: str = None) -> bool:
        """
        Invalidate cached model and related data.
        
        Args:
            model_name: Name of the model
            model_version: Specific version to invalidate (None for all versions)
        """
        try:
            if model_version:
                # Invalidate specific version
                keys_to_delete = [
                    f"ml_model:{model_name}:{model_version}",
                    f"ml_model_meta:{model_name}:{model_version}"
                ]
            else:
                # Invalidate all versions
                patterns = [
                    f"ml_model:{model_name}:*",
                    f"ml_model_meta:{model_name}:*",
                    f"prediction:{model_name}:*",
                    f"performance:{model_name}"
                ]
                
                keys_to_delete = []
                for pattern in patterns:
                    keys = await self.cache.keys(pattern)
                    keys_to_delete.extend(keys)
            
            # Delete all keys
            for key in keys_to_delete:
                await self.cache.delete(key)
            
            # Update registry
            await self._unregister_cached_model(model_name, model_version)
            
            logger.info(f"Invalidated cache for model: {model_name}" + 
                       (f" v{model_version}" if model_version else " (all versions)"))
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate model cache: {str(e)}")
            return False
    
    async def get_cache_statistics(self) -> dict:
        """
        Get cache usage statistics.
        """
        try:
            stats = {
                "cached_models": 0,
                "cached_predictions": 0,
                "cached_performance_metrics": 0,
                "memory_usage_mb": 0,
                "hit_rate": 0.0
            }
            
            # Count cached items
            model_keys = await self.cache.keys("ml_model:*")
            prediction_keys = await self.cache.keys("prediction:*")
            performance_keys = await self.cache.keys("performance:*")
            
            stats["cached_models"] = len(model_keys)
            stats["cached_predictions"] = len(prediction_keys)
            stats["cached_performance_metrics"] = len(performance_keys)
            
            # Get memory usage (approximate)
            try:
                if hasattr(self.cache.client, 'info'):
                    redis_info = self.cache.client.info('memory')
                    stats["memory_usage_mb"] = redis_info.get('used_memory', 0) / (1024 * 1024)
            except:
                pass
            
            # Get hit rate from counters
            hit_rate_data = await self.cache.get_json("cache_stats:hit_rate")
            if hit_rate_data:
                hits = hit_rate_data.get("hits", 0)
                misses = hit_rate_data.get("misses", 0)
                total = hits + misses
                stats["hit_rate"] = hits / total if total > 0 else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {str(e)}")
            return {}
    
    # Private helper methods
    
    async def _cleanup_old_models_if_needed(self):
        """
        Clean up old cached models if memory limit is exceeded.
        """
        try:
            # Get current memory usage
            stats = await self.get_cache_statistics()
            current_memory = stats.get("memory_usage_mb", 0) * 1024 * 1024
            
            if current_memory > self.model_memory_limit:
                logger.info("Memory limit exceeded, cleaning up old models")
                
                # Get all cached models with access times
                model_registry = await self.cache.get_json("ml_model_registry") or {}
                
                # Sort by last access time (LRU)
                sorted_models = sorted(
                    model_registry.items(),
                    key=lambda x: x[1].get("last_access", 0)
                )
                
                # Remove oldest models until under limit
                for model_key, model_info in sorted_models:
                    await self.invalidate_model_cache(
                        model_info["model_name"], 
                        model_info["model_version"]
                    )
                    
                    # Check if we're under limit
                    stats = await self.get_cache_statistics()
                    current_memory = stats.get("memory_usage_mb", 0) * 1024 * 1024
                    if current_memory <= self.model_memory_limit * 0.8:  # 80% of limit
                        break
                        
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {str(e)}")
    
    async def _register_cached_model(self, model_name: str, model_version: str):
        """
        Register a cached model in the registry.
        """
        try:
            registry = await self.cache.get_json("ml_model_registry") or {}
            
            model_key = f"{model_name}:{model_version}"
            registry[model_key] = {
                "model_name": model_name,
                "model_version": model_version,
                "cached_at": datetime.utcnow().timestamp(),
                "last_access": datetime.utcnow().timestamp()
            }
            
            await self.cache.set_json("ml_model_registry", registry, ex=86400)
            
        except Exception as e:
            logger.error(f"Failed to register cached model: {str(e)}")
    
    async def _unregister_cached_model(self, model_name: str, model_version: str = None):
        """
        Unregister a cached model from the registry.
        """
        try:
            registry = await self.cache.get_json("ml_model_registry") or {}
            
            if model_version:
                model_key = f"{model_name}:{model_version}"
                registry.pop(model_key, None)
            else:
                # Remove all versions
                keys_to_remove = [k for k in registry.keys() if k.startswith(f"{model_name}:")]
                for key in keys_to_remove:
                    registry.pop(key, None)
            
            await self.cache.set_json("ml_model_registry", registry, ex=86400)
            
        except Exception as e:
            logger.error(f"Failed to unregister cached model: {str(e)}")
    
    async def _update_model_access_time(self, model_name: str, model_version: str):
        """
        Update the last access time for a cached model.
        """
        try:
            registry = await self.cache.get_json("ml_model_registry") or {}
            
            model_key = f"{model_name}:{model_version}"
            if model_key in registry:
                registry[model_key]["last_access"] = datetime.utcnow().timestamp()
                await self.cache.set_json("ml_model_registry", registry, ex=86400)
                
        except Exception as e:
            logger.error(f"Failed to update model access time: {str(e)}")
    
    async def _get_latest_model_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a cached model.
        """
        try:
            registry = await self.cache.get_json("ml_model_registry") or {}
            
            # Find all versions of the model
            model_versions = []
            for key, info in registry.items():
                if info["model_name"] == model_name:
                    model_versions.append((info["model_version"], info["cached_at"]))
            
            if model_versions:
                # Return the most recently cached version
                latest = max(model_versions, key=lambda x: x[1])
                return latest[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest model version: {str(e)}")
            return None

# Performance Monitoring and Cache Invalidation

class CacheInvalidationManager:
    """
    Manages cache invalidation strategies based on data changes and time-based rules.
    """
    
    def __init__(self, redis_client: RedisCache = None):
        self.cache = redis_client or get_redis_client()
        self.invalidation_rules = {}
        
    async def register_invalidation_rule(self, rule_name: str, pattern: str, 
                                       trigger_events: List[str], ttl: int = None):
        """
        Register a cache invalidation rule.
        
        Args:
            rule_name: Name of the rule
            pattern: Cache key pattern to invalidate
            trigger_events: List of events that trigger invalidation
            ttl: Optional TTL for the rule
        """
        rule = {
            "pattern": pattern,
            "trigger_events": trigger_events,
            "created_at": datetime.utcnow().isoformat(),
            "ttl": ttl
        }
        
        self.invalidation_rules[rule_name] = rule
        
        # Store in cache for persistence
        await self.cache.set_json("invalidation_rules", self.invalidation_rules, ex=86400)
        
        logger.info(f"Registered invalidation rule: {rule_name}")
    
    async def trigger_invalidation(self, event_name: str, context: dict = None):
        """
        Trigger cache invalidation based on an event.
        
        Args:
            event_name: Name of the event
            context: Optional context data for pattern substitution
        """
        try:
            # Load rules from cache
            cached_rules = await self.cache.get_json("invalidation_rules")
            if cached_rules:
                self.invalidation_rules.update(cached_rules)
            
            invalidated_patterns = []
            
            for rule_name, rule in self.invalidation_rules.items():
                if event_name in rule["trigger_events"]:
                    pattern = rule["pattern"]
                    
                    # Substitute context variables in pattern
                    if context:
                        for key, value in context.items():
                            pattern = pattern.replace(f"{{{key}}}", str(value))
                    
                    # Find and delete matching keys
                    keys_to_delete = await self.cache.keys(pattern)
                    for key in keys_to_delete:
                        await self.cache.delete(key)
                    
                    invalidated_patterns.append(pattern)
                    logger.info(f"Invalidated {len(keys_to_delete)} keys matching pattern: {pattern}")
            
            return invalidated_patterns
            
        except Exception as e:
            logger.error(f"Failed to trigger invalidation for event {event_name}: {str(e)}")
            return []
    
    async def setup_default_rules(self):
        """
        Set up default invalidation rules for the AI backend.
        """
        # Business data invalidation when transactions change
        await self.register_invalidation_rule(
            "business_data_on_transaction",
            "business_data:{business_id}:*",
            ["transaction_created", "transaction_updated", "transaction_deleted"]
        )
        
        # Model predictions invalidation when model is retrained
        await self.register_invalidation_rule(
            "predictions_on_model_update",
            "prediction:{model_name}:*",
            ["model_trained", "model_deployed"]
        )
        
        # Performance metrics invalidation when feedback is received
        await self.register_invalidation_rule(
            "performance_on_feedback",
            "performance:{model_name}",
            ["feedback_received", "model_performance_updated"]
        )
        
        # User preferences invalidation
        await self.register_invalidation_rule(
            "user_preferences_on_update",
            "user_preferences:{user_id}",
            ["user_preferences_updated"]
        )

class CachePerformanceMonitor:
    """
    Monitors cache performance and provides optimization recommendations.
    """
    
    def __init__(self, redis_client: RedisCache = None):
        self.cache = redis_client or get_redis_client()
        
    async def record_cache_hit(self, cache_type: str):
        """Record a cache hit for performance tracking."""
        await self._update_hit_miss_counter(cache_type, "hits")
    
    async def record_cache_miss(self, cache_type: str):
        """Record a cache miss for performance tracking."""
        await self._update_hit_miss_counter(cache_type, "misses")
    
    async def _update_hit_miss_counter(self, cache_type: str, counter_type: str):
        """Update hit/miss counters."""
        try:
            counter_key = f"cache_stats:{cache_type}:{counter_type}"
            await self.cache.incr(counter_key)
            await self.cache.expire(counter_key, 86400)  # Expire after 24 hours
            
            # Also update global counter
            global_key = f"cache_stats:global:{counter_type}"
            await self.cache.incr(global_key)
            await self.cache.expire(global_key, 86400)
            
        except Exception as e:
            logger.error(f"Failed to update cache counter: {str(e)}")
    
    async def get_performance_report(self) -> dict:
        """
        Generate a comprehensive cache performance report.
        """
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "global_stats": {},
                "cache_type_stats": {},
                "recommendations": []
            }
            
            # Global hit/miss rates
            global_hits = int(await self.cache.get("cache_stats:global:hits") or 0)
            global_misses = int(await self.cache.get("cache_stats:global:misses") or 0)
            total_requests = global_hits + global_misses
            
            report["global_stats"] = {
                "total_requests": total_requests,
                "hits": global_hits,
                "misses": global_misses,
                "hit_rate": global_hits / total_requests if total_requests > 0 else 0.0
            }
            
            # Per-cache-type statistics
            cache_types = ["ml_model", "prediction", "performance", "business_data"]
            for cache_type in cache_types:
                hits = int(await self.cache.get(f"cache_stats:{cache_type}:hits") or 0)
                misses = int(await self.cache.get(f"cache_stats:{cache_type}:misses") or 0)
                total = hits + misses
                
                report["cache_type_stats"][cache_type] = {
                    "hits": hits,
                    "misses": misses,
                    "total_requests": total,
                    "hit_rate": hits / total if total > 0 else 0.0
                }
            
            # Generate recommendations
            report["recommendations"] = await self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            return {}
    
    async def _generate_recommendations(self, report: dict) -> List[str]:
        """
        Generate optimization recommendations based on performance data.
        """
        recommendations = []
        
        try:
            global_hit_rate = report["global_stats"].get("hit_rate", 0.0)
            
            # Low hit rate recommendations
            if global_hit_rate < 0.5:
                recommendations.append(
                    "Global cache hit rate is low (<50%). Consider increasing cache TTL values."
                )
            
            # Per-cache-type recommendations
            for cache_type, stats in report["cache_type_stats"].items():
                hit_rate = stats.get("hit_rate", 0.0)
                total_requests = stats.get("total_requests", 0)
                
                if total_requests > 100:  # Only recommend for active cache types
                    if hit_rate < 0.3:
                        recommendations.append(
                            f"{cache_type} cache has low hit rate ({hit_rate:.1%}). "
                            f"Consider optimizing cache keys or increasing TTL."
                        )
                    elif hit_rate > 0.9:
                        recommendations.append(
                            f"{cache_type} cache has excellent hit rate ({hit_rate:.1%}). "
                            f"Current configuration is optimal."
                        )
            
            # Memory usage recommendations
            ml_cache = MLModelCache(self.cache)
            cache_stats = await ml_cache.get_cache_statistics()
            memory_usage = cache_stats.get("memory_usage_mb", 0)
            
            if memory_usage > 400:  # 400MB threshold
                recommendations.append(
                    f"High memory usage ({memory_usage:.1f}MB). "
                    f"Consider reducing model cache TTL or implementing more aggressive cleanup."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["Unable to generate recommendations due to error."]

# Global instances
ml_model_cache = MLModelCache()
cache_invalidation_manager = CacheInvalidationManager()
cache_performance_monitor = CachePerformanceMonitor()

# Utility functions for common caching patterns

async def cache_business_data(business_id: str, data_type: str, data: dict, expiration: int = 1800):
    """
    Cache business-specific data.
    
    Args:
        business_id: Business identifier
        data_type: Type of data being cached
        data: Data to cache
        expiration: Cache expiration in seconds (default 30 minutes)
    """
    cache_key = f"business_data:{business_id}:{data_type}"
    cache = get_redis_client()
    result = await cache.set_json(cache_key, data, ex=expiration)
    
    # Record cache operation
    if result:
        await cache_performance_monitor.record_cache_hit("business_data")
    else:
        await cache_performance_monitor.record_cache_miss("business_data")
    
    return result

async def get_cached_business_data(business_id: str, data_type: str) -> Optional[dict]:
    """
    Get cached business-specific data.
    
    Args:
        business_id: Business identifier
        data_type: Type of data to retrieve
    
    Returns:
        Cached data or None if not found
    """
    cache_key = f"business_data:{business_id}:{data_type}"
    cache = get_redis_client()
    return await cache.get_json(cache_key)

async def invalidate_business_cache(business_id: str, data_type: str = None):
    """
    Invalidate cached business data.
    
    Args:
        business_id: Business identifier
        data_type: Specific data type to invalidate, or None for all
    """
    cache = get_redis_client()
    
    if data_type:
        cache_key = f"business_data:{business_id}:{data_type}"
        await cache.delete(cache_key)
    else:
        # Invalidate all business data
        pattern = f"business_data:{business_id}:*"
        keys = await cache.keys(pattern)
        for key in keys:
            await cache.delete(key)

async def cache_user_preferences(user_id: str, preferences: dict, expiration: int = 86400):
    """
    Cache user preferences.
    
    Args:
        user_id: User identifier
        preferences: User preferences data
        expiration: Cache expiration in seconds (default 24 hours)
    """
    cache_key = f"user_preferences:{user_id}"
    cache = get_redis_client()
    return await cache.set_json(cache_key, preferences, ex=expiration)

async def get_cached_user_preferences(user_id: str) -> Optional[dict]:
    """
    Get cached user preferences.
    
    Args:
        user_id: User identifier
    
    Returns:
        Cached preferences or None if not found
    """
    cache_key = f"user_preferences:{user_id}"
    cache = get_redis_client()
    return await cache.get_json(cache_key)
# Global cache instance for easy import
cache = get_redis_client()
ml_cache = MLModelCache()
invalidation_manager = CacheInvalidationManager()
performance_monitor = CachePerformanceMonitor()

# Global instances
cache = get_redis_client()
ml_model_cache = MLModelCache(cache)
cache_invalidation_manager = CacheInvalidationManager(cache)
cache_performance_monitor = CachePerformanceMonitor(cache)

# Export commonly used instances
__all__ = [
    'cache',
    'ml_model_cache', 
    'cache_invalidation_manager',
    'cache_performance_monitor',
    'get_redis_client',
    'cache_result',
    'cache_ml_model'
]