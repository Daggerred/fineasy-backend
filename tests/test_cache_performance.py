"""
Tests for Redis caching and performance optimization functionality.

Tests ML model caching, resource management, cache invalidation,
and performance monitoring features.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.utils.cache import (
    MLModelCache, 
    CacheInvalidationManager, 
    CachePerformanceMonitor,
    get_redis_client
)
from app.utils.resource_manager import (
    ResourceManager, 
    ResourceType, 
    ResourceStatus,
    ModelResourceUsage
)


class TestMLModelCache:
    """Test ML model caching functionality"""
    
    @pytest.fixture
    async def ml_cache(self):
        """Create ML model cache instance for testing"""
        cache = MLModelCache()
        yield cache
        # Cleanup
        await cache.cache.delete("test_*")
    
    @pytest.fixture
    def mock_model(self):
        """Mock ML model for testing"""
        model = Mock()
        model.predict = Mock(return_value=[1, 0, 1])
        return model
    
    @pytest.mark.asyncio
    async def test_cache_model(self, ml_cache, mock_model):
        """Test caching ML models"""
        model_name = "test_fraud_model"
        model_version = "v1.0"
        metadata = {"accuracy": 0.85, "trained_at": datetime.utcnow().isoformat()}
        
        # Cache the model
        result = await ml_cache.cache_model(model_name, model_version, mock_model, metadata)
        assert result is True
        
        # Verify model is cached
        cached_model = await ml_cache.get_cached_model(model_name, model_version)
        assert cached_model is not None
        assert cached_model.predict([1, 2, 3]) == [1, 0, 1]
    
    @pytest.mark.asyncio
    async def test_cache_prediction_result(self, ml_cache):
        """Test caching prediction results"""
        model_name = "test_model"
        input_hash = "abc123"
        prediction = {"result": "fraud", "confidence": 0.9}
        
        # Cache prediction
        result = await ml_cache.cache_prediction_result(model_name, input_hash, prediction, 0.9)
        assert result is True
        
        # Retrieve cached prediction
        cached_prediction = await ml_cache.get_cached_prediction(model_name, input_hash)
        assert cached_prediction is not None
        assert cached_prediction["prediction"]["result"] == "fraud"
        assert cached_prediction["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_cache_performance_metrics(self, ml_cache):
        """Test caching performance metrics"""
        model_name = "test_model"
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        
        # Cache metrics
        result = await ml_cache.cache_performance_metrics(model_name, metrics)
        assert result is True
        
        # Retrieve cached metrics
        cached_metrics = await ml_cache.get_cached_performance_metrics(model_name)
        assert cached_metrics is not None
        assert cached_metrics["accuracy"] == 0.85
        assert "cached_at" in cached_metrics
    
    @pytest.mark.asyncio
    async def test_invalidate_model_cache(self, ml_cache, mock_model):
        """Test cache invalidation"""
        model_name = "test_model"
        model_version = "v1.0"
        
        # Cache model and prediction
        await ml_cache.cache_model(model_name, model_version, mock_model)
        await ml_cache.cache_prediction_result(model_name, "hash123", {"result": "test"})
        
        # Verify cached
        assert await ml_cache.get_cached_model(model_name, model_version) is not None
        
        # Invalidate cache
        result = await ml_cache.invalidate_model_cache(model_name)
        assert result is True
        
        # Verify invalidated
        assert await ml_cache.get_cached_model(model_name, model_version) is None
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, ml_cache, mock_model):
        """Test cache statistics collection"""
        # Cache some models and predictions
        await ml_cache.cache_model("model1", "v1", mock_model)
        await ml_cache.cache_model("model2", "v1", mock_model)
        await ml_cache.cache_prediction_result("model1", "hash1", {"result": "test"})
        
        # Get statistics
        stats = await ml_cache.get_cache_statistics()
        
        assert isinstance(stats, dict)
        assert "cached_models" in stats
        assert "cached_predictions" in stats
        assert stats["cached_models"] >= 2
        assert stats["cached_predictions"] >= 1


class TestCacheInvalidationManager:
    """Test cache invalidation management"""
    
    @pytest.fixture
    async def invalidation_manager(self):
        """Create invalidation manager for testing"""
        manager = CacheInvalidationManager()
        yield manager
        # Cleanup
        await manager.cache.delete("invalidation_rules")
    
    @pytest.mark.asyncio
    async def test_register_invalidation_rule(self, invalidation_manager):
        """Test registering invalidation rules"""
        rule_name = "test_rule"
        pattern = "test_data:{business_id}:*"
        trigger_events = ["data_updated", "data_deleted"]
        
        await invalidation_manager.register_invalidation_rule(
            rule_name, pattern, trigger_events
        )
        
        assert rule_name in invalidation_manager.invalidation_rules
        rule = invalidation_manager.invalidation_rules[rule_name]
        assert rule["pattern"] == pattern
        assert rule["trigger_events"] == trigger_events
    
    @pytest.mark.asyncio
    async def test_trigger_invalidation(self, invalidation_manager):
        """Test triggering cache invalidation"""
        # Set up test data
        redis_client = get_redis_client()
        await redis_client.set("test_data:123:item1", "value1")
        await redis_client.set("test_data:123:item2", "value2")
        
        # Register rule
        await invalidation_manager.register_invalidation_rule(
            "test_rule",
            "test_data:{business_id}:*",
            ["data_updated"]
        )
        
        # Trigger invalidation
        invalidated_patterns = await invalidation_manager.trigger_invalidation(
            "data_updated", {"business_id": "123"}
        )
        
        assert len(invalidated_patterns) > 0
        assert "test_data:123:*" in invalidated_patterns
        
        # Verify data was invalidated
        assert await redis_client.get("test_data:123:item1") is None
        assert await redis_client.get("test_data:123:item2") is None
    
    @pytest.mark.asyncio
    async def test_setup_default_rules(self, invalidation_manager):
        """Test setting up default invalidation rules"""
        await invalidation_manager.setup_default_rules()
        
        # Verify default rules are registered
        assert "business_data_on_transaction" in invalidation_manager.invalidation_rules
        assert "predictions_on_model_update" in invalidation_manager.invalidation_rules
        assert "performance_on_feedback" in invalidation_manager.invalidation_rules


class TestCachePerformanceMonitor:
    """Test cache performance monitoring"""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor for testing"""
        monitor = CachePerformanceMonitor()
        yield monitor
        # Cleanup
        redis_client = get_redis_client()
        await redis_client.delete("cache_stats:*")
    
    @pytest.mark.asyncio
    async def test_record_cache_operations(self, performance_monitor):
        """Test recording cache hits and misses"""
        cache_type = "test_cache"
        
        # Record some hits and misses
        await performance_monitor.record_cache_hit(cache_type)
        await performance_monitor.record_cache_hit(cache_type)
        await performance_monitor.record_cache_miss(cache_type)
        
        # Verify counters
        redis_client = get_redis_client()
        hits = int(await redis_client.get(f"cache_stats:{cache_type}:hits") or 0)
        misses = int(await redis_client.get(f"cache_stats:{cache_type}:misses") or 0)
        
        assert hits == 2
        assert misses == 1
    
    @pytest.mark.asyncio
    async def test_performance_report(self, performance_monitor):
        """Test generating performance reports"""
        # Record some operations
        await performance_monitor.record_cache_hit("ml_model")
        await performance_monitor.record_cache_hit("ml_model")
        await performance_monitor.record_cache_miss("ml_model")
        
        # Generate report
        report = await performance_monitor.get_performance_report()
        
        assert isinstance(report, dict)
        assert "global_stats" in report
        assert "cache_type_stats" in report
        assert "recommendations" in report
        
        # Check ML model stats
        ml_stats = report["cache_type_stats"].get("ml_model", {})
        assert ml_stats["hits"] == 2
        assert ml_stats["misses"] == 1
        assert ml_stats["hit_rate"] == 2/3


class TestResourceManager:
    """Test resource management functionality"""
    
    @pytest.fixture
    async def resource_manager(self):
        """Create resource manager for testing"""
        manager = ResourceManager()
        yield manager
        await manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, resource_manager):
        """Test getting system resource metrics"""
        metrics = await resource_manager.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert ResourceType.MEMORY in metrics
        assert ResourceType.CPU in metrics
        assert ResourceType.DISK in metrics
        assert ResourceType.CACHE in metrics
        
        # Check memory metrics
        memory_metric = metrics[ResourceType.MEMORY]
        assert memory_metric.resource_type == ResourceType.MEMORY
        assert 0 <= memory_metric.usage_percentage <= 100
        assert memory_metric.status in [ResourceStatus.OPTIMAL, ResourceStatus.WARNING, ResourceStatus.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_track_model_usage(self, resource_manager):
        """Test tracking ML model usage"""
        model_name = "test_model"
        model_version = "v1.0"
        
        # Track some operations
        await resource_manager.track_model_usage(model_name, model_version, "load", 1500.0)
        await resource_manager.track_model_usage(model_name, model_version, "predict", 50.0)
        await resource_manager.track_model_usage(model_name, model_version, "cache_hit", 10.0)
        
        # Verify tracking
        model_key = f"{model_name}:{model_version}"
        assert model_key in resource_manager.model_usage_tracking
        
        usage = resource_manager.model_usage_tracking[model_key]
        assert usage.model_name == model_name
        assert usage.model_version == model_version
        assert usage.access_count == 3
        assert usage.load_time_ms == 1500.0
    
    @pytest.mark.asyncio
    async def test_optimize_resources(self, resource_manager):
        """Test resource optimization"""
        # Add some mock model usage data
        model_usage = ModelResourceUsage(
            model_name="old_model",
            model_version="v1.0",
            memory_mb=50.0,
            cpu_percentage=5.0,
            cache_size_mb=25.0,
            last_accessed=datetime.utcnow() - timedelta(days=10),  # Old access
            access_count=2,  # Low usage
            load_time_ms=1000.0
        )
        resource_manager.model_usage_tracking["old_model:v1.0"] = model_usage
        
        # Run optimization
        results = await resource_manager.optimize_resources()
        
        assert isinstance(results, dict)
        assert "actions_taken" in results
        assert "recommendations" in results
        assert "metrics_before" in results
        assert "metrics_after" in results
        assert "timestamp" in results
    
    @pytest.mark.asyncio
    async def test_model_performance_recommendations(self, resource_manager):
        """Test getting model performance recommendations"""
        model_name = "test_model"
        
        # Add mock usage data
        high_memory_usage = ModelResourceUsage(
            model_name=model_name,
            model_version="v1.0",
            memory_mb=150.0,  # High memory usage
            cpu_percentage=10.0,
            cache_size_mb=75.0,
            last_accessed=datetime.utcnow(),
            access_count=1000,  # High usage
            load_time_ms=8000.0  # Slow loading
        )
        resource_manager.model_usage_tracking[f"{model_name}:v1.0"] = high_memory_usage
        
        # Get recommendations
        recommendations = await resource_manager.get_model_performance_recommendations(model_name)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend model compression due to high memory usage
        memory_recommendation = any("compression" in rec.lower() for rec in recommendations)
        assert memory_recommendation
        
        # Should recommend keeping cached due to high usage
        cache_recommendation = any("cached" in rec.lower() for rec in recommendations)
        assert cache_recommendation
    
    @pytest.mark.asyncio
    async def test_resource_health_report(self, resource_manager):
        """Test generating resource health reports"""
        report = await resource_manager.get_resource_health_report()
        
        assert isinstance(report, dict)
        assert "overall_status" in report
        assert "system_metrics" in report
        assert "cache_performance" in report
        assert "model_usage" in report
        assert "recommendations" in report
        assert "timestamp" in report
        
        # Overall status should be a valid ResourceStatus
        assert report["overall_status"] in [status.value for status in ResourceStatus]
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, resource_manager):
        """Test starting and stopping resource monitoring"""
        # Start monitoring
        await resource_manager.start_monitoring()
        assert resource_manager._monitoring_task is not None
        assert not resource_manager._monitoring_task.done()
        
        # Stop monitoring
        await resource_manager.stop_monitoring()
        assert resource_manager._monitoring_task.done()


class TestCacheIntegration:
    """Integration tests for cache and resource management"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_caching_workflow(self):
        """Test complete caching workflow with ML models"""
        ml_cache = MLModelCache()
        performance_monitor = CachePerformanceMonitor()
        
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1, 0, 1])
        
        model_name = "integration_test_model"
        model_version = "v1.0"
        
        try:
            # 1. Cache model
            await ml_cache.cache_model(model_name, model_version, mock_model)
            await performance_monitor.record_cache_miss("ml_model")  # First time caching
            
            # 2. Retrieve model (cache hit)
            cached_model = await ml_cache.get_cached_model(model_name, model_version)
            assert cached_model is not None
            await performance_monitor.record_cache_hit("ml_model")
            
            # 3. Cache prediction result
            input_hash = "test_input_hash"
            prediction = {"result": "no_fraud", "confidence": 0.95}
            await ml_cache.cache_prediction_result(model_name, input_hash, prediction)
            
            # 4. Retrieve prediction (cache hit)
            cached_prediction = await ml_cache.get_cached_prediction(model_name, input_hash)
            assert cached_prediction is not None
            assert cached_prediction["prediction"]["result"] == "no_fraud"
            await performance_monitor.record_cache_hit("prediction")
            
            # 5. Get performance report
            report = await performance_monitor.get_performance_report()
            assert report["cache_type_stats"]["ml_model"]["hits"] >= 1
            
            # 6. Get cache statistics
            stats = await ml_cache.get_cache_statistics()
            assert stats["cached_models"] >= 1
            assert stats["cached_predictions"] >= 1
            
        finally:
            # Cleanup
            await ml_cache.invalidate_model_cache(model_name)
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_workflow(self):
        """Test cache invalidation workflow"""
        invalidation_manager = CacheInvalidationManager()
        redis_client = get_redis_client()
        
        try:
            # Set up test data
            business_id = "test_business_123"
            await redis_client.set_json(f"business_data:{business_id}:transactions", {"count": 100})
            await redis_client.set_json(f"business_data:{business_id}:insights", {"revenue": 50000})
            
            # Register invalidation rule
            await invalidation_manager.register_invalidation_rule(
                "test_business_data",
                "business_data:{business_id}:*",
                ["transaction_created", "transaction_updated"]
            )
            
            # Verify data exists
            assert await redis_client.get_json(f"business_data:{business_id}:transactions") is not None
            assert await redis_client.get_json(f"business_data:{business_id}:insights") is not None
            
            # Trigger invalidation
            await invalidation_manager.trigger_invalidation(
                "transaction_created", {"business_id": business_id}
            )
            
            # Verify data was invalidated
            assert await redis_client.get_json(f"business_data:{business_id}:transactions") is None
            assert await redis_client.get_json(f"business_data:{business_id}:insights") is None
            
        finally:
            # Cleanup
            await redis_client.delete("invalidation_rules")
    
    @pytest.mark.asyncio
    async def test_resource_optimization_workflow(self):
        """Test complete resource optimization workflow"""
        resource_manager = ResourceManager()
        ml_cache = MLModelCache()
        
        try:
            # Create mock models with different usage patterns
            models = [
                ("active_model", "v1.0", datetime.utcnow(), 1000),  # Active, high usage
                ("old_model", "v1.0", datetime.utcnow() - timedelta(days=10), 5),  # Old, low usage
                ("medium_model", "v1.0", datetime.utcnow() - timedelta(days=2), 50)  # Medium usage
            ]
            
            for model_name, version, last_access, access_count in models:
                usage = ModelResourceUsage(
                    model_name=model_name,
                    model_version=version,
                    memory_mb=25.0,
                    cpu_percentage=5.0,
                    cache_size_mb=12.5,
                    last_accessed=last_access,
                    access_count=access_count,
                    load_time_ms=1000.0
                )
                resource_manager.model_usage_tracking[f"{model_name}:{version}"] = usage
            
            # Get initial metrics
            initial_metrics = await resource_manager.get_system_metrics()
            assert ResourceType.MEMORY in initial_metrics
            
            # Run optimization
            optimization_results = await resource_manager.optimize_resources()
            
            # Verify optimization results
            assert "actions_taken" in optimization_results
            assert "recommendations" in optimization_results
            assert "metrics_before" in optimization_results
            assert "metrics_after" in optimization_results
            
            # Get health report
            health_report = await resource_manager.get_resource_health_report()
            assert health_report["overall_status"] in [status.value for status in ResourceStatus]
            
        finally:
            # Cleanup
            resource_manager.model_usage_tracking.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])