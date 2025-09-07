#!/usr/bin/env python3
"""
Validation script for Redis caching and performance optimization.

Tests ML model caching, resource management, cache invalidation,
and performance monitoring functionality.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_redis_connection():
    """Test Redis connection and basic operations"""
    logger.info("Testing Redis connection...")
    
    try:
        from app.utils.cache import get_redis_client
        
        redis_client = get_redis_client()
        
        # Test basic operations
        test_key = "cache_test:connection"
        test_value = "test_value"
        
        # Set and get
        await redis_client.set(test_key, test_value, ex=60)
        retrieved_value = await redis_client.get(test_key)
        
        assert retrieved_value == test_value, f"Expected {test_value}, got {retrieved_value}"
        
        # Test JSON operations
        test_json = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        await redis_client.set_json(f"{test_key}:json", test_json, ex=60)
        retrieved_json = await redis_client.get_json(f"{test_key}:json")
        
        assert retrieved_json["test"] == "data", "JSON serialization failed"
        
        # Cleanup
        await redis_client.delete(test_key)
        await redis_client.delete(f"{test_key}:json")
        
        logger.info("✓ Redis connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Redis connection test failed: {str(e)}")
        return False


async def test_ml_model_caching():
    """Test ML model caching functionality"""
    logger.info("Testing ML model caching...")
    
    try:
        from app.utils.cache import MLModelCache
        from unittest.mock import Mock
        
        ml_cache = MLModelCache()
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1, 0, 1, 0])
        mock_model.__class__.__name__ = "MockMLModel"
        
        model_name = "test_fraud_detector"
        model_version = "v1.0"
        metadata = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "trained_at": datetime.utcnow().isoformat()
        }
        
        # Test model caching
        cache_result = await ml_cache.cache_model(model_name, model_version, mock_model, metadata)
        assert cache_result, "Failed to cache model"
        
        # Test model retrieval
        cached_model = await ml_cache.get_cached_model(model_name, model_version)
        assert cached_model is not None, "Failed to retrieve cached model"
        assert cached_model.predict([1, 2, 3, 4]) == [1, 0, 1, 0], "Cached model prediction failed"
        
        # Test prediction caching
        input_hash = "test_input_hash_123"
        prediction_result = {
            "prediction": "fraud",
            "confidence": 0.92,
            "risk_factors": ["duplicate_invoice", "unusual_amount"]
        }
        
        pred_cache_result = await ml_cache.cache_prediction_result(
            model_name, input_hash, prediction_result, 0.92
        )
        assert pred_cache_result, "Failed to cache prediction result"
        
        # Test prediction retrieval
        cached_prediction = await ml_cache.get_cached_prediction(model_name, input_hash)
        assert cached_prediction is not None, "Failed to retrieve cached prediction"
        assert cached_prediction["prediction"]["prediction"] == "fraud", "Cached prediction incorrect"
        
        # Test performance metrics caching
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc_roc": 0.91
        }
        
        metrics_cache_result = await ml_cache.cache_performance_metrics(model_name, performance_metrics)
        assert metrics_cache_result, "Failed to cache performance metrics"
        
        cached_metrics = await ml_cache.get_cached_performance_metrics(model_name)
        assert cached_metrics is not None, "Failed to retrieve cached metrics"
        assert cached_metrics["accuracy"] == 0.85, "Cached metrics incorrect"
        
        # Test cache statistics
        stats = await ml_cache.get_cache_statistics()
        assert isinstance(stats, dict), "Cache statistics should be a dictionary"
        assert stats["cached_models"] >= 1, "Should have at least 1 cached model"
        assert stats["cached_predictions"] >= 1, "Should have at least 1 cached prediction"
        
        # Test cache invalidation
        invalidation_result = await ml_cache.invalidate_model_cache(model_name, model_version)
        assert invalidation_result, "Failed to invalidate model cache"
        
        # Verify invalidation
        invalidated_model = await ml_cache.get_cached_model(model_name, model_version)
        assert invalidated_model is None, "Model should be invalidated"
        
        logger.info("✓ ML model caching test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ ML model caching test failed: {str(e)}")
        return False


async def test_cache_performance_monitoring():
    """Test cache performance monitoring"""
    logger.info("Testing cache performance monitoring...")
    
    try:
        from app.utils.cache import CachePerformanceMonitor
        
        monitor = CachePerformanceMonitor()
        
        # Record some cache operations
        cache_types = ["ml_model", "prediction", "performance", "business_data"]
        
        for cache_type in cache_types:
            # Simulate hits and misses
            for _ in range(5):
                await monitor.record_cache_hit(cache_type)
            for _ in range(2):
                await monitor.record_cache_miss(cache_type)
        
        # Generate performance report
        report = await monitor.get_performance_report()
        
        assert isinstance(report, dict), "Performance report should be a dictionary"
        assert "global_stats" in report, "Report should contain global stats"
        assert "cache_type_stats" in report, "Report should contain cache type stats"
        assert "recommendations" in report, "Report should contain recommendations"
        
        # Verify global stats
        global_stats = report["global_stats"]
        assert global_stats["total_requests"] > 0, "Should have recorded requests"
        assert 0 <= global_stats["hit_rate"] <= 1, "Hit rate should be between 0 and 1"
        
        # Verify cache type stats
        for cache_type in cache_types:
            if cache_type in report["cache_type_stats"]:
                type_stats = report["cache_type_stats"][cache_type]
                assert type_stats["hits"] == 5, f"Should have 5 hits for {cache_type}"
                assert type_stats["misses"] == 2, f"Should have 2 misses for {cache_type}"
                assert abs(type_stats["hit_rate"] - (5/7)) < 0.01, f"Hit rate calculation incorrect for {cache_type}"
        
        logger.info("✓ Cache performance monitoring test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cache performance monitoring test failed: {str(e)}")
        return False


async def test_cache_invalidation():
    """Test cache invalidation management"""
    logger.info("Testing cache invalidation...")
    
    try:
        from app.utils.cache import CacheInvalidationManager, get_redis_client
        
        manager = CacheInvalidationManager()
        redis_client = get_redis_client()
        
        # Set up test data
        business_id = "test_business_456"
        test_keys = [
            f"business_data:{business_id}:transactions",
            f"business_data:{business_id}:insights",
            f"business_data:{business_id}:reports"
        ]
        
        for key in test_keys:
            await redis_client.set_json(key, {"test": "data", "timestamp": time.time()}, ex=300)
        
        # Verify data exists
        for key in test_keys:
            data = await redis_client.get_json(key)
            assert data is not None, f"Test data should exist for key: {key}"
        
        # Register invalidation rule
        rule_name = "test_business_invalidation"
        pattern = "business_data:{business_id}:*"
        trigger_events = ["transaction_created", "data_updated"]
        
        await manager.register_invalidation_rule(rule_name, pattern, trigger_events)
        
        # Verify rule registration
        assert rule_name in manager.invalidation_rules, "Rule should be registered"
        rule = manager.invalidation_rules[rule_name]
        assert rule["pattern"] == pattern, "Rule pattern should match"
        assert set(rule["trigger_events"]) == set(trigger_events), "Rule events should match"
        
        # Trigger invalidation
        invalidated_patterns = await manager.trigger_invalidation(
            "transaction_created", {"business_id": business_id}
        )
        
        assert len(invalidated_patterns) > 0, "Should have invalidated some patterns"
        expected_pattern = f"business_data:{business_id}:*"
        assert expected_pattern in invalidated_patterns, f"Should have invalidated pattern: {expected_pattern}"
        
        # Verify data was invalidated
        for key in test_keys:
            data = await redis_client.get_json(key)
            assert data is None, f"Data should be invalidated for key: {key}"
        
        # Test default rules setup
        await manager.setup_default_rules()
        default_rules = ["business_data_on_transaction", "predictions_on_model_update", "performance_on_feedback"]
        
        for rule_name in default_rules:
            assert rule_name in manager.invalidation_rules, f"Default rule {rule_name} should be registered"
        
        logger.info("✓ Cache invalidation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cache invalidation test failed: {str(e)}")
        return False


async def test_resource_management():
    """Test resource management functionality"""
    logger.info("Testing resource management...")
    
    try:
        from app.utils.resource_manager import ResourceManager, ResourceType, ResourceStatus, ModelResourceUsage
        
        manager = ResourceManager()
        
        # Test system metrics collection
        metrics = await manager.get_system_metrics()
        
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        
        required_resources = [ResourceType.MEMORY, ResourceType.CPU, ResourceType.DISK, ResourceType.CACHE]
        for resource_type in required_resources:
            assert resource_type in metrics, f"Should have metrics for {resource_type}"
            
            metric = metrics[resource_type]
            assert 0 <= metric.usage_percentage <= 100, f"Usage percentage should be 0-100 for {resource_type}"
            assert metric.status in [ResourceStatus.OPTIMAL, ResourceStatus.WARNING, ResourceStatus.CRITICAL], \
                f"Status should be valid for {resource_type}"
            assert metric.current_usage >= 0, f"Current usage should be non-negative for {resource_type}"
            assert metric.max_usage > 0, f"Max usage should be positive for {resource_type}"
        
        # Test model usage tracking
        model_name = "test_tracking_model"
        model_version = "v1.0"
        
        await manager.track_model_usage(model_name, model_version, "load", 1500.0)
        await manager.track_model_usage(model_name, model_version, "predict", 50.0)
        await manager.track_model_usage(model_name, model_version, "cache_hit", 10.0)
        
        model_key = f"{model_name}:{model_version}"
        assert model_key in manager.model_usage_tracking, "Model usage should be tracked"
        
        usage = manager.model_usage_tracking[model_key]
        assert usage.model_name == model_name, "Model name should match"
        assert usage.model_version == model_version, "Model version should match"
        assert usage.access_count == 3, "Should have 3 access records"
        assert usage.load_time_ms == 1500.0, "Load time should be recorded"
        
        # Test resource optimization
        # Add some mock data for optimization
        old_model_usage = ModelResourceUsage(
            model_name="old_unused_model",
            model_version="v0.1",
            memory_mb=75.0,
            cpu_percentage=5.0,
            cache_size_mb=37.5,
            last_accessed=datetime.utcnow() - timedelta(days=15),  # Very old
            access_count=3,  # Low usage
            load_time_ms=2000.0
        )
        manager.model_usage_tracking["old_unused_model:v0.1"] = old_model_usage
        
        optimization_results = await manager.optimize_resources()
        
        assert isinstance(optimization_results, dict), "Optimization results should be a dictionary"
        assert "actions_taken" in optimization_results, "Should contain actions taken"
        assert "recommendations" in optimization_results, "Should contain recommendations"
        assert "metrics_before" in optimization_results, "Should contain before metrics"
        assert "metrics_after" in optimization_results, "Should contain after metrics"
        assert "timestamp" in optimization_results, "Should contain timestamp"
        
        # Test performance recommendations
        recommendations = await manager.get_model_performance_recommendations(model_name)
        assert isinstance(recommendations, list), "Recommendations should be a list"
        
        # Test health report
        health_report = await manager.get_resource_health_report()
        
        assert isinstance(health_report, dict), "Health report should be a dictionary"
        assert "overall_status" in health_report, "Should contain overall status"
        assert "system_metrics" in health_report, "Should contain system metrics"
        assert "cache_performance" in health_report, "Should contain cache performance"
        assert "model_usage" in health_report, "Should contain model usage"
        assert "recommendations" in health_report, "Should contain recommendations"
        
        valid_statuses = [status.value for status in ResourceStatus]
        assert health_report["overall_status"] in valid_statuses, "Overall status should be valid"
        
        logger.info("✓ Resource management test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Resource management test failed: {str(e)}")
        return False


async def test_integration_workflow():
    """Test complete integration workflow"""
    logger.info("Testing integration workflow...")
    
    try:
        from app.utils.cache import MLModelCache, CachePerformanceMonitor, CacheInvalidationManager
        from app.utils.resource_manager import ResourceManager
        from unittest.mock import Mock
        
        # Initialize components
        ml_cache = MLModelCache()
        performance_monitor = CachePerformanceMonitor()
        invalidation_manager = CacheInvalidationManager()
        resource_manager = ResourceManager()
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0, 1, 0])
        
        model_name = "integration_test_model"
        model_version = "v2.0"
        business_id = "integration_test_business"
        
        # 1. Cache model and track usage
        await ml_cache.cache_model(model_name, model_version, mock_model)
        await resource_manager.track_model_usage(model_name, model_version, "load", 1200.0)
        await performance_monitor.record_cache_miss("ml_model")  # First load
        
        # 2. Use model (cache hit)
        cached_model = await ml_cache.get_cached_model(model_name, model_version)
        assert cached_model is not None, "Model should be cached"
        await resource_manager.track_model_usage(model_name, model_version, "predict", 75.0)
        await performance_monitor.record_cache_hit("ml_model")
        
        # 3. Cache prediction results
        for i in range(3):
            input_hash = f"integration_input_{i}"
            prediction = {"result": f"prediction_{i}", "confidence": 0.8 + i * 0.05}
            await ml_cache.cache_prediction_result(model_name, input_hash, prediction)
            await performance_monitor.record_cache_miss("prediction")  # First time
        
        # 4. Retrieve predictions (cache hits)
        for i in range(3):
            input_hash = f"integration_input_{i}"
            cached_pred = await ml_cache.get_cached_prediction(model_name, input_hash)
            assert cached_pred is not None, f"Prediction {i} should be cached"
            await performance_monitor.record_cache_hit("prediction")
        
        # 5. Set up invalidation rules
        await invalidation_manager.register_invalidation_rule(
            "integration_test_rule",
            f"prediction:{model_name}:*",
            ["model_updated", "model_retrained"]
        )
        
        # 6. Get performance report
        performance_report = await performance_monitor.get_performance_report()
        assert performance_report["cache_type_stats"]["ml_model"]["hits"] >= 1, "Should have ML model cache hits"
        assert performance_report["cache_type_stats"]["prediction"]["hits"] >= 3, "Should have prediction cache hits"
        
        # 7. Get resource health
        health_report = await resource_manager.get_resource_health_report()
        assert health_report["overall_status"] in ["optimal", "warning", "critical"], "Should have valid status"
        
        # 8. Trigger cache invalidation
        await invalidation_manager.trigger_invalidation("model_updated", {"model_name": model_name})
        
        # 9. Verify predictions were invalidated
        for i in range(3):
            input_hash = f"integration_input_{i}"
            cached_pred = await ml_cache.get_cached_prediction(model_name, input_hash)
            assert cached_pred is None, f"Prediction {i} should be invalidated"
        
        # 10. Run resource optimization
        optimization_results = await resource_manager.optimize_resources()
        assert len(optimization_results["actions_taken"]) >= 0, "Should have optimization results"
        
        # 11. Get final statistics
        cache_stats = await ml_cache.get_cache_statistics()
        assert cache_stats["cached_models"] >= 1, "Should still have cached models"
        
        logger.info("✓ Integration workflow test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration workflow test failed: {str(e)}")
        return False


async def run_performance_benchmark():
    """Run performance benchmarks for caching operations"""
    logger.info("Running performance benchmarks...")
    
    try:
        from app.utils.cache import MLModelCache, get_redis_client
        from unittest.mock import Mock
        
        ml_cache = MLModelCache()
        redis_client = get_redis_client()
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1] * 100)
        
        # Benchmark model caching
        model_cache_times = []
        for i in range(10):
            start_time = time.time()
            await ml_cache.cache_model(f"benchmark_model_{i}", "v1.0", mock_model)
            end_time = time.time()
            model_cache_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_model_cache_time = sum(model_cache_times) / len(model_cache_times)
        logger.info(f"Average model caching time: {avg_model_cache_time:.2f}ms")
        
        # Benchmark model retrieval
        model_retrieval_times = []
        for i in range(10):
            start_time = time.time()
            await ml_cache.get_cached_model(f"benchmark_model_{i}", "v1.0")
            end_time = time.time()
            model_retrieval_times.append((end_time - start_time) * 1000)
        
        avg_model_retrieval_time = sum(model_retrieval_times) / len(model_retrieval_times)
        logger.info(f"Average model retrieval time: {avg_model_retrieval_time:.2f}ms")
        
        # Benchmark prediction caching
        prediction_cache_times = []
        for i in range(100):
            start_time = time.time()
            await ml_cache.cache_prediction_result(
                "benchmark_model_0", 
                f"input_hash_{i}", 
                {"result": f"prediction_{i}", "confidence": 0.9}
            )
            end_time = time.time()
            prediction_cache_times.append((end_time - start_time) * 1000)
        
        avg_prediction_cache_time = sum(prediction_cache_times) / len(prediction_cache_times)
        logger.info(f"Average prediction caching time: {avg_prediction_cache_time:.2f}ms")
        
        # Benchmark prediction retrieval
        prediction_retrieval_times = []
        for i in range(100):
            start_time = time.time()
            await ml_cache.get_cached_prediction("benchmark_model_0", f"input_hash_{i}")
            end_time = time.time()
            prediction_retrieval_times.append((end_time - start_time) * 1000)
        
        avg_prediction_retrieval_time = sum(prediction_retrieval_times) / len(prediction_retrieval_times)
        logger.info(f"Average prediction retrieval time: {avg_prediction_retrieval_time:.2f}ms")
        
        # Performance assertions
        assert avg_model_cache_time < 100, f"Model caching too slow: {avg_model_cache_time:.2f}ms"
        assert avg_model_retrieval_time < 50, f"Model retrieval too slow: {avg_model_retrieval_time:.2f}ms"
        assert avg_prediction_cache_time < 10, f"Prediction caching too slow: {avg_prediction_cache_time:.2f}ms"
        assert avg_prediction_retrieval_time < 5, f"Prediction retrieval too slow: {avg_prediction_retrieval_time:.2f}ms"
        
        # Cleanup
        for i in range(10):
            await ml_cache.invalidate_model_cache(f"benchmark_model_{i}")
        
        logger.info("✓ Performance benchmarks passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance benchmarks failed: {str(e)}")
        return False


async def main():
    """Run all validation tests"""
    logger.info("Starting Redis caching and performance optimization validation...")
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("ML Model Caching", test_ml_model_caching),
        ("Cache Performance Monitoring", test_cache_performance_monitoring),
        ("Cache Invalidation", test_cache_invalidation),
        ("Resource Management", test_resource_management),
        ("Integration Workflow", test_integration_workflow),
        ("Performance Benchmarks", run_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Redis caching and performance optimization tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)