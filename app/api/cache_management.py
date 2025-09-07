"""
Cache Management API Endpoints

Provides endpoints for monitoring and managing Redis cache performance,
ML model caching, and resource optimization.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging

from ..utils.cache import (
    ml_model_cache, 
    cache_performance_monitor, 
    cache_invalidation_manager,
    get_redis_client
)
from ..utils.resource_manager import resource_manager, ResourceType
from ..utils.auth import get_current_user
from ..models.responses import (
    CacheStatsResponse,
    ResourceHealthResponse,
    OptimizationResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_statistics(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive cache statistics and performance metrics.
    """
    try:
        # Get ML model cache statistics
        ml_stats = await ml_model_cache.get_cache_statistics()
        
        # Get performance report
        performance_report = await cache_performance_monitor.get_performance_report()
        
        # Get Redis info
        redis_client = get_redis_client()
        redis_info = {}
        try:
            if hasattr(redis_client.client, 'info'):
                redis_info = redis_client.client.info()
        except:
            pass
        
        return CacheStatsResponse(
            ml_model_stats=ml_stats,
            performance_report=performance_report,
            redis_info=redis_info,
            timestamp=performance_report.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")


@router.get("/health", response_model=ResourceHealthResponse)
async def get_resource_health(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive resource health report including system metrics,
    cache performance, and optimization recommendations.
    """
    try:
        health_report = await resource_manager.get_resource_health_report()
        
        return ResourceHealthResponse(
            overall_status=health_report.get("overall_status", "unknown"),
            system_metrics=health_report.get("system_metrics", {}),
            cache_performance=health_report.get("cache_performance", {}),
            model_usage=health_report.get("model_usage", {}),
            recommendations=health_report.get("recommendations", []),
            timestamp=health_report.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get resource health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource health: {str(e)}")


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_resources(
    background_tasks: BackgroundTasks,
    force: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger resource optimization including cache cleanup,
    memory management, and performance tuning.
    
    Args:
        force: Force optimization even if not needed
    """
    try:
        # Run optimization
        optimization_results = await resource_manager.optimize_resources()
        
        # Start monitoring if not already running
        background_tasks.add_task(resource_manager.start_monitoring)
        
        return OptimizationResponse(
            success=True,
            actions_taken=optimization_results.get("actions_taken", []),
            recommendations=optimization_results.get("recommendations", []),
            metrics_before=optimization_results.get("metrics_before", {}),
            metrics_after=optimization_results.get("metrics_after", {}),
            timestamp=optimization_results.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Resource optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resource optimization failed: {str(e)}")


@router.delete("/models/{model_name}")
async def invalidate_model_cache(
    model_name: str,
    model_version: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Invalidate cached ML model and related prediction results.
    
    Args:
        model_name: Name of the model to invalidate
        model_version: Specific version to invalidate (optional)
    """
    try:
        success = await ml_model_cache.invalidate_model_cache(model_name, model_version)
        
        if success:
            return {
                "success": True,
                "message": f"Cache invalidated for model: {model_name}" + 
                          (f" v{model_version}" if model_version else " (all versions)")
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to invalidate model cache")
            
    except Exception as e:
        logger.error(f"Failed to invalidate model cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to invalidate model cache: {str(e)}")


@router.delete("/predictions/{model_name}")
async def clear_prediction_cache(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Clear cached prediction results for a specific model.
    
    Args:
        model_name: Name of the model whose predictions to clear
    """
    try:
        redis_client = get_redis_client()
        
        # Find and delete prediction cache keys
        pattern = f"prediction:{model_name}:*"
        keys = await redis_client.keys(pattern)
        
        deleted_count = 0
        for key in keys:
            if await redis_client.delete(key):
                deleted_count += 1
        
        return {
            "success": True,
            "message": f"Cleared {deleted_count} prediction cache entries for model: {model_name}"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear prediction cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear prediction cache: {str(e)}")


@router.post("/invalidation/trigger")
async def trigger_cache_invalidation(
    event_name: str,
    context: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger cache invalidation based on an event.
    
    Args:
        event_name: Name of the event that triggers invalidation
        context: Optional context data for pattern substitution
    """
    try:
        invalidated_patterns = await cache_invalidation_manager.trigger_invalidation(
            event_name, context or {}
        )
        
        return {
            "success": True,
            "event_name": event_name,
            "invalidated_patterns": invalidated_patterns,
            "message": f"Triggered invalidation for event: {event_name}"
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger cache invalidation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger cache invalidation: {str(e)}")


@router.get("/models/{model_name}/recommendations")
async def get_model_recommendations(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get performance recommendations for a specific ML model.
    
    Args:
        model_name: Name of the model to analyze
    """
    try:
        recommendations = await resource_manager.get_model_performance_recommendations(model_name)
        
        return {
            "model_name": model_name,
            "recommendations": recommendations,
            "timestamp": performance_report.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Failed to get model recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model recommendations: {str(e)}")


@router.get("/monitoring/start")
async def start_resource_monitoring(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Start background resource monitoring.
    """
    try:
        background_tasks.add_task(resource_manager.start_monitoring)
        
        return {
            "success": True,
            "message": "Resource monitoring started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.get("/monitoring/stop")
async def stop_resource_monitoring(current_user: dict = Depends(get_current_user)):
    """
    Stop background resource monitoring.
    """
    try:
        await resource_manager.stop_monitoring()
        
        return {
            "success": True,
            "message": "Resource monitoring stopped"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/config")
async def get_cache_configuration(current_user: dict = Depends(get_current_user)):
    """
    Get current cache configuration settings.
    """
    try:
        config = {
            "redis_url": "***hidden***",  # Don't expose sensitive info
            "model_memory_limit_mb": ml_model_cache.model_memory_limit / (1024**2),
            "prediction_cache_ttl": ml_model_cache.prediction_cache_ttl,
            "model_cache_ttl": ml_model_cache.model_cache_ttl,
            "performance_cache_ttl": ml_model_cache.performance_cache_ttl,
            "monitoring_interval": resource_manager.monitoring_interval,
            "resource_thresholds": resource_manager.resource_thresholds
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get cache configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache configuration: {str(e)}")


@router.put("/config")
async def update_cache_configuration(
    config_updates: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Update cache configuration settings.
    
    Args:
        config_updates: Dictionary of configuration updates
    """
    try:
        updated_settings = []
        
        # Update ML cache settings
        if "model_memory_limit_mb" in config_updates:
            ml_model_cache.model_memory_limit = config_updates["model_memory_limit_mb"] * 1024**2
            updated_settings.append("model_memory_limit_mb")
        
        if "prediction_cache_ttl" in config_updates:
            ml_model_cache.prediction_cache_ttl = config_updates["prediction_cache_ttl"]
            updated_settings.append("prediction_cache_ttl")
        
        if "model_cache_ttl" in config_updates:
            ml_model_cache.model_cache_ttl = config_updates["model_cache_ttl"]
            updated_settings.append("model_cache_ttl")
        
        # Update resource manager settings
        if "monitoring_interval" in config_updates:
            resource_manager.monitoring_interval = config_updates["monitoring_interval"]
            updated_settings.append("monitoring_interval")
        
        if "resource_thresholds" in config_updates:
            resource_manager.resource_thresholds.update(config_updates["resource_thresholds"])
            updated_settings.append("resource_thresholds")
        
        return {
            "success": True,
            "updated_settings": updated_settings,
            "message": f"Updated {len(updated_settings)} configuration settings"
        }
        
    except Exception as e:
        logger.error(f"Failed to update cache configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update cache configuration: {str(e)}")