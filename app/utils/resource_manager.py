import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from .cache import MLModelCache, get_redis_client, cache_performance_monitor
from ..config import settings

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    CACHE = "cache"


class ResourceStatus(str, Enum):
    """Resource status levels"""
    OPTIMAL = "optimal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    resource_type: ResourceType
    current_usage: float
    max_usage: float
    usage_percentage: float
    status: ResourceStatus
    timestamp: datetime
    details: Dict[str, Any] = None


@dataclass
class ModelResourceUsage:
    model_name: str
    model_version: str
    memory_mb: float
    cpu_percentage: float
    cache_size_mb: float
    last_accessed: datetime
    access_count: int
    load_time_ms: float


class ResourceManager:
    """
    Manages system resources and optimizes ML model performance.
    """
    
    def __init__(self):
        self.cache = get_redis_client()
        self.ml_cache = MLModelCache()
        self.monitoring_interval = 30  # seconds
        self.resource_thresholds = {
            ResourceType.MEMORY: {"warning": 70, "critical": 85},
            ResourceType.CPU: {"warning": 80, "critical": 95},
            ResourceType.DISK: {"warning": 80, "critical": 90},
            ResourceType.CACHE: {"warning": 75, "critical": 90}
        }
        self.model_usage_tracking = {}
        self.optimization_history = []
        self._monitoring_task = None
        
    async def start_monitoring(self):
        """Start resource monitoring background task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring background task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Resource monitoring stopped")
    
    async def get_system_metrics(self) -> Dict[ResourceType, ResourceMetrics]:
        """
        Get current system resource metrics.
        """
        metrics = {}
        
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_status = self._determine_status(
                memory.percent, self.resource_thresholds[ResourceType.MEMORY]
            )
            
            # Build details dict with platform-specific attributes
            details = {
                "available_gb": memory.available / (1024**3)
            }
            
            # Add cached and buffers only if available (Linux/Windows)
            if hasattr(memory, 'cached'):
                details["cached_gb"] = memory.cached / (1024**3)
            if hasattr(memory, 'buffers'):
                details["buffers_gb"] = memory.buffers / (1024**3)
            
            metrics[ResourceType.MEMORY] = ResourceMetrics(
                resource_type=ResourceType.MEMORY,
                current_usage=memory.used / (1024**3),  # GB
                max_usage=memory.total / (1024**3),  # GB
                usage_percentage=memory.percent,
                status=memory_status,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._determine_status(
                cpu_percent, self.resource_thresholds[ResourceType.CPU]
            )
            
            metrics[ResourceType.CPU] = ResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=cpu_percent,
                max_usage=100.0,
                usage_percentage=cpu_percent,
                status=cpu_status,
                timestamp=datetime.utcnow(),
                details={
                    "cpu_count": psutil.cpu_count(),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_status = self._determine_status(
                disk.percent, self.resource_thresholds[ResourceType.DISK]
            )
            
            metrics[ResourceType.DISK] = ResourceMetrics(
                resource_type=ResourceType.DISK,
                current_usage=disk.used / (1024**3),  # GB
                max_usage=disk.total / (1024**3),  # GB
                usage_percentage=disk.percent,
                status=disk_status,
                timestamp=datetime.utcnow(),
                details={
                    "free_gb": disk.free / (1024**3)
                }
            )
            
            # Cache metrics
            cache_stats = await self.ml_cache.get_cache_statistics()
            cache_memory_mb = cache_stats.get("memory_usage_mb", 0)
            cache_limit_mb = self.ml_cache.model_memory_limit / (1024**2)
            cache_percentage = (cache_memory_mb / cache_limit_mb * 100) if cache_limit_mb > 0 else 0
            
            cache_status = self._determine_status(
                cache_percentage, self.resource_thresholds[ResourceType.CACHE]
            )
            
            metrics[ResourceType.CACHE] = ResourceMetrics(
                resource_type=ResourceType.CACHE,
                current_usage=cache_memory_mb,
                max_usage=cache_limit_mb,
                usage_percentage=cache_percentage,
                status=cache_status,
                timestamp=datetime.utcnow(),
                details=cache_stats
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
        
        return metrics
    
    async def track_model_usage(self, model_name: str, model_version: str, 
                              operation: str, duration_ms: float = None):
        """
        Track ML model resource usage.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            operation: Operation performed (load, predict, cache_hit, etc.)
            duration_ms: Duration of the operation in milliseconds
        """
        try:
            model_key = f"{model_name}:{model_version}"
            
            # Get or create usage tracking entry
            if model_key not in self.model_usage_tracking:
                self.model_usage_tracking[model_key] = ModelResourceUsage(
                    model_name=model_name,
                    model_version=model_version,
                    memory_mb=0.0,
                    cpu_percentage=0.0,
                    cache_size_mb=0.0,
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    load_time_ms=0.0
                )
            
            usage = self.model_usage_tracking[model_key]
            usage.last_accessed = datetime.utcnow()
            usage.access_count += 1
            
            if operation == "load" and duration_ms:
                usage.load_time_ms = duration_ms
            
            # Store in cache for persistence
            await self._store_model_usage(model_key, usage)
            
            logger.debug(f"Tracked model usage: {model_name} - {operation}")
            
        except Exception as e:
            logger.error(f"Failed to track model usage: {str(e)}")
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """
        Perform resource optimization based on current usage patterns.
        """
        optimization_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "actions_taken": [],
            "recommendations": [],
            "metrics_before": {},
            "metrics_after": {}
        }
        
        try:
            # Get current metrics
            optimization_results["metrics_before"] = await self.get_system_metrics()
            
            # Perform optimizations
            await self._optimize_memory_usage(optimization_results)
            await self._optimize_cache_performance(optimization_results)
            await self._cleanup_unused_models(optimization_results)
            
            # Get metrics after optimization
            optimization_results["metrics_after"] = await self.get_system_metrics()
            
            # Store optimization history
            self.optimization_history.append(optimization_results)
            
            # Keep only last 100 optimization records
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info(f"Resource optimization completed: {len(optimization_results['actions_taken'])} actions taken")
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {str(e)}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def get_model_performance_recommendations(self, model_name: str) -> List[str]:
        """
        Get performance recommendations for a specific model.
        
        Args:
            model_name: Name of the model to analyze
        """
        recommendations = []
        
        try:
            # Get model usage data
            model_usage = await self._get_model_usage_data(model_name)
            if not model_usage:
                return ["No usage data available for this model"]
            
            # Analyze usage patterns
            for usage in model_usage:
                # High memory usage
                if usage.memory_mb > 100:  # 100MB threshold
                    recommendations.append(
                        f"Model {usage.model_name} v{usage.model_version} uses {usage.memory_mb:.1f}MB. "
                        f"Consider model compression or quantization."
                    )
                
                # Slow loading
                if usage.load_time_ms > 5000:  # 5 second threshold
                    recommendations.append(
                        f"Model {usage.model_name} v{usage.model_version} takes {usage.load_time_ms:.0f}ms to load. "
                        f"Consider keeping it cached or using a smaller model variant."
                    )
                
                # Low access frequency
                days_since_access = (datetime.utcnow() - usage.last_accessed).days
                if days_since_access > 7 and usage.access_count < 10:
                    recommendations.append(
                        f"Model {usage.model_name} v{usage.model_version} has low usage "
                        f"({usage.access_count} accesses, last used {days_since_access} days ago). "
                        f"Consider removing from cache."
                    )
                
                # High access frequency
                if usage.access_count > 1000:
                    recommendations.append(
                        f"Model {usage.model_name} v{usage.model_version} is heavily used "
                        f"({usage.access_count} accesses). Consider keeping permanently cached."
                    )
            
        except Exception as e:
            logger.error(f"Failed to get model recommendations: {str(e)}")
            recommendations.append("Unable to generate recommendations due to error")
        
        return recommendations
    
    async def get_resource_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive resource health report.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": ResourceStatus.OPTIMAL,
            "system_metrics": {},
            "model_usage": {},
            "cache_performance": {},
            "optimization_history": [],
            "recommendations": []
        }
        
        try:
            # System metrics
            metrics = await self.get_system_metrics()
            report["system_metrics"] = {
                resource_type.value: {
                    "usage_percentage": metric.usage_percentage,
                    "status": metric.status.value,
                    "current_usage": metric.current_usage,
                    "max_usage": metric.max_usage
                }
                for resource_type, metric in metrics.items()
            }
            
            # Determine overall status
            statuses = [metric.status for metric in metrics.values()]
            if ResourceStatus.CRITICAL in statuses:
                report["overall_status"] = ResourceStatus.CRITICAL
            elif ResourceStatus.WARNING in statuses:
                report["overall_status"] = ResourceStatus.WARNING
            
            # Cache performance
            cache_report = await cache_performance_monitor.get_performance_report()
            report["cache_performance"] = cache_report
            
            # Model usage summary
            model_usage_summary = await self._get_model_usage_summary()
            report["model_usage"] = model_usage_summary
            
            # Recent optimization history
            report["optimization_history"] = self.optimization_history[-5:]  # Last 5 optimizations
            
            # Generate recommendations
            report["recommendations"] = await self._generate_health_recommendations(metrics)
            
        except Exception as e:
            logger.error(f"Failed to generate health report: {str(e)}")
            report["error"] = str(e)
        
        return report
    
    # Private helper methods
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Get current metrics
                metrics = await self.get_system_metrics()
                
                # Check for critical conditions
                for resource_type, metric in metrics.items():
                    if metric.status == ResourceStatus.CRITICAL:
                        logger.warning(f"Critical resource usage: {resource_type.value} at {metric.usage_percentage:.1f}%")
                        
                        # Trigger emergency optimization
                        await self._emergency_optimization(resource_type, metric)
                
                # Store metrics for historical analysis
                await self._store_metrics(metrics)
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _determine_status(self, usage_percentage: float, thresholds: Dict[str, float]) -> ResourceStatus:
        """Determine resource status based on usage percentage."""
        if usage_percentage >= thresholds["critical"]:
            return ResourceStatus.CRITICAL
        elif usage_percentage >= thresholds["warning"]:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.OPTIMAL
    
    async def _optimize_memory_usage(self, results: Dict[str, Any]):
        """Optimize memory usage by cleaning up unused resources."""
        try:
            # Get current memory usage
            memory = psutil.virtual_memory()
            
            if memory.percent > self.resource_thresholds[ResourceType.MEMORY]["warning"]:
                # Clean up old cached models
                cleaned_models = await self.ml_cache._cleanup_old_models_if_needed()
                
                results["actions_taken"].append(f"Cleaned up old cached models due to high memory usage ({memory.percent:.1f}%)")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                results["actions_taken"].append("Forced garbage collection")
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    async def _optimize_cache_performance(self, results: Dict[str, Any]):
        """Optimize cache performance based on usage patterns."""
        try:
            cache_stats = await self.ml_cache.get_cache_statistics()
            
            # If hit rate is low, suggest increasing TTL
            if cache_stats.get("hit_rate", 0) < 0.5:
                results["recommendations"].append(
                    "Cache hit rate is low. Consider increasing TTL values or optimizing cache keys."
                )
            
            # If memory usage is high, clean up
            memory_usage_mb = cache_stats.get("memory_usage_mb", 0)
            if memory_usage_mb > 400:  # 400MB threshold
                await self.ml_cache._cleanup_old_models_if_needed()
                results["actions_taken"].append(f"Cleaned up cache due to high memory usage ({memory_usage_mb:.1f}MB)")
                
        except Exception as e:
            logger.error(f"Cache optimization failed: {str(e)}")
    
    async def _cleanup_unused_models(self, results: Dict[str, Any]):
        """Clean up models that haven't been used recently."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            cleaned_count = 0
            
            for model_key, usage in self.model_usage_tracking.items():
                if usage.last_accessed < cutoff_date and usage.access_count < 5:
                    # Remove from cache
                    await self.ml_cache.invalidate_model_cache(usage.model_name, usage.model_version)
                    
                    # Remove from tracking
                    del self.model_usage_tracking[model_key]
                    cleaned_count += 1
            
            if cleaned_count > 0:
                results["actions_taken"].append(f"Cleaned up {cleaned_count} unused models")
                
        except Exception as e:
            logger.error(f"Model cleanup failed: {str(e)}")
    
    async def _emergency_optimization(self, resource_type: ResourceType, metric: ResourceMetrics):
        """Perform emergency optimization for critical resource usage."""
        try:
            if resource_type == ResourceType.MEMORY:
                # Aggressive memory cleanup
                await self.ml_cache._cleanup_old_models_if_needed()
                
                # Clear prediction cache
                prediction_keys = await self.cache.keys("prediction:*")
                for key in prediction_keys[:100]:  # Clear first 100 prediction cache entries
                    await self.cache.delete(key)
                
                logger.warning(f"Emergency memory optimization: cleared {len(prediction_keys[:100])} prediction cache entries")
                
            elif resource_type == ResourceType.CACHE:
                # Clear half of the cached models (LRU)
                registry = await self.cache.get_json("ml_model_registry") or {}
                
                # Sort by last access time
                sorted_models = sorted(
                    registry.items(),
                    key=lambda x: x[1].get("last_access", 0)
                )
                
                # Remove oldest half
                models_to_remove = sorted_models[:len(sorted_models)//2]
                for model_key, model_info in models_to_remove:
                    await self.ml_cache.invalidate_model_cache(
                        model_info["model_name"], 
                        model_info["model_version"]
                    )
                
                logger.warning(f"Emergency cache optimization: removed {len(models_to_remove)} cached models")
                
        except Exception as e:
            logger.error(f"Emergency optimization failed: {str(e)}")
    
    async def _store_metrics(self, metrics: Dict[ResourceType, ResourceMetrics]):
        """Store metrics for historical analysis."""
        try:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    resource_type.value: {
                        "usage_percentage": metric.usage_percentage,
                        "status": metric.status.value,
                        "current_usage": metric.current_usage,
                        "max_usage": metric.max_usage
                    }
                    for resource_type, metric in metrics.items()
                }
            }
            
            # Store in Redis with TTL
            await self.cache.set_json(
                f"resource_metrics:{int(time.time())}", 
                metrics_data, 
                ex=86400  # Keep for 24 hours
            )
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
    
    async def _store_model_usage(self, model_key: str, usage: ModelResourceUsage):
        """Store model usage data."""
        try:
            usage_data = {
                "model_name": usage.model_name,
                "model_version": usage.model_version,
                "memory_mb": usage.memory_mb,
                "cpu_percentage": usage.cpu_percentage,
                "cache_size_mb": usage.cache_size_mb,
                "last_accessed": usage.last_accessed.isoformat(),
                "access_count": usage.access_count,
                "load_time_ms": usage.load_time_ms
            }
            
            await self.cache.set_json(f"model_usage:{model_key}", usage_data, ex=86400)
            
        except Exception as e:
            logger.error(f"Failed to store model usage: {str(e)}")
    
    async def _get_model_usage_data(self, model_name: str) -> List[ModelResourceUsage]:
        """Get usage data for a specific model."""
        try:
            usage_data = []
            
            # Get all usage keys for the model
            pattern = f"model_usage:{model_name}:*"
            keys = await self.cache.keys(pattern)
            
            for key in keys:
                data = await self.cache.get_json(key)
                if data:
                    usage = ModelResourceUsage(
                        model_name=data["model_name"],
                        model_version=data["model_version"],
                        memory_mb=data["memory_mb"],
                        cpu_percentage=data["cpu_percentage"],
                        cache_size_mb=data["cache_size_mb"],
                        last_accessed=datetime.fromisoformat(data["last_accessed"]),
                        access_count=data["access_count"],
                        load_time_ms=data["load_time_ms"]
                    )
                    usage_data.append(usage)
            
            return usage_data
            
        except Exception as e:
            logger.error(f"Failed to get model usage data: {str(e)}")
            return []
    
    async def _get_model_usage_summary(self) -> Dict[str, Any]:
        """Get summary of model usage across all models."""
        try:
            summary = {
                "total_models": 0,
                "active_models": 0,
                "total_memory_mb": 0.0,
                "total_accesses": 0,
                "most_used_models": [],
                "least_used_models": []
            }
            
            all_usage = []
            for usage in self.model_usage_tracking.values():
                all_usage.append(usage)
                summary["total_models"] += 1
                summary["total_memory_mb"] += usage.memory_mb
                summary["total_accesses"] += usage.access_count
                
                # Consider active if accessed in last 24 hours
                if (datetime.utcnow() - usage.last_accessed).total_seconds() < 86400:
                    summary["active_models"] += 1
            
            # Sort by access count
            all_usage.sort(key=lambda x: x.access_count, reverse=True)
            
            # Most used models (top 5)
            summary["most_used_models"] = [
                {
                    "model_name": usage.model_name,
                    "model_version": usage.model_version,
                    "access_count": usage.access_count,
                    "memory_mb": usage.memory_mb
                }
                for usage in all_usage[:5]
            ]
            
            # Least used models (bottom 5)
            summary["least_used_models"] = [
                {
                    "model_name": usage.model_name,
                    "model_version": usage.model_version,
                    "access_count": usage.access_count,
                    "last_accessed": usage.last_accessed.isoformat()
                }
                for usage in all_usage[-5:]
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model usage summary: {str(e)}")
            return {}
    
    async def _generate_health_recommendations(self, metrics: Dict[ResourceType, ResourceMetrics]) -> List[str]:
        """Generate health recommendations based on current metrics."""
        recommendations = []
        
        try:
            for resource_type, metric in metrics.items():
                if metric.status == ResourceStatus.CRITICAL:
                    if resource_type == ResourceType.MEMORY:
                        recommendations.append(
                            f"Critical memory usage ({metric.usage_percentage:.1f}%). "
                            f"Consider increasing system memory or reducing model cache size."
                        )
                    elif resource_type == ResourceType.CPU:
                        recommendations.append(
                            f"Critical CPU usage ({metric.usage_percentage:.1f}%). "
                            f"Consider scaling horizontally or optimizing model inference."
                        )
                    elif resource_type == ResourceType.CACHE:
                        recommendations.append(
                            f"Critical cache usage ({metric.usage_percentage:.1f}%). "
                            f"Consider increasing cache limits or implementing more aggressive cleanup."
                        )
                
                elif metric.status == ResourceStatus.WARNING:
                    recommendations.append(
                        f"{resource_type.value.title()} usage is high ({metric.usage_percentage:.1f}%). "
                        f"Monitor closely and consider optimization."
                    )
            
        except Exception as e:
            logger.error(f"Failed to generate health recommendations: {str(e)}")
        
        return recommendations


# Global resource manager instance
resource_manager = ResourceManager()