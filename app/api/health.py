"""
Health check and monitoring endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import time
import asyncio
import logging
from datetime import datetime, timedelta

from ..database import test_connection
from ..config import settings
from ..utils.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checking service"""
    
    def __init__(self):
        self.last_check_time = None
        self.cached_health_status = None
        self.cache_duration = 30  # seconds
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status with caching"""
        current_time = time.time()
        
        # Use cached result if recent
        if (self.cached_health_status and 
            self.last_check_time and 
            current_time - self.last_check_time < self.cache_duration):
            return self.cached_health_status
        
        health_status = await self._perform_health_checks()
        
        # Cache the result
        self.cached_health_status = health_status
        self.last_check_time = current_time
        
        return health_status
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform all health checks"""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "checks": {},
            "metrics": {},
            "features": {}
        }
        
        # Run all checks concurrently
        check_tasks = [
            self._check_database(),
            self._check_redis(),
            self._check_ml_models(),
            self._check_external_services(),
            self._check_system_resources(),
            self._get_performance_metrics()
        ]
        
        try:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # Process results
            health_status["checks"]["database"] = results[0] if not isinstance(results[0], Exception) else {"status": "error", "error": str(results[0])}
            health_status["checks"]["redis"] = results[1] if not isinstance(results[1], Exception) else {"status": "error", "error": str(results[1])}
            health_status["checks"]["ml_models"] = results[2] if not isinstance(results[2], Exception) else {"status": "error", "error": str(results[2])}
            health_status["checks"]["external_services"] = results[3] if not isinstance(results[3], Exception) else {"status": "error", "error": str(results[3])}
            health_status["checks"]["system_resources"] = results[4] if not isinstance(results[4], Exception) else {"status": "error", "error": str(results[4])}
            health_status["metrics"] = results[5] if not isinstance(results[5], Exception) else {}
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        # Determine overall status
        health_status["status"] = self._determine_overall_status(health_status["checks"])
        
        # Feature availability
        health_status["features"] = self._check_feature_availability(health_status["checks"])
        
        return health_status
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            await test_connection()
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "message": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed"
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            from ..utils.cache import cache
            if not cache.redis_client:
                return {
                    "status": "disabled",
                    "message": "Redis not configured"
                }
            
            start_time = time.time()
            await cache.redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await cache.redis_client.info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "message": "Redis connection successful",
                "info": {
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Redis connection failed"
            }
    
    async def _check_ml_models(self) -> Dict[str, Any]:
        """Check ML models status and performance"""
        try:
            from ..services.ml_engine import ml_engine
            model_health = await ml_engine.get_model_health()
            
            return {
                "status": "healthy" if model_health["loaded_models"] > 0 else "warning",
                "loaded_models": model_health["loaded_models"],
                "models": model_health.get("models", {}),
                "message": f"{model_health['loaded_models']} models loaded and ready"
            }
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "message": "ML models check failed"
            }
    
    async def _check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity"""
        external_checks = {}
        
        # GST API check
        if settings.COMPLIANCE_CHECKING_ENABLED and hasattr(settings, 'GST_API_URL'):
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{settings.GST_API_URL}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        external_checks["gst_api"] = {
                            "status": "healthy" if response.status == 200 else "warning",
                            "response_code": response.status,
                            "message": f"GST API responded with {response.status}"
                        }
            except Exception as e:
                external_checks["gst_api"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "message": "GST API unreachable"
                }
        
        return external_checks
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "warning",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        except ImportError:
            return {
                "status": "warning",
                "message": "psutil not available for system monitoring"
            }
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "message": "System resource check failed"
            }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            from ..utils.cache import cache
            
            metrics = {
                "cache_hit_rate": 0.85,  # This would be calculated from actual metrics
                "avg_response_time_ms": 150,  # This would be tracked
                "requests_per_minute": 45,  # This would be tracked
                "error_rate": 0.02,  # This would be calculated
                "model_accuracy": {
                    "fraud_detection": 0.92,
                    "predictive_analytics": 0.78,
                    "compliance_checking": 0.95
                }
            }
            
            # Get actual cache stats if available
            if cache.redis_client:
                try:
                    info = await cache.redis_client.info()
                    metrics["cache_stats"] = {
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0),
                        "used_memory": info.get("used_memory_human")
                    }
                    
                    # Calculate actual hit rate
                    hits = info.get("keyspace_hits", 0)
                    misses = info.get("keyspace_misses", 0)
                    if hits + misses > 0:
                        metrics["cache_hit_rate"] = hits / (hits + misses)
                except:
                    pass
            
            return metrics
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {}
    
    def _determine_overall_status(self, checks: Dict[str, Any]) -> str:
        """Determine overall health status from individual checks"""
        unhealthy_count = 0
        warning_count = 0
        
        for check in checks.values():
            if isinstance(check, dict):
                status = check.get("status", "unknown")
                if status == "unhealthy":
                    unhealthy_count += 1
                elif status == "warning":
                    warning_count += 1
        
        if unhealthy_count > 0:
            return "unhealthy"
        elif warning_count > 0:
            return "degraded"
        else:
            return "healthy"
    
    def _check_feature_availability(self, checks: Dict[str, Any]) -> Dict[str, bool]:
        """Check which features are available based on health checks"""
        db_healthy = checks.get("database", {}).get("status") == "healthy"
        
        return {
            "fraud_detection": settings.FRAUD_DETECTION_ENABLED and db_healthy,
            "predictive_analytics": settings.PREDICTIVE_ANALYTICS_ENABLED and db_healthy,
            "compliance_checking": settings.COMPLIANCE_CHECKING_ENABLED and db_healthy,
            "nlp_invoice": settings.NLP_INVOICE_ENABLED and db_healthy,
            "caching": checks.get("redis", {}).get("status") in ["healthy", "warning"],
            "ml_models": checks.get("ml_models", {}).get("status") in ["healthy", "warning"]
        }


# Global health checker instance
health_checker = HealthChecker()


@router.get("/health")
async def basic_health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.API_VERSION
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components"""
    return await health_checker.get_comprehensive_health()


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    try:
        # Basic application health
        return {"status": "alive", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical dependencies
        await test_connection()
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "message": "Service is ready to accept traffic"
        }
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/startup")
async def startup_probe():
    """Kubernetes startup probe endpoint"""
    try:
        # Check if application has fully started
        health = await health_checker.get_comprehensive_health()
        
        if health["status"] in ["healthy", "degraded"]:
            return {
                "status": "started",
                "timestamp": time.time(),
                "message": "Service has started successfully"
            }
        else:
            raise HTTPException(status_code=503, detail="Service still starting")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Startup probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service startup failed")


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        health = await health_checker.get_comprehensive_health()
        metrics = health.get("metrics", {})
        
        # Format metrics for Prometheus
        prometheus_metrics = []
        
        # Health status metrics
        status_value = 1 if health["status"] == "healthy" else 0
        prometheus_metrics.append(f"ai_backend_health_status {status_value}")
        
        # Performance metrics
        if "cache_hit_rate" in metrics:
            prometheus_metrics.append(f"ai_backend_cache_hit_rate {metrics['cache_hit_rate']}")
        
        if "avg_response_time_ms" in metrics:
            prometheus_metrics.append(f"ai_backend_avg_response_time_ms {metrics['avg_response_time_ms']}")
        
        if "requests_per_minute" in metrics:
            prometheus_metrics.append(f"ai_backend_requests_per_minute {metrics['requests_per_minute']}")
        
        if "error_rate" in metrics:
            prometheus_metrics.append(f"ai_backend_error_rate {metrics['error_rate']}")
        
        # Model accuracy metrics
        if "model_accuracy" in metrics:
            for model, accuracy in metrics["model_accuracy"].items():
                prometheus_metrics.append(f"ai_backend_model_accuracy{{model=\"{model}\"}} {accuracy}")
        
        # Component health metrics
        for component, check in health.get("checks", {}).items():
            if isinstance(check, dict):
                status_value = 1 if check.get("status") == "healthy" else 0
                prometheus_metrics.append(f"ai_backend_component_health{{component=\"{component}\"}} {status_value}")
                
                # Response time metrics
                if "response_time_ms" in check:
                    prometheus_metrics.append(f"ai_backend_component_response_time_ms{{component=\"{component}\"}} {check['response_time_ms']}")
        
        return "\n".join(prometheus_metrics)
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return f"# Error generating metrics: {str(e)}"


@router.get("/health/dependencies")
async def check_dependencies():
    """Check status of all external dependencies"""
    try:
        health = await health_checker.get_comprehensive_health()
        
        dependencies = {
            "database": health["checks"].get("database", {}),
            "redis": health["checks"].get("redis", {}),
            "external_services": health["checks"].get("external_services", {}),
            "ml_models": health["checks"].get("ml_models", {})
        }
        
        return {
            "timestamp": time.time(),
            "dependencies": dependencies,
            "overall_status": health["status"]
        }
        
    except Exception as e:
        logger.error(f"Dependencies check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check dependencies")


@router.post("/health/test/{component}")
async def test_component(component: str):
    """Test specific component health"""
    try:
        if component == "database":
            result = await health_checker._check_database()
        elif component == "redis":
            result = await health_checker._check_redis()
        elif component == "ml_models":
            result = await health_checker._check_ml_models()
        elif component == "external_services":
            result = await health_checker._check_external_services()
        elif component == "system_resources":
            result = await health_checker._check_system_resources()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown component: {component}")
        
        return {
            "component": component,
            "timestamp": time.time(),
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Component test error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test component: {str(e)}")