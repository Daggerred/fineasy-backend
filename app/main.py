"""
FastAPI application for AI-powered business intelligence
"""
# TO:DO Uvicorn to be fixed and merged with Dockerfile and deployment scripts
from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
try:
    from fastapi.middleware.base import BaseHTTPMiddleware
except ImportError:
    from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

from .config import settings, validate_configuration
from .database import init_database, test_connection

# Just added this here to fix logging error 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utilities with error handling
try:
    from .utils.auth import AuthMiddleware
except ImportError:
    AuthMiddleware = None

try:
    from .utils.security_middleware import security_middleware
except ImportError:
    security_middleware = None

try:
    from .utils.audit_logger import audit_logger, AuditEventType, AuditSeverity
except ImportError:
    audit_logger = None
    AuditEventType = None
    AuditSeverity = None

try:
    from .utils.data_retention import get_retention_manager
except ImportError:
    get_retention_manager = None

# Import API modules with detailed error handling
print("Starting API module imports...")

# Import individual modules with detailed logging
try:
    from .api import health
    print("✅ Health module imported successfully")
except ImportError as e:
    print(f"❌ Health module import failed: {e}")
    health = None

try:
    from .api import fraud
    print("✅ Fraud module imported successfully")
except ImportError as e:
    print(f"❌ Fraud module import failed: {e}")
    fraud = None

try:
    from .api import insights
    print("✅ Insights module imported successfully")
except ImportError as e:
    print(f"❌ Insights module import failed: {e}")
    insights = None

try:
    from .api import compliance
    print("✅ Compliance module imported successfully")
except ImportError as e:
    print(f"❌ Compliance module import failed: {e}")
    compliance = None

try:
    from .api import invoice
    print("✅ Invoice module imported successfully")
except ImportError as e:
    print(f"❌ Invoice module import failed: {e}")
    invoice = None

try:
    from .api import ml_engine
    print("✅ ML Engine module imported successfully")
except ImportError as e:
    print(f"❌ ML Engine module import failed: {e}")
    ml_engine = None

try:
    from .api import notifications
    print("✅ Notifications module imported successfully")
except ImportError as e:
    print(f"❌ Notifications module import failed: {e}")
    notifications = None

try:
    from .api import cache_management
    print("✅ Cache Management module imported successfully")
except ImportError as e:
    print(f"❌ Cache Management module import failed: {e}")
    cache_management = None

try:
    from .api import feature_flags
    print("✅ Feature Flags module imported successfully")
except ImportError as e:
    print(f"❌ Feature Flags module import failed: {e}")
    feature_flags = None

print("API module imports completed.")


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with comprehensive protection"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            # Skip security checks for health endpoints
            if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
                response = await call_next(request)
                self._add_security_headers(response)
                return response
            
            # Extract business_id from request if available
            business_id = None
            if hasattr(request, 'path_params') and 'business_id' in request.path_params:
                business_id = request.path_params['business_id']
            elif request.method == "POST" and hasattr(request, '_body'):
                # Try to extract from request body (this is a simplified approach)
                try:
                    import json
                    body = await request.body()
                    if body:
                        data = json.loads(body)
                        business_id = data.get('business_id')
                except:
                    pass
            
            # Determine operation type from path
            operation_type = self._determine_operation_type(request.url.path)
            
            # Apply security validation for AI endpoints
            if operation_type and business_id and security_middleware:
                try:
                    security_context = await security_middleware.validate_request(
                        request, business_id, operation_type
                    )
                    
                    # Log security validation
                    if audit_logger and AuditEventType and AuditSeverity:
                        audit_logger.log_ai_operation(
                            event_type=AuditEventType.DATA_ACCESS,
                            business_id=business_id,
                            operation_details={
                                "endpoint": request.url.path,
                                "method": request.method,
                                "operation_type": operation_type,
                                "security_score": security_context.get("security_score")
                            },
                            severity=AuditSeverity.LOW,
                            ip_address=security_context.get("client_ip"),
                            user_agent=security_context.get("user_agent")
                        )
                    
                except HTTPException as e:
                    # Security validation failed
                    if audit_logger and AuditEventType and AuditSeverity:
                        audit_logger.log_ai_operation(
                            event_type=AuditEventType.SECURITY_VIOLATION,
                            business_id=business_id or "unknown",
                            operation_details={
                                "endpoint": request.url.path,
                                "method": request.method,
                                "error": str(e.detail),
                                "status_code": e.status_code
                            },
                            severity=AuditSeverity.HIGH,
                            ip_address=request.client.host if request.client else None
                        )
                    raise
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log successful request
            process_time = time.time() - start_time
            if operation_type and business_id and audit_logger and AuditEventType and AuditSeverity:
                audit_logger.log_ai_operation(
                    event_type=AuditEventType.AI_PROCESSING_END,
                    business_id=business_id,
                    operation_details={
                        "endpoint": request.url.path,
                        "method": request.method,
                        "status_code": response.status_code,
                        "processing_time_seconds": process_time
                    },
                    severity=AuditSeverity.LOW
                )
            
            return response
            
        except Exception as e:
            # Log error
            if audit_logger and AuditEventType and AuditSeverity:
                audit_logger.log_ai_operation(
                    event_type=AuditEventType.ERROR_OCCURRED,
                    business_id=business_id or "unknown",
                    operation_details={
                        "endpoint": request.url.path,
                        "method": request.method,
                        "error": str(e),
                        "processing_time_seconds": time.time() - start_time
                    },
                    severity=AuditSeverity.HIGH,
                    ip_address=request.client.host if request.client else None
                )
            raise
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    
    def _determine_operation_type(self, path: str) -> str:
        """Determine operation type from request path"""
        if "/fraud" in path:
            return "fraud_detection"
        elif "/insights" in path:
            return "predictive_analysis"
        elif "/compliance" in path:
            return "compliance_check"
        elif "/invoice" in path and "nlp" in path:
            return "nlp_processing"
        elif "/ml" in path:
            return "model_operation"
        return None


class BackgroundTaskManager:
    """Manager for background tasks and analytics processing"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running_tasks = set()
    
    async def start(self):
        """Start background task manager"""
        logger.info("Starting background task manager...")
        # Start periodic tasks
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_analytics_refresh())
        asyncio.create_task(self._start_notification_scheduler())
        asyncio.create_task(self._periodic_data_retention_cleanup())
        asyncio.create_task(self._start_resource_monitoring())
        asyncio.create_task(self._setup_cache_invalidation_rules())
    
    async def stop(self):
        """Stop background task manager"""
        logger.info("Stopping background task manager...")
        self.executor.shutdown(wait=True)
    
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                from .utils.cache import cache
                # Clean up expired cache entries
                logger.info("Running periodic cache cleanup")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _periodic_analytics_refresh(self):
        """Periodic analytics refresh for active businesses"""
        while True:
            try:
                await asyncio.sleep(21600)  # Run every 6 hours
                logger.info("Running periodic analytics refresh")
                # This would refresh analytics for active businesses
            except Exception as e:
                logger.error(f"Analytics refresh error: {e}")
    
    async def _start_notification_scheduler(self):
        """Start the smart notification scheduler"""
        try:
            from .services.smart_notifications import NotificationScheduler
            scheduler = NotificationScheduler()
            logger.info("Starting notification scheduler...")
            await scheduler.start()
        except ImportError:
            logger.warning("Smart notifications service not available")
        except Exception as e:
            logger.error(f"Notification scheduler error: {e}")
    
    async def _periodic_data_retention_cleanup(self):
        """Periodic data retention cleanup task"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                logger.info("Running periodic data retention cleanup")
                
                # Run retention cleanup if available
                if get_retention_manager:
                    retention_manager = get_retention_manager()
                    cleanup_stats = await retention_manager.run_retention_cleanup()
                    
                    # Log cleanup results
                    if audit_logger and AuditEventType and AuditSeverity:
                        audit_logger.log_ai_operation(
                            event_type=AuditEventType.AI_PROCESSING_END,
                            business_id="system",
                            operation_details={
                                "operation": "data_retention_cleanup",
                                "cleanup_stats": cleanup_stats
                            },
                            severity=AuditSeverity.LOW
                        )
                    
                    logger.info(f"Data retention cleanup completed: {cleanup_stats}")
                else:
                    logger.warning("Data retention manager not available")
                
            except Exception as e:
                logger.error(f"Data retention cleanup error: {e}")
                if audit_logger and AuditEventType and AuditSeverity:
                    audit_logger.log_ai_operation(
                        event_type=AuditEventType.ERROR_OCCURRED,
                        business_id="system",
                        operation_details={
                            "operation": "data_retention_cleanup",
                            "error": str(e)
                        },
                        severity=AuditSeverity.HIGH
                    )
    
    async def _start_resource_monitoring(self):
        """Start resource monitoring for cache and ML models"""
        try:
            from .utils.resource_manager import resource_manager
            logger.info("Starting resource monitoring...")
            await resource_manager.start_monitoring()
        except ImportError:
            logger.warning("Resource manager not available")
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
    
    async def _setup_cache_invalidation_rules(self):
        """Set up default cache invalidation rules"""
        try:
            from .utils.cache import cache_invalidation_manager
            logger.info("Setting up cache invalidation rules...")
            await cache_invalidation_manager.setup_default_rules()
        except ImportError:
            logger.warning("Cache invalidation manager not available")
        except Exception as e:
            logger.error(f"Cache invalidation setup error: {e}")


# Global background task manager
background_manager = BackgroundTaskManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AI Backend Service...")
    
    try:
        # Validate configuration
        validate_configuration()
        logger.info("Configuration validated")
        
        # Initialize database
        await init_database()
        logger.info("Database initialized")
        
        # Test database connection
        await test_connection()
        logger.info("Database connection verified")
        
        # Initialize retention manager
        global retention_manager
        if get_retention_manager:
            retention_manager = get_retention_manager()
            logger.info("Data retention manager initialized")
        else:
            retention_manager = None
            logger.warning("Data retention manager not available")
        
        # Start background task manager
        await background_manager.start()
        logger.info("Background task manager started")
        
        logger.info("AI Backend Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start AI Backend Service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Backend Service...")
    await background_manager.stop()
    logger.info("Background task manager stopped")


# Create FastAPI app
app = FastAPI(
    title="FinEasy AI Backend",
    description="AI-powered business intelligence for FinEasy",
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware (order matters!)
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include API routers with detailed logging
print("Starting router inclusion...")

if health:
    try:
        app.include_router(health.router, tags=["health"])
        print("✅ Health router included")
    except Exception as e:
        print(f"❌ Health router inclusion failed: {e}")
else:
    print("⚠️  Health module not available")

if fraud:
    try:
        app.include_router(fraud.router, tags=["fraud"])
        print("✅ Fraud router included")
    except Exception as e:
        print(f"❌ Fraud router inclusion failed: {e}")
else:
    print("⚠️  Fraud module not available")

if insights:
    try:
        app.include_router(insights.router, tags=["insights"])
        print("✅ Insights router included")
    except Exception as e:
        print(f"❌ Insights router inclusion failed: {e}")
else:
    print("⚠️  Insights module not available")

if compliance:
    try:
        app.include_router(compliance.router, tags=["compliance"])
        print("✅ Compliance router included")
    except Exception as e:
        print(f"❌ Compliance router inclusion failed: {e}")
else:
    print("⚠️  Compliance module not available")

if invoice:
    try:
        app.include_router(invoice.router, tags=["invoice"])
        print("✅ Invoice router included")
    except Exception as e:
        print(f"❌ Invoice router inclusion failed: {e}")
else:
    print("⚠️  Invoice module not available")

if ml_engine:
    try:
        app.include_router(ml_engine.router, tags=["ml-engine"])
        print("✅ ML Engine router included")
    except Exception as e:
        print(f"❌ ML Engine router inclusion failed: {e}")
else:
    print("⚠️  ML Engine module not available")

print("Router inclusion completed.")

# Log final route count
print(f"Total routes registered: {len(app.routes)}")
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        print(f"  {list(route.methods)} {route.path}")

if notifications:
    app.include_router(notifications.router, tags=["notifications"])

if cache_management:
    app.include_router(cache_management.router, tags=["cache"])

if feature_flags:
    app.include_router(feature_flags.router, tags=["feature-flags"])

try:
    from .api import models
    app.include_router(models.router, tags=["models"])
except ImportError:
    logger.warning("Models API module could not be imported")

# Add NLP router
try:
    from .api import nlp
    app.include_router(nlp.router, prefix="/api/nlp", tags=["nlp"])
    logger.info("NLP API router loaded successfully")
except ImportError:
    logger.warning("NLP API module could not be imported")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "FinEasy AI Backend",
        "version": settings.API_VERSION,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check with detailed status"""
    health_status = {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION,
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Database health check
    try:
        await test_connection()
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        health_status["status"] = "unhealthy"
    
    # Redis health check
    try:
        from .utils.cache import cache
        if hasattr(cache, 'redis_client') and cache.redis_client:
            await cache.redis_client.ping()
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "message": "Redis connection successful"
            }
        else:
            health_status["checks"]["redis"] = {
                "status": "warning",
                "message": "Redis not configured"
            }
    except ImportError:
        health_status["checks"]["redis"] = {
            "status": "warning",
            "message": "Redis cache module not available"
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}"
        }
    
    # ML Models health check
    try:
        from .services.ml_engine import ml_engine
        model_status = await ml_engine.get_model_health()
        health_status["checks"]["ml_models"] = {
            "status": "healthy" if model_status["loaded_models"] > 0 else "warning",
            "message": f"{model_status['loaded_models']} models loaded",
            "details": model_status
        }
    except ImportError:
        health_status["checks"]["ml_models"] = {
            "status": "warning",
            "message": "ML engine not available"
        }
    except Exception as e:
        logger.error(f"ML models health check failed: {e}")
        health_status["checks"]["ml_models"] = {
            "status": "warning",
            "message": f"ML models check failed: {str(e)}"
        }
    
    # External services health check
    external_services = {}
    
    # GST API check (if enabled)
    if settings.COMPLIANCE_CHECKING_ENABLED and hasattr(settings, 'GST_API_URL'):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.GST_API_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        external_services["gst_api"] = {
                            "status": "healthy",
                            "message": "GST API accessible"
                        }
                    else:
                        external_services["gst_api"] = {
                            "status": "warning",
                            "message": f"GST API returned status {response.status}"
                        }
        except Exception as e:
            external_services["gst_api"] = {
                "status": "unhealthy",
                "message": f"GST API unreachable: {str(e)}"
            }
    
    health_status["checks"]["external_services"] = external_services
    
    # Feature availability
    health_status["features"] = {
        "fraud_detection": settings.FRAUD_DETECTION_ENABLED and health_status["checks"]["database"]["status"] == "healthy",
        "predictive_analytics": settings.PREDICTIVE_ANALYTICS_ENABLED and health_status["checks"]["database"]["status"] == "healthy",
        "compliance_checking": settings.COMPLIANCE_CHECKING_ENABLED and health_status["checks"]["database"]["status"] == "healthy",
        "nlp_invoice": settings.NLP_INVOICE_ENABLED and health_status["checks"]["database"]["status"] == "healthy"
    }
    
    # Overall status determination
    unhealthy_checks = [
        check for check in health_status["checks"].values() 
        if isinstance(check, dict) and check.get("status") == "unhealthy"
    ]
    
    if unhealthy_checks:
        health_status["status"] = "unhealthy"
    elif any(
        check.get("status") == "warning" 
        for check in health_status["checks"].values() 
        if isinstance(check, dict)
    ):
        health_status["status"] = "degraded"
    
    return health_status


@app.get(f"/api/{settings.API_VERSION}/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_version": settings.API_VERSION,
        "timestamp": time.time(),
        "uptime": "healthy"
    }


@app.get(f"/api/{settings.API_VERSION}/analytics/performance")
async def analytics_performance():
    """Analytics performance metrics endpoint"""
    try:
        from .utils.cache import cache
        
        # Get cache statistics
        cache_stats = {
            "cache_enabled": cache.redis_client is not None,
            "background_tasks_running": len(background_manager.running_tasks),
            "executor_threads": background_manager.executor._threads if hasattr(background_manager.executor, '_threads') else 0
        }
        
        return {
            "status": "healthy",
            "cache_stats": cache_stats,
            "performance_metrics": {
                "avg_response_time_ms": 150,  # This would be tracked in production
                "cache_hit_rate": 0.85,      # This would be calculated from actual metrics
                "prediction_accuracy": 0.82   # This would be tracked from validation
            }
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get(f"/api/{settings.API_VERSION}/health/detailed")
async def detailed_health_check():
    """Detailed health check with component-level status"""
    try:
        health_details = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": {},
            "error_rates": {},
            "performance_metrics": {}
        }
        
        # Database component
        try:
            await test_connection()
            health_details["components"]["database"] = {
                "status": "healthy",
                "response_time_ms": 10,  # This would be measured
                "last_error": None
            }
        except Exception as e:
            health_details["components"]["database"] = {
                "status": "unhealthy",
                "response_time_ms": None,
                "last_error": str(e),
                "error_time": time.time()
            }
            health_details["overall_status"] = "unhealthy"
        
        # Cache component
        try:
            from .utils.cache import cache
            if cache.redis_client:
                await cache.redis_client.ping()
                health_details["components"]["cache"] = {
                    "status": "healthy",
                    "response_time_ms": 5,
                    "last_error": None
                }
            else:
                health_details["components"]["cache"] = {
                    "status": "disabled",
                    "response_time_ms": None,
                    "last_error": "Redis not configured"
                }
        except Exception as e:
            health_details["components"]["cache"] = {
                "status": "unhealthy",
                "response_time_ms": None,
                "last_error": str(e),
                "error_time": time.time()
            }
        
        # AI Services component
        try:
            from .services.ml_engine import ml_engine
            model_health = await ml_engine.get_model_health()
            health_details["components"]["ai_services"] = {
                "status": "healthy" if model_health["loaded_models"] > 0 else "degraded",
                "loaded_models": model_health["loaded_models"],
                "model_details": model_health.get("models", {}),
                "last_error": None
            }
        except Exception as e:
            health_details["components"]["ai_services"] = {
                "status": "unhealthy",
                "loaded_models": 0,
                "last_error": str(e),
                "error_time": time.time()
            }
        
        # Error rates (would be tracked from actual metrics in production)
        health_details["error_rates"] = {
            "fraud_detection": 0.02,
            "predictive_analytics": 0.01,
            "compliance_checking": 0.03,
            "nlp_invoice": 0.05
        }
        
        # Performance metrics
        health_details["performance_metrics"] = {
            "avg_response_time_ms": 150,
            "p95_response_time_ms": 300,
            "requests_per_minute": 45,
            "cache_hit_rate": 0.85,
            "model_accuracy": {
                "fraud_detection": 0.92,
                "predictive_analytics": 0.78,
                "compliance_checking": 0.95
            }
        }
        
        return health_details
        
    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        return {
            "overall_status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


@app.post(f"/api/{settings.API_VERSION}/errors/report")
async def report_error(request: Request):
    """Endpoint for clients to report errors for monitoring"""
    try:
        error_data = await request.json()
        
        # Log the client error
        logger.error(f"Client error reported: {error_data}")
        
        # In production, this would be sent to monitoring system
        if audit_logger and AuditEventType and AuditSeverity:
            audit_logger.log_ai_operation(
                event_type=AuditEventType.ERROR_OCCURRED,
                business_id=error_data.get("business_id", "unknown"),
                operation_details={
                    "client_error": error_data,
                    "user_agent": request.headers.get("user-agent"),
                    "timestamp": time.time()
                },
                severity=AuditSeverity.MEDIUM,
                ip_address=request.client.host if request.client else None
            )
        
        return {
            "status": "received",
            "message": "Error report received and logged"
        }
        
    except Exception as e:
        logger.error(f"Error reporting endpoint failed: {e}")
        return {
            "status": "error",
            "message": "Failed to process error report"
        }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )