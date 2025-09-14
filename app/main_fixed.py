"""
Fixed FastAPI application for AI-powered business intelligence
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Callable

from .config import settings, validate_configuration
from .database import init_database, test_connection

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
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
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
        
        logger.info("AI Backend Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start AI Backend Service: {e}")
        # Don't raise in development mode
        if settings.ENVIRONMENT == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Backend Service...")

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

# Add middleware
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

# Import and include API routers with error handling
try:
    from .api import health
    app.include_router(health.router, tags=["health"])
    logger.info("Health router included")
except ImportError as e:
    logger.warning(f"Health module not available: {e}")

try:
    from .api import fraud
    app.include_router(fraud.router, tags=["fraud"])
    logger.info("Fraud router included")
except ImportError as e:
    logger.warning(f"Fraud module not available: {e}")

try:
    from .api import insights
    app.include_router(insights.router, tags=["insights"])
    logger.info("Insights router included")
except ImportError as e:
    logger.warning(f"Insights module not available: {e}")

try:
    from .api import compliance
    app.include_router(compliance.router, tags=["compliance"])
    logger.info("Compliance router included")
except ImportError as e:
    logger.warning(f"Compliance module not available: {e}")

try:
    from .api import invoice
    app.include_router(invoice.router, tags=["invoice"])
    logger.info("Invoice router included")
except ImportError as e:
    logger.warning(f"Invoice module not available: {e}")

try:
    from .api import ml_engine
    app.include_router(ml_engine.router, tags=["ml-engine"])
    logger.info("ML Engine router included")
except ImportError as e:
    logger.warning(f"ML Engine module not available: {e}")

try:
    from .api import notifications
    app.include_router(notifications.router, tags=["notifications"])
    logger.info("Notifications router included")
except ImportError as e:
    logger.warning(f"Notifications module not available: {e}")

try:
    from .api import whatsapp
    app.include_router(whatsapp.router, tags=["whatsapp"])
    logger.info("WhatsApp router included")
except ImportError as e:
    logger.warning(f"WhatsApp module not available: {e}")

try:
    from .api import nlp
    app.include_router(nlp.router, prefix="/api/nlp", tags=["nlp"])
    logger.info("NLP router included")
except ImportError as e:
    logger.warning(f"NLP module not available: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "FinEasy AI Backend",
        "version": settings.API_VERSION,
        "status": "healthy",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    try:
        await test_connection()
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "version": settings.API_VERSION,
            "timestamp": time.time(),
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "environment": settings.ENVIRONMENT,
            "version": settings.API_VERSION,
            "timestamp": time.time(),
            "database": f"error: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main_fixed:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )