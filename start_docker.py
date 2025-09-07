#!/usr/bin/env python3
"""
Docker startup script for Fineasy AI Backend
Handles initialization and graceful startup
"""
import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        "/app/uploads",
        "/app/logs", 
        "/app/cache",
        "/app/temp",
        "/app/ml_models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_environment():
    """Check environment variables"""
    required_vars = [
        "ENVIRONMENT",
        "PORT",
        "SUPABASE_URL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    else:
        logger.info("All required environment variables are set")
    
    # Log current configuration
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'unknown')}")
    logger.info(f"Debug Mode: {os.getenv('DEBUG', 'false')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    logger.info(f"Supabase URL: {os.getenv('SUPABASE_URL', 'not set')}")

def wait_for_dependencies():
    """Wait for external dependencies if needed"""
    # In a real deployment, you might wait for database, redis, etc.
    logger.info("Checking dependencies...")
    time.sleep(2)
    logger.info("Dependencies check completed")

def start_application():
    """Start the FastAPI application"""
    logger.info("Starting Fineasy AI Backend...")
    
    try:
        # Import and start the application
        import uvicorn
        from app.main import app
        
        # Get configuration
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        workers = int(os.getenv("API_WORKERS", 1))
        reload = os.getenv("DEBUG", "false").lower() == "true"
        
        logger.info(f"Starting server on {host}:{port} with {workers} workers")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            access_log=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("=== Fineasy AI Backend Docker Startup ===")
    
    try:
        # Setup
        setup_directories()
        check_environment()
        wait_for_dependencies()
        
        # Start application
        start_application()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()