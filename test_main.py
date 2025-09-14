#!/usr/bin/env python3
"""
Simplified main.py for testing router imports
"""
from fastapi import FastAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create simple FastAPI app
app = FastAPI(
    title="FinEasy AI Backend - Test",
    description="Test version to debug router issues",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Test individual router imports
print("Testing router imports...")

try:
    from app.api.health import router as health_router
    app.include_router(health_router, tags=["health"])
    print("✅ Health router included")
except Exception as e:
    print(f"❌ Health router failed: {e}")

try:
    from app.api.fraud import router as fraud_router
    app.include_router(fraud_router, tags=["fraud"])
    print("✅ Fraud router included")
except Exception as e:
    print(f"❌ Fraud router failed: {e}")

try:
    from app.api.insights import router as insights_router
    app.include_router(insights_router, tags=["insights"])
    print("✅ Insights router included")
except Exception as e:
    print(f"❌ Insights router failed: {e}")

@app.get("/")
async def root():
    return {"message": "Test FinEasy AI Backend", "status": "healthy"}

@app.get("/test")
async def test():
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({"methods": list(route.methods), "path": route.path})
    return {"routes": routes, "total": len(routes)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)