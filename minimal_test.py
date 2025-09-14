#!/usr/bin/env python3
"""
Minimal FastAPI app to test docs functionality
"""
from fastapi import FastAPI
import uvicorn

# Create minimal app
app = FastAPI(
    title="FinEasy AI Backend",
    description="AI-powered business intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/")
async def root():
    return {"message": "FinEasy AI Backend", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/test")
async def test():
    return {"message": "Test endpoint", "docs": "Available at /docs"}

if __name__ == "__main__":
    print("Starting minimal test server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)