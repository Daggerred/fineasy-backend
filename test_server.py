#!/usr/bin/env python3
"""
Simple test script to check FastAPI server and docs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Create a minimal FastAPI app for testing
app = FastAPI(
    title="FinEasy AI Backend Test",
    description="Test server for debugging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/")
async def root():
    return {"message": "Test server is running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Test server health check"}

@app.get("/test")
async def test_endpoint():
    return {
        "message": "Test endpoint working",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    }

if __name__ == "__main__":
    print("Starting test server on http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)