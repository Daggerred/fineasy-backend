#!/usr/bin/env python3
"""
Minimal FastAPI app to test docs generation
"""
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List

# Create minimal app
app = FastAPI(
    title="Minimal Test App",
    description="Test app for debugging docs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Simple models
class SimpleResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = datetime.utcnow()

class TestData(BaseModel):
    id: str
    name: str
    value: float

@app.get("/")
async def root():
    return {"message": "Minimal test app"}

@app.get("/test", response_model=SimpleResponse)
async def test_endpoint():
    return SimpleResponse(message="Test successful")

@app.post("/data", response_model=SimpleResponse)
async def create_data(data: TestData):
    return SimpleResponse(message=f"Created {data.name}")

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal app...")
    print("Visit: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)