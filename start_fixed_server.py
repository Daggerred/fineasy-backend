#!/usr/bin/env python3
"""
Start the fixed FastAPI server
"""
import sys
import os
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting FinEasy AI Backend Server...")
    print("Docs will be available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    
    # Start the server using the fixed main module
    uvicorn.run(
        "app.main_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )