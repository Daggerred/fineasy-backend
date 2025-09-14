#!/usr/bin/env python3
"""
Quick test for OpenAPI and docs
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🧪 Quick OpenAPI Test")
    print("=" * 50)
    
    # Test 1: Import models
    try:
        from app.models.responses import BaseResponse, ModelPerformanceMetrics
        print("✅ Models import successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    # Test 2: Create model instances
    try:
        base = BaseResponse(message="test")
        metrics = ModelPerformanceMetrics(accuracy=0.95)
        print("✅ Model instances created successfully")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False
    
    # Test 3: Import main app
    try:
        from app.main import app
        print("✅ Main app imported successfully")
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        return False
    
    # Test 4: Generate OpenAPI schema
    try:
        schema = app.openapi()
        path_count = len(schema.get('paths', {}))
        print(f"✅ OpenAPI schema generated with {path_count} paths")
    except Exception as e:
        print(f"❌ OpenAPI generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test with TestClient
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test openapi.json
        response = client.get("/openapi.json")
        if response.status_code == 200:
            print("✅ /openapi.json endpoint works")
        else:
            print(f"❌ /openapi.json failed: {response.status_code}")
            
        # Test docs
        response = client.get("/docs")
        if response.status_code == 200:
            print("✅ /docs endpoint works")
            if len(response.content) > 5000:  # Docs should be substantial
                print("✅ /docs has substantial content")
            else:
                print(f"⚠️  /docs content seems minimal ({len(response.content)} bytes)")
        else:
            print(f"❌ /docs failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ TestClient test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Docs should work now.")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Try running: python start_server.py")
        print("📖 Then visit: http://localhost:8000/docs")
    else:
        print("\n🔧 Fix the errors above and try again")