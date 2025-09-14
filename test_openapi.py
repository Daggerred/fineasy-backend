#!/usr/bin/env python3
"""
Test OpenAPI schema generation
"""
import sys
import os
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_openapi():
    print("Testing OpenAPI schema generation...")
    
    try:
        from app.main import app
        print("✅ App imported successfully")
        
        # Try to generate OpenAPI schema
        openapi_schema = app.openapi()
        print("✅ OpenAPI schema generated successfully")
        
        # Check if schema has paths
        if 'paths' in openapi_schema:
            path_count = len(openapi_schema['paths'])
            print(f"✅ OpenAPI schema has {path_count} paths")
            
            # Show some paths
            for i, path in enumerate(list(openapi_schema['paths'].keys())[:5]):
                print(f"  - {path}")
            if path_count > 5:
                print(f"  ... and {path_count - 5} more")
        else:
            print("❌ No paths found in OpenAPI schema")
            
        # Check for components/schemas
        if 'components' in openapi_schema and 'schemas' in openapi_schema['components']:
            schema_count = len(openapi_schema['components']['schemas'])
            print(f"✅ OpenAPI has {schema_count} component schemas")
        else:
            print("⚠️  No component schemas found")
            
        return True
        
    except Exception as e:
        print(f"❌ OpenAPI schema generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docs_endpoint():
    print("\nTesting docs endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test OpenAPI JSON endpoint
        response = client.get("/openapi.json")
        print(f"OpenAPI JSON status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ OpenAPI JSON endpoint works")
            data = response.json()
            if 'paths' in data:
                print(f"✅ OpenAPI JSON has {len(data['paths'])} paths")
            else:
                print("❌ OpenAPI JSON missing paths")
        else:
            print(f"❌ OpenAPI JSON failed: {response.text}")
            
        # Test docs endpoint
        response = client.get("/docs")
        print(f"Docs page status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Docs endpoint accessible")
            if len(response.content) > 1000:  # Docs page should be substantial
                print("✅ Docs page has content")
            else:
                print("⚠️  Docs page seems empty or minimal")
        else:
            print(f"❌ Docs endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Docs endpoint test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = test_openapi()
    if success:
        test_docs_endpoint()
    else:
        print("\nSkipping docs test due to OpenAPI generation failure")