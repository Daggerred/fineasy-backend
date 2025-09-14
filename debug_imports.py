#!/usr/bin/env python3
"""
Debug script to check import issues
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    
    # Test basic imports
    try:
        from app.main import app
        print("✅ Main app import successful")
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        return
    
    # Test individual API module imports
    modules = ['health', 'fraud', 'insights', 'compliance', 'invoice', 'ml_engine']
    
    for module_name in modules:
        try:
            module = __import__(f'app.api.{module_name}', fromlist=[module_name])
            router = getattr(module, 'router', None)
            if router:
                print(f"✅ {module_name} module and router imported successfully")
            else:
                print(f"⚠️  {module_name} module imported but no router found")
        except Exception as e:
            print(f"❌ {module_name} module import failed: {e}")
    
    # Test FastAPI routes
    print(f"\nRegistered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  {route.methods} {route.path}")
    
    print(f"\nTotal routes: {len(app.routes)}")

if __name__ == "__main__":
    test_imports()