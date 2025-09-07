#!/usr/bin/env python3
"""
Test all imports without database dependency
"""
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing Basic Imports...")
    
    try:
        # Test config
        from app.config import settings
        print("✅ Config imported")
        
        # Test models
        from app.models.responses import BaseResponse
        print("✅ Models imported")
        
        # Test utilities (without database dependency)
        from app.utils.auth import AuthMiddleware
        print("✅ Auth utils imported")
        
        from app.utils.security_middleware import SecurityMiddleware
        print("✅ Security middleware imported")
        
        from app.utils.audit_logger import AIAuditLogger
        print("✅ Audit logger imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_service_imports():
    """Test service imports"""
    print("\n🔍 Testing Service Imports...")
    
    try:
        # Test services
        from app.services.smart_notifications import NotificationScheduler
        print("✅ Smart notifications imported")
        
        from app.services.gemini_service import GeminiService
        print("✅ Gemini service imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Service imports failed: {e}")
        return False

def test_api_imports():
    """Test API imports"""
    print("\n🔍 Testing API Imports...")
    
    try:
        # Test API modules
        from app.api.health import router as health_router
        print("✅ Health API imported")
        
        return True
        
    except Exception as e:
        print(f"❌ API imports failed: {e}")
        return False

def test_main_app_import():
    """Test main app import"""
    print("\n🔍 Testing Main App Import...")
    
    try:
        # This should work without database connection
        from app.main import app
        print("✅ Main FastAPI app imported")
        print(f"   App type: {type(app)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all import tests"""
    print("🚀 Import Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Service Imports", test_service_imports),
        ("API Imports", test_api_imports),
        ("Main App Import", test_main_app_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All import tests passed!")
        return 0
    else:
        print("⚠️  Some import tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)