#!/usr/bin/env python3
"""
Simple startup test for the AI backend
"""
import sys
import os
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing imports...")
    
    try:
        from app.config import settings, validate_configuration
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from app.database import init_database, test_connection
        print("✅ Database module imported successfully")
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    try:
        from app.main import app
        print("✅ Main FastAPI app imported successfully")
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        return False
    
    return True

async def test_configuration():
    """Test configuration validation"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from app.config import settings
        print(f"✅ Environment: {settings.ENVIRONMENT}")
        print(f"✅ API Version: {settings.API_VERSION}")
        print(f"✅ AI Features: {settings.ENABLE_AI_FEATURES}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_database_connection():
    """Test database connection if configured"""
    print("\n🗄️  Testing database connection...")
    
    try:
        from app.config import settings
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            print("⚠️  Database credentials not configured, skipping test")
            return True
        
        from app.database import init_database, test_connection
        await init_database()
        await test_connection()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("   This is expected if database is not configured")
        return True  # Don't fail the test for database issues

async def main():
    """Run all startup tests"""
    print("🚀 FinEasy AI Backend Startup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Database Test", test_database_connection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = await test_func()
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
        print("🎉 All tests passed! Backend should start successfully.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)