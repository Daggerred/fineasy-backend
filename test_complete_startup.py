#!/usr/bin/env python3
"""
Complete startup test for AI backend
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.append('app')

async def test_complete_startup():
    """Test complete startup process"""
    print("🚀 Testing complete AI backend startup...")
    
    try:
        # Test 1: Environment variables
        print("\n1. Testing environment variables...")
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'AI_ENCRYPTION_KEY']
        for var in required_vars:
            if os.environ.get(var):
                print(f"   ✅ {var}: Present")
            else:
                print(f"   ❌ {var}: Missing")
                return False
        
        # Test 2: Encryption service
        print("\n2. Testing encryption service...")
        from app.utils.encryption import encryption_service
        if encryption_service.enabled:
            print("   ✅ Encryption service: Initialized")
        else:
            print("   ❌ Encryption service: Failed")
            return False
        
        # Test 3: Database connection
        print("\n3. Testing database connection...")
        try:
            from app.database import init_database
            supabase = await init_database()
            if supabase:
                print("   ✅ Database connection: Successful")
            else:
                print("   ⚠️  Database connection: Not available (continuing without DB)")
        except Exception as e:
            print(f"   ⚠️  Database connection: {e} (continuing without DB)")
        
        # Test 4: Import all modules
        print("\n4. Testing module imports...")
        modules_to_test = [
            'app.config',
            'app.utils.auth',
            'app.utils.security_middleware',
            'app.utils.audit_logger',
            'app.services.fraud_detection',
            'app.services.compliance',
            'app.services.predictive_analytics',
            'app.api.health'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"   ✅ {module}: Imported successfully")
            except Exception as e:
                print(f"   ❌ {module}: Import failed - {e}")
                return False
        
        # Test 5: FastAPI app creation
        print("\n5. Testing FastAPI app creation...")
        from app.main import app
        if app:
            print("   ✅ FastAPI app: Created successfully")
        else:
            print("   ❌ FastAPI app: Creation failed")
            return False
        
        print("\n🎉 All startup tests passed! Backend is ready to start.")
        return True
        
    except Exception as e:
        print(f"\n❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_startup())
    sys.exit(0 if success else 1)