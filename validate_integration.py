#!/usr/bin/env python3
"""
Validation script for Supabase integration
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from config import settings, validate_configuration
    from database import init_database, test_connection, DatabaseManager
    from utils.auth import AuthToken, verify_supabase_token
    from utils.database import AIDataUtils
    
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def validate_configuration_settings():
    """Validate configuration settings"""
    print("\n🔧 Validating configuration...")
    
    try:
        validate_configuration()
        print("✅ Configuration validation passed")
        
        # Check required settings
        required_settings = [
            'SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'SUPABASE_ANON_KEY'
        ]
        
        for setting in required_settings:
            value = getattr(settings, setting, None)
            if value:
                print(f"✅ {setting}: {'*' * 10}...{value[-10:]}")
            else:
                print(f"❌ {setting}: Not set")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


async def validate_database_connection():
    """Validate database connection"""
    print("\n🗄️  Validating database connection...")
    
    try:
        # Initialize database
        client = await init_database()
        print("✅ Database client initialized")
        
        # Test connection
        await test_connection()
        print("✅ Database connection test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   This might be expected if Supabase is not accessible")
        return False


async def validate_database_manager():
    """Validate DatabaseManager functionality"""
    print("\n📊 Validating DatabaseManager...")
    
    try:
        db_manager = DatabaseManager()
        print("✅ DatabaseManager initialized")
        
        # Test basic operations (these should not crash even with no data)
        business_data = await db_manager.get_business_data("test-id")
        print(f"✅ get_business_data returned: {type(business_data)}")
        
        transactions = await db_manager.get_transactions("test-id", limit=1)
        print(f"✅ get_transactions returned: {len(transactions)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager validation failed: {e}")
        return False


async def validate_auth_utilities():
    """Validate authentication utilities"""
    print("\n🔐 Validating authentication utilities...")
    
    try:
        # Test AuthToken creation
        auth_token = AuthToken("user-123", "test@example.com", "business-456")
        print(f"✅ AuthToken created: {auth_token.user_id}")
        
        # Test token verification (this will fail with invalid token, but shouldn't crash)
        try:
            # This should raise an exception for invalid token
            await verify_supabase_token("invalid-token")
        except Exception:
            print("✅ Token verification properly rejects invalid tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Auth utilities validation failed: {e}")
        return False


async def validate_ai_data_utils():
    """Validate AI data utilities"""
    print("\n🤖 Validating AI data utilities...")
    
    try:
        # Test data hashing
        test_data = {"amount": 100, "description": "test", "id": "123"}
        hash_result = AIDataUtils.generate_data_hash(test_data)
        print(f"✅ Data hash generated: {hash_result[:16]}...")
        
        # Test data anonymization
        sensitive_data = {
            "amount": 100,
            "customer_name": "John Doe",
            "email": "john@example.com"
        }
        anonymized = AIDataUtils.anonymize_financial_data(sensitive_data)
        print(f"✅ Data anonymized: customer_name -> {anonymized['customer_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI data utils validation failed: {e}")
        return False


async def main():
    """Main validation function"""
    print("🚀 Starting Supabase Integration Validation")
    print("=" * 50)
    
    results = []
    
    # Run all validations
    results.append(await validate_configuration_settings())
    results.append(await validate_database_connection())
    results.append(await validate_database_manager())
    results.append(await validate_auth_utilities())
    results.append(await validate_ai_data_utils())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} validations PASSED!")
        print("✅ Supabase integration is ready for use")
        return 0
    else:
        print(f"⚠️  {passed}/{total} validations passed")
        print("❌ Some issues need to be resolved")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)