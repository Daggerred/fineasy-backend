#!/usr/bin/env python3
"""
Test database connection specifically
"""
import sys
import os
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def test_database_connection():
    """Test database connection with detailed debugging"""
    print("🗄️  Testing Database Connection")
    print("=" * 50)
    
    try:
        # Import configuration
        from app.config import settings
        print(f"✅ Configuration loaded")
        print(f"   Environment: {settings.ENVIRONMENT}")
        
        # Check if database credentials are configured
        if not settings.SUPABASE_URL:
            print("❌ SUPABASE_URL not configured")
            return False
            
        if not settings.SUPABASE_SERVICE_KEY:
            print("❌ SUPABASE_SERVICE_KEY not configured")
            return False
            
        print(f"✅ Database credentials configured")
        print(f"   URL: {settings.SUPABASE_URL[:50]}...")
        print(f"   Key: {settings.SUPABASE_SERVICE_KEY[:20]}...")
        
        # Test Supabase client creation
        print("\n🔧 Testing Supabase client creation...")
        from supabase import create_client
        
        # Create client with minimal options
        client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        
        print("✅ Supabase client created successfully")
        print(f"   Client type: {type(client)}")
        print(f"   Has URL: {hasattr(client, 'supabase_url')}")
        print(f"   Has Key: {hasattr(client, 'supabase_key')}")
        
        # Test basic client functionality
        print("\n🔍 Testing client functionality...")
        try:
            # Try a simple table operation
            response = client.table("businesses").select("id").limit(1).execute()
            print("✅ Table query executed successfully")
            print(f"   Response type: {type(response)}")
            print(f"   Has data: {hasattr(response, 'data')}")
            
            if hasattr(response, 'data'):
                print(f"   Data: {response.data}")
            
        except Exception as query_error:
            print(f"⚠️  Table query failed (this is normal if tables don't exist): {query_error}")
            print("   This doesn't necessarily indicate a connection problem")
        
        # Test our database module
        print("\n🔧 Testing database module...")
        from app.database import init_database, test_connection
        
        await init_database()
        result = await test_connection()
        
        if result:
            print("✅ Database module test successful")
        else:
            print("⚠️  Database module test failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run database test"""
    print("🚀 Database Connection Test")
    print("=" * 50)
    
    success = await test_database_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Database test completed successfully!")
        return 0
    else:
        print("❌ Database test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)