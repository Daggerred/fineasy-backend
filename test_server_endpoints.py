#!/usr/bin/env python3
"""
Test server endpoints to verify functionality
"""
import asyncio
import aiohttp
import time
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_endpoints():
    """Test various server endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints_to_test = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/docs", "API Documentation"),
        ("/redoc", "ReDoc Documentation"),
        ("/openapi.json", "OpenAPI Schema"),
        ("/api/v1/health/detailed", "Detailed health check")
    ]
    
    print("🧪 Testing server endpoints...")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        for endpoint, description in endpoints_to_test:
            try:
                url = f"{base_url}{endpoint}"
                async with session.get(url, timeout=10) as response:
                    status = response.status
                    content_type = response.headers.get('content-type', '')
                    
                    if status == 200:
                        print(f"✅ {description}: {status} - {content_type}")
                        
                        # For JSON endpoints, check if response is valid JSON
                        if 'application/json' in content_type:
                            try:
                                data = await response.json()
                                if endpoint == "/":
                                    print(f"   📄 Message: {data.get('message', 'N/A')}")
                                elif endpoint == "/health":
                                    print(f"   📊 Status: {data.get('status', 'N/A')}")
                                    print(f"   🔧 Environment: {data.get('environment', 'N/A')}")
                            except:
                                print(f"   ⚠️  Response is not valid JSON")
                        
                        # For HTML endpoints, check if content exists
                        elif 'text/html' in content_type:
                            text = await response.text()
                            if len(text) > 100:
                                print(f"   📄 HTML content loaded ({len(text)} chars)")
                            else:
                                print(f"   ⚠️  HTML content seems empty or minimal")
                    
                    else:
                        print(f"❌ {description}: {status}")
                        
            except asyncio.TimeoutError:
                print(f"⏰ {description}: Timeout")
            except Exception as e:
                print(f"❌ {description}: Error - {e}")
    
    print("=" * 50)
    print("🎯 Endpoint testing completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_endpoints())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        sys.exit(1)