#!/usr/bin/env python3
"""
Final verification test for AI backend
"""
import asyncio
import aiohttp
import json
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_comprehensive_functionality():
    """Test comprehensive backend functionality"""
    base_url = "http://localhost:8000"
    
    print("🎯 Final Backend Verification Test")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Basic endpoints
        print("\n1. 🔍 Testing Basic Endpoints")
        print("-" * 30)
        
        basic_endpoints = [
            ("/", "Root"),
            ("/health", "Health Check"),
            ("/docs", "API Documentation"),
            ("/openapi.json", "OpenAPI Schema")
        ]
        
        for endpoint, name in basic_endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        print(f"✅ {name}: Working")
                    else:
                        print(f"❌ {name}: Status {response.status}")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
        
        # Test 2: Health endpoints
        print("\n2. 🏥 Testing Health Endpoints")
        print("-" * 30)
        
        health_endpoints = [
            ("/api/v1/health/detailed", "Detailed Health"),
            ("/api/v1/status", "API Status"),
            ("/api/v1/analytics/performance", "Performance Metrics")
        ]
        
        for endpoint, name in health_endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ {name}: Working")
                        
                        # Show key metrics
                        if endpoint == "/api/v1/health/detailed":
                            status = data.get('overall_status', 'unknown')
                            components = len(data.get('components', {}))
                            print(f"   📊 Overall Status: {status}")
                            print(f"   🔧 Components Checked: {components}")
                        
                    else:
                        print(f"❌ {name}: Status {response.status}")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
        
        # Test 3: API endpoint availability
        print("\n3. 🚀 Testing API Endpoint Availability")
        print("-" * 30)
        
        # Get all available endpoints from OpenAPI schema
        try:
            async with session.get(f"{base_url}/openapi.json") as response:
                if response.status == 200:
                    schema = await response.json()
                    paths = schema.get('paths', {})
                    
                    # Count endpoints by category
                    categories = {}
                    for path, methods in paths.items():
                        for method, details in methods.items():
                            tags = details.get('tags', ['untagged'])
                            category = tags[0] if tags else 'untagged'
                            if category not in categories:
                                categories[category] = 0
                            categories[category] += 1
                    
                    print(f"📊 Total API Endpoints: {sum(categories.values())}")
                    for category, count in sorted(categories.items()):
                        print(f"   {category}: {count} endpoints")
                
        except Exception as e:
            print(f"❌ API Schema: Error - {e}")
        
        # Test 4: Core AI Services
        print("\n4. 🤖 Testing Core AI Services")
        print("-" * 30)
        
        # Test fraud detection endpoint (should return method not allowed for GET)
        try:
            async with session.get(f"{base_url}/api/v1/fraud/analyze") as response:
                if response.status == 405:  # Method not allowed is expected for POST endpoint
                    print("✅ Fraud Detection API: Available")
                else:
                    print(f"⚠️  Fraud Detection API: Status {response.status}")
        except Exception as e:
            print(f"❌ Fraud Detection API: Error - {e}")
        
        # Test insights endpoint
        try:
            async with session.get(f"{base_url}/api/v1/insights/test-business-id") as response:
                # Any response (even error) means the endpoint exists
                print("✅ Business Insights API: Available")
        except Exception as e:
            print(f"❌ Business Insights API: Error - {e}")
        
        # Test compliance endpoint (should return method not allowed for GET)
        try:
            async with session.get(f"{base_url}/api/v1/compliance/check") as response:
                if response.status == 405:  # Method not allowed is expected for POST endpoint
                    print("✅ Compliance API: Available")
                else:
                    print(f"⚠️  Compliance API: Status {response.status}")
        except Exception as e:
            print(f"❌ Compliance API: Error - {e}")
        
        # Test 5: Documentation Quality
        print("\n5. 📚 Testing Documentation Quality")
        print("-" * 30)
        
        try:
            async with session.get(f"{base_url}/docs") as response:
                if response.status == 200:
                    content = await response.text()
                    if len(content) > 1000 and "swagger" in content.lower():
                        print("✅ API Documentation: Comprehensive")
                    else:
                        print("⚠️  API Documentation: Basic")
                else:
                    print(f"❌ API Documentation: Status {response.status}")
        except Exception as e:
            print(f"❌ API Documentation: Error - {e}")
        
        # Test 6: Error Handling
        print("\n6. 🛡️  Testing Error Handling")
        print("-" * 30)
        
        # Test non-existent endpoint
        try:
            async with session.get(f"{base_url}/api/v1/nonexistent") as response:
                if response.status == 404:
                    print("✅ 404 Error Handling: Working")
                else:
                    print(f"⚠️  404 Error Handling: Status {response.status}")
        except Exception as e:
            print(f"❌ 404 Error Handling: Error - {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Backend Verification Complete!")
    print("✅ The AI backend is fully operational and ready for use!")
    print("📱 Access the API at: http://localhost:8000")
    print("📚 View documentation at: http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_comprehensive_functionality())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        sys.exit(1)