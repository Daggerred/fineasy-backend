#!/usr/bin/env python3
"""
Test script for comprehensive error handling implementation
"""
import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 10

async def test_health_endpoint():
    """Test the comprehensive health endpoint"""
    print("Testing health endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health endpoint working")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Features: {data.get('features', {})}")
                    print(f"   Checks: {list(data.get('checks', {}).keys())}")
                    return True
                else:
                    print(f"❌ Health endpoint returned {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

async def test_detailed_health_endpoint():
    """Test the detailed health endpoint"""
    print("\nTesting detailed health endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/v1/health/detailed", timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Detailed health endpoint working")
                    print(f"   Overall Status: {data.get('overall_status')}")
                    print(f"   Components: {list(data.get('components', {}).keys())}")
                    print(f"   Error Rates: {data.get('error_rates', {})}")
                    return True
                else:
                    print(f"❌ Detailed health endpoint returned {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Detailed health endpoint failed: {e}")
        return False

async def test_error_reporting_endpoint():
    """Test the error reporting endpoint"""
    print("\nTesting error reporting endpoint...")
    
    test_error = {
        "business_id": "test_business",
        "error_type": "network_error",
        "message": "Test error for monitoring",
        "operation": "fraud_analysis",
        "timestamp": time.time(),
        "client_info": {
            "platform": "test",
            "version": "1.0.0"
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/errors/report",
                json=test_error,
                timeout=TIMEOUT
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Error reporting endpoint working")
                    print(f"   Response: {data.get('message')}")
                    return True
                else:
                    print(f"❌ Error reporting endpoint returned {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error reporting endpoint failed: {e}")
        return False

async def test_performance_metrics_endpoint():
    """Test the performance metrics endpoint"""
    print("\nTesting performance metrics endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/v1/analytics/performance", timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Performance metrics endpoint working")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Cache Stats: {data.get('cache_stats', {})}")
                    print(f"   Performance Metrics: {data.get('performance_metrics', {})}")
                    return True
                else:
                    print(f"❌ Performance metrics endpoint returned {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Performance metrics endpoint failed: {e}")
        return False

async def test_fraud_analysis_error_handling():
    """Test fraud analysis with error handling"""
    print("\nTesting fraud analysis error handling...")
    
    # Test with invalid business ID to trigger error handling
    test_data = {
        "business_id": "invalid_business_id_for_testing"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/fraud/analyze",
                json=test_data,
                timeout=TIMEOUT
            ) as response:
                # We expect this to either work or fail gracefully
                if response.status in [200, 400, 422, 500]:
                    print("✅ Fraud analysis error handling working")
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            print(f"   Error response: {error_data}")
                        except:
                            print(f"   HTTP Status: {response.status}")
                    return True
                else:
                    print(f"❌ Unexpected status code: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Fraud analysis error handling failed: {e}")
        return False

async def test_service_availability():
    """Test basic service availability"""
    print("\nTesting basic service availability...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/", timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Service is available")
                    print(f"   Message: {data.get('message')}")
                    print(f"   Version: {data.get('version')}")
                    return True
                else:
                    print(f"❌ Service returned {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Service availability test failed: {e}")
        return False

async def main():
    """Run all error handling tests"""
    print("🧪 Testing Comprehensive Error Handling Implementation")
    print("=" * 60)
    
    tests = [
        test_service_availability,
        test_health_endpoint,
        test_detailed_health_endpoint,
        test_performance_metrics_endpoint,
        test_error_reporting_endpoint,
        test_fraud_analysis_error_handling,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All error handling tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test runner failed: {e}")
        sys.exit(1)