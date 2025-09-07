#!/usr/bin/env python3
"""
Deployment validation script for AI Backend
"""
import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, Any, List

class DeploymentValidator:
    """Validates AI Backend deployment"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def validate_endpoint(self, endpoint: str, method: str = "GET", 
                              expected_status: int = 200,
                              description: str = None) -> Dict[str, Any]:
        """Validate a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        description = description or f"{method} {endpoint}"
        
        try:
            start_time = time.time()
            
            if method == "GET":
                async with self.session.get(url, timeout=10) as response:
                    response_time = (time.time() - start_time) * 1000
                    content = await response.text()
                    
                    result = {
                        "endpoint": endpoint,
                        "method": method,
                        "description": description,
                        "status_code": response.status,
                        "expected_status": expected_status,
                        "response_time_ms": round(response_time, 2),
                        "success": response.status == expected_status,
                        "content_length": len(content)
                    }
                    
                    # Try to parse JSON response
                    try:
                        json_content = json.loads(content)
                        result["response_type"] = "json"
                        result["response_keys"] = list(json_content.keys()) if isinstance(json_content, dict) else None
                    except:
                        result["response_type"] = "text"
                    
                    return result
            
            elif method == "POST":
                async with self.session.post(url, json={}, timeout=10) as response:
                    response_time = (time.time() - start_time) * 1000
                    content = await response.text()
                    
                    return {
                        "endpoint": endpoint,
                        "method": method,
                        "description": description,
                        "status_code": response.status,
                        "expected_status": expected_status,
                        "response_time_ms": round(response_time, 2),
                        "success": response.status == expected_status,
                        "content_length": len(content)
                    }
        
        except asyncio.TimeoutError:
            return {
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status_code": 0,
                "expected_status": expected_status,
                "response_time_ms": 10000,
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status_code": 0,
                "expected_status": expected_status,
                "response_time_ms": 0,
                "success": False,
                "error": str(e)
            }
    
    async def validate_all_endpoints(self) -> List[Dict[str, Any]]:
        """Validate all critical endpoints"""
        endpoints_to_test = [
            # Basic endpoints
            ("/", "GET", 200, "Root endpoint"),
            ("/health", "GET", 200, "Basic health check"),
            ("/health/detailed", "GET", 200, "Detailed health check"),
            ("/health/live", "GET", 200, "Liveness probe"),
            ("/health/ready", "GET", 200, "Readiness probe"),
            ("/health/startup", "GET", 200, "Startup probe"),
            ("/metrics", "GET", 200, "Prometheus metrics"),
            
            # API endpoints (these might return 401/422 without auth, which is expected)
            ("/api/v1/status", "GET", 200, "API status"),
            ("/api/v1/analytics/performance", "GET", 200, "Performance metrics"),
            ("/health/dependencies", "GET", 200, "Dependencies check"),
            
            # Component test endpoints
            ("/health/test/database", "POST", 200, "Database component test"),
            ("/health/test/redis", "POST", 200, "Redis component test"),
            ("/health/test/ml_models", "POST", 200, "ML models component test"),
        ]
        
        print("🔍 Validating AI Backend deployment...")
        print("=" * 50)
        
        results = []
        for endpoint, method, expected_status, description in endpoints_to_test:
            result = await self.validate_endpoint(endpoint, method, expected_status, description)
            results.append(result)
            
            # Print result
            status_icon = "✅" if result["success"] else "❌"
            print(f"{status_icon} {description}")
            print(f"   {method} {endpoint} -> {result['status_code']} ({result['response_time_ms']}ms)")
            
            if not result["success"] and "error" in result:
                print(f"   Error: {result['error']}")
            
            print()
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print validation summary"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print("📊 Validation Summary")
        print("=" * 50)
        print(f"Total endpoints tested: {len(results)}")
        print(f"Successful: {len(successful)} ✅")
        print(f"Failed: {len(failed)} ❌")
        print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_response_time = sum(r["response_time_ms"] for r in successful) / len(successful)
            print(f"Average response time: {avg_response_time:.1f}ms")
        
        if failed:
            print("\n❌ Failed endpoints:")
            for result in failed:
                print(f"   - {result['description']}: {result.get('error', 'Status ' + str(result['status_code']))}")
        
        print("\n🎯 Deployment Status:", "HEALTHY ✅" if len(failed) == 0 else "ISSUES DETECTED ⚠️")
        
        return len(failed) == 0


async def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate AI Backend deployment")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--wait", type=int, default=0, help="Wait time before testing (seconds)")
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for services to start...")
        await asyncio.sleep(args.wait)
    
    async with DeploymentValidator(args.url) as validator:
        results = await validator.validate_all_endpoints()
        success = validator.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())