#!/usr/bin/env python3
"""
Validation script for feature flags and gradual rollout system.
"""
import asyncio
import aiohttp
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
API_VERSION = "v1"


class FeatureFlagValidator:
    """Validates feature flag functionality."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", details: Dict = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if message:
            logger.info(f"   {message}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def test_health_check(self):
        """Test basic health check."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test_result(
                        "Health Check", 
                        True, 
                        f"Service is {data.get('status', 'unknown')}"
                    )
                    return True
                else:
                    self.log_test_result(
                        "Health Check", 
                        False, 
                        f"HTTP {response.status}"
                    )
                    return False
        except Exception as e:
            self.log_test_result("Health Check", False, str(e))
            return False
    
    async def test_feature_flag_check(self):
        """Test feature flag checking endpoint."""
        try:
            test_data = {
                "feature_name": "fraud_detection",
                "user_id": "test_user_123",
                "business_id": "test_business_456"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/{API_VERSION}/feature-flags/check",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate response structure
                    required_fields = ["feature_name", "enabled"]
                    if all(field in data for field in required_fields):
                        self.log_test_result(
                            "Feature Flag Check", 
                            True, 
                            f"Feature '{data['feature_name']}' is {'enabled' if data['enabled'] else 'disabled'}"
                        )
                        return True
                    else:
                        self.log_test_result(
                            "Feature Flag Check", 
                            False, 
                            "Missing required fields in response"
                        )
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Feature Flag Check", 
                        False, 
                        f"HTTP {response.status}: {error_text}"
                    )
                    return False
        except Exception as e:
            self.log_test_result("Feature Flag Check", False, str(e))
            return False
    
    async def test_interaction_tracking(self):
        """Test interaction tracking endpoint."""
        try:
            test_data = {
                "feature_name": "fraud_detection",
                "user_id": "test_user_123",
                "interaction_type": "view"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/{API_VERSION}/feature-flags/track/interaction",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.log_test_result(
                            "Interaction Tracking", 
                            True, 
                            "Interaction tracked successfully"
                        )
                        return True
                    else:
                        self.log_test_result(
                            "Interaction Tracking", 
                            False, 
                            data.get("message", "Unknown error")
                        )
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Interaction Tracking", 
                        False, 
                        f"HTTP {response.status}: {error_text}"
                    )
                    return False
        except Exception as e:
            self.log_test_result("Interaction Tracking", False, str(e))
            return False
    
    async def test_conversion_tracking(self):
        """Test conversion tracking endpoint."""
        try:
            test_data = {
                "feature_name": "fraud_detection",
                "user_id": "test_user_123",
                "conversion_value": 1.0
            }
            
            async with self.session.post(
                f"{self.base_url}/api/{API_VERSION}/feature-flags/track/conversion",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.log_test_result(
                            "Conversion Tracking", 
                            True, 
                            "Conversion tracked successfully"
                        )
                        return True
                    else:
                        self.log_test_result(
                            "Conversion Tracking", 
                            False, 
                            data.get("message", "Unknown error")
                        )
                        return False
                else:
                    error_text = await response.text()
                    self.log_test_result(
                        "Conversion Tracking", 
                        False, 
                        f"HTTP {response.status}: {error_text}"
                    )
                    return False
        except Exception as e:
            self.log_test_result("Conversion Tracking", False, str(e))
            return False
    
    async def test_multiple_users_consistency(self):
        """Test that same user gets consistent feature flag results."""
        try:
            user_id = "consistency_test_user"
            feature_name = "smart_notifications"
            
            results = []
            
            # Make multiple requests for the same user
            for i in range(5):
                test_data = {
                    "feature_name": feature_name,
                    "user_id": user_id,
                    "business_id": "test_business"
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/{API_VERSION}/feature-flags/check",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results.append({
                            "enabled": data.get("enabled"),
                            "variant": data.get("variant")
                        })
                    else:
                        self.log_test_result(
                            "User Consistency Test", 
                            False, 
                            f"Request {i+1} failed with HTTP {response.status}"
                        )
                        return False
            
            # Check consistency
            first_result = results[0]
            all_consistent = all(
                r["enabled"] == first_result["enabled"] and 
                r["variant"] == first_result["variant"] 
                for r in results
            )
            
            if all_consistent:
                self.log_test_result(
                    "User Consistency Test", 
                    True, 
                    f"User consistently gets enabled={first_result['enabled']}, variant={first_result['variant']}"
                )
                return True
            else:
                self.log_test_result(
                    "User Consistency Test", 
                    False, 
                    f"Inconsistent results: {results}"
                )
                return False
                
        except Exception as e:
            self.log_test_result("User Consistency Test", False, str(e))
            return False
    
    async def test_ab_test_distribution(self):
        """Test A/B test variant distribution."""
        try:
            feature_name = "smart_notifications"
            variant_counts = {}
            total_users = 100
            
            # Test with multiple users
            for i in range(total_users):
                user_id = f"ab_test_user_{i}"
                test_data = {
                    "feature_name": feature_name,
                    "user_id": user_id,
                    "business_id": "test_business"
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/{API_VERSION}/feature-flags/check",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("enabled"):
                            variant = data.get("variant", "control")
                            variant_counts[variant] = variant_counts.get(variant, 0) + 1
                    else:
                        self.log_test_result(
                            "A/B Test Distribution", 
                            False, 
                            f"Request for user {i} failed"
                        )
                        return False
            
            # Check if we have reasonable distribution
            if variant_counts:
                total_enabled = sum(variant_counts.values())
                distribution = {
                    variant: (count / total_enabled) * 100 
                    for variant, count in variant_counts.items()
                }
                
                self.log_test_result(
                    "A/B Test Distribution", 
                    True, 
                    f"Distribution: {distribution} (total enabled: {total_enabled}/{total_users})"
                )
                return True
            else:
                self.log_test_result(
                    "A/B Test Distribution", 
                    False, 
                    "No users were assigned to any variant"
                )
                return False
                
        except Exception as e:
            self.log_test_result("A/B Test Distribution", False, str(e))
            return False
    
    async def test_performance_tracking(self):
        """Test performance tracking integration."""
        try:
            # This would test the performance monitoring endpoints
            # For now, just check if the endpoint exists
            async with self.session.get(
                f"{self.base_url}/api/{API_VERSION}/analytics/performance"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "performance_metrics" in data:
                        self.log_test_result(
                            "Performance Tracking", 
                            True, 
                            "Performance metrics endpoint accessible"
                        )
                        return True
                    else:
                        self.log_test_result(
                            "Performance Tracking", 
                            False, 
                            "Performance metrics not found in response"
                        )
                        return False
                else:
                    self.log_test_result(
                        "Performance Tracking", 
                        False, 
                        f"HTTP {response.status}"
                    )
                    return False
        except Exception as e:
            self.log_test_result("Performance Tracking", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all validation tests."""
        logger.info("🚀 Starting Feature Flag Validation Tests")
        logger.info("=" * 50)
        
        tests = [
            self.test_health_check,
            self.test_feature_flag_check,
            self.test_interaction_tracking,
            self.test_conversion_tracking,
            self.test_multiple_users_consistency,
            self.test_ab_test_distribution,
            self.test_performance_tracking,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
        
        logger.info("=" * 50)
        logger.info(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Feature flag system is working correctly.")
            return True
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Please check the issues above.")
            return False
    
    def print_summary(self):
        """Print test summary."""
        logger.info("\n📋 Detailed Test Summary:")
        logger.info("-" * 30)
        
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            logger.info(f"{status} {result['test']}")
            if result["message"]:
                logger.info(f"   {result['message']}")


async def main():
    """Main validation function."""
    try:
        async with FeatureFlagValidator() as validator:
            success = await validator.run_all_tests()
            validator.print_summary()
            
            if success:
                logger.info("\n🎯 Feature flag system validation completed successfully!")
                sys.exit(0)
            else:
                logger.error("\n💥 Feature flag system validation failed!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())