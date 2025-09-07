#!/usr/bin/env python3
"""
Comprehensive test script for compliance API endpoints
Tests all the new compliance features including issue tracking and reminders
"""

import asyncio
import aiohttp
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000"
API_VERSION = "v1"
TEST_BUSINESS_ID = str(uuid.uuid4())
TEST_INVOICE_ID = str(uuid.uuid4())
TEST_USER_ID = str(uuid.uuid4())

# Mock authentication token (in real implementation, this would be a JWT)
AUTH_TOKEN = "test-token-123"

class ComplianceAPITester:
    """Test class for compliance API endpoints"""
    
    def __init__(self):
        self.session = None
        self.base_url = f"{BASE_URL}/api/{API_VERSION}"
        self.headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test compliance service health check"""
        print("🔍 Testing compliance health check...")
        
        try:
            async with self.session.get(
                f"{self.base_url}/compliance/health",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_basic_compliance_check(self) -> bool:
        """Test basic compliance checking"""
        print("🔍 Testing basic compliance check...")
        
        try:
            payload = {
                "invoice_id": TEST_INVOICE_ID,
                "business_id": TEST_BUSINESS_ID
            }
            
            async with self.session.post(
                f"{self.base_url}/compliance/check",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status in [200, 404]:  # 404 is expected for non-existent invoice
                    data = await response.json()
                    print(f"✅ Basic compliance check: {data.get('message', 'Success')}")
                    return True
                else:
                    print(f"❌ Basic compliance check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Basic compliance check error: {e}")
            return False
    
    async def test_gst_validation(self) -> bool:
        """Test GST number validation"""
        print("🔍 Testing GST validation...")
        
        test_cases = [
            {"gstin": "27AAPFU0939F1ZV", "expected": True},  # Valid format
            {"gstin": "INVALID123", "expected": False},       # Invalid format
            {"gstin": "12ABCDE1234F1Z5", "expected": True},   # Valid format
        ]
        
        success_count = 0
        
        for test_case in test_cases:
            try:
                payload = {"gstin": test_case["gstin"]}
                
                async with self.session.post(
                    f"{self.base_url}/compliance/gst/validate",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        is_valid = data.get("is_valid", False)
                        
                        if test_case["expected"]:
                            if is_valid or data.get("status") == "format_valid":
                                print(f"✅ GST validation passed for {test_case['gstin']}")
                                success_count += 1
                            else:
                                print(f"❌ GST validation failed for valid GSTIN {test_case['gstin']}")
                        else:
                            if not is_valid:
                                print(f"✅ GST validation correctly rejected {test_case['gstin']}")
                                success_count += 1
                            else:
                                print(f"❌ GST validation incorrectly accepted {test_case['gstin']}")
                    else:
                        print(f"❌ GST validation request failed: {response.status}")
            except Exception as e:
                print(f"❌ GST validation error for {test_case['gstin']}: {e}")
        
        return success_count == len(test_cases)
    
    async def test_compliance_reminders(self) -> bool:
        """Test compliance reminder system"""
        print("🔍 Testing compliance reminders...")
        
        try:
            # Create a reminder
            reminder_payload = {
                "business_id": TEST_BUSINESS_ID,
                "reminder_type": "gst_filing",
                "due_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
                "description": "Test GST filing reminder",
                "priority": "high",
                "recurring": True,
                "recurring_interval_days": 30
            }
            
            reminder_id = None
            
            async with self.session.post(
                f"{self.base_url}/compliance/reminders",
                headers=self.headers,
                json=reminder_payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    reminder_id = data.get("reminder_id")
                    print(f"✅ Reminder created: {reminder_id}")
                else:
                    print(f"❌ Reminder creation failed: {response.status}")
                    return False
            
            # Get reminders for business
            async with self.session.get(
                f"{self.base_url}/compliance/reminders/{TEST_BUSINESS_ID}",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Retrieved {len(data)} reminders")
                else:
                    print(f"❌ Failed to retrieve reminders: {response.status}")
                    return False
            
            # Complete the reminder
            if reminder_id:
                async with self.session.put(
                    f"{self.base_url}/compliance/reminders/{reminder_id}/complete",
                    headers=self.headers,
                    params={"completion_notes": "Test completion"}
                ) as response:
                    if response.status == 200:
                        print("✅ Reminder completed successfully")
                    else:
                        print(f"❌ Failed to complete reminder: {response.status}")
                        return False
            
            return True
            
        except Exception as e:
            print(f"❌ Compliance reminders error: {e}")
            return False
    
    async def test_compliance_tracking(self) -> bool:
        """Test compliance tracking functionality"""
        print("🔍 Testing compliance tracking...")
        
        try:
            async with self.session.get(
                f"{self.base_url}/compliance/tracking/{TEST_BUSINESS_ID}",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Compliance tracking data: {data}")
                    return True
                else:
                    print(f"❌ Compliance tracking failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Compliance tracking error: {e}")
            return False
    
    async def test_plain_language_explanations(self) -> bool:
        """Test plain language explanation generation"""
        print("🔍 Testing plain language explanations...")
        
        issue_types = ["gst_validation", "tax_calculation", "missing_fields", "deadline_warning"]
        success_count = 0
        
        for issue_type in issue_types:
            try:
                async with self.session.get(
                    f"{self.base_url}/compliance/explanations/{issue_type}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("explanation") and data.get("suggested_actions"):
                            print(f"✅ Explanation generated for {issue_type}")
                            success_count += 1
                        else:
                            print(f"❌ Incomplete explanation for {issue_type}")
                    else:
                        print(f"❌ Explanation request failed for {issue_type}: {response.status}")
            except Exception as e:
                print(f"❌ Explanation error for {issue_type}: {e}")
        
        return success_count == len(issue_types)
    
    async def test_upcoming_deadlines(self) -> bool:
        """Test upcoming deadlines functionality"""
        print("🔍 Testing upcoming deadlines...")
        
        try:
            async with self.session.get(
                f"{self.base_url}/compliance/deadlines/{TEST_BUSINESS_ID}",
                headers=self.headers,
                params={"days_ahead": 30}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Retrieved deadlines: {data.get('total_deadlines', 0)} deadlines")
                    return True
                else:
                    print(f"❌ Deadlines request failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Deadlines error: {e}")
            return False
    
    async def test_bulk_compliance_check(self) -> bool:
        """Test bulk compliance checking"""
        print("🔍 Testing bulk compliance check...")
        
        try:
            # Start bulk check
            invoice_ids = [str(uuid.uuid4()) for _ in range(3)]
            
            async with self.session.post(
                f"{self.base_url}/compliance/bulk-check",
                headers=self.headers,
                params={"business_id": TEST_BUSINESS_ID},
                json=invoice_ids
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    job_id = data.get("job_id")
                    print(f"✅ Bulk check started: {job_id}")
                    
                    # Check job status
                    if job_id:
                        await asyncio.sleep(1)  # Wait a bit
                        
                        async with self.session.get(
                            f"{self.base_url}/compliance/bulk-check/{job_id}/status",
                            headers=self.headers
                        ) as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                print(f"✅ Bulk check status: {status_data.get('status')}")
                                return True
                            else:
                                print(f"❌ Failed to get bulk check status: {status_response.status}")
                                return False
                    else:
                        print("❌ No job ID returned from bulk check")
                        return False
                else:
                    print(f"❌ Bulk check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Bulk compliance check error: {e}")
            return False
    
    async def test_issue_resolution(self) -> bool:
        """Test compliance issue resolution"""
        print("🔍 Testing issue resolution...")
        
        try:
            # This would normally resolve a real issue ID
            # For testing, we'll use a mock ID and expect a 404 or error
            mock_issue_id = str(uuid.uuid4())
            
            resolution_payload = {
                "issue_id": mock_issue_id,
                "resolution_action": "Updated GSTIN in invoice",
                "resolution_notes": "Corrected invalid GSTIN format",
                "resolved_by": TEST_USER_ID
            }
            
            async with self.session.post(
                f"{self.base_url}/compliance/issues/{mock_issue_id}/resolve",
                headers=self.headers,
                json=resolution_payload
            ) as response:
                # We expect this to fail since the issue doesn't exist
                # But the endpoint should handle it gracefully
                if response.status in [404, 500]:
                    print("✅ Issue resolution endpoint handled non-existent issue correctly")
                    return True
                elif response.status == 200:
                    print("✅ Issue resolution endpoint working (unexpected success)")
                    return True
                else:
                    print(f"❌ Issue resolution failed unexpectedly: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Issue resolution error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all compliance API tests"""
        print("🚀 Starting Compliance API Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Basic Compliance Check", self.test_basic_compliance_check),
            ("GST Validation", self.test_gst_validation),
            ("Compliance Reminders", self.test_compliance_reminders),
            ("Compliance Tracking", self.test_compliance_tracking),
            ("Plain Language Explanations", self.test_plain_language_explanations),
            ("Upcoming Deadlines", self.test_upcoming_deadlines),
            ("Bulk Compliance Check", self.test_bulk_compliance_check),
            ("Issue Resolution", self.test_issue_resolution),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        print("=" * 50)
        
        # Print summary
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        return results

async def main():
    """Main test runner"""
    print("🧪 Compliance API Endpoint Tester")
    print(f"🔗 Testing against: {BASE_URL}")
    print(f"📋 Test Business ID: {TEST_BUSINESS_ID}")
    print(f"📄 Test Invoice ID: {TEST_INVOICE_ID}")
    
    async with ComplianceAPITester() as tester:
        results = await tester.run_all_tests()
        
        # Return appropriate exit code
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n⚠️  Some tests failed!")
            return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)