#!/usr/bin/env python3
"""
Simple validation script for compliance API endpoints
Tests the core functionality without requiring a full database setup
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.compliance import ComplianceChecker
from app.models.base import ComplianceIssue, ComplianceType, ComplianceSeverity

class MockDatabaseManager:
    """Mock database manager for testing"""
    
    async def fetch_one(self, query: str, params=None):
        """Mock fetch_one method"""
        if "invoices" in query:
            # Return mock invoice data
            return {
                'id': 'test-invoice-id',
                'invoice_number': 'INV-001',
                'invoice_date': datetime.utcnow(),
                'supplier_gstin': '27AAPFU0939F1ZV',
                'customer_gstin': '12ABCDE1234F1Z5',
                'supplier_state': 'Maharashtra',
                'customer_state': 'Karnataka',
                'place_of_supply': 'Karnataka',
                'tax_amount': 180.0,
                'items': [
                    {
                        'name': 'Test Product',
                        'taxable_value': 1000.0,
                        'tax_rate': 18.0
                    }
                ]
            }
        elif "compliance_reminders" in query:
            return {
                'id': 'test-reminder-id',
                'business_id': 'test-business-id',
                'reminder_type': 'gst_filing',
                'due_date': datetime.utcnow() + timedelta(days=7),
                'description': 'Test reminder',
                'priority': 'high',
                'recurring': True,
                'recurring_interval_days': 30,
                'created_at': datetime.utcnow()
            }
        elif "bulk_compliance_jobs" in query:
            return {
                'id': 'test-job-id',
                'business_id': 'test-business-id',
                'status': 'completed',
                'progress': 100.0,
                'results': []
            }
        return None
    
    async def fetch_all(self, query: str, params=None):
        """Mock fetch_all method"""
        if "compliance_issues" in query and "GROUP BY" in query:
            return [
                {'issue_type': 'gst_validation', 'count': 2},
                {'issue_type': 'tax_calculation', 'count': 1}
            ]
        elif "compliance_reminders" in query:
            return [
                {
                    'id': 'reminder-1',
                    'reminder_type': 'gst_filing',
                    'description': 'GSTR-1 filing',
                    'due_date': datetime.utcnow() + timedelta(days=5),
                    'priority': 'high',
                    'recurring': False,
                    'recurring_interval_days': None
                }
            ]
        elif "compliance_issues" in query:
            return [
                {
                    'total_issues': 5,
                    'resolved_issues': 2,
                    'pending_issues': 3,
                    'critical_issues': 1,
                    'resolution_rate': 0.4
                }
            ]
        return []
    
    async def execute(self, query: str, params=None):
        """Mock execute method"""
        return True

class ComplianceValidator:
    """Validator for compliance service functionality"""
    
    def __init__(self):
        self.checker = ComplianceChecker()
        # Replace the database manager with our mock
        self.checker.db = MockDatabaseManager()
    
    async def test_gst_format_validation(self) -> bool:
        """Test GST number format validation"""
        print("🔍 Testing GST format validation...")
        
        test_cases = [
            ("27AAPFU0939F1ZV", True),   # Valid GSTIN
            ("12ABCDE1234F1Z5", True),   # Valid GSTIN
            ("INVALID123", False),       # Invalid format
            ("27AAPFU0939F1Z", False),   # Too short
            ("27AAPFU0939F1ZVX", False), # Too long
            ("", False),                 # Empty
        ]
        
        success_count = 0
        for gstin, expected in test_cases:
            try:
                is_valid = self.checker._validate_gstin_format(gstin)
                if is_valid == expected:
                    print(f"✅ GST format validation passed for: {gstin}")
                    success_count += 1
                else:
                    print(f"❌ GST format validation failed for: {gstin} (expected {expected}, got {is_valid})")
            except Exception as e:
                print(f"❌ GST format validation error for {gstin}: {e}")
        
        return success_count == len(test_cases)
    
    async def test_compliance_score_calculation(self) -> bool:
        """Test compliance score calculation"""
        print("🔍 Testing compliance score calculation...")
        
        try:
            # Test with no issues
            score = self.checker._calculate_compliance_score([])
            if score == 1.0:
                print("✅ Compliance score calculation: No issues = 1.0")
            else:
                print(f"❌ Compliance score calculation failed: Expected 1.0, got {score}")
                return False
            
            # Test with various severity issues
            issues = [
                ComplianceIssue(
                    type=ComplianceType.GST_VALIDATION,
                    description="Test issue",
                    plain_language_explanation="Test explanation",
                    severity=ComplianceSeverity.HIGH
                ),
                ComplianceIssue(
                    type=ComplianceType.TAX_CALCULATION,
                    description="Test issue 2",
                    plain_language_explanation="Test explanation 2",
                    severity=ComplianceSeverity.MEDIUM
                )
            ]
            
            score = self.checker._calculate_compliance_score(issues)
            if 0.0 <= score <= 1.0:
                print(f"✅ Compliance score calculation: With issues = {score}")
            else:
                print(f"❌ Compliance score calculation failed: Score {score} out of range")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Compliance score calculation error: {e}")
            return False
    
    async def test_plain_language_explanations(self) -> bool:
        """Test plain language explanation generation"""
        print("🔍 Testing plain language explanations...")
        
        issue_types = ["gst_validation", "tax_calculation", "missing_fields", "deadline_warning"]
        success_count = 0
        
        for issue_type in issue_types:
            try:
                explanation = await self.checker.generate_plain_language_explanation(issue_type)
                
                if (explanation.get("explanation") and 
                    explanation.get("suggested_actions") and
                    isinstance(explanation.get("suggested_actions"), list)):
                    print(f"✅ Plain language explanation generated for: {issue_type}")
                    success_count += 1
                else:
                    print(f"❌ Incomplete explanation for: {issue_type}")
            except Exception as e:
                print(f"❌ Plain language explanation error for {issue_type}: {e}")
        
        return success_count == len(issue_types)
    
    async def test_tax_calculation(self) -> bool:
        """Test tax calculation logic"""
        print("🔍 Testing tax calculation...")
        
        try:
            # Mock invoice data
            invoice_data = {
                'items': [
                    {
                        'name': 'Product 1',
                        'taxable_value': 1000.0,
                        'tax_rate': 18.0
                    },
                    {
                        'name': 'Product 2', 
                        'taxable_value': 500.0,
                        'tax_rate': 12.0
                    }
                ],
                'supplier_state': 'Maharashtra',
                'customer_state': 'Maharashtra'  # Same state = CGST + SGST
            }
            
            expected_tax, breakdown = await self.checker._calculate_expected_tax(invoice_data)
            
            # Expected: (1000 * 0.18) + (500 * 0.12) = 180 + 60 = 240
            if abs(expected_tax - 240.0) < 0.01:
                print(f"✅ Tax calculation correct: {expected_tax}")
                
                # Check breakdown for intra-state (CGST + SGST)
                if breakdown.get('cgst') > 0 and breakdown.get('sgst') > 0 and breakdown.get('igst') == 0:
                    print("✅ Tax breakdown correct for intra-state transaction")
                    return True
                else:
                    print(f"❌ Tax breakdown incorrect: {breakdown}")
                    return False
            else:
                print(f"❌ Tax calculation incorrect: Expected 240.0, got {expected_tax}")
                return False
                
        except Exception as e:
            print(f"❌ Tax calculation error: {e}")
            return False
    
    async def test_reminder_message_generation(self) -> bool:
        """Test reminder message generation"""
        print("🔍 Testing reminder message generation...")
        
        try:
            reminder = {
                'reminder_type': 'gst_filing',
                'description': 'GSTR-1 filing for March 2024'
            }
            
            # Test different days before due date
            test_cases = [
                (0, "⚠️ Due Today"),
                (1, "⏰ Due Tomorrow"),
                (7, "📅 Due in 7 days")
            ]
            
            success_count = 0
            for days_before, expected_prefix in test_cases:
                message = self.checker._generate_reminder_message(reminder, days_before)
                if message.startswith(expected_prefix):
                    print(f"✅ Reminder message correct for {days_before} days: {message}")
                    success_count += 1
                else:
                    print(f"❌ Reminder message incorrect for {days_before} days: {message}")
            
            return success_count == len(test_cases)
            
        except Exception as e:
            print(f"❌ Reminder message generation error: {e}")
            return False
    
    async def test_compliance_status_determination(self) -> bool:
        """Test compliance status determination"""
        print("🔍 Testing compliance status determination...")
        
        try:
            from app.models.base import ComplianceStatus
            
            # Test no issues
            status = self.checker._determine_compliance_status([])
            if status == ComplianceStatus.COMPLIANT:
                print("✅ Status determination: No issues = COMPLIANT")
            else:
                print(f"❌ Status determination failed: Expected COMPLIANT, got {status}")
                return False
            
            # Test critical issues
            critical_issues = [
                ComplianceIssue(
                    type=ComplianceType.GST_VALIDATION,
                    description="Critical issue",
                    plain_language_explanation="Critical explanation",
                    severity=ComplianceSeverity.CRITICAL
                )
            ]
            
            status = self.checker._determine_compliance_status(critical_issues)
            if status == ComplianceStatus.CRITICAL_ISSUES:
                print("✅ Status determination: Critical issues = CRITICAL_ISSUES")
            else:
                print(f"❌ Status determination failed: Expected CRITICAL_ISSUES, got {status}")
                return False
            
            # Test medium issues
            medium_issues = [
                ComplianceIssue(
                    type=ComplianceType.MISSING_FIELDS,
                    description="Medium issue",
                    plain_language_explanation="Medium explanation",
                    severity=ComplianceSeverity.MEDIUM
                )
            ]
            
            status = self.checker._determine_compliance_status(medium_issues)
            if status == ComplianceStatus.ISSUES_FOUND:
                print("✅ Status determination: Medium issues = ISSUES_FOUND")
                return True
            else:
                print(f"❌ Status determination failed: Expected ISSUES_FOUND, got {status}")
                return False
                
        except Exception as e:
            print(f"❌ Compliance status determination error: {e}")
            return False
    
    async def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation tests"""
        print("🚀 Starting Compliance Service Validation")
        print("=" * 50)
        
        validations = [
            ("GST Format Validation", self.test_gst_format_validation),
            ("Compliance Score Calculation", self.test_compliance_score_calculation),
            ("Plain Language Explanations", self.test_plain_language_explanations),
            ("Tax Calculation", self.test_tax_calculation),
            ("Reminder Message Generation", self.test_reminder_message_generation),
            ("Compliance Status Determination", self.test_compliance_status_determination),
        ]
        
        results = {}
        passed = 0
        total = len(validations)
        
        for test_name, test_func in validations:
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
        print(f"📊 Validation Results: {passed}/{total} tests passed")
        print("=" * 50)
        
        # Print summary
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        return results

async def main():
    """Main validation runner"""
    print("🧪 Compliance Service Validator")
    print("🔧 Testing core compliance functionality")
    
    validator = ComplianceValidator()
    results = await validator.run_all_validations()
    
    # Return appropriate exit code
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print("\n⚠️  Some validations failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)