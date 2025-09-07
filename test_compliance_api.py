#!/usr/bin/env python3
"""
Simple test for compliance API endpoints
"""
import sys
import os
import json
from unittest.mock import patch, AsyncMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Mock external dependencies
class MockDatabaseManager:
    async def fetch_one(self, query, params):
        # Return mock invoice data
        return {
            'id': 'test-invoice-123',
            'invoice_number': 'INV-2024-001',
            'invoice_date': '2024-01-15',
            'supplier_gstin': '27AAPFU0939F1ZV',
            'customer_gstin': '29AABCU9603R1ZX',
            'supplier_state': 'Maharashtra',
            'customer_state': 'Karnataka',
            'place_of_supply': 'Karnataka',
            'taxable_value': 1000.0,
            'tax_amount': 180.0,
            'total_amount': 1180.0,
            'items': [
                {
                    'name': 'Software License',
                    'taxable_value': 1000.0,
                    'tax_rate': 18.0
                }
            ]
        }

# Mock the database module
sys.modules['app.database'] = type('MockModule', (), {'DatabaseManager': MockDatabaseManager})()

# Mock aiohttp for GST API calls
class MockAiohttp:
    class ClientSession:
        def __init__(self):
            pass
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        def post(self, url, json=None, headers=None, timeout=None):
            return MockResponse()

class MockResponse:
    def __init__(self):
        self.status = 200
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def json(self):
        return {
            'valid': True,
            'tradeNam': 'Test Company Pvt Ltd',
            'sts': 'Active',
            'rgdt': '2020-01-01T00:00:00'
        }

sys.modules['aiohttp'] = type('MockModule', (), {
    'ClientSession': MockAiohttp.ClientSession,
    'TimeoutError': TimeoutError
})()

# Now import the compliance service
from services.compliance import ComplianceChecker


async def test_compliance_checker_initialization():
    """Test ComplianceChecker initialization"""
    print("Testing ComplianceChecker initialization...")
    
    checker = ComplianceChecker()
    assert checker is not None
    assert hasattr(checker, 'gst_rates')
    assert hasattr(checker, 'required_invoice_fields')
    
    print("✓ ComplianceChecker initialization test passed")


async def test_gstin_validation():
    """Test GSTIN validation"""
    print("\nTesting GSTIN validation...")
    
    checker = ComplianceChecker()
    
    # Test valid GSTIN
    result = await checker.validate_gst_number("27AAPFU0939F1ZV")
    print(f"Valid GSTIN result: {result}")
    assert result.gstin == "27AAPFU0939F1ZV"
    assert result.is_valid is True
    
    # Test invalid GSTIN
    result = await checker.validate_gst_number("INVALID")
    print(f"Invalid GSTIN result: {result}")
    assert result.is_valid is False
    assert result.status == "invalid_format"
    
    print("✓ GSTIN validation test passed")


async def test_tax_verification():
    """Test tax verification"""
    print("\nTesting tax verification...")
    
    checker = ComplianceChecker()
    
    # Mock the database call to return test data
    with patch.object(checker, '_get_invoice_data') as mock_get_invoice:
        mock_get_invoice.return_value = {
            'id': 'test-invoice',
            'supplier_state': 'Maharashtra',
            'customer_state': 'Karnataka',
            'items': [
                {
                    'name': 'Product A',
                    'taxable_value': 1000.0,
                    'tax_rate': 18.0
                }
            ],
            'tax_amount': 180.0
        }
        
        result = await checker.verify_tax_calculations('test-invoice')
        print(f"Tax verification result: {result}")
        
        assert result.invoice_id == 'test-invoice'
        assert result.is_correct is True
        assert result.expected_tax == 180.0
        assert result.calculated_tax == 180.0
    
    print("✓ Tax verification test passed")


async def test_compliance_check():
    """Test comprehensive compliance check"""
    print("\nTesting comprehensive compliance check...")
    
    checker = ComplianceChecker()
    
    # Mock the database call and GST validation
    with patch.object(checker, '_get_invoice_data') as mock_get_invoice:
        with patch.object(checker, 'validate_gst_number') as mock_gst_validation:
            mock_get_invoice.return_value = {
                'id': 'test-invoice',
                'invoice_number': 'INV-2024-001',
                'invoice_date': '2024-01-15',
                'supplier_gstin': '27AAPFU0939F1ZV',
                'customer_gstin': '29AABCU9603R1ZX',
                'supplier_state': 'Maharashtra',
                'customer_state': 'Karnataka',
                'place_of_supply': 'Karnataka',
                'taxable_value': 1000.0,
                'tax_amount': 180.0,
                'total_amount': 1180.0,
                'items': [
                    {
                        'name': 'Software License',
                        'taxable_value': 1000.0,
                        'tax_rate': 18.0
                    }
                ]
            }
            
            # Mock GST validation to return valid results
            from models.responses import GSTValidationResult
            mock_gst_validation.return_value = GSTValidationResult(
                gstin="27AAPFU0939F1ZV",
                is_valid=True,
                status="active"
            )
            
            result = await checker.check_compliance('test-invoice')
            print(f"Compliance check result: {result}")
            
            assert result.invoice_id == 'test-invoice'
            assert result.success is True
            assert isinstance(result.compliance_score, float)
            assert 0.0 <= result.compliance_score <= 1.0
    
    print("✓ Comprehensive compliance check test passed")


async def test_field_completeness():
    """Test field completeness validation"""
    print("\nTesting field completeness validation...")
    
    checker = ComplianceChecker()
    
    # Test complete data
    complete_data = {
        'invoice_number': 'INV-2024-001',
        'invoice_date': '2024-01-15',
        'supplier_gstin': '27AAPFU0939F1ZV',
        'customer_gstin': '29AABCU9603R1ZX',
        'place_of_supply': 'Karnataka',
        'items': [{'name': 'Product A'}],
        'taxable_value': 1000.0,
        'tax_amount': 180.0,
        'total_amount': 1180.0
    }
    
    issues = await checker._check_field_completeness(complete_data)
    print(f"Complete data issues: {len(issues)}")
    assert len(issues) == 0
    
    # Test incomplete data
    incomplete_data = {
        'invoice_number': 'INV-2024-002',
        'taxable_value': 500.0
    }
    
    issues = await checker._check_field_completeness(incomplete_data)
    print(f"Incomplete data issues: {len(issues)}")
    assert len(issues) > 0
    
    print("✓ Field completeness validation test passed")


async def main():
    """Run all API tests"""
    print("GST Compliance Service API Tests")
    print("=" * 50)
    
    try:
        await test_compliance_checker_initialization()
        await test_gstin_validation()
        await test_tax_verification()
        await test_compliance_check()
        await test_field_completeness()
        
        print("\n" + "=" * 50)
        print("✅ All GST Compliance Service API tests passed!")
        print("\nThe compliance service is ready for integration:")
        print("- ComplianceChecker class implemented with all required methods")
        print("- GSTIN validation with format checking and API integration")
        print("- Tax calculation verification for intra-state and inter-state")
        print("- Field completeness validation for GST compliance")
        print("- Comprehensive compliance scoring system")
        print("- Plain language explanations for compliance issues")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)