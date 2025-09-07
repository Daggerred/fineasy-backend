"""
Tests for GST Compliance Service
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from decimal import Decimal

from app.services.compliance import ComplianceChecker
from app.models.base import ComplianceType, ComplianceSeverity, ComplianceStatus
from app.models.responses import GSTValidationResult, TaxVerificationResult


class TestComplianceChecker:
    """Test cases for ComplianceChecker"""
    
    @pytest.fixture
    def compliance_checker(self):
        """Create ComplianceChecker instance for testing"""
        return ComplianceChecker()
    
    @pytest.fixture
    def sample_invoice_data(self):
        """Sample invoice data for testing"""
        return {
            'id': 'inv-123',
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
                    'name': 'Product A',
                    'taxable_value': 1000.0,
                    'tax_rate': 18.0,
                    'quantity': 1
                }
            ]
        }
    
    @pytest.fixture
    def incomplete_invoice_data(self):
        """Incomplete invoice data for testing validation"""
        return {
            'id': 'inv-456',
            'invoice_number': 'INV-2024-002',
            # Missing required fields
            'taxable_value': 500.0,
            'items': []
        }
    
    def test_validate_gstin_format_valid(self, compliance_checker):
        """Test valid GSTIN format validation"""
        valid_gstin = "27AAPFU0939F1ZV"
        assert compliance_checker._validate_gstin_format(valid_gstin) is True
    
    def test_validate_gstin_format_invalid(self, compliance_checker):
        """Test invalid GSTIN format validation"""
        invalid_gstins = [
            "27AAPFU0939F1Z",  # Too short
            "27AAPFU0939F1ZVX",  # Too long
            "27aapfu0939f1zv",  # Lowercase
            "27AAPFU0939F1ZA",  # Invalid check digit position
            "",  # Empty
            None  # None
        ]
        
        for gstin in invalid_gstins:
            assert compliance_checker._validate_gstin_format(gstin) is False
    
    @pytest.mark.asyncio
    async def test_validate_gst_number_format_only(self, compliance_checker):
        """Test GST validation with format validation only (no API key)"""
        compliance_checker.gst_api_key = None
        
        result = await compliance_checker.validate_gst_number("27AAPFU0939F1ZV")
        
        assert result.gstin == "27AAPFU0939F1ZV"
        assert result.is_valid is True
        assert result.status == "format_valid"
    
    @pytest.mark.asyncio
    async def test_validate_gst_number_invalid_format(self, compliance_checker):
        """Test GST validation with invalid format"""
        result = await compliance_checker.validate_gst_number("INVALID")
        
        assert result.gstin == "INVALID"
        assert result.is_valid is False
        assert result.status == "invalid_format"
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_validate_gst_number_api_success(self, mock_post, compliance_checker):
        """Test GST validation via API - success case"""
        compliance_checker.gst_api_key = "test-api-key"
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'valid': True,
            'tradeNam': 'Test Company',
            'sts': 'Active',
            'rgdt': '2020-01-01T00:00:00'
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        result = await compliance_checker.validate_gst_number("27AAPFU0939F1ZV")
        
        assert result.is_valid is True
        assert result.business_name == 'Test Company'
        assert result.status == 'Active'
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_validate_gst_number_api_failure(self, mock_post, compliance_checker):
        """Test GST validation via API - failure case"""
        compliance_checker.gst_api_key = "test-api-key"
        
        # Mock API error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_post.return_value.__aenter__.return_value = mock_response
        
        result = await compliance_checker.validate_gst_number("27AAPFU0939F1ZV")
        
        assert result.is_valid is False
        assert result.status == "api_error"
    
    @pytest.mark.asyncio
    async def test_check_field_completeness_complete(self, compliance_checker, sample_invoice_data):
        """Test field completeness check with complete data"""
        issues = await compliance_checker._check_field_completeness(sample_invoice_data)
        
        # Should have no issues for complete data
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_check_field_completeness_incomplete(self, compliance_checker, incomplete_invoice_data):
        """Test field completeness check with incomplete data"""
        issues = await compliance_checker._check_field_completeness(incomplete_invoice_data)
        
        # Should have issues for missing fields
        assert len(issues) > 0
        
        # Check for specific missing fields
        missing_fields = [issue.field_name for issue in issues if issue.type == ComplianceType.MISSING_FIELDS]
        expected_missing = ['invoice_date', 'supplier_gstin', 'customer_gstin', 'place_of_supply', 'tax_amount', 'total_amount']
        
        for field in expected_missing:
            assert field in missing_fields
    
    @pytest.mark.asyncio
    async def test_calculate_expected_tax_intra_state(self, compliance_checker):
        """Test tax calculation for intra-state transaction (CGST + SGST)"""
        invoice_data = {
            'supplier_state': 'Maharashtra',
            'customer_state': 'Maharashtra',  # Same state
            'items': [
                {
                    'name': 'Product A',
                    'taxable_value': 1000.0,
                    'tax_rate': 18.0
                }
            ]
        }
        
        expected_tax, breakdown = await compliance_checker._calculate_expected_tax(invoice_data)
        
        assert expected_tax == 180.0
        assert breakdown['cgst'] == 90.0  # Half of total tax
        assert breakdown['sgst'] == 90.0  # Half of total tax
        assert breakdown['igst'] == 0.0   # No IGST for intra-state
    
    @pytest.mark.asyncio
    async def test_calculate_expected_tax_inter_state(self, compliance_checker):
        """Test tax calculation for inter-state transaction (IGST)"""
        invoice_data = {
            'supplier_state': 'Maharashtra',
            'customer_state': 'Karnataka',  # Different state
            'items': [
                {
                    'name': 'Product A',
                    'taxable_value': 1000.0,
                    'tax_rate': 18.0
                }
            ]
        }
        
        expected_tax, breakdown = await compliance_checker._calculate_expected_tax(invoice_data)
        
        assert expected_tax == 180.0
        assert breakdown['cgst'] == 0.0   # No CGST for inter-state
        assert breakdown['sgst'] == 0.0   # No SGST for inter-state
        assert breakdown['igst'] == 180.0 # Full tax as IGST
    
    @pytest.mark.asyncio
    async def test_verify_tax_calculations_correct(self, compliance_checker, sample_invoice_data):
        """Test tax verification with correct calculations"""
        with patch.object(compliance_checker, '_get_invoice_data', return_value=sample_invoice_data):
            result = await compliance_checker.verify_tax_calculations('inv-123')
            
            assert result.invoice_id == 'inv-123'
            assert result.is_correct is True
            assert result.variance < 0.01
    
    @pytest.mark.asyncio
    async def test_verify_tax_calculations_incorrect(self, compliance_checker, sample_invoice_data):
        """Test tax verification with incorrect calculations"""
        # Modify sample data to have incorrect tax
        sample_invoice_data['tax_amount'] = 150.0  # Should be 180.0
        
        with patch.object(compliance_checker, '_get_invoice_data', return_value=sample_invoice_data):
            result = await compliance_checker.verify_tax_calculations('inv-123')
            
            assert result.invoice_id == 'inv-123'
            assert result.is_correct is False
            assert result.variance == 30.0  # 180 - 150
            assert result.calculated_tax == 150.0
            assert result.expected_tax == 180.0
    
    @pytest.mark.asyncio
    async def test_validate_place_of_supply_correct(self, compliance_checker, sample_invoice_data):
        """Test place of supply validation - correct case"""
        issues = await compliance_checker._validate_place_of_supply(sample_invoice_data)
        
        # Should have no issues as place_of_supply matches customer_state
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_place_of_supply_mismatch(self, compliance_checker, sample_invoice_data):
        """Test place of supply validation - mismatch case"""
        sample_invoice_data['place_of_supply'] = 'Tamil Nadu'  # Different from customer_state
        
        issues = await compliance_checker._validate_place_of_supply(sample_invoice_data)
        
        # Should have mismatch issue
        assert len(issues) > 0
        assert any(issue.type == ComplianceType.GST_VALIDATION for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_place_of_supply_missing(self, compliance_checker, sample_invoice_data):
        """Test place of supply validation - missing case"""
        sample_invoice_data['place_of_supply'] = None
        
        issues = await compliance_checker._validate_place_of_supply(sample_invoice_data)
        
        # Should have missing field issue
        assert len(issues) > 0
        assert any(issue.type == ComplianceType.MISSING_FIELDS for issue in issues)
    
    def test_calculate_compliance_score_no_issues(self, compliance_checker):
        """Test compliance score calculation with no issues"""
        score = compliance_checker._calculate_compliance_score([])
        assert score == 1.0
    
    def test_calculate_compliance_score_with_issues(self, compliance_checker):
        """Test compliance score calculation with various severity issues"""
        from app.models.base import ComplianceIssue
        
        issues = [
            ComplianceIssue(
                type=ComplianceType.MISSING_FIELDS,
                description="Test issue 1",
                plain_language_explanation="Test",
                severity=ComplianceSeverity.HIGH
            ),
            ComplianceIssue(
                type=ComplianceType.GST_VALIDATION,
                description="Test issue 2",
                plain_language_explanation="Test",
                severity=ComplianceSeverity.MEDIUM
            )
        ]
        
        score = compliance_checker._calculate_compliance_score(issues)
        assert 0.0 <= score < 1.0  # Should be less than 1 but not negative
    
    def test_determine_compliance_status_compliant(self, compliance_checker):
        """Test compliance status determination - compliant case"""
        status = compliance_checker._determine_compliance_status([])
        assert status == ComplianceStatus.COMPLIANT
    
    def test_determine_compliance_status_critical(self, compliance_checker):
        """Test compliance status determination - critical case"""
        from app.models.base import ComplianceIssue
        
        issues = [
            ComplianceIssue(
                type=ComplianceType.MISSING_FIELDS,
                description="Critical issue",
                plain_language_explanation="Test",
                severity=ComplianceSeverity.CRITICAL
            )
        ]
        
        status = compliance_checker._determine_compliance_status(issues)
        assert status == ComplianceStatus.CRITICAL_ISSUES
    
    def test_determine_compliance_status_issues_found(self, compliance_checker):
        """Test compliance status determination - issues found case"""
        from app.models.base import ComplianceIssue
        
        issues = [
            ComplianceIssue(
                type=ComplianceType.GST_VALIDATION,
                description="Medium issue",
                plain_language_explanation="Test",
                severity=ComplianceSeverity.MEDIUM
            )
        ]
        
        status = compliance_checker._determine_compliance_status(issues)
        assert status == ComplianceStatus.ISSUES_FOUND
    
    @pytest.mark.asyncio
    async def test_check_compliance_complete_flow(self, compliance_checker, sample_invoice_data):
        """Test complete compliance checking flow"""
        with patch.object(compliance_checker, '_get_invoice_data', return_value=sample_invoice_data):
            with patch.object(compliance_checker, 'validate_gst_number') as mock_gst_validation:
                # Mock GST validation to return valid results
                mock_gst_validation.return_value = GSTValidationResult(
                    gstin="27AAPFU0939F1ZV",
                    is_valid=True,
                    status="active"
                )
                
                result = await compliance_checker.check_compliance('inv-123')
                
                assert result.invoice_id == 'inv-123'
                assert result.success is True
                assert isinstance(result.compliance_score, float)
                assert result.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.ISSUES_FOUND]
    
    @pytest.mark.asyncio
    async def test_check_compliance_invoice_not_found(self, compliance_checker):
        """Test compliance checking when invoice is not found"""
        with patch.object(compliance_checker, '_get_invoice_data', return_value=None):
            result = await compliance_checker.check_compliance('nonexistent-invoice')
            
            assert result.success is False
            assert result.overall_status == ComplianceStatus.CRITICAL_ISSUES
            assert result.compliance_score == 0.0


@pytest.mark.asyncio
async def test_compliance_integration():
    """Integration test for compliance service"""
    checker = ComplianceChecker()
    
    # Test with mock data
    sample_data = {
        'id': 'test-invoice',
        'invoice_number': 'TEST-001',
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
                'name': 'Test Product',
                'taxable_value': 1000.0,
                'tax_rate': 18.0
            }
        ]
    }
    
    with patch.object(checker, '_get_invoice_data', return_value=sample_data):
        with patch.object(checker, 'validate_gst_number') as mock_gst:
            mock_gst.return_value = GSTValidationResult(
                gstin="27AAPFU0939F1ZV",
                is_valid=True,
                status="active"
            )
            
            # Test compliance check
            compliance_result = await checker.check_compliance('test-invoice')
            assert compliance_result.success is True
            
            # Test tax verification
            tax_result = await checker.verify_tax_calculations('test-invoice')
            assert tax_result.is_correct is True
            
            # Test GST validation
            gst_result = await checker.validate_gst_number('27AAPFU0939F1ZV')
            assert gst_result.is_valid is True