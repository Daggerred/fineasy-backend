"""
Tests for security and privacy features
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from app.utils.encryption import DataEncryption, SecureDataProcessor
from app.utils.anonymization import DataAnonymizer, PrivacyPreservingAnalyzer
from app.utils.audit_logger import AIAuditLogger, AuditEventType, AuditSeverity
from app.utils.data_retention import DataRetentionManager, DataCategory, RetentionPolicy
from app.utils.security_middleware import AISecurityMiddleware, RateLimiter, SecurityMonitor


class TestDataEncryption:
    """Test data encryption functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.encryption = DataEncryption()
        self.secure_processor = SecureDataProcessor()
    
    def test_encrypt_decrypt_data(self):
        """Test basic encryption and decryption"""
        test_data = {
            "customer_name": "John Doe",
            "amount": 1000.50,
            "account_number": "1234567890",
            "phone": "+91-9876543210"
        }
        
        # Encrypt data
        encrypted_data = self.encryption.encrypt_data(test_data)
        assert isinstance(encrypted_data, str)
        assert encrypted_data != json.dumps(test_data)
        
        # Decrypt data
        decrypted_data = self.encryption.decrypt_data(encrypted_data)
        assert decrypted_data == test_data
    
    def test_encrypt_decrypt_field(self):
        """Test single field encryption"""
        test_value = "sensitive_data_123"
        
        encrypted_value = self.encryption.encrypt_field(test_value)
        assert encrypted_value != test_value
        
        decrypted_value = self.encryption.decrypt_field(encrypted_value)
        assert decrypted_value == test_value
    
    def test_hash_data(self):
        """Test data hashing"""
        test_data = "test_data_for_hashing"
        hash_value = self.encryption.hash_data(test_data)
        
        assert len(hash_value) == 64  # SHA256 hex length
        assert hash_value != test_data
        
        # Same data should produce same hash
        hash_value2 = self.encryption.hash_data(test_data)
        assert hash_value == hash_value2
    
    def test_secure_data_processor(self):
        """Test secure data processing"""
        financial_data = {
            "customer_name": "Jane Smith",
            "amount": 2500.75,
            "account_number": "9876543210",
            "upi_id": "jane@paytm",
            "phone": "+91-8765432109",
            "email": "jane@example.com"
        }
        
        # Prepare for AI processing
        processed_data = self.secure_processor.prepare_for_ai_processing(financial_data)
        
        assert processed_data["_encrypted"] is True
        assert "amount_encrypted" in processed_data
        assert "account_number_encrypted" in processed_data
        
        # Original sensitive data should be hashed
        assert processed_data["amount"] != financial_data["amount"]
        assert processed_data["account_number"] != financial_data["account_number"]
        
        # Restore from processing
        restored_data = self.secure_processor.restore_from_ai_processing(processed_data)
        
        # Should match original data
        for key in ["customer_name", "amount", "account_number", "upi_id", "phone", "email"]:
            assert str(restored_data[key]) == str(financial_data[key])


class TestDataAnonymization:
    """Test data anonymization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.anonymizer = DataAnonymizer()
        self.privacy_analyzer = PrivacyPreservingAnalyzer()
    
    def test_anonymize_financial_data(self):
        """Test financial data anonymization"""
        original_data = {
            "customer_name": "Rajesh Kumar",
            "supplier_name": "ABC Suppliers Ltd",
            "phone": "+91-9876543210",
            "email": "rajesh@example.com",
            "address": "123 Main Street, Mumbai 400001",
            "account_number": "1234567890123456",
            "upi_id": "rajesh@paytm",
            "gstin": "27ABCDE1234F1Z5",
            "amount": 15000.50
        }
        
        anonymized_data = self.anonymizer.anonymize_financial_data(original_data)
        
        # Check that data is anonymized
        assert anonymized_data["customer_name"] != original_data["customer_name"]
        assert anonymized_data["supplier_name"] != original_data["supplier_name"]
        assert anonymized_data["phone"] != original_data["phone"]
        assert anonymized_data["email"] != original_data["email"]
        assert anonymized_data["account_number"] != original_data["account_number"]
        assert anonymized_data["upi_id"] != original_data["upi_id"]
        assert anonymized_data["gstin"] != original_data["gstin"]
        
        # Check that anonymization is consistent
        anonymized_data2 = self.anonymizer.anonymize_financial_data(original_data)
        assert anonymized_data["customer_name"] == anonymized_data2["customer_name"]
        
        # Check metadata
        assert anonymized_data["_anonymized"] is True
        assert "_anonymization_timestamp" in anonymized_data
    
    def test_anonymize_phone_number(self):
        """Test phone number anonymization"""
        phone_numbers = [
            "+91-9876543210",
            "9876543210",
            "+1-555-123-4567",
            "555.123.4567"
        ]
        
        for phone in phone_numbers:
            anonymized = self.anonymizer._anonymize_phone(phone)
            assert anonymized != phone
            assert "X" in anonymized
    
    def test_anonymize_email(self):
        """Test email anonymization"""
        emails = [
            "user@example.com",
            "test.user@company.co.in",
            "admin@domain.org"
        ]
        
        for email in emails:
            anonymized = self.anonymizer._anonymize_email(email)
            assert anonymized != email
            assert "@" in anonymized
            assert anonymized.endswith(".com") or anonymized.endswith(".org")
    
    def test_anonymize_amount(self):
        """Test amount anonymization"""
        amounts = [1000.50, 25000.75, 500.00, 100000.00]
        
        for amount in amounts:
            anonymized = self.anonymizer._anonymize_amount(amount)
            assert anonymized != amount
            # Should be within 10-20% of original
            assert 0.8 * amount <= anonymized <= 1.2 * amount
    
    def test_privacy_preserving_analyzer(self):
        """Test privacy-preserving analysis"""
        dataset = [
            {"customer_name": "Customer A", "amount": 1000, "category": "supplies"},
            {"customer_name": "Customer B", "amount": 2000, "category": "services"},
            {"customer_name": "Customer C", "amount": 1500, "category": "supplies"}
        ]
        
        anonymized_dataset = self.privacy_analyzer.prepare_dataset_for_ml(dataset)
        
        assert len(anonymized_dataset) == len(dataset)
        for record in anonymized_dataset:
            assert record["_anonymized"] is True
            assert record["customer_name"] != "Customer A"  # Should be anonymized


class TestAuditLogging:
    """Test audit logging functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.audit_logger = AIAuditLogger()
    
    def test_log_ai_operation(self):
        """Test basic AI operation logging"""
        audit_id = self.audit_logger.log_ai_operation(
            event_type=AuditEventType.FRAUD_DETECTION,
            business_id="test-business-123",
            user_id="test-user-456",
            operation_details={"test": "data"},
            severity=AuditSeverity.MEDIUM
        )
        
        assert audit_id is not None
        assert len(audit_id) == 36  # UUID length
    
    def test_log_data_processing(self):
        """Test data processing logging"""
        audit_id = self.audit_logger.log_data_processing(
            business_id="test-business-123",
            data_type="financial_transactions",
            operation="fraud_analysis",
            record_count=100,
            processing_duration=5.5,
            success=True
        )
        
        assert audit_id is not None
    
    def test_log_security_event(self):
        """Test security event logging"""
        audit_id = self.audit_logger.log_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            business_id="test-business-123",
            security_details={
                "violation_type": "rate_limit_exceeded",
                "client_ip": "192.168.1.100"
            },
            ip_address="192.168.1.100"
        )
        
        assert audit_id is not None
    
    def test_log_model_operation(self):
        """Test ML model operation logging"""
        audit_id = self.audit_logger.log_model_operation(
            business_id="test-business-123",
            model_name="fraud_detector_v1",
            operation="predict",
            model_version="1.0.0",
            input_features=["amount", "frequency", "pattern"],
            prediction_confidence=0.85
        )
        
        assert audit_id is not None
    
    def test_log_privacy_operation(self):
        """Test privacy operation logging"""
        audit_id = self.audit_logger.log_privacy_operation(
            business_id="test-business-123",
            privacy_operation="data_anonymization",
            data_fields=["customer_name", "phone", "email"],
            anonymization_method="hash_based"
        )
        
        assert audit_id is not None


class TestDataRetention:
    """Test data retention and purging functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.retention_manager = DataRetentionManager()
    
    @pytest.mark.asyncio
    async def test_apply_retention_policy(self):
        """Test applying retention policy"""
        # Test with expired data
        expired_date = datetime.utcnow() - timedelta(days=100)
        
        with patch.object(self.retention_manager, '_purge_data', return_value=True) as mock_purge:
            result = await self.retention_manager.apply_retention_policy(
                business_id="test-business-123",
                data_category=DataCategory.PROCESSED_AI_RESULTS,
                data_id="test-data-456",
                created_at=expired_date
            )
            
            assert result is True
            mock_purge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_user_requested_purge(self):
        """Test user-requested data purging"""
        categories = [DataCategory.PROCESSED_AI_RESULTS, DataCategory.ANONYMIZED_PATTERNS]
        
        with patch.object(self.retention_manager, '_get_ai_results_for_review', return_value=[]):
            with patch.object(self.retention_manager, '_get_anonymized_data_for_review', return_value=[]):
                result = await self.retention_manager.user_requested_purge(
                    business_id="test-business-123",
                    user_id="test-user-456",
                    data_categories=categories
                )
                
                assert "business_id" in result
                assert "user_id" in result
                assert "categories_purged" in result
    
    def test_get_retention_policy_info(self):
        """Test getting retention policy information"""
        policy_info = self.retention_manager.get_retention_policy_info(
            DataCategory.PROCESSED_AI_RESULTS
        )
        
        assert policy_info["category"] == "processed_ai_results"
        assert policy_info["retention_days"] == 90
        assert policy_info["auto_purge"] is True


class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.security_middleware = AISecurityMiddleware()
        self.rate_limiter = RateLimiter()
        self.security_monitor = SecurityMonitor()
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiting functionality"""
        client_ip = "192.168.1.100"
        business_id = "test-business-123"
        
        # Should allow initial requests
        for i in range(10):
            result = await self.rate_limiter.check_rate_limit(client_ip, business_id)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_security_assessment(self):
        """Test security score assessment"""
        score = await self.security_monitor.assess_request_security(
            client_ip="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            business_id="test-business-123",
            operation_type="fraud_detection"
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be considered safe
    
    @pytest.mark.asyncio
    async def test_security_assessment_suspicious(self):
        """Test security assessment with suspicious indicators"""
        score = await self.security_monitor.assess_request_security(
            client_ip="192.168.1.100",
            user_agent="bot",  # Bot indicator
            business_id="test-business-123",
            operation_type="fraud_detection"
        )
        
        assert score < 1.0  # Should be penalized for bot indicator
    
    @pytest.mark.asyncio
    async def test_secure_data_processing(self):
        """Test secure data processing"""
        test_data = {
            "customer_name": "Test Customer",
            "amount": 1000.50,
            "phone": "+91-9876543210"
        }
        
        secured_data = await self.security_middleware.secure_data_processing(
            data=test_data,
            business_id="test-business-123",
            operation_type="fraud_detection",
            require_anonymization=True
        )
        
        assert secured_data["_anonymized"] is True
        assert secured_data["customer_name"] != test_data["customer_name"]


class TestIntegrationSecurity:
    """Integration tests for security features"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_flow(self):
        """Test complete security flow from request to audit"""
        # Initialize components
        security_middleware = AISecurityMiddleware()
        audit_logger = AIAuditLogger()
        
        # Mock request data
        financial_data = {
            "customer_name": "Integration Test Customer",
            "amount": 5000.00,
            "account_number": "1234567890",
            "phone": "+91-9876543210"
        }
        
        business_id = "integration-test-business"
        
        # 1. Secure data processing
        secured_data = await security_middleware.secure_data_processing(
            data=financial_data,
            business_id=business_id,
            operation_type="fraud_detection",
            require_anonymization=True
        )
        
        # 2. Log the operation
        audit_id = audit_logger.log_ai_operation(
            event_type=AuditEventType.FRAUD_DETECTION,
            business_id=business_id,
            operation_details={"data_secured": True},
            severity=AuditSeverity.LOW
        )
        
        # 3. Verify security measures applied
        assert secured_data["_anonymized"] is True
        assert secured_data["customer_name"] != financial_data["customer_name"]
        assert audit_id is not None
    
    @pytest.mark.asyncio
    async def test_data_lifecycle_security(self):
        """Test complete data lifecycle with security measures"""
        # Initialize components
        encryption = DataEncryption()
        anonymizer = DataAnonymizer()
        retention_manager = DataRetentionManager()
        
        # Original data
        original_data = {
            "customer_name": "Lifecycle Test Customer",
            "amount": 7500.25,
            "account_number": "9876543210",
            "email": "test@example.com"
        }
        
        # 1. Encrypt sensitive data
        encrypted_data = encryption.encrypt_data(original_data)
        assert isinstance(encrypted_data, str)
        
        # 2. Anonymize for ML processing
        anonymized_data = anonymizer.anonymize_financial_data(original_data)
        assert anonymized_data["_anonymized"] is True
        
        # 3. Check retention policy
        policy_info = retention_manager.get_retention_policy_info(
            DataCategory.PROCESSED_AI_RESULTS
        )
        assert policy_info["auto_purge"] is True
        
        # 4. Decrypt data when needed
        decrypted_data = encryption.decrypt_data(encrypted_data)
        assert decrypted_data == original_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])