"""
Tests for Fraud Detection Service
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np

from app.services.fraud_detection import FraudDetector
from app.models.base import FraudAlert, FraudType
from app.models.responses import FraudAnalysisResponse


class TestFraudDetector:
    """Test cases for FraudDetector class"""
    
    @pytest.fixture
    def fraud_detector(self):
        """Create a FraudDetector instance with mocked database"""
        with patch('app.services.fraud_detection.DatabaseManager') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            detector = FraudDetector()
            return detector
    
    @pytest.fixture
    def sample_invoices(self):
        """Sample invoice data for testing"""
        return [
            {
                'id': 'inv1',
                'invoice_number': 'INV-001',
                'customer_id': 'cust1',
                'total_amount': 1000.0,
                'description': 'Software development services',
                'created_at': '2024-01-15T10:00:00Z'
            },
            {
                'id': 'inv2',
                'invoice_number': 'INV-002',
                'customer_id': 'cust1',
                'total_amount': 1000.0,
                'description': 'Software development services',
                'created_at': '2024-01-15T11:00:00Z'
            },
            {
                'id': 'inv3',
                'invoice_number': 'INV-003',
                'customer_id': 'cust2',
                'total_amount': 500.0,
                'description': 'Consulting services',
                'created_at': '2024-01-16T10:00:00Z'
            }
        ]
    
    @pytest.fixture
    def sample_transactions(self):
        """Sample transaction data for testing"""
        return [
            {
                'id': 'trans1',
                'amount': 1000.0,
                'type': 'income',
                'description': 'Payment for INV-001',
                'created_at': '2024-01-15T12:00:00Z'
            },
            {
                'id': 'trans2',
                'amount': 1000.0,
                'type': 'income',
                'description': 'Payment for INV-001',
                'created_at': '2024-01-15T12:30:00Z'
            },
            {
                'id': 'trans3',
                'amount': 500.0,
                'type': 'expense',
                'description': 'Office supplies',
                'created_at': '2024-01-16T09:00:00Z'
            }
        ]
    
    @pytest.fixture
    def sample_suppliers(self):
        """Sample supplier data for testing"""
        return [
            {'id': 'sup1', 'name': 'Tech Solutions Ltd'},
            {'id': 'sup2', 'name': 'Office Supplies Co'}
        ]

    # =============================================================================
    # Test Duplicate Detection Algorithms
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_detect_duplicate_invoices(self, fraud_detector, sample_invoices):
        """Test duplicate invoice detection using fuzzy matching"""
        # Mock database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=sample_invoices)
        fraud_detector.db.get_transactions = AsyncMock(return_value=[])
        fraud_detector.db.get_suppliers = AsyncMock(return_value=[])
        
        # Test duplicate detection
        alerts = await fraud_detector.detect_duplicates('business123')
        
        # Should detect duplicate between inv1 and inv2
        assert len(alerts) > 0
        duplicate_alerts = [a for a in alerts if a.type == FraudType.DUPLICATE_INVOICE]
        assert len(duplicate_alerts) > 0
        
        # Check alert properties
        alert = duplicate_alerts[0]
        assert alert.confidence_score >= fraud_detector.duplicate_threshold
        assert 'similar' in alert.message.lower()
        assert len(alert.evidence) > 0
        assert 'similarity_score' in alert.evidence
    
    @pytest.mark.asyncio
    async def test_detect_duplicate_transactions(self, fraud_detector, sample_transactions):
        """Test duplicate transaction detection"""
        # Mock database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=[])
        fraud_detector.db.get_transactions = AsyncMock(return_value=sample_transactions)
        fraud_detector.db.get_suppliers = AsyncMock(return_value=[])
        
        # Test duplicate detection
        alerts = await fraud_detector.detect_duplicates('business123')
        
        # Should detect duplicate between trans1 and trans2
        duplicate_alerts = [a for a in alerts if a.type == FraudType.DUPLICATE_INVOICE]
        assert len(duplicate_alerts) > 0
        
        # Check alert details
        alert = duplicate_alerts[0]
        assert alert.confidence_score >= fraud_detector.duplicate_threshold
        assert 'duplicate transaction' in alert.message.lower()
    
    @pytest.mark.asyncio
    async def test_detect_supplier_duplicates(self, fraud_detector, sample_suppliers):
        """Test supplier duplicate billing detection"""
        # Create invoices from same supplier
        supplier_invoices = [
            {
                'id': 'inv1',
                'invoice_number': 'SUP-001',
                'supplier_id': 'sup1',
                'total_amount': 2000.0,
                'description': 'Hardware purchase',
                'created_at': '2024-01-15T10:00:00Z'
            },
            {
                'id': 'inv2',
                'invoice_number': 'SUP-002',
                'supplier_id': 'sup1',
                'total_amount': 2000.0,
                'description': 'Hardware purchase',
                'created_at': '2024-01-15T11:00:00Z'
            }
        ]
        
        # Mock database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=supplier_invoices)
        fraud_detector.db.get_transactions = AsyncMock(return_value=[])
        fraud_detector.db.get_suppliers = AsyncMock(return_value=sample_suppliers)
        
        # Test supplier duplicate detection
        alerts = await fraud_detector.detect_duplicates('business123')
        
        # Should detect supplier duplicate billing
        supplier_alerts = [a for a in alerts if a.type == FraudType.SUPPLIER_DUPLICATE]
        assert len(supplier_alerts) > 0
        
        # Check alert message
        alert = supplier_alerts[0]
        assert 'billed you twice' in alert.message.lower()
        assert 'Tech Solutions Ltd' in alert.message
    
    # =============================================================================
    # Test Fuzzy Matching Logic
    # =============================================================================
    
    def test_calculate_invoice_similarity(self, fraud_detector):
        """Test fuzzy matching logic for invoice comparison"""
        invoice1 = {
            'total_amount': 1000.0,
            'description': 'Software development services',
            'customer_id': 'cust1',
            'created_at': '2024-01-15T10:00:00Z'
        }
        
        invoice2 = {
            'total_amount': 1000.0,
            'description': 'Software development services',
            'customer_id': 'cust1',
            'created_at': '2024-01-15T11:00:00Z'
        }
        
        # Test high similarity
        similarity = fraud_detector._calculate_invoice_similarity(invoice1, invoice2)
        assert similarity >= 0.8  # Should be high similarity
        
        # Test low similarity
        invoice3 = {
            'total_amount': 500.0,
            'description': 'Consulting services',
            'customer_id': 'cust2',
            'created_at': '2024-01-20T10:00:00Z'
        }
        
        similarity_low = fraud_detector._calculate_invoice_similarity(invoice1, invoice3)
        assert similarity_low < 0.5  # Should be low similarity
    
    def test_calculate_transaction_similarity(self, fraud_detector):
        """Test fuzzy matching logic for transaction comparison"""
        trans1 = {
            'amount': 1000.0,
            'description': 'Payment for services',
            'type': 'income'
        }
        
        trans2 = {
            'amount': 1000.0,
            'description': 'Payment for services',
            'type': 'income'
        }
        
        # Test exact match
        similarity = fraud_detector._calculate_transaction_similarity(trans1, trans2)
        assert similarity >= 0.9  # Should be very high similarity
        
        # Test different amounts
        trans3 = {
            'amount': 500.0,
            'description': 'Different payment',
            'type': 'expense'
        }
        
        similarity_low = fraud_detector._calculate_transaction_similarity(trans1, trans3)
        assert similarity_low < 0.5  # Should be low similarity
    
    # =============================================================================
    # Test Payment Mismatch Detection
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_detect_payment_mismatches(self, fraud_detector):
        """Test payment mismatch detection between invoices and payments"""
        invoices = [
            {
                'id': 'inv1',
                'invoice_number': 'INV-001',
                'total_amount': 1000.0,
                'created_at': '2024-01-15T10:00:00Z'
            }
        ]
        
        transactions = [
            {
                'id': 'trans1',
                'amount': 800.0,  # Mismatch: less than invoice amount
                'type': 'income',
                'description': 'Payment for INV-001',
                'created_at': '2024-01-15T12:00:00Z'
            }
        ]
        
        # Mock database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=invoices)
        fraud_detector.db.get_transactions = AsyncMock(return_value=transactions)
        
        # Test payment mismatch detection
        alerts = await fraud_detector.detect_payment_mismatches('business123')
        
        # Should detect payment mismatch
        mismatch_alerts = [a for a in alerts if a.type == FraudType.PAYMENT_MISMATCH]
        assert len(mismatch_alerts) > 0
        
        # Check alert details
        alert = mismatch_alerts[0]
        assert 'mismatch' in alert.message.lower() or 'orphaned' in alert.message.lower()
        # Could be either payment mismatch or orphaned payment depending on invoice reference extraction
    
    @pytest.mark.asyncio
    async def test_detect_orphaned_payments(self, fraud_detector):
        """Test detection of payments without matching invoices"""
        invoices = [
            {
                'id': 'inv1',
                'invoice_number': 'INV-001',
                'total_amount': 1000.0
            }
        ]
        
        transactions = [
            {
                'id': 'trans1',
                'amount': 500.0,
                'type': 'income',
                'description': 'Payment for INV-999',  # Non-existent invoice
                'created_at': '2024-01-15T12:00:00Z'
            }
        ]
        
        # Mock database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=invoices)
        fraud_detector.db.get_transactions = AsyncMock(return_value=transactions)
        
        # Test orphaned payment detection
        alerts = await fraud_detector.detect_payment_mismatches('business123')
        
        # Should detect orphaned payment
        orphaned_alerts = [a for a in alerts if 'orphaned' in a.message.lower()]
        assert len(orphaned_alerts) > 0
        
        # Check alert details
        alert = orphaned_alerts[0]
        assert 'INV-999' in alert.message
    
    # =============================================================================
    # Test Pattern Analysis Functions
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_analyze_transaction_patterns(self, fraud_detector):
        """Test suspicious transaction pattern detection"""
        # Create transactions with suspicious patterns
        transactions = []
        base_time = datetime.now()
        
        # Create round number pattern (suspicious)
        for i in range(15):
            transactions.append({
                'id': f'trans{i}',
                'amount': 1000.0,  # Round number
                'type': 'income',
                'description': f'Payment {i}',
                'created_at': (base_time + timedelta(hours=i)).isoformat()
            })
        
        # Mock database calls
        fraud_detector.db.get_transactions = AsyncMock(return_value=transactions)
        
        # Test pattern analysis
        alerts = await fraud_detector.analyze_transaction_patterns('business123')
        
        # Should detect suspicious patterns
        pattern_alerts = [a for a in alerts if a.type == FraudType.SUSPICIOUS_PATTERN]
        assert len(pattern_alerts) > 0
        
        # Check for round number pattern detection
        round_alerts = [a for a in pattern_alerts if 'round number' in a.message.lower()]
        assert len(round_alerts) > 0
    
    def test_extract_transaction_features(self, fraud_detector):
        """Test feature extraction for ML analysis"""
        transactions = [
            {
                'amount': 1000.0,
                'type': 'income',
                'description': 'Payment for services',
                'created_at': '2024-01-15T14:30:00Z'
            },
            {
                'amount': 500.0,
                'type': 'expense',
                'description': 'Office supplies',
                'created_at': '2024-01-15T09:15:00Z'
            }
        ]
        
        # Test feature extraction
        features = fraud_detector._extract_transaction_features(transactions)
        
        # Should return numpy array with features
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 2  # Two transactions
        assert features.shape[1] == 6  # Six features per transaction
        
        # Check feature values
        assert features[0][0] == 1000.0  # Amount
        assert features[1][0] == 500.0   # Amount
    
    @pytest.mark.asyncio
    async def test_detect_velocity_anomalies(self, fraud_detector):
        """Test detection of high transaction velocity"""
        # Create many transactions in short time (suspicious velocity)
        transactions = []
        base_time = datetime.now()
        
        for i in range(15):  # More than 10 transactions in 1 hour
            transactions.append({
                'id': f'trans{i}',
                'amount': 100.0 + i,
                'type': 'income',
                'description': f'Quick payment {i}',
                'created_at': (base_time + timedelta(minutes=i*3)).isoformat()
            })
        
        # Test velocity detection
        alerts = await fraud_detector._detect_velocity_anomalies(transactions, 'business123')
        
        # Should detect high velocity
        velocity_alerts = [a for a in alerts if 'velocity' in a.message.lower()]
        assert len(velocity_alerts) > 0
        
        # Check alert details
        alert = velocity_alerts[0]
        assert alert.evidence['transaction_count'] > 10
        assert alert.evidence['time_window_minutes'] == 60
    
    # =============================================================================
    # Test Fraud Alert Data Models and Storage
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_save_fraud_alert(self, fraud_detector):
        """Test fraud alert storage"""
        # Mock database save method
        fraud_detector.db.save_fraud_alert = AsyncMock(return_value='alert123')
        
        # Create test alert
        alert = FraudAlert(
            type=FraudType.DUPLICATE_INVOICE,
            message='Test duplicate invoice',
            confidence_score=0.9,
            evidence={'test': 'data'},
            business_id='business123',
            entity_id='invoice123'
        )
        
        # Test saving alert
        alert_id = await fraud_detector._save_fraud_alert(alert)
        
        # Verify save was called
        assert alert_id == 'alert123'
        fraud_detector.db.save_fraud_alert.assert_called_once()
        
        # Check saved data structure
        call_args = fraud_detector.db.save_fraud_alert.call_args[0][0]
        assert call_args['business_id'] == 'business123'
        assert call_args['alert_type'] == FraudType.DUPLICATE_INVOICE.value
        assert call_args['message'] == 'Test duplicate invoice'
        assert call_args['confidence_score'] == 0.9
        assert call_args['status'] == 'active'
    
    def test_fraud_alert_model_validation(self):
        """Test FraudAlert model validation"""
        # Test valid alert creation
        alert = FraudAlert(
            type=FraudType.PAYMENT_MISMATCH,
            message='Payment mismatch detected',
            confidence_score=0.85,
            business_id='business123'
        )
        
        assert alert.type == FraudType.PAYMENT_MISMATCH
        assert alert.confidence_score == 0.85
        assert alert.business_id == 'business123'
        assert isinstance(alert.detected_at, datetime)
        assert isinstance(alert.id, str)
        
        # Test confidence score validation
        with pytest.raises(ValueError):
            FraudAlert(
                type=FraudType.DUPLICATE_INVOICE,
                message='Test',
                confidence_score=1.5,  # Invalid: > 1.0
                business_id='business123'
            )
    
    # =============================================================================
    # Test Comprehensive Fraud Analysis
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_analyze_fraud_comprehensive(self, fraud_detector, sample_invoices, 
                                             sample_transactions, sample_suppliers):
        """Test comprehensive fraud analysis workflow"""
        # Mock all database calls
        fraud_detector.db.get_invoices = AsyncMock(return_value=sample_invoices)
        fraud_detector.db.get_transactions = AsyncMock(return_value=sample_transactions)
        fraud_detector.db.get_suppliers = AsyncMock(return_value=sample_suppliers)
        fraud_detector.db.save_fraud_alert = AsyncMock(return_value='alert123')
        fraud_detector.db.log_ai_operation = AsyncMock()
        
        # Test comprehensive analysis
        result = await fraud_detector.analyze_fraud('business123')
        
        # Verify response structure
        assert isinstance(result, FraudAnalysisResponse)
        assert result.business_id == 'business123'
        assert isinstance(result.alerts, list)
        assert 0.0 <= result.risk_score <= 1.0
        assert isinstance(result.analysis_metadata, dict)
        
        # Verify metadata contains expected fields
        metadata = result.analysis_metadata
        assert 'duplicate_alerts' in metadata
        assert 'mismatch_alerts' in metadata
        assert 'pattern_alerts' in metadata
        assert 'total_alerts' in metadata
        assert 'analysis_timestamp' in metadata
    
    def test_calculate_risk_score(self, fraud_detector):
        """Test risk score calculation"""
        # Create test alerts with different types and confidence scores
        alerts = [
            FraudAlert(
                type=FraudType.DUPLICATE_INVOICE,
                message='Duplicate invoice',
                confidence_score=0.9,
                business_id='business123'
            ),
            FraudAlert(
                type=FraudType.PAYMENT_MISMATCH,
                message='Payment mismatch',
                confidence_score=0.8,
                business_id='business123'
            ),
            FraudAlert(
                type=FraudType.SUSPICIOUS_PATTERN,
                message='Suspicious pattern',
                confidence_score=0.7,
                business_id='business123'
            )
        ]
        
        # Test risk score calculation
        risk_score = fraud_detector._calculate_risk_score(alerts)
        
        # Should be weighted average, capped at 1.0
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.0  # Should be positive with alerts
        
        # Test empty alerts
        empty_risk_score = fraud_detector._calculate_risk_score([])
        assert empty_risk_score == 0.0
    
    # =============================================================================
    # Test Utility Functions
    # =============================================================================
    
    def test_extract_invoice_reference(self, fraud_detector):
        """Test invoice reference extraction from descriptions"""
        # Test various invoice reference patterns
        test_cases = [
            ('Payment for INV-001', '001'),
            ('Invoice 123 payment', '123'),
            ('Bill 456 received', '456'),
            ('Payment #789', '789'),
            ('Regular payment', None)  # No reference
        ]
        
        for description, expected in test_cases:
            result = fraud_detector._extract_invoice_reference(description)
            assert result == expected
    
    def test_date_calculations(self, fraud_detector):
        """Test date difference calculations"""
        date1 = '2024-01-15T10:00:00Z'
        date2 = '2024-01-17T10:00:00Z'
        
        # Test date difference in days
        diff_days = fraud_detector._calculate_date_difference(date1, date2)
        assert diff_days == 2
        
        # Test time difference in minutes
        time1 = '2024-01-15T10:00:00Z'
        time2 = '2024-01-15T12:30:00Z'
        diff_minutes = fraud_detector._calculate_time_difference_minutes(time1, time2)
        assert diff_minutes == 150  # 2.5 hours = 150 minutes
    
    def test_time_feature_extraction(self, fraud_detector):
        """Test time-based feature extraction"""
        timestamp = '2024-01-15T14:30:00Z'  # Monday, 2:30 PM
        
        # Test hour extraction
        hour = fraud_detector._extract_hour_from_timestamp(timestamp)
        assert hour == 14
        
        # Test day of week extraction
        day_of_week = fraud_detector._extract_day_of_week_from_timestamp(timestamp)
        assert 0 <= day_of_week <= 6  # Monday = 0, Sunday = 6
    
    # =============================================================================
    # Test Error Handling
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_fraud_analysis_error_handling(self, fraud_detector):
        """Test error handling in fraud analysis"""
        # Mock database to raise exception
        fraud_detector.db.get_invoices = AsyncMock(side_effect=Exception('Database error'))
        
        # Test error handling
        result = await fraud_detector.analyze_fraud('business123')
        
        # Should return error response
        assert result.success == False
        assert 'failed' in result.message.lower()
        assert result.risk_score == 0.0
        assert len(result.alerts) == 0
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_error_handling(self, fraud_detector):
        """Test error handling in duplicate detection"""
        # Mock database to raise exception
        fraud_detector.db.get_invoices = AsyncMock(side_effect=Exception('Database error'))
        fraud_detector.db.get_transactions = AsyncMock(return_value=[])
        fraud_detector.db.get_suppliers = AsyncMock(return_value=[])
        
        # Test error handling
        alerts = await fraud_detector.detect_duplicates('business123')
        
        # Should return empty list on error
        assert alerts == []
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_insufficient_data(self, fraud_detector):
        """Test pattern analysis with insufficient data"""
        # Mock database to return minimal data
        fraud_detector.db.get_transactions = AsyncMock(return_value=[
            {'id': '1', 'amount': 100, 'type': 'income', 'description': 'test', 'created_at': '2024-01-15T10:00:00Z'}
        ])
        
        # Test pattern analysis with insufficient data
        alerts = await fraud_detector.analyze_transaction_patterns('business123')
        
        # Should handle insufficient data gracefully
        assert isinstance(alerts, list)
        # May be empty or contain limited alerts due to insufficient data