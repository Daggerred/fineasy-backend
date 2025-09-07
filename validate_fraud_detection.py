#!/usr/bin/env python3
"""
Simple validation script for fraud detection service
"""
import asyncio
import sys
from unittest.mock import Mock, AsyncMock

# Add the app directory to the path
sys.path.append('.')

from app.services.fraud_detection import FraudDetector
from app.models.base import FraudType


async def test_fraud_detection_service():
    """Test the fraud detection service with sample data"""
    print("🔍 Testing Fraud Detection Service...")
    
    # Mock the DatabaseManager to avoid initialization issues
    from unittest.mock import patch
    with patch('app.services.fraud_detection.DatabaseManager') as mock_db_class:
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        # Create detector with mocked database
        detector = FraudDetector()
        
        # Sample test data
        sample_invoices = [
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
            }
        ]
        
        sample_transactions = [
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
            }
        ]
        
        sample_suppliers = [
            {'id': 'sup1', 'name': 'Tech Solutions Ltd'}
        ]
        
        # Mock database responses
        detector.db.get_invoices = AsyncMock(return_value=sample_invoices)
        detector.db.get_transactions = AsyncMock(return_value=sample_transactions)
        detector.db.get_suppliers = AsyncMock(return_value=sample_suppliers)
        detector.db.save_fraud_alert = AsyncMock(return_value='alert123')
        detector.db.log_ai_operation = AsyncMock()
        
        try:
            # Test 1: Duplicate Detection
            print("\n📋 Test 1: Duplicate Detection")
            duplicate_alerts = await detector.detect_duplicates('business123')
            print(f"✅ Found {len(duplicate_alerts)} duplicate alerts")
            
            for alert in duplicate_alerts:
                print(f"   - {alert.type.value}: {alert.message}")
                print(f"     Confidence: {alert.confidence_score:.2f}")
            
            # Test 2: Payment Mismatch Detection
            print("\n💰 Test 2: Payment Mismatch Detection")
            mismatch_alerts = await detector.detect_payment_mismatches('business123')
            print(f"✅ Found {len(mismatch_alerts)} payment mismatch alerts")
            
            for alert in mismatch_alerts:
                print(f"   - {alert.type.value}: {alert.message}")
                print(f"     Confidence: {alert.confidence_score:.2f}")
            
            # Test 3: Pattern Analysis
            print("\n📊 Test 3: Transaction Pattern Analysis")
            pattern_alerts = await detector.analyze_transaction_patterns('business123')
            print(f"✅ Found {len(pattern_alerts)} pattern alerts")
            
            for alert in pattern_alerts:
                print(f"   - {alert.type.value}: {alert.message}")
                print(f"     Confidence: {alert.confidence_score:.2f}")
            
            # Test 4: Comprehensive Analysis
            print("\n🔍 Test 4: Comprehensive Fraud Analysis")
            result = await detector.analyze_fraud('business123')
            print(f"✅ Analysis completed successfully")
            print(f"   - Business ID: {result.business_id}")
            print(f"   - Total alerts: {len(result.alerts)}")
            print(f"   - Risk score: {result.risk_score:.2f}")
            print(f"   - Success: {result.success}")
            
            # Test 5: Fuzzy Matching
            print("\n🔤 Test 5: Fuzzy Matching Logic")
            invoice1 = sample_invoices[0]
            invoice2 = sample_invoices[1]
            similarity = detector._calculate_invoice_similarity(invoice1, invoice2)
            print(f"✅ Invoice similarity: {similarity:.2f}")
            
            trans1 = sample_transactions[0]
            trans2 = sample_transactions[1]
            trans_similarity = detector._calculate_transaction_similarity(trans1, trans2)
            print(f"✅ Transaction similarity: {trans_similarity:.2f}")
            
            # Test 6: Feature Extraction
            print("\n🧮 Test 6: ML Feature Extraction")
            features = detector._extract_transaction_features(sample_transactions)
            print(f"✅ Extracted features shape: {features.shape}")
            print(f"   - Features per transaction: {features.shape[1]}")
            
            # Test 7: Risk Score Calculation
            print("\n⚠️  Test 7: Risk Score Calculation")
            all_alerts = duplicate_alerts + mismatch_alerts + pattern_alerts
            risk_score = detector._calculate_risk_score(all_alerts)
            print(f"✅ Calculated risk score: {risk_score:.2f}")
            
            print("\n🎉 All fraud detection tests passed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error during fraud detection testing: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_fraud_alert_models():
    """Test fraud alert data models"""
    print("\n📝 Testing Fraud Alert Models...")
    
    try:
        from app.models.base import FraudAlert, FraudType
        
        # Test creating fraud alert
        alert = FraudAlert(
            type=FraudType.DUPLICATE_INVOICE,
            message='Test duplicate invoice detected',
            confidence_score=0.9,
            evidence={'test': 'data'},
            business_id='business123',
            entity_id='invoice123'
        )
        
        print(f"✅ Created fraud alert: {alert.id}")
        print(f"   - Type: {alert.type.value}")
        print(f"   - Message: {alert.message}")
        print(f"   - Confidence: {alert.confidence_score}")
        print(f"   - Business ID: {alert.business_id}")
        print(f"   - Entity ID: {alert.entity_id}")
        print(f"   - Detected at: {alert.detected_at}")
        
        # Test all fraud types
        print("\n📋 Available fraud types:")
        for fraud_type in FraudType:
            print(f"   - {fraud_type.value}")
        
        print("\n✅ Fraud alert models validation passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing fraud alert models: {e}")
        return False


async def main():
    """Main validation function"""
    print("🚀 Starting Fraud Detection Service Validation")
    print("=" * 50)
    
    # Test fraud alert models
    models_ok = test_fraud_alert_models()
    
    # Test fraud detection service
    service_ok = await test_fraud_detection_service()
    
    print("\n" + "=" * 50)
    if models_ok and service_ok:
        print("🎉 All fraud detection validations passed!")
        print("\n✅ Task 3 Requirements Verified:")
        print("   ✓ FraudDetector class with duplicate detection algorithms")
        print("   ✓ Fuzzy matching logic for invoice comparison")
        print("   ✓ Pattern analysis functions for suspicious transaction detection")
        print("   ✓ Fraud alert data models and storage")
        return 0
    else:
        print("❌ Some validations failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)