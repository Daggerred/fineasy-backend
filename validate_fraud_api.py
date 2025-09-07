#!/usr/bin/env python3
"""
Validation script for Fraud Detection API endpoints
"""
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from app.api.fraud import (
    FraudAnalysisRequest, AlertUpdateRequest, BulkAnalysisRequest,
    _generate_alert_summary, _generate_recommendations, _calculate_next_analysis_time
)
from app.models.base import FraudAlert, FraudType
from app.models.responses import FraudAnalysisResponse
from app.services.fraud_detection import FraudDetector


def test_request_models():
    """Test request model validation"""
    print("Testing request model validation...")
    
    # Test FraudAnalysisRequest
    try:
        request = FraudAnalysisRequest(
            business_id="test_business_123",
            analysis_types=["duplicates", "mismatches", "patterns"],
            date_range_days=30,
            include_resolved=False
        )
        print(f"✓ FraudAnalysisRequest validation passed: {request.business_id}")
    except Exception as e:
        print(f"✗ FraudAnalysisRequest validation failed: {e}")
    
    # Test AlertUpdateRequest
    try:
        update_request = AlertUpdateRequest(
            status="resolved",
            resolution_notes="False positive - customer confirmed legitimate transaction"
        )
        print(f"✓ AlertUpdateRequest validation passed: {update_request.status}")
    except Exception as e:
        print(f"✗ AlertUpdateRequest validation failed: {e}")
    
    # Test BulkAnalysisRequest
    try:
        bulk_request = BulkAnalysisRequest(
            business_ids=["business1", "business2", "business3"],
            analysis_types=["duplicates", "mismatches"]
        )
        print(f"✓ BulkAnalysisRequest validation passed: {len(bulk_request.business_ids)} businesses")
    except Exception as e:
        print(f"✗ BulkAnalysisRequest validation failed: {e}")
    
    # Test invalid requests
    try:
        invalid_request = FraudAnalysisRequest(
            business_id="test",
            analysis_types=["invalid_type"]  # Should fail
        )
        print("✗ Invalid analysis type validation should have failed")
    except ValueError:
        print("✓ Invalid analysis type correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def test_helper_functions():
    """Test helper functions"""
    print("\nTesting helper functions...")
    
    # Create sample alerts
    sample_alerts = [
        FraudAlert(
            type=FraudType.DUPLICATE_INVOICE,
            message="Duplicate invoice detected between INV-001 and INV-002",
            confidence_score=0.95,
            evidence={"similarity_score": 0.95, "amount_difference": 0},
            business_id="test_business",
            entity_id="invoice_002"
        ),
        FraudAlert(
            type=FraudType.PAYMENT_MISMATCH,
            message="Payment mismatch: Invoice ₹1000 vs Payment ₹800",
            confidence_score=0.85,
            evidence={"invoice_amount": 1000, "payment_amount": 800},
            business_id="test_business",
            entity_id="payment_123"
        ),
        FraudAlert(
            type=FraudType.SUSPICIOUS_PATTERN,
            message="High transaction velocity detected",
            confidence_score=0.70,
            evidence={"transaction_count": 15, "time_window_minutes": 60},
            business_id="test_business"
        )
    ]
    
    # Test alert summary generation
    try:
        summary = _generate_alert_summary(sample_alerts)
        print(f"✓ Alert summary generated: {summary.total_alerts} total, {summary.high_risk_alerts} high risk")
        print(f"  Alert types: {summary.alert_types}")
    except Exception as e:
        print(f"✗ Alert summary generation failed: {e}")
    
    # Test recommendations generation
    try:
        recommendations = _generate_recommendations(sample_alerts, 0.8)
        print(f"✓ Recommendations generated: {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")
    except Exception as e:
        print(f"✗ Recommendations generation failed: {e}")
    
    # Test next analysis time calculation
    try:
        high_risk_time = _calculate_next_analysis_time(0.9)
        medium_risk_time = _calculate_next_analysis_time(0.6)
        low_risk_time = _calculate_next_analysis_time(0.2)
        
        print(f"✓ Next analysis times calculated:")
        print(f"  High risk (0.9): {high_risk_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Medium risk (0.6): {medium_risk_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Low risk (0.2): {low_risk_time.strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        print(f"✗ Next analysis time calculation failed: {e}")


async def test_fraud_detector_integration():
    """Test fraud detector service integration"""
    print("\nTesting fraud detector integration...")
    
    try:
        # Mock the database manager
        with Mock() as mock_db:
            # Create fraud detector instance
            detector = FraudDetector()
            detector.db = mock_db
            
            # Mock database responses
            mock_db.get_invoices = AsyncMock(return_value=[
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
            ])
            
            mock_db.get_transactions = AsyncMock(return_value=[
                {
                    'id': 'trans1',
                    'amount': 1000.0,
                    'type': 'income',
                    'description': 'Payment for INV-001',
                    'created_at': '2024-01-15T12:00:00Z'
                }
            ])
            
            mock_db.get_suppliers = AsyncMock(return_value=[])
            mock_db.save_fraud_alert = AsyncMock(return_value='alert_123')
            mock_db.log_ai_operation = AsyncMock(return_value='log_123')
            
            # Test duplicate detection
            duplicate_alerts = await detector.detect_duplicates('test_business')
            print(f"✓ Duplicate detection completed: {len(duplicate_alerts)} alerts found")
            
            # Test payment mismatch detection
            mismatch_alerts = await detector.detect_payment_mismatches('test_business')
            print(f"✓ Payment mismatch detection completed: {len(mismatch_alerts)} alerts found")
            
            # Test comprehensive fraud analysis
            analysis_result = await detector.analyze_fraud('test_business')
            print(f"✓ Comprehensive fraud analysis completed:")
            print(f"  Business ID: {analysis_result.business_id}")
            print(f"  Risk Score: {analysis_result.risk_score:.2f}")
            print(f"  Total Alerts: {len(analysis_result.alerts)}")
            print(f"  Success: {analysis_result.success}")
            
    except Exception as e:
        print(f"✗ Fraud detector integration test failed: {e}")


def test_response_models():
    """Test response model serialization"""
    print("\nTesting response models...")
    
    try:
        # Create sample fraud analysis response
        sample_alerts = [
            FraudAlert(
                type=FraudType.DUPLICATE_INVOICE,
                message="Test duplicate",
                confidence_score=0.9,
                business_id="test_business"
            )
        ]
        
        response = FraudAnalysisResponse(
            business_id="test_business",
            alerts=sample_alerts,
            risk_score=0.8,
            analysis_metadata={
                "duplicate_alerts": 1,
                "total_alerts": 1,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Test JSON serialization
        response_json = response.model_dump()
        print(f"✓ FraudAnalysisResponse serialization successful")
        print(f"  Business ID: {response_json['business_id']}")
        print(f"  Risk Score: {response_json['risk_score']}")
        print(f"  Alert Count: {len(response_json['alerts'])}")
        
    except Exception as e:
        print(f"✗ Response model test failed: {e}")


def test_error_handling():
    """Test error handling scenarios"""
    print("\nTesting error handling...")
    
    # Test invalid field values
    test_cases = [
        {
            "name": "Invalid analysis type",
            "data": {"business_id": "test", "analysis_types": ["invalid"]},
            "model": FraudAnalysisRequest,
            "should_fail": True
        },
        {
            "name": "Invalid date range",
            "data": {"business_id": "test", "date_range_days": 400},
            "model": FraudAnalysisRequest,
            "should_fail": True
        },
        {
            "name": "Invalid alert status",
            "data": {"status": "invalid_status"},
            "model": AlertUpdateRequest,
            "should_fail": True
        },
        {
            "name": "Too many business IDs",
            "data": {"business_ids": [f"business{i}" for i in range(15)]},
            "model": BulkAnalysisRequest,
            "should_fail": True
        }
    ]
    
    for test_case in test_cases:
        try:
            model_instance = test_case["model"](**test_case["data"])
            if test_case["should_fail"]:
                print(f"✗ {test_case['name']}: Should have failed but didn't")
            else:
                print(f"✓ {test_case['name']}: Passed as expected")
        except (ValueError, TypeError) as e:
            if test_case["should_fail"]:
                print(f"✓ {test_case['name']}: Correctly rejected - {str(e)[:50]}...")
            else:
                print(f"✗ {test_case['name']}: Unexpected failure - {e}")
        except Exception as e:
            print(f"✗ {test_case['name']}: Unexpected error - {e}")


async def main():
    """Run all validation tests"""
    print("=" * 60)
    print("FRAUD DETECTION API VALIDATION")
    print("=" * 60)
    
    test_request_models()
    test_helper_functions()
    await test_fraud_detector_integration()
    test_response_models()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())