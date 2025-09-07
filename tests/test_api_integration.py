"""
Comprehensive integration tests for all AI API endpoints
"""
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.main import app
from app.models.responses import (
    FraudAnalysisResponse, BusinessInsightsResponse, 
    ComplianceResponse, InvoiceGenerationResponse
)
from app.models.base import FraudAlert, FraudType, BusinessInsight, InsightType


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers"""
    return {"Authorization": "Bearer test_token_123"}


@pytest.fixture
def sample_business_id():
    """Sample business ID"""
    return "business_123"


class TestFraudAPIIntegration:
    """Integration tests for Fraud Detection API"""
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector.analyze_fraud')
    def test_analyze_fraud_endpoint(self, mock_analyze_fraud, mock_verify_token, client, auth_headers, sample_business_id):
        """Test fraud analysis endpoint integration"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_fraud_response = FraudAnalysisResponse(
            business_id=sample_business_id,
            alerts=[
                FraudAlert(
                    type=FraudType.DUPLICATE_INVOICE,
                    message="Duplicate invoice detected",
                    confidence_score=0.9,
                    business_id=sample_business_id
                )
            ],
            risk_score=0.7,
            analysis_metadata={"total_alerts": 1},
            analyzed_at=datetime.utcnow()
        )
        mock_analyze_fraud.return_value = mock_fraud_response
        
        # Make request
        response = client.post(
            f"/api/v1/fraud/analyze",
            json={"business_id": sample_business_id},
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == sample_business_id
        assert len(data["alerts"]) == 1
        assert data["risk_score"] == 0.7
        assert data["alerts"][0]["type"] == FraudType.DUPLICATE_INVOICE.value
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector.detect_duplicates')
    def test_detect_duplicates_endpoint(self, mock_detect_duplicates, mock_verify_token, client, auth_headers, sample_business_id):
        """Test duplicate detection endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_alerts = [
            FraudAlert(
                type=FraudType.DUPLICATE_INVOICE,
                message="Duplicate invoice found",
                confidence_score=0.85,
                business_id=sample_business_id
            )
        ]
        mock_detect_duplicates.return_value = mock_alerts
        
        # Make request
        response = client.get(
            f"/api/v1/fraud/duplicates/{sample_business_id}",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["confidence_score"] == 0.85
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector.detect_payment_mismatches')
    def test_payment_mismatches_endpoint(self, mock_detect_mismatches, mock_verify_token, client, auth_headers, sample_business_id):
        """Test payment mismatch detection endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_alerts = [
            FraudAlert(
                type=FraudType.PAYMENT_MISMATCH,
                message="Payment amount mismatch detected",
                confidence_score=0.8,
                business_id=sample_business_id
            )
        ]
        mock_detect_mismatches.return_value = mock_alerts
        
        # Make request
        response = client.get(
            f"/api/v1/fraud/payment-mismatches/{sample_business_id}",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["type"] == FraudType.PAYMENT_MISMATCH.value


class TestInsightsAPIIntegration:
    """Integration tests for Insights API"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_get_insights_endpoint(self, mock_cache_set, mock_cache_get, mock_generate_insights, 
                                 mock_verify_token, client, auth_headers, sample_business_id):
        """Test business insights endpoint integration"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None  # Not cached
        
        mock_insights_response = BusinessInsightsResponse(
            business_id=sample_business_id,
            insights=[
                BusinessInsight(
                    type=InsightType.CASH_FLOW,
                    title="Cash Flow Warning",
                    description="Negative cash flow predicted",
                    recommendations=["Reduce expenses"],
                    impact_score=0.8
                )
            ],
            generated_at=datetime.utcnow(),
            next_update=datetime.utcnow() + timedelta(hours=24)
        )
        mock_generate_insights.return_value = mock_insights_response
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == sample_business_id
        assert len(data["insights"]) == 1
        assert data["insights"][0]["type"] == InsightType.CASH_FLOW.value
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.predict_cash_flow')
    def test_cash_flow_prediction_endpoint(self, mock_predict_cash_flow, mock_verify_token, 
                                         client, auth_headers, sample_business_id):
        """Test cash flow prediction endpoint"""
        from app.models.responses import CashFlowPrediction
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_prediction = CashFlowPrediction(
            predicted_inflow=50000.0,
            predicted_outflow=45000.0,
            net_cash_flow=5000.0,
            confidence=0.85,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Historical trends"]
        )
        mock_predict_cash_flow.return_value = mock_prediction
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/cash-flow?months=3",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_inflow"] == 50000.0
        assert data["net_cash_flow"] == 5000.0
        assert data["confidence"] == 0.85
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.analyze_customer_revenue')
    def test_customer_analysis_endpoint(self, mock_analyze_customer, mock_verify_token, 
                                      client, auth_headers, sample_business_id):
        """Test customer analysis endpoint"""
        from app.models.responses import CustomerAnalysis
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_analysis = CustomerAnalysis(
            top_customers=[
                {"name": "Customer A", "revenue": 30000, "percentage": 30}
            ],
            revenue_concentration=0.75,
            pareto_analysis={"top_20_percent": 0.8},
            recommendations=["Diversify customer base"]
        )
        mock_analyze_customer.return_value = mock_analysis
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/customer-analysis",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["top_customers"]) == 1
        assert data["revenue_concentration"] == 0.75


class TestComplianceAPIIntegration:
    """Integration tests for Compliance API"""
    
    @patch('app.api.compliance.verify_token')
    @patch('app.services.compliance.ComplianceChecker.check_invoice_compliance')
    def test_check_compliance_endpoint(self, mock_check_compliance, mock_verify_token, 
                                     client, auth_headers):
        """Test compliance checking endpoint"""
        from app.models.base import ComplianceIssue, ComplianceType, ComplianceSeverity
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_compliance_response = ComplianceResponse(
            invoice_id="invoice_123",
            issues=[
                ComplianceIssue(
                    type=ComplianceType.GST_VALIDATION,
                    description="Invalid GSTIN format",
                    plain_language_explanation="The GST number format is incorrect",
                    suggested_fixes=["Verify GST number format"],
                    severity=ComplianceSeverity.HIGH
                )
            ],
            overall_status="non_compliant",
            last_checked=datetime.utcnow()
        )
        mock_check_compliance.return_value = mock_compliance_response
        
        # Make request
        response = client.post(
            "/api/v1/compliance/check",
            json={"invoice_id": "invoice_123"},
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["invoice_id"] == "invoice_123"
        assert len(data["issues"]) == 1
        assert data["overall_status"] == "non_compliant"
    
    @patch('app.api.compliance.verify_token')
    @patch('app.services.compliance.ComplianceChecker.validate_gst_number')
    def test_validate_gst_endpoint(self, mock_validate_gst, mock_verify_token, 
                                 client, auth_headers):
        """Test GST validation endpoint"""
        from app.models.responses import GSTValidationResult
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_validation_result = GSTValidationResult(
            gstin="27AAPFU0939F1ZV",
            is_valid=True,
            business_name="Test Business",
            status="Active",
            registration_date="2020-01-01"
        )
        mock_validate_gst.return_value = mock_validation_result
        
        # Make request
        response = client.post(
            "/api/v1/compliance/validate-gst",
            json={"gstin": "27AAPFU0939F1ZV"},
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["gstin"] == "27AAPFU0939F1ZV"
        assert data["is_valid"] is True
        assert data["business_name"] == "Test Business"


class TestNLPInvoiceAPIIntegration:
    """Integration tests for NLP Invoice API"""
    
    @patch('app.api.invoice.verify_token')
    @patch('app.services.nlp_invoice.NLPInvoiceGenerator.parse_invoice_request')
    @patch('app.services.nlp_invoice.NLPInvoiceGenerator.resolve_entities')
    @patch('app.services.nlp_invoice.NLPInvoiceGenerator.generate_invoice')
    def test_generate_invoice_endpoint(self, mock_generate_invoice, mock_resolve_entities, 
                                     mock_parse_request, mock_verify_token, 
                                     client, auth_headers, sample_business_id):
        """Test NLP invoice generation endpoint"""
        from app.models.base import InvoiceRequest, InvoiceItem, ResolvedEntities
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        
        mock_request = InvoiceRequest(
            raw_input="Generate invoice for Test Customer, 5 laptops",
            business_id=sample_business_id,
            customer_name="Test Customer",
            items=[InvoiceItem(name="Laptop", quantity=5)]
        )
        mock_parse_request.return_value = mock_request
        
        mock_resolved = ResolvedEntities(
            customer_id="cust_123",
            resolved_items=[InvoiceItem(name="Laptop", quantity=5, unit_price=50000.0)]
        )
        mock_resolve_entities.return_value = mock_resolved
        
        mock_response = InvoiceGenerationResponse(
            success=True,
            invoice_id="inv_123",
            invoice_data={"total_amount": 295000.0},
            errors=[],
            suggestions=[]
        )
        mock_generate_invoice.return_value = mock_response
        
        # Make request
        response = client.post(
            "/api/v1/invoice/generate",
            json={
                "text": "Generate invoice for Test Customer, 5 laptops",
                "business_id": sample_business_id
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["invoice_id"] == "inv_123"
        assert data["invoice_data"]["total_amount"] == 295000.0
    
    @patch('app.api.invoice.verify_token')
    @patch('app.services.nlp_invoice.NLPInvoiceGenerator.parse_invoice_request')
    def test_parse_invoice_text_endpoint(self, mock_parse_request, mock_verify_token, 
                                       client, auth_headers, sample_business_id):
        """Test invoice text parsing endpoint"""
        from app.models.base import InvoiceRequest, InvoiceItem
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_request = InvoiceRequest(
            raw_input="Generate invoice for Test Customer, 3 chairs",
            business_id=sample_business_id,
            customer_name="Test Customer",
            items=[InvoiceItem(name="Chair", quantity=3)]
        )
        mock_parse_request.return_value = mock_request
        
        # Make request
        response = client.post(
            "/api/v1/invoice/parse",
            json={
                "text": "Generate invoice for Test Customer, 3 chairs",
                "business_id": sample_business_id
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["customer_name"] == "Test Customer"
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Chair"
        assert data["items"][0]["quantity"] == 3


class TestMLEngineAPIIntegration:
    """Integration tests for ML Engine API"""
    
    @patch('app.api.ml_engine.verify_token')
    @patch('app.services.ml_engine.ml_engine.train_model')
    def test_train_model_endpoint(self, mock_train_model, mock_verify_token, 
                                client, auth_headers, sample_business_id):
        """Test model training endpoint"""
        from app.models.responses import MLModelMetadata
        from app.services.ml_engine import ModelType, ModelStatus
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_metadata = MLModelMetadata(
            model_name=f"fraud_detection_{sample_business_id}",
            model_version="v1.0",
            model_type=ModelType.FRAUD_DETECTION,
            business_id=sample_business_id,
            training_data_hash="test_hash",
            status=ModelStatus.TRAINING
        )
        mock_train_model.return_value = mock_metadata
        
        # Make request
        response = client.post(
            "/api/v1/ml/train",
            json={
                "model_type": "fraud_detection",
                "business_id": sample_business_id,
                "feature_columns": ["amount", "frequency"],
                "target_column": "is_fraud"
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == f"fraud_detection_{sample_business_id}"
        assert data["status"] == ModelStatus.TRAINING.value
    
    @patch('app.api.ml_engine.verify_token')
    @patch('app.services.ml_engine.ml_engine.deploy_model')
    def test_deploy_model_endpoint(self, mock_deploy_model, mock_verify_token, 
                                 client, auth_headers):
        """Test model deployment endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_deploy_model.return_value = True
        
        # Make request
        response = client.post(
            "/api/v1/ml/deploy",
            json={
                "model_name": "fraud_detection_business_123",
                "model_version": "v1.0"
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deployed successfully" in data["message"]
    
    @patch('app.api.ml_engine.verify_token')
    @patch('app.services.ml_engine.ml_engine.record_feedback')
    def test_record_feedback_endpoint(self, mock_record_feedback, mock_verify_token, 
                                    client, auth_headers):
        """Test feedback recording endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_record_feedback.return_value = True
        
        # Make request
        response = client.post(
            "/api/v1/ml/feedback",
            json={
                "model_name": "fraud_detection_business_123",
                "prediction_id": "pred_123",
                "actual_outcome": True,
                "user_feedback": "Correctly identified fraud"
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestNotificationsAPIIntegration:
    """Integration tests for Notifications API"""
    
    @patch('app.api.notifications.verify_token')
    @patch('app.services.smart_notifications.SmartNotificationManager.generate_notifications')
    def test_generate_notifications_endpoint(self, mock_generate_notifications, mock_verify_token, 
                                           client, auth_headers, sample_business_id):
        """Test notification generation endpoint"""
        from app.models.responses import NotificationResponse
        from app.models.base import SmartNotification, NotificationType, NotificationPriority
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_notifications = [
            SmartNotification(
                type=NotificationType.FRAUD_ALERT,
                title="Fraud Alert",
                message="Duplicate invoice detected",
                priority=NotificationPriority.HIGH,
                business_id=sample_business_id
            )
        ]
        mock_generate_notifications.return_value = mock_notifications
        
        # Make request
        response = client.post(
            "/api/v1/notifications/generate",
            json={"business_id": sample_business_id},
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["notifications"]) == 1
        assert data["notifications"][0]["type"] == NotificationType.FRAUD_ALERT.value
        assert data["notifications"][0]["priority"] == NotificationPriority.HIGH.value


class TestCacheManagementAPIIntegration:
    """Integration tests for Cache Management API"""
    
    @patch('app.api.cache_management.verify_token')
    @patch('app.utils.cache.cache.clear_pattern')
    def test_clear_cache_endpoint(self, mock_clear_pattern, mock_verify_token, 
                                client, auth_headers, sample_business_id):
        """Test cache clearing endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_clear_pattern.return_value = 5  # 5 entries cleared
        
        # Make request
        response = client.delete(
            f"/api/v1/cache/clear/{sample_business_id}",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "5 cache entries" in data["message"]
    
    @patch('app.api.cache_management.verify_token')
    @patch('app.utils.cache.cache.get_stats')
    def test_cache_stats_endpoint(self, mock_get_stats, mock_verify_token, 
                                client, auth_headers):
        """Test cache statistics endpoint"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_get_stats.return_value = {
            "total_keys": 150,
            "memory_usage": "25MB",
            "hit_rate": 0.85,
            "miss_rate": 0.15
        }
        
        # Make request
        response = client.get(
            "/api/v1/cache/stats",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total_keys"] == 150
        assert data["hit_rate"] == 0.85


class TestAPIErrorHandling:
    """Test error handling across all API endpoints"""
    
    def test_unauthorized_access(self, client, sample_business_id):
        """Test unauthorized access to protected endpoints"""
        endpoints = [
            f"/api/v1/fraud/analyze",
            f"/api/v1/insights/{sample_business_id}",
            f"/api/v1/compliance/check",
            f"/api/v1/invoice/generate"
        ]
        
        for endpoint in endpoints:
            if "analyze" in endpoint or "check" in endpoint or "generate" in endpoint:
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 401
    
    @patch('app.api.fraud.verify_token')
    def test_invalid_business_id(self, mock_verify_token, client, auth_headers):
        """Test handling of invalid business ID"""
        mock_verify_token.return_value = "user_123"
        
        response = client.post(
            "/api/v1/fraud/analyze",
            json={"business_id": ""},  # Empty business ID
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    def test_service_error_handling(self, mock_generate_insights, mock_verify_token, 
                                  client, auth_headers, sample_business_id):
        """Test handling of service errors"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_generate_insights.side_effect = Exception("Service unavailable")
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers=auth_headers
        )
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Service unavailable" in response.json()["detail"]


class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    def test_cached_response_performance(self, mock_cache_get, mock_generate_insights, 
                                       mock_verify_token, client, auth_headers, sample_business_id):
        """Test performance of cached responses"""
        import time
        
        # Setup mocks for cached response
        mock_verify_token.return_value = "user_123"
        cached_response = {
            "business_id": sample_business_id,
            "insights": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        mock_cache_get.return_value = cached_response
        
        # Measure response time
        start_time = time.time()
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers=auth_headers
        )
        end_time = time.time()
        
        # Assertions
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Cached responses should be fast
        
        # Verify insights generation was not called (cached)
        mock_generate_insights.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])