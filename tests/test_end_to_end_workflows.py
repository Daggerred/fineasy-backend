"""
End-to-end tests for complete AI workflows
"""
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.main import app
from app.services.fraud_detection import FraudDetector
from app.services.predictive_analytics import PredictiveAnalyzer
from app.services.compliance import ComplianceChecker
from app.services.nlp_invoice import NLPInvoiceGenerator
from app.services.ml_engine import ml_engine


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers"""
    return {"Authorization": "Bearer test_token_123"}


@pytest.fixture
def sample_business_data():
    """Complete sample business data for end-to-end testing"""
    return {
        "business_id": "e2e_test_business",
        "transactions": [
            {
                "id": "trans_1",
                "amount": 10000.0,
                "type": "income",
                "description": "Payment from Customer A",
                "created_at": "2024-01-15T10:00:00Z"
            },
            {
                "id": "trans_2",
                "amount": 10000.0,
                "type": "income",
                "description": "Payment from Customer A",  # Potential duplicate
                "created_at": "2024-01-15T11:00:00Z"
            },
            {
                "id": "trans_3",
                "amount": 5000.0,
                "type": "expense",
                "description": "Office supplies",
                "created_at": "2024-01-16T09:00:00Z"
            }
        ],
        "invoices": [
            {
                "id": "inv_1",
                "invoice_number": "INV-001",
                "customer_id": "cust_1",
                "total_amount": 10000.0,
                "status": "paid",
                "gstin": "27AAPFU0939F1ZV",
                "created_at": "2024-01-15T09:00:00Z"
            },
            {
                "id": "inv_2",
                "invoice_number": "INV-002",
                "customer_id": "cust_2",
                "total_amount": 15000.0,
                "status": "pending",
                "gstin": "INVALID_GST",  # Invalid GST for compliance testing
                "created_at": "2024-01-16T10:00:00Z"
            }
        ],
        "customers": [
            {
                "id": "cust_1",
                "name": "Customer A",
                "total_revenue": 50000.0,
                "transaction_count": 10
            },
            {
                "id": "cust_2",
                "name": "Customer B",
                "total_revenue": 30000.0,
                "transaction_count": 5
            }
        ]
    }


class TestCompleteAIWorkflow:
    """Test complete AI analysis workflow from start to finish"""
    
    @patch('app.api.fraud.verify_token')
    @patch('app.api.insights.verify_token')
    @patch('app.api.compliance.verify_token')
    @patch('app.services.fraud_detection.DatabaseManager')
    @patch('app.services.predictive_analytics.DatabaseManager')
    @patch('app.services.compliance.DatabaseManager')
    def test_complete_business_analysis_workflow(
        self, mock_compliance_db, mock_analytics_db, mock_fraud_db,
        mock_compliance_verify, mock_insights_verify, mock_fraud_verify,
        client, auth_headers, sample_business_data
    ):
        """Test complete business analysis workflow"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mocks
        mock_fraud_verify.return_value = "user_123"
        mock_insights_verify.return_value = "user_123"
        mock_compliance_verify.return_value = "user_123"
        
        # Setup database mocks for fraud detection
        mock_fraud_db_instance = Mock()
        mock_fraud_db.return_value = mock_fraud_db_instance
        mock_fraud_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_fraud_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_fraud_db_instance.get_suppliers = AsyncMock(return_value=[])
        mock_fraud_db_instance.save_fraud_alert = AsyncMock(return_value="alert_123")
        mock_fraud_db_instance.log_ai_operation = AsyncMock()
        
        # Setup database mocks for predictive analytics
        mock_analytics_db_instance = Mock()
        mock_analytics_db.return_value = mock_analytics_db_instance
        mock_analytics_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_analytics_db_instance.get_customer_revenue_data = AsyncMock(return_value=sample_business_data["customers"])
        mock_analytics_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_analytics_db_instance.get_current_cash_balance = AsyncMock(return_value=75000.0)
        
        # Setup database mocks for compliance
        mock_compliance_db_instance = Mock()
        mock_compliance_db.return_value = mock_compliance_db_instance
        mock_compliance_db_instance.get_invoice = AsyncMock(return_value=sample_business_data["invoices"][1])  # Invalid GST invoice
        
        # Step 1: Run fraud analysis
        fraud_response = client.post(
            "/api/v1/fraud/analyze",
            json={"business_id": business_id},
            headers=auth_headers
        )
        
        assert fraud_response.status_code == 200
        fraud_data = fraud_response.json()
        assert fraud_data["business_id"] == business_id
        assert len(fraud_data["alerts"]) > 0  # Should detect duplicate transactions
        assert fraud_data["risk_score"] > 0.0
        
        # Step 2: Generate business insights
        with patch('app.utils.cache.cache.get', return_value=None):
            with patch('app.utils.cache.cache.set'):
                insights_response = client.get(
                    f"/api/v1/insights/{business_id}",
                    headers=auth_headers
                )
        
        assert insights_response.status_code == 200
        insights_data = insights_response.json()
        assert insights_data["business_id"] == business_id
        assert len(insights_data["insights"]) > 0
        
        # Step 3: Check compliance for problematic invoice
        compliance_response = client.post(
            "/api/v1/compliance/check",
            json={"invoice_id": "inv_2"},
            headers=auth_headers
        )
        
        assert compliance_response.status_code == 200
        compliance_data = compliance_response.json()
        assert compliance_data["invoice_id"] == "inv_2"
        assert len(compliance_data["issues"]) > 0  # Should detect invalid GST
        assert compliance_data["overall_status"] == "non_compliant"
        
        # Verify workflow completion
        assert fraud_data["analysis_metadata"]["total_alerts"] > 0
        assert len(insights_data["insights"]) >= 2  # Should have multiple insight types
        assert any(issue["type"] == "GST_VALIDATION" for issue in compliance_data["issues"])
    
    @patch('app.api.invoice.verify_token')
    @patch('app.services.nlp_invoice.DatabaseManager')
    @patch('spacy.load')
    def test_nlp_invoice_generation_workflow(
        self, mock_spacy_load, mock_nlp_db, mock_verify_token,
        client, auth_headers, sample_business_data
    ):
        """Test complete NLP invoice generation workflow"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mock
        mock_verify_token.return_value = "user_123"
        
        # Setup spaCy mock
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        # Mock entity extraction
        mock_doc = Mock()
        mock_entities = [
            Mock(text="Customer A", label_="ORG"),
            Mock(text="5", label_="CARDINAL"),
            Mock(text="laptops", label_="PRODUCT"),
            Mock(text="50000", label_="MONEY"),
            Mock(text="UPI", label_="PAYMENT")
        ]
        mock_doc.ents = mock_entities
        mock_nlp.return_value = mock_doc
        
        # Setup database mocks
        mock_db_instance = Mock()
        mock_nlp_db.return_value = mock_db_instance
        mock_db_instance.get_customers = AsyncMock(return_value=sample_business_data["customers"])
        mock_db_instance.get_products = AsyncMock(return_value=[
            {"id": "prod_1", "name": "Laptop Computer", "price": 50000.0, "unit": "piece"}
        ])
        mock_db_instance.create_invoice = AsyncMock(return_value={
            "id": "inv_generated",
            "invoice_number": "INV-003",
            "total_amount": 295000.0,
            "customer_name": "Customer A"
        })
        
        # Step 1: Parse natural language input
        parse_response = client.post(
            "/api/v1/invoice/parse",
            json={
                "text": "Generate invoice for Customer A, 5 laptops at 50000 each, UPI payment",
                "business_id": business_id
            },
            headers=auth_headers
        )
        
        assert parse_response.status_code == 200
        parse_data = parse_response.json()
        assert parse_data["customer_name"] == "Customer A"
        assert len(parse_data["items"]) == 1
        assert parse_data["items"][0]["quantity"] == 5
        assert parse_data["payment_preference"] == "UPI"
        
        # Step 2: Generate complete invoice
        generate_response = client.post(
            "/api/v1/invoice/generate",
            json={
                "text": "Generate invoice for Customer A, 5 laptops at 50000 each, UPI payment",
                "business_id": business_id
            },
            headers=auth_headers
        )
        
        assert generate_response.status_code == 200
        generate_data = generate_response.json()
        assert generate_data["success"] is True
        assert generate_data["invoice_id"] == "inv_generated"
        assert generate_data["invoice_data"]["total_amount"] == 295000.0
        
        # Verify workflow completion
        assert parse_data["raw_input"] is not None
        assert generate_data["invoice_data"]["customer_name"] == "Customer A"


class TestMLModelTrainingWorkflow:
    """Test ML model training and deployment workflow"""
    
    @patch('app.api.ml_engine.verify_token')
    @patch('app.services.ml_engine.DatabaseManager')
    def test_model_training_deployment_workflow(
        self, mock_ml_db, mock_verify_token, client, auth_headers, sample_business_data
    ):
        """Test complete ML model training and deployment workflow"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mock
        mock_verify_token.return_value = "user_123"
        
        # Setup database mocks
        mock_db_instance = Mock()
        mock_ml_db.return_value = mock_db_instance
        
        # Mock training data
        training_data = []
        for i, trans in enumerate(sample_business_data["transactions"]):
            training_data.append({
                **trans,
                "is_fraud": 1 if i == 1 else 0  # Mark second transaction as fraud (duplicate)
            })
        
        mock_db_instance.execute_query = AsyncMock(return_value=training_data)
        
        # Mock model storage operations
        with patch('app.services.ml_engine.MLModelManager._save_model') as mock_save_model:
            with patch('app.services.ml_engine.MLModelManager._store_model_metadata') as mock_store_metadata:
                with patch('app.services.ml_engine.MLModelManager._load_model') as mock_load_model:
                    
                    mock_save_model.return_value = "model_path.joblib"
                    mock_store_metadata.return_value = True
                    mock_load_model.return_value = Mock()
                    
                    # Step 1: Train fraud detection model
                    train_response = client.post(
                        "/api/v1/ml/train",
                        json={
                            "model_type": "fraud_detection",
                            "business_id": business_id,
                            "feature_columns": ["amount", "frequency", "time_diff"],
                            "target_column": "is_fraud"
                        },
                        headers=auth_headers
                    )
                    
                    assert train_response.status_code == 200
                    train_data = train_response.json()
                    assert train_data["model_name"] == f"fraud_detection_{business_id}"
                    assert train_data["status"] == "validating"
                    
                    # Step 2: Deploy trained model
                    deploy_response = client.post(
                        "/api/v1/ml/deploy",
                        json={
                            "model_name": train_data["model_name"],
                            "model_version": train_data["model_version"]
                        },
                        headers=auth_headers
                    )
                    
                    assert deploy_response.status_code == 200
                    deploy_data = deploy_response.json()
                    assert deploy_data["success"] is True
                    
                    # Step 3: Record feedback for model improvement
                    feedback_response = client.post(
                        "/api/v1/ml/feedback",
                        json={
                            "model_name": train_data["model_name"],
                            "prediction_id": "pred_123",
                            "actual_outcome": True,
                            "user_feedback": "Correctly identified duplicate transaction"
                        },
                        headers=auth_headers
                    )
                    
                    assert feedback_response.status_code == 200
                    feedback_data = feedback_response.json()
                    assert feedback_data["success"] is True
                    
                    # Verify workflow completion
                    assert train_data["business_id"] == business_id
                    assert "deployed successfully" in deploy_data["message"]
                    assert "recorded successfully" in feedback_data["message"]


class TestNotificationWorkflow:
    """Test smart notification generation workflow"""
    
    @patch('app.api.notifications.verify_token')
    @patch('app.api.fraud.verify_token')
    @patch('app.api.insights.verify_token')
    @patch('app.services.smart_notifications.DatabaseManager')
    @patch('app.services.fraud_detection.DatabaseManager')
    @patch('app.services.predictive_analytics.DatabaseManager')
    def test_smart_notification_workflow(
        self, mock_analytics_db, mock_fraud_db, mock_notifications_db,
        mock_insights_verify, mock_fraud_verify, mock_notifications_verify,
        client, auth_headers, sample_business_data
    ):
        """Test smart notification generation based on AI analysis"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mocks
        mock_fraud_verify.return_value = "user_123"
        mock_insights_verify.return_value = "user_123"
        mock_notifications_verify.return_value = "user_123"
        
        # Setup database mocks
        mock_fraud_db_instance = Mock()
        mock_fraud_db.return_value = mock_fraud_db_instance
        mock_fraud_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_fraud_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_fraud_db_instance.get_suppliers = AsyncMock(return_value=[])
        mock_fraud_db_instance.save_fraud_alert = AsyncMock(return_value="alert_123")
        mock_fraud_db_instance.log_ai_operation = AsyncMock()
        
        mock_analytics_db_instance = Mock()
        mock_analytics_db.return_value = mock_analytics_db_instance
        mock_analytics_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_analytics_db_instance.get_customer_revenue_data = AsyncMock(return_value=sample_business_data["customers"])
        mock_analytics_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_analytics_db_instance.get_current_cash_balance = AsyncMock(return_value=25000.0)  # Low balance
        
        mock_notifications_db_instance = Mock()
        mock_notifications_db.return_value = mock_notifications_db_instance
        mock_notifications_db_instance.get_user_preferences = AsyncMock(return_value={
            "fraud_alerts": True,
            "cash_flow_warnings": True,
            "compliance_reminders": True
        })
        mock_notifications_db_instance.save_notification = AsyncMock(return_value="notif_123")
        
        # Step 1: Generate fraud alerts
        fraud_response = client.post(
            "/api/v1/fraud/analyze",
            json={"business_id": business_id},
            headers=auth_headers
        )
        
        assert fraud_response.status_code == 200
        fraud_data = fraud_response.json()
        
        # Step 2: Generate business insights (should detect cash flow issues)
        with patch('app.utils.cache.cache.get', return_value=None):
            with patch('app.utils.cache.cache.set'):
                insights_response = client.get(
                    f"/api/v1/insights/{business_id}",
                    headers=auth_headers
                )
        
        assert insights_response.status_code == 200
        insights_data = insights_response.json()
        
        # Step 3: Generate smart notifications based on analysis
        notifications_response = client.post(
            "/api/v1/notifications/generate",
            json={"business_id": business_id},
            headers=auth_headers
        )
        
        assert notifications_response.status_code == 200
        notifications_data = notifications_response.json()
        
        # Verify notifications were generated based on analysis
        assert len(notifications_data["notifications"]) > 0
        
        # Should have fraud alert notification
        fraud_notifications = [n for n in notifications_data["notifications"] 
                             if n["type"] == "FRAUD_ALERT"]
        assert len(fraud_notifications) > 0
        
        # Should have cash flow warning (due to low balance)
        cash_flow_notifications = [n for n in notifications_data["notifications"] 
                                 if n["type"] == "CASH_FLOW_WARNING"]
        assert len(cash_flow_notifications) > 0
        
        # Verify notification priorities
        high_priority_notifications = [n for n in notifications_data["notifications"] 
                                     if n["priority"] == "HIGH"]
        assert len(high_priority_notifications) > 0


class TestDataPrivacyWorkflow:
    """Test data privacy and security workflow"""
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.DatabaseManager')
    @patch('app.utils.anonymization.anonymize_data')
    @patch('app.utils.audit_logger.log_ai_operation')
    def test_privacy_compliant_analysis_workflow(
        self, mock_audit_log, mock_anonymize, mock_fraud_db, mock_verify_token,
        client, auth_headers, sample_business_data
    ):
        """Test privacy-compliant AI analysis workflow"""
        business_id = sample_business_data["business_id"]
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        
        # Mock data anonymization
        anonymized_data = []
        for trans in sample_business_data["transactions"]:
            anonymized_trans = trans.copy()
            anonymized_trans["description"] = "ANONYMIZED_DESCRIPTION"
            anonymized_data.append(anonymized_trans)
        mock_anonymize.return_value = anonymized_data
        
        # Setup database mocks
        mock_db_instance = Mock()
        mock_fraud_db.return_value = mock_db_instance
        mock_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_db_instance.get_suppliers = AsyncMock(return_value=[])
        mock_db_instance.save_fraud_alert = AsyncMock(return_value="alert_123")
        mock_db_instance.log_ai_operation = AsyncMock()
        
        # Mock audit logging
        mock_audit_log.return_value = True
        
        # Run fraud analysis with privacy protection
        response = client.post(
            "/api/v1/fraud/analyze",
            json={"business_id": business_id, "privacy_mode": True},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify analysis completed
        assert data["business_id"] == business_id
        assert len(data["alerts"]) > 0
        
        # Verify privacy measures were applied
        mock_anonymize.assert_called()  # Data was anonymized
        mock_audit_log.assert_called()  # Operation was logged
        
        # Verify audit log contains privacy information
        audit_calls = mock_audit_log.call_args_list
        assert len(audit_calls) > 0
        audit_data = audit_calls[0][1]  # Get keyword arguments
        assert audit_data.get("privacy_mode") is True


class TestErrorRecoveryWorkflow:
    """Test error recovery and graceful degradation workflow"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.DatabaseManager')
    @patch('app.utils.cache.cache.get')
    def test_graceful_degradation_workflow(
        self, mock_cache_get, mock_analytics_db, mock_verify_token,
        client, auth_headers, sample_business_data
    ):
        """Test graceful degradation when services fail"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mock
        mock_verify_token.return_value = "user_123"
        
        # Mock cache miss
        mock_cache_get.return_value = None
        
        # Mock database failure
        mock_db_instance = Mock()
        mock_analytics_db.return_value = mock_db_instance
        mock_db_instance.get_transactions = AsyncMock(side_effect=Exception("Database connection failed"))
        
        # Request insights despite database failure
        response = client.get(
            f"/api/v1/insights/{business_id}",
            headers=auth_headers
        )
        
        # Should handle error gracefully
        assert response.status_code in [200, 500]  # Either graceful degradation or proper error
        
        if response.status_code == 200:
            # Graceful degradation - limited insights
            data = response.json()
            assert data["business_id"] == business_id
            # May have limited or cached insights
        else:
            # Proper error handling
            error_data = response.json()
            assert "detail" in error_data
            assert "failed" in error_data["detail"].lower()


class TestPerformanceWorkflow:
    """Test performance optimization workflow"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.DatabaseManager')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_caching_performance_workflow(
        self, mock_cache_set, mock_cache_get, mock_analytics_db, mock_verify_token,
        client, auth_headers, sample_business_data
    ):
        """Test caching performance optimization workflow"""
        business_id = sample_business_data["business_id"]
        
        # Setup authentication mock
        mock_verify_token.return_value = "user_123"
        
        # Setup database mocks
        mock_db_instance = Mock()
        mock_analytics_db.return_value = mock_db_instance
        mock_db_instance.get_transactions = AsyncMock(return_value=sample_business_data["transactions"])
        mock_db_instance.get_customer_revenue_data = AsyncMock(return_value=sample_business_data["customers"])
        mock_db_instance.get_invoices = AsyncMock(return_value=sample_business_data["invoices"])
        mock_db_instance.get_current_cash_balance = AsyncMock(return_value=75000.0)
        
        # First request - cache miss
        mock_cache_get.return_value = None
        
        response1 = client.get(
            f"/api/v1/insights/{business_id}",
            headers=auth_headers
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Verify cache was set
        mock_cache_set.assert_called()
        
        # Second request - cache hit
        cached_data = data1.copy()
        mock_cache_get.return_value = cached_data
        
        response2 = client.get(
            f"/api/v1/insights/{business_id}",
            headers=auth_headers
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should return same data from cache
        assert data1["business_id"] == data2["business_id"]
        
        # Database should not be called again for cached request
        # (This is implicit in the mock setup)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])