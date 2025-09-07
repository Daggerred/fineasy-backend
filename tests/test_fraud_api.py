"""
Tests for Fraud Detection API endpoints
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
from datetime import datetime, timedelta
import json

from app.main import app
from app.api.fraud import (
    FraudAnalysisRequest, AlertUpdateRequest, BulkAnalysisRequest,
    _generate_alert_summary, _generate_recommendations, _calculate_next_analysis_time
)
from app.models.base import FraudAlert, FraudType
from app.models.responses import FraudAnalysisResponse


class TestFraudDetectionAPI:
    """Test cases for Fraud Detection API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth_token(self):
        """Mock authentication token"""
        return "Bearer test_token_123"
    
    @pytest.fixture
    def mock_user_info(self):
        """Mock user information"""
        return {
            "user_id": "user123",
            "email": "test@example.com",
            "business_id": "business123"
        }
    
    @pytest.fixture
    def sample_fraud_alerts(self):
        """Sample fraud alerts for testing"""
        return [
            FraudAlert(
                type=FraudType.DUPLICATE_INVOICE,
                message="Duplicate invoice detected",
                confidence_score=0.9,
                evidence={"similarity_score": 0.9},
                business_id="business123",
                entity_id="invoice123"
            ),
            FraudAlert(
                type=FraudType.PAYMENT_MISMATCH,
                message="Payment mismatch detected",
                confidence_score=0.8,
                evidence={"amount_difference": 100},
                business_id="business123",
                entity_id="payment123"
            ),
            FraudAlert(
                type=FraudType.SUSPICIOUS_PATTERN,
                message="Suspicious pattern detected",
                confidence_score=0.7,
                evidence={"pattern_type": "velocity"},
                business_id="business123"
            )
        ]
    
    # =============================================================================
    # Test Request Model Validation
    # =============================================================================
    
    def test_fraud_analysis_request_validation(self):
        """Test FraudAnalysisRequest model validation"""
        # Test valid request
        valid_request = FraudAnalysisRequest(
            business_id="business123",
            analysis_types=["duplicates", "mismatches"],
            date_range_days=30,
            include_resolved=False
        )
        assert valid_request.business_id == "business123"
        assert valid_request.analysis_types == ["duplicates", "mismatches"]
        assert valid_request.date_range_days == 30
        
        # Test invalid analysis types
        with pytest.raises(ValueError):
            FraudAnalysisRequest(
                business_id="business123",
                analysis_types=["invalid_type"]
            )
        
        # Test invalid date range
        with pytest.raises(ValueError):
            FraudAnalysisRequest(
                business_id="business123",
                date_range_days=400  # > 365
            )
    
    def test_alert_update_request_validation(self):
        """Test AlertUpdateRequest model validation"""
        # Test valid request
        valid_request = AlertUpdateRequest(
            status="resolved",
            resolution_notes="False positive"
        )
        assert valid_request.status == "resolved"
        assert valid_request.resolution_notes == "False positive"
        
        # Test invalid status
        with pytest.raises(ValueError):
            AlertUpdateRequest(status="invalid_status")
    
    def test_bulk_analysis_request_validation(self):
        """Test BulkAnalysisRequest model validation"""
        # Test valid request
        valid_request = BulkAnalysisRequest(
            business_ids=["business1", "business2"],
            analysis_types=["duplicates"]
        )
        assert len(valid_request.business_ids) == 2
        
        # Test too many business IDs
        with pytest.raises(ValueError):
            BulkAnalysisRequest(
                business_ids=[f"business{i}" for i in range(15)]  # > 10
            )
    
    # =============================================================================
    # Test Fraud Analysis Endpoint
    # =============================================================================
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_analyze_fraud_success(self, mock_detector_class, mock_verify_token, 
                                  client, mock_auth_token, mock_user_info, sample_fraud_alerts):
        """Test successful fraud analysis"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock analysis result
        mock_result = FraudAnalysisResponse(
            business_id="business123",
            alerts=sample_fraud_alerts,
            risk_score=0.8,
            analysis_metadata={
                "duplicate_alerts": 1,
                "mismatch_alerts": 1,
                "pattern_alerts": 1,
                "total_alerts": 3
            }
        )
        mock_detector.analyze_fraud = AsyncMock(return_value=mock_result)
        
        # Test request
        request_data = {
            "business_id": "business123",
            "analysis_types": ["duplicates", "mismatches"],
            "date_range_days": 30
        }
        
        response = client.post(
            "/api/v1/fraud/analyze",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == "business123"
        assert data["risk_score"] == 0.8
        assert len(data["alerts"]) == 3
        assert "alert_summary" in data
        assert "recommendations" in data
        assert "next_analysis_recommended" in data
        
        # Verify detector was called
        mock_detector.analyze_fraud.assert_called_once_with("business123")
    
    @patch('app.api.fraud.verify_token')
    def test_analyze_fraud_authentication_failure(self, mock_verify_token, client, mock_auth_token):
        """Test fraud analysis with authentication failure"""
        # Mock authentication failure
        mock_verify_token.side_effect = HTTPException(status_code=401, detail="Invalid token")
        
        request_data = {
            "business_id": "business123",
            "analysis_types": ["duplicates"]
        }
        
        response = client.post(
            "/api/v1/fraud/analyze",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_analyze_fraud_service_error(self, mock_detector_class, mock_verify_token,
                                       client, mock_auth_token, mock_user_info):
        """Test fraud analysis with service error"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector error
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.analyze_fraud = AsyncMock(side_effect=Exception("Service error"))
        
        request_data = {
            "business_id": "business123",
            "analysis_types": ["duplicates"]
        }
        
        response = client.post(
            "/api/v1/fraud/analyze",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        assert response.status_code == 500
        assert "Service error" in response.json()["detail"]
    
    # =============================================================================
    # Test Get Fraud Alerts Endpoint
    # =============================================================================
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_get_fraud_alerts_success(self, mock_detector_class, mock_verify_token,
                                    client, mock_auth_token, mock_user_info, sample_fraud_alerts):
        """Test successful fraud alerts retrieval"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.get_fraud_alerts = AsyncMock(return_value=sample_fraud_alerts)
        mock_detector._calculate_risk_score = Mock(return_value=0.8)
        
        response = client.get(
            "/api/v1/fraud/alerts/business123",
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == "business123"
        assert len(data["alerts"]) == 3
        assert data["risk_score"] == 0.8
        
        # Verify detector was called with correct parameters
        mock_detector.get_fraud_alerts.assert_called_once_with(
            business_id="business123",
            include_resolved=False,
            alert_type=None,
            limit=50,
            offset=0
        )
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_get_fraud_alerts_with_filters(self, mock_detector_class, mock_verify_token,
                                         client, mock_auth_token, mock_user_info):
        """Test fraud alerts retrieval with filters"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.get_fraud_alerts = AsyncMock(return_value=[])
        mock_detector._calculate_risk_score = Mock(return_value=0.0)
        
        response = client.get(
            "/api/v1/fraud/alerts/business123?include_resolved=true&alert_type=duplicate_invoice&limit=20&offset=10",
            headers={"Authorization": mock_auth_token}
        )
        
        assert response.status_code == 200
        
        # Verify detector was called with filters
        mock_detector.get_fraud_alerts.assert_called_once_with(
            business_id="business123",
            include_resolved=True,
            alert_type="duplicate_invoice",
            limit=20,
            offset=10
        )
    
    def test_get_fraud_alerts_invalid_alert_type(self, client, mock_auth_token):
        """Test fraud alerts retrieval with invalid alert type"""
        with patch('app.api.fraud.verify_token') as mock_verify_token:
            mock_verify_token.return_value = {"user_id": "user123"}
            
            response = client.get(
                "/api/v1/fraud/alerts/business123?alert_type=invalid_type",
                headers={"Authorization": mock_auth_token}
            )
            
            assert response.status_code == 400
            assert "Invalid alert type" in response.json()["detail"]
    
    # =============================================================================
    # Test Update Fraud Alert Endpoint
    # =============================================================================
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_update_fraud_alert_success(self, mock_detector_class, mock_verify_token,
                                      client, mock_auth_token, mock_user_info):
        """Test successful fraud alert update"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        updated_alert = {
            "id": "alert123",
            "status": "resolved",
            "resolution_notes": "False positive",
            "updated_at": datetime.utcnow().isoformat()
        }
        mock_detector.update_fraud_alert = AsyncMock(return_value=updated_alert)
        
        request_data = {
            "status": "resolved",
            "resolution_notes": "False positive"
        }
        
        response = client.put(
            "/api/v1/fraud/alerts/alert123",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "updated successfully" in data["message"]
        assert data["alert"]["id"] == "alert123"
        
        # Verify detector was called
        mock_detector.update_fraud_alert.assert_called_once_with(
            alert_id="alert123",
            status="resolved",
            resolution_notes="False positive",
            updated_by="user123"
        )
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_update_fraud_alert_not_found(self, mock_detector_class, mock_verify_token,
                                        client, mock_auth_token, mock_user_info):
        """Test fraud alert update when alert not found"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.update_fraud_alert = AsyncMock(return_value=None)
        
        request_data = {
            "status": "resolved"
        }
        
        response = client.put(
            "/api/v1/fraud/alerts/nonexistent",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    # =============================================================================
    # Test Bulk Fraud Analysis Endpoint
    # =============================================================================
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_bulk_fraud_analysis_success(self, mock_detector_class, mock_verify_token,
                                       client, mock_auth_token, mock_user_info):
        """Test successful bulk fraud analysis"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock analysis results for different businesses
        def mock_analyze_fraud(business_id):
            if business_id == "business1":
                return FraudAnalysisResponse(
                    business_id=business_id,
                    alerts=[],
                    risk_score=0.2,
                    analysis_metadata={}
                )
            else:
                return FraudAnalysisResponse(
                    business_id=business_id,
                    alerts=[FraudAlert(
                        type=FraudType.DUPLICATE_INVOICE,
                        message="Test alert",
                        confidence_score=0.9,
                        business_id=business_id
                    )],
                    risk_score=0.9,
                    analysis_metadata={}
                )
        
        mock_detector.analyze_fraud = AsyncMock(side_effect=mock_analyze_fraud)
        
        request_data = {
            "business_ids": ["business1", "business2"],
            "analysis_types": ["duplicates"]
        }
        
        response = client.post(
            "/api/v1/fraud/analyze/bulk",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["summary"]["successful_analyses"] == 2
        assert data["summary"]["total_alerts"] == 1
        assert "business1" in data["results"]
        assert "business2" in data["results"]
        assert data["results"]["business1"]["risk_score"] == 0.2
        assert data["results"]["business2"]["risk_score"] == 0.9
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_bulk_fraud_analysis_partial_failure(self, mock_detector_class, mock_verify_token,
                                                client, mock_auth_token, mock_user_info):
        """Test bulk fraud analysis with partial failures"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock analysis with one success and one failure
        def mock_analyze_fraud(business_id):
            if business_id == "business1":
                return FraudAnalysisResponse(
                    business_id=business_id,
                    alerts=[],
                    risk_score=0.1,
                    analysis_metadata={}
                )
            else:
                raise Exception("Analysis failed")
        
        mock_detector.analyze_fraud = AsyncMock(side_effect=mock_analyze_fraud)
        
        request_data = {
            "business_ids": ["business1", "business2"]
        }
        
        response = client.post(
            "/api/v1/fraud/analyze/bulk",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["successful_analyses"] == 1
        assert data["summary"]["failed_analyses"] == 1
        assert data["results"]["business1"]["success"] == True
        assert data["results"]["business2"]["success"] == False
        assert "error" in data["results"]["business2"]
    
    # =============================================================================
    # Test Fraud Statistics Endpoint
    # =============================================================================
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_get_fraud_statistics_success(self, mock_detector_class, mock_verify_token,
                                        client, mock_auth_token, mock_user_info):
        """Test successful fraud statistics retrieval"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        mock_stats = {
            "period": {"start_date": "2024-01-01", "end_date": "2024-01-31", "days": 30},
            "alert_counts": {"total": 10, "active": 5, "resolved": 4, "false_positives": 1},
            "alert_types": {"duplicate_invoice": 6, "payment_mismatch": 4},
            "risk_levels": {"high": 2, "medium": 5, "low": 3},
            "metrics": {"average_risk_score": 0.65, "resolution_rate": 50.0, "false_positive_rate": 10.0}
        }
        mock_detector.get_fraud_statistics = AsyncMock(return_value=mock_stats)
        
        response = client.get(
            "/api/v1/fraud/stats/business123?days=30",
            headers={"Authorization": mock_auth_token}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["business_id"] == "business123"
        assert data["period_days"] == 30
        assert data["statistics"]["alert_counts"]["total"] == 10
        assert data["statistics"]["metrics"]["average_risk_score"] == 0.65
        
        # Verify detector was called
        mock_detector.get_fraud_statistics.assert_called_once_with("business123", 30)
    
    # =============================================================================
    # Test Helper Functions
    # =============================================================================
    
    def test_generate_alert_summary(self, sample_fraud_alerts):
        """Test alert summary generation"""
        summary = _generate_alert_summary(sample_fraud_alerts)
        
        assert summary.total_alerts == 3
        assert summary.high_risk_alerts == 2  # confidence >= 0.8
        assert summary.medium_risk_alerts == 1  # 0.5 <= confidence < 0.8
        assert summary.low_risk_alerts == 0
        assert summary.alert_types["duplicate_invoice"] == 1
        assert summary.alert_types["payment_mismatch"] == 1
        assert summary.alert_types["suspicious_pattern"] == 1
        assert summary.latest_alert_time is not None
    
    def test_generate_alert_summary_empty(self):
        """Test alert summary generation with empty alerts"""
        summary = _generate_alert_summary([])
        
        assert summary.total_alerts == 0
        assert summary.high_risk_alerts == 0
        assert summary.medium_risk_alerts == 0
        assert summary.low_risk_alerts == 0
        assert summary.alert_types == {}
        assert summary.latest_alert_time is None
    
    def test_generate_recommendations(self, sample_fraud_alerts):
        """Test recommendation generation"""
        recommendations = _generate_recommendations(sample_fraud_alerts, 0.8)
        
        assert len(recommendations) > 0
        assert any("High fraud risk" in rec for rec in recommendations)
        assert any("invoice number validation" in rec for rec in recommendations)
        assert any("payment records" in rec for rec in recommendations)
        assert any("transaction patterns" in rec for rec in recommendations)
    
    def test_generate_recommendations_low_risk(self):
        """Test recommendation generation for low risk"""
        low_risk_alerts = [
            FraudAlert(
                type=FraudType.SUSPICIOUS_PATTERN,
                message="Low risk pattern",
                confidence_score=0.3,
                business_id="business123"
            )
        ]
        
        recommendations = _generate_recommendations(low_risk_alerts, 0.2)
        
        # Should not include high risk recommendations
        assert not any("High fraud risk" in rec for rec in recommendations)
        assert not any("Medium fraud risk" in rec for rec in recommendations)
    
    def test_calculate_next_analysis_time(self):
        """Test next analysis time calculation"""
        # High risk - should be 1 day
        high_risk_time = _calculate_next_analysis_time(0.9)
        expected_high = datetime.utcnow() + timedelta(days=1)
        assert abs((high_risk_time - expected_high).total_seconds()) < 60  # Within 1 minute
        
        # Medium risk - should be 3 days
        medium_risk_time = _calculate_next_analysis_time(0.6)
        expected_medium = datetime.utcnow() + timedelta(days=3)
        assert abs((medium_risk_time - expected_medium).total_seconds()) < 60
        
        # Low risk - should be 7 days
        low_risk_time = _calculate_next_analysis_time(0.2)
        expected_low = datetime.utcnow() + timedelta(days=7)
        assert abs((low_risk_time - expected_low).total_seconds()) < 60
    
    # =============================================================================
    # Test Error Handling and Edge Cases
    # =============================================================================
    
    def test_missing_authorization_header(self, client):
        """Test API calls without authorization header"""
        request_data = {
            "business_id": "business123"
        }
        
        response = client.post("/api/v1/fraud/analyze", json=request_data)
        assert response.status_code == 403  # FastAPI HTTPBearer returns 403 for missing auth
    
    def test_invalid_json_payload(self, client, mock_auth_token):
        """Test API calls with invalid JSON payload"""
        response = client.post(
            "/api/v1/fraud/analyze",
            data="invalid json",
            headers={"Authorization": mock_auth_token, "Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client, mock_auth_token):
        """Test API calls with missing required fields"""
        # Missing business_id
        response = client.post(
            "/api/v1/fraud/analyze",
            json={},
            headers={"Authorization": mock_auth_token}
        )
        assert response.status_code == 422
    
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.FraudDetector')
    def test_service_timeout_handling(self, mock_detector_class, mock_verify_token,
                                    client, mock_auth_token, mock_user_info):
        """Test handling of service timeouts"""
        # Mock authentication
        mock_verify_token.return_value = mock_user_info
        
        # Mock fraud detector timeout
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.analyze_fraud = AsyncMock(side_effect=asyncio.TimeoutError("Service timeout"))
        
        request_data = {
            "business_id": "business123"
        }
        
        response = client.post(
            "/api/v1/fraud/analyze",
            json=request_data,
            headers={"Authorization": mock_auth_token}
        )
        
        assert response.status_code == 500
        assert "timeout" in response.json()["detail"].lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestFraudDetectionAPIIntegration:
    """Integration tests for fraud detection API"""
    
    @pytest.mark.integration
    @patch('app.utils.auth.verify_supabase_token')
    @patch('app.database.DatabaseManager')
    def test_full_fraud_analysis_workflow(self, mock_db_manager, mock_verify_token):
        """Test complete fraud analysis workflow"""
        # This would be a full integration test with real database
        # For now, we'll mock the database interactions
        pass
    
    @pytest.mark.integration
    def test_api_response_schema_compliance(self):
        """Test that API responses comply with OpenAPI schema"""
        # This would validate response schemas against OpenAPI spec
        pass
    
    @pytest.mark.performance
    def test_fraud_analysis_performance(self):
        """Test fraud analysis performance under load"""
        # This would test API performance with large datasets
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])