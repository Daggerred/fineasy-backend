"""
Tests for predictive analytics API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from app.main import app
from app.models.responses import (
    BusinessInsightsResponse, CashFlowPrediction, 
    CustomerAnalysis, WorkingCapitalAnalysis
)
from app.models.base import BusinessInsight, InsightType


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_token():
    """Mock authentication token"""
    return "Bearer test_token_123"


@pytest.fixture
def sample_business_id():
    """Sample business ID"""
    return "business_123"


@pytest.fixture
def sample_insights_response():
    """Sample insights response"""
    return BusinessInsightsResponse(
        business_id="business_123",
        insights=[
            BusinessInsight(
                id="insight_1",
                type=InsightType.CASH_FLOW,
                title="Cash Flow Warning",
                description="Your expenses will exceed income next month",
                recommendations=["Reduce expenses", "Increase collections"],
                impact_score=0.8,
                valid_until=datetime.utcnow() + timedelta(days=30)
            )
        ],
        generated_at=datetime.utcnow(),
        next_update=datetime.utcnow() + timedelta(hours=24)
    )


@pytest.fixture
def sample_cash_flow_prediction():
    """Sample cash flow prediction"""
    return CashFlowPrediction(
        predicted_inflow=50000.0,
        predicted_outflow=45000.0,
        net_cash_flow=5000.0,
        confidence=0.85,
        period_start=datetime.utcnow(),
        period_end=datetime.utcnow() + timedelta(days=90),
        factors=["Seasonal trends", "Historical patterns"]
    )


@pytest.fixture
def sample_customer_analysis():
    """Sample customer analysis"""
    return CustomerAnalysis(
        top_customers=[
            {"name": "Customer A", "revenue": 30000, "percentage": 30},
            {"name": "Customer B", "revenue": 25000, "percentage": 25},
            {"name": "Customer C", "revenue": 20000, "percentage": 20}
        ],
        revenue_concentration=0.75,
        pareto_analysis={"top_20_percent": 0.8},
        recommendations=["Diversify customer base", "Protect top customers"]
    )


@pytest.fixture
def sample_working_capital_analysis():
    """Sample working capital analysis"""
    return WorkingCapitalAnalysis(
        current_working_capital=100000.0,
        trend_direction="decreasing",
        days_until_depletion=45,
        recommendations=["Improve collections", "Extend payment terms"],
        risk_level="medium"
    )


class TestInsightsAPI:
    """Test cases for insights API endpoints"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    def test_get_business_insights_cached(
        self, mock_cache_get, mock_generate_insights, mock_verify_token, 
        client, mock_token, sample_business_id, sample_insights_response
    ):
        """Test getting cached business insights"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = sample_insights_response.dict()
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == sample_business_id
        assert len(data["insights"]) == 1
        assert data["insights"][0]["title"] == "Cash Flow Warning"
        
        # Verify cache was checked
        mock_cache_get.assert_called_once_with(f"insights:{sample_business_id}")
        # Verify insights generation was not called (cached result)
        mock_generate_insights.assert_not_called()
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_get_business_insights_not_cached(
        self, mock_cache_set, mock_cache_get, mock_generate_insights, 
        mock_verify_token, client, mock_token, sample_business_id, sample_insights_response
    ):
        """Test getting business insights when not cached"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None  # Not cached
        mock_generate_insights.return_value = sample_insights_response
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == sample_business_id
        
        # Verify insights were generated and cached
        mock_generate_insights.assert_called_once_with(sample_business_id)
        mock_cache_set.assert_called_once()
    
    @patch('app.api.insights.verify_token')
    def test_get_business_insights_unauthorized(self, mock_verify_token, client, sample_business_id):
        """Test unauthorized access to business insights"""
        # Setup mock
        mock_verify_token.return_value = None
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Assertions
        assert response.status_code == 401
        assert "Invalid authentication token" in response.json()["detail"]
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.delete')
    @patch('app.utils.cache.cache.set')
    def test_generate_insights_force_refresh(
        self, mock_cache_set, mock_cache_delete, mock_generate_insights, 
        mock_verify_token, client, mock_token, sample_business_id, sample_insights_response
    ):
        """Test force generating new insights"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_generate_insights.return_value = sample_insights_response
        
        # Make request
        response = client.post(
            f"/api/v1/insights/generate?business_id={sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == sample_business_id
        
        # Verify cache was cleared and new insights generated
        mock_cache_delete.assert_called_once_with(f"insights:{sample_business_id}")
        mock_generate_insights.assert_called_once_with(sample_business_id)
        mock_cache_set.assert_called_once()
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.predict_cash_flow')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_get_cash_flow_prediction(
        self, mock_cache_set, mock_cache_get, mock_predict_cash_flow, 
        mock_verify_token, client, mock_token, sample_business_id, sample_cash_flow_prediction
    ):
        """Test getting cash flow prediction"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None  # Not cached
        mock_predict_cash_flow.return_value = sample_cash_flow_prediction
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/cash-flow?months=3",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_inflow"] == 50000.0
        assert data["predicted_outflow"] == 45000.0
        assert data["net_cash_flow"] == 5000.0
        assert data["confidence"] == 0.85
        
        # Verify prediction was generated and cached
        mock_predict_cash_flow.assert_called_once_with(sample_business_id, 3)
        mock_cache_set.assert_called_once()
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.analyze_customer_revenue')
    @patch('app.utils.cache.cache.get')
    def test_get_customer_analysis_cached(
        self, mock_cache_get, mock_analyze_customer_revenue, mock_verify_token, 
        client, mock_token, sample_business_id, sample_customer_analysis
    ):
        """Test getting cached customer analysis"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = sample_customer_analysis.dict()
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/customer-analysis",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["top_customers"]) == 3
        assert data["revenue_concentration"] == 0.75
        assert "Diversify customer base" in data["recommendations"]
        
        # Verify analysis was not called (cached result)
        mock_analyze_customer_revenue.assert_not_called()
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.calculate_working_capital_trend')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_get_working_capital_analysis(
        self, mock_cache_set, mock_cache_get, mock_calculate_working_capital, 
        mock_verify_token, client, mock_token, sample_business_id, sample_working_capital_analysis
    ):
        """Test getting working capital analysis"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None  # Not cached
        mock_calculate_working_capital.return_value = sample_working_capital_analysis
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/working-capital",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["current_working_capital"] == 100000.0
        assert data["trend_direction"] == "decreasing"
        assert data["days_until_depletion"] == 45
        assert data["risk_level"] == "medium"
        
        # Verify analysis was generated and cached
        mock_calculate_working_capital.assert_called_once_with(sample_business_id)
        mock_cache_set.assert_called_once()
    
    @patch('app.api.insights.verify_token')
    @patch('app.utils.cache.cache.clear_pattern')
    def test_clear_insights_cache(
        self, mock_clear_pattern, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test clearing insights cache"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_clear_pattern.return_value = 5  # 5 entries cleared
        
        # Make request
        response = client.delete(
            f"/api/v1/insights/{sample_business_id}/cache",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "5 cache entries" in data["message"]
        assert data["business_id"] == sample_business_id
        
        # Verify cache patterns were cleared
        assert mock_clear_pattern.call_count == 4  # 4 different patterns
    
    @patch('app.api.insights.verify_token')
    def test_batch_generate_insights(self, mock_verify_token, client, mock_token):
        """Test batch generating insights for multiple businesses"""
        # Setup mock
        mock_verify_token.return_value = "user_123"
        business_ids = ["business_1", "business_2", "business_3"]
        
        # Make request
        response = client.post(
            "/api/v1/insights/batch-generate",
            json=business_ids,
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "3 businesses" in data["message"]
        assert data["business_ids"] == business_ids
    
    @patch('app.api.insights.verify_token')
    def test_batch_generate_insights_too_many(self, mock_verify_token, client, mock_token):
        """Test batch generating insights with too many businesses"""
        # Setup mock
        mock_verify_token.return_value = "user_123"
        business_ids = [f"business_{i}" for i in range(15)]  # 15 businesses (over limit)
        
        # Make request
        response = client.post(
            "/api/v1/insights/batch-generate",
            json=business_ids,
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 400
        assert "Maximum 10 businesses" in response.json()["detail"]
    
    def test_cash_flow_prediction_invalid_months(self, client, mock_token, sample_business_id):
        """Test cash flow prediction with invalid months parameter"""
        # Make request with invalid months
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/cash-flow?months=15",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    def test_insights_service_error(
        self, mock_generate_insights, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test handling of service errors"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_generate_insights.side_effect = Exception("Service unavailable")
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 500
        assert "Service unavailable" in response.json()["detail"]


class TestInsightsPerformance:
    """Performance tests for insights API"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_insights_response_time(
        self, mock_cache_set, mock_cache_get, mock_generate_insights, 
        mock_verify_token, client, mock_token, sample_business_id, sample_insights_response
    ):
        """Test insights API response time"""
        import time
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None
        mock_generate_insights.return_value = sample_insights_response
        
        # Measure response time
        start_time = time.time()
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        end_time = time.time()
        
        # Assertions
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 2.0  # Should respond within 2 seconds
    
    @patch('app.api.insights.verify_token')
    @patch('app.utils.cache.cache.get')
    def test_cache_hit_performance(
        self, mock_cache_get, mock_verify_token, client, mock_token, 
        sample_business_id, sample_insights_response
    ):
        """Test performance when cache hit occurs"""
        import time
        
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = sample_insights_response.dict()
        
        # Measure response time for cached result
        start_time = time.time()
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        end_time = time.time()
        
        # Assertions
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 0.5  # Cached responses should be very fast


class TestInsightsAccuracy:
    """Tests for prediction accuracy validation"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.predict_cash_flow')
    def test_cash_flow_prediction_accuracy_validation(
        self, mock_predict_cash_flow, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test cash flow prediction accuracy validation"""
        # Setup mock with high confidence prediction
        high_confidence_prediction = CashFlowPrediction(
            predicted_inflow=50000.0,
            predicted_outflow=45000.0,
            net_cash_flow=5000.0,
            confidence=0.95,  # High confidence
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Strong historical data", "Stable patterns"]
        )
        
        mock_verify_token.return_value = "user_123"
        mock_predict_cash_flow.return_value = high_confidence_prediction
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/cash-flow",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] >= 0.7  # Minimum acceptable confidence
        assert len(data["factors"]) > 0  # Should have supporting factors
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.analyze_customer_revenue')
    def test_customer_analysis_data_validation(
        self, mock_analyze_customer_revenue, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test customer analysis data validation"""
        # Setup mock with validated data
        validated_analysis = CustomerAnalysis(
            top_customers=[
                {"name": "Customer A", "revenue": 30000, "percentage": 30},
                {"name": "Customer B", "revenue": 25000, "percentage": 25},
                {"name": "Customer C", "revenue": 20000, "percentage": 20}
            ],
            revenue_concentration=0.75,
            pareto_analysis={"top_20_percent": 0.8, "validation": "passed"},
            recommendations=["Diversify customer base", "Protect top customers"]
        )
        
        mock_verify_token.return_value = "user_123"
        mock_analyze_customer_revenue.return_value = validated_analysis
        
        # Make request
        response = client.get(
            f"/api/v1/insights/{sample_business_id}/customer-analysis",
            headers={"Authorization": mock_token}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Validate data consistency
        total_percentage = sum(customer["percentage"] for customer in data["top_customers"])
        assert total_percentage <= 100  # Percentages should not exceed 100%
        
        # Validate revenue concentration makes sense
        assert 0 <= data["revenue_concentration"] <= 1
        
        # Validate recommendations are provided
        assert len(data["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])