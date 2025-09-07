"""
Integration tests for predictive analytics workflow
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import time
import pandas as pd
import numpy as np

from app.main import app
from app.models.responses import BusinessInsightsResponse, CashFlowPrediction, CustomerAnalysis, WorkingCapitalAnalysis
from app.models.base import BusinessInsight, InsightType
from app.services.predictive_analytics import PredictiveAnalyzer


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_token():
    """Mock authentication token"""
    return "Bearer integration_test_token"


@pytest.fixture
def sample_business_id():
    """Sample business ID for integration tests"""
    return "integration_business_123"


class TestPredictiveAnalyticsIntegration:
    """Integration tests for the complete predictive analytics workflow"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_complete_insights_workflow(
        self, mock_cache_set, mock_cache_get, mock_generate_insights, 
        mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test complete insights generation and caching workflow"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None  # Not cached initially
        
        sample_insights = BusinessInsightsResponse(
            business_id=sample_business_id,
            insights=[
                BusinessInsight(
                    id="insight_1",
                    type=InsightType.CASH_FLOW,
                    title="Cash Flow Alert",
                    description="Expenses trending higher than income",
                    recommendations=["Review expenses", "Increase revenue"],
                    impact_score=0.8,
                    valid_until=datetime.utcnow() + timedelta(days=30)
                ),
                BusinessInsight(
                    id="insight_2",
                    type=InsightType.CUSTOMER_ANALYSIS,
                    title="Customer Concentration Risk",
                    description="Top 3 customers represent 80% of revenue",
                    recommendations=["Diversify customer base"],
                    impact_score=0.7,
                    valid_until=datetime.utcnow() + timedelta(days=30)
                )
            ],
            generated_at=datetime.utcnow(),
            next_update=datetime.utcnow() + timedelta(hours=24)
        )
        mock_generate_insights.return_value = sample_insights
        
        # Step 1: Initial request (not cached)
        response1 = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["insights"]) == 2
        assert data1["insights"][0]["title"] == "Cash Flow Alert"
        
        # Verify insights were generated and cached
        mock_generate_insights.assert_called_once_with(sample_business_id)
        mock_cache_set.assert_called_once()
        
        # Step 2: Second request (should use cache)
        mock_cache_get.return_value = sample_insights.dict()
        mock_generate_insights.reset_mock()
        
        response2 = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2 == data1  # Should be identical (cached)
        
        # Verify cache was used (no new generation)
        mock_generate_insights.assert_not_called()
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.predict_cash_flow')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.analyze_customer_revenue')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.calculate_working_capital_trend')
    def test_multiple_analytics_endpoints(
        self, mock_working_capital, mock_customer_analysis, mock_cash_flow, 
        mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test multiple analytics endpoints working together"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        
        mock_cash_flow.return_value = CashFlowPrediction(
            predicted_inflow=100000.0,
            predicted_outflow=85000.0,
            net_cash_flow=15000.0,
            confidence=0.88,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Historical trends", "Seasonal patterns"]
        )
        
        from app.models.responses import CustomerAnalysis, WorkingCapitalAnalysis
        
        mock_customer_analysis.return_value = CustomerAnalysis(
            top_customers=[
                {"name": "Customer A", "revenue": 50000, "percentage": 40},
                {"name": "Customer B", "revenue": 30000, "percentage": 24},
                {"name": "Customer C", "revenue": 25000, "percentage": 20}
            ],
            revenue_concentration=0.84,
            pareto_analysis={"top_20_percent": 0.8},
            recommendations=["Reduce dependency on top customer"]
        )
        
        mock_working_capital.return_value = WorkingCapitalAnalysis(
            current_working_capital=150000.0,
            trend_direction="stable",
            days_until_depletion=None,
            recommendations=["Maintain current levels"],
            risk_level="low"
        )
        
        # Test all endpoints
        endpoints = [
            f"/api/v1/insights/{sample_business_id}/cash-flow",
            f"/api/v1/insights/{sample_business_id}/customer-analysis",
            f"/api/v1/insights/{sample_business_id}/working-capital"
        ]
        
        responses = []
        for endpoint in endpoints:
            response = client.get(endpoint, headers={"Authorization": mock_token})
            assert response.status_code == 200
            responses.append(response.json())
        
        # Verify all responses have expected data
        cash_flow_data, customer_data, working_capital_data = responses
        
        assert cash_flow_data["net_cash_flow"] == 15000.0
        assert customer_data["revenue_concentration"] == 0.84
        assert working_capital_data["risk_level"] == "low"
    
    @patch('app.api.insights.verify_token')
    def test_batch_processing_workflow(self, mock_verify_token, client, mock_token):
        """Test batch processing of multiple businesses"""
        mock_verify_token.return_value = "user_123"
        
        business_ids = ["business_1", "business_2", "business_3", "business_4"]
        
        # Test batch generation
        response = client.post(
            "/api/v1/insights/batch-generate",
            json=business_ids,
            headers={"Authorization": mock_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["business_ids"]) == 4
        assert "4 businesses" in data["message"]
    
    @patch('app.api.insights.verify_token')
    @patch('app.utils.cache.cache.clear_pattern')
    def test_cache_management_workflow(
        self, mock_clear_pattern, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test cache management and invalidation workflow"""
        mock_verify_token.return_value = "user_123"
        mock_clear_pattern.return_value = 8  # 8 cache entries cleared
        
        # Clear cache
        response = client.delete(
            f"/api/v1/insights/{sample_business_id}/cache",
            headers={"Authorization": mock_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "8 cache entries" in data["message"]
        
        # Verify all cache patterns were cleared
        expected_patterns = [
            f"insights:{sample_business_id}",
            f"cash_flow:{sample_business_id}:*",
            f"customer_analysis:{sample_business_id}",
            f"working_capital:{sample_business_id}"
        ]
        
        assert mock_clear_pattern.call_count == len(expected_patterns)
    
    def test_performance_monitoring_endpoint(self, client):
        """Test performance monitoring endpoint"""
        response = client.get("/api/v1/analytics/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "cache_stats" in data
        assert "performance_metrics" in data
        
        # Verify performance metrics structure
        metrics = data["performance_metrics"]
        assert "avg_response_time_ms" in metrics
        assert "cache_hit_rate" in metrics
        assert "prediction_accuracy" in metrics
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    def test_error_handling_and_recovery(
        self, mock_generate_insights, mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test error handling and recovery mechanisms"""
        mock_verify_token.return_value = "user_123"
        
        # Test service error
        mock_generate_insights.side_effect = Exception("Database connection failed")
        
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]
        
        # Test recovery after error
        mock_generate_insights.side_effect = None
        mock_generate_insights.return_value = BusinessInsightsResponse(
            business_id=sample_business_id,
            insights=[],
            generated_at=datetime.utcnow()
        )
        
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        
        assert response.status_code == 200
    
    @patch('app.api.insights.verify_token')
    def test_authentication_across_endpoints(self, mock_verify_token, client, sample_business_id):
        """Test authentication requirements across all endpoints"""
        # Test without token
        endpoints = [
            f"/api/v1/insights/{sample_business_id}",
            f"/api/v1/insights/{sample_business_id}/cash-flow",
            f"/api/v1/insights/{sample_business_id}/customer-analysis",
            f"/api/v1/insights/{sample_business_id}/working-capital"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 403  # No authorization header
        
        # Test with invalid token
        mock_verify_token.return_value = None
        
        for endpoint in endpoints:
            response = client.get(endpoint, headers={"Authorization": "Bearer invalid_token"})
            assert response.status_code == 401  # Invalid token
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_concurrent_requests_handling(
        self, mock_cache_set, mock_cache_get, mock_generate_insights, 
        mock_verify_token, client, mock_token, sample_business_id
    ):
        """Test handling of concurrent requests"""
        mock_verify_token.return_value = "user_123"
        mock_cache_get.return_value = None
        
        sample_insights = BusinessInsightsResponse(
            business_id=sample_business_id,
            insights=[],
            generated_at=datetime.utcnow()
        )
        mock_generate_insights.return_value = sample_insights
        
        # Simulate concurrent requests
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = client.get(
                f"/api/v1/insights/{sample_business_id}",
                headers={"Authorization": mock_token}
            )
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 5
        assert all(code == 200 for code in status_codes)


class TestPredictiveAnalyticsPerformance:
    """Performance tests for predictive analytics"""
    
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer.generate_insights')
    @patch('app.utils.cache.cache.get')
    def test_response_time_requirements(
        self, mock_cache_get, mock_generate_insights, mock_verify_token, 
        client, mock_token, sample_business_id
    ):
        """Test that response times meet requirements"""
        mock_verify_token.return_value = "user_123"
        
        # Test cached response time
        mock_cache_get.return_value = {
            "business_id": sample_business_id,
            "insights": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        start_time = time.time()
        response = client.get(
            f"/api/v1/insights/{sample_business_id}",
            headers={"Authorization": mock_token}
        )
        cached_time = time.time() - start_time
        
        assert response.status_code == 200
        assert cached_time < 0.5  # Cached responses should be under 500ms
        
        # Test uncached response time
        mock_cache_get.return_value = None
        mock_generate_insights.return_value = BusinessInsightsResponse(
            business_id=sample_business_id,
            insights=[],
            generated_at=datetime.utcnow()
        )
        
        start_time = time.time()
        response = client.get(
            f"/api/v1/insights/{sample_business_id}?force_refresh=true",
            headers={"Authorization": mock_token}
        )
        uncached_time = time.time() - start_time
        
        assert response.status_code == 200
        assert uncached_time < 3.0  # Uncached responses should be under 3 seconds
    
    @patch('app.api.insights.verify_token')
    def test_batch_processing_performance(self, mock_verify_token, client, mock_token):
        """Test batch processing performance"""
        mock_verify_token.return_value = "user_123"
        
        # Test maximum batch size
        business_ids = [f"business_{i}" for i in range(10)]  # Maximum allowed
        
        start_time = time.time()
        response = client.post(
            "/api/v1/insights/batch-generate",
            json=business_ids,
            headers={"Authorization": mock_token}
        )
        batch_time = time.time() - start_time
        
        assert response.status_code == 200
        assert batch_time < 2.0  # Batch scheduling should be fast


class TestPredictiveAnalyticsCore:
    """Tests for core predictive analytics functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create PredictiveAnalyzer instance for testing"""
        return PredictiveAnalyzer()
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Sample transaction data for testing"""
        return [
            {'amount': 10000, 'created_at': datetime.utcnow() - timedelta(days=30), 'transaction_type': 'income'},
            {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=29), 'transaction_type': 'expense'},
            {'amount': 15000, 'created_at': datetime.utcnow() - timedelta(days=25), 'transaction_type': 'income'},
            {'amount': -7000, 'created_at': datetime.utcnow() - timedelta(days=20), 'transaction_type': 'expense'},
            {'amount': 12000, 'created_at': datetime.utcnow() - timedelta(days=15), 'transaction_type': 'income'},
            {'amount': -6000, 'created_at': datetime.utcnow() - timedelta(days=10), 'transaction_type': 'expense'},
            {'amount': 18000, 'created_at': datetime.utcnow() - timedelta(days=5), 'transaction_type': 'income'},
            {'amount': -8000, 'created_at': datetime.utcnow() - timedelta(days=2), 'transaction_type': 'expense'},
        ]
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer revenue data for testing"""
        return [
            {'customer_id': 'cust_1', 'customer_name': 'Customer A', 'amount': 50000, 'transaction_count': 10},
            {'customer_id': 'cust_2', 'customer_name': 'Customer B', 'amount': 30000, 'transaction_count': 8},
            {'customer_id': 'cust_3', 'customer_name': 'Customer C', 'amount': 25000, 'transaction_count': 6},
            {'customer_id': 'cust_4', 'customer_name': 'Customer D', 'amount': 15000, 'transaction_count': 4},
            {'customer_id': 'cust_5', 'customer_name': 'Customer E', 'amount': 10000, 'transaction_count': 3},
        ]
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_transaction_history')
    async def test_cash_flow_prediction_with_sufficient_data(self, mock_get_data, analyzer, sample_transaction_data):
        """Test cash flow prediction with sufficient historical data"""
        mock_get_data.return_value = sample_transaction_data
        
        result = await analyzer.predict_cash_flow("test_business", months=3)
        
        assert isinstance(result, CashFlowPrediction)
        assert result.confidence > 0.0
        assert result.predicted_inflow > 0
        assert result.predicted_outflow > 0
        assert len(result.factors) > 0
        assert result.period_start < result.period_end
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_transaction_history')
    async def test_cash_flow_prediction_insufficient_data(self, mock_get_data, analyzer):
        """Test cash flow prediction with insufficient data"""
        mock_get_data.return_value = [
            {'amount': 1000, 'created_at': datetime.utcnow(), 'transaction_type': 'income'}
        ]
        
        result = await analyzer.predict_cash_flow("test_business", months=3)
        
        assert isinstance(result, CashFlowPrediction)
        assert result.confidence == 0.0
        assert "Insufficient historical data" in result.factors[0]
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_customer_revenue_data')
    async def test_customer_revenue_analysis_pareto(self, mock_get_data, analyzer, sample_customer_data):
        """Test customer revenue analysis using Pareto principle"""
        mock_get_data.return_value = sample_customer_data
        
        result = await analyzer.analyze_customer_revenue("test_business")
        
        assert isinstance(result, CustomerAnalysis)
        assert len(result.top_customers) > 0
        assert result.revenue_concentration >= 0.0
        assert result.revenue_concentration <= 1.0
        assert 'total_customers' in result.pareto_analysis
        assert len(result.recommendations) > 0
        
        # Verify Pareto analysis
        assert result.pareto_analysis['total_customers'] == 5
        
        # Check that top customers are sorted by revenue
        revenues = [customer['revenue'] for customer in result.top_customers]
        assert revenues == sorted(revenues, reverse=True)
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_customer_revenue_data')
    async def test_customer_revenue_analysis_no_data(self, mock_get_data, analyzer):
        """Test customer revenue analysis with no data"""
        mock_get_data.return_value = []
        
        result = await analyzer.analyze_customer_revenue("test_business")
        
        assert isinstance(result, CustomerAnalysis)
        assert len(result.top_customers) == 0
        assert "No customer data available" in result.recommendations[0]
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_working_capital_data')
    async def test_working_capital_analysis_with_trend(self, mock_get_data, analyzer):
        """Test working capital analysis with trend data"""
        mock_get_data.return_value = {
            'current_assets': 100000,
            'current_liabilities': 60000,
            'historical_working_capital': [
                {'date': (datetime.utcnow() - timedelta(days=90)).isoformat(), 'working_capital': 35000},
                {'date': (datetime.utcnow() - timedelta(days=60)).isoformat(), 'working_capital': 38000},
                {'date': (datetime.utcnow() - timedelta(days=30)).isoformat(), 'working_capital': 40000},
            ]
        }
        
        result = await analyzer.calculate_working_capital_trend("test_business")
        
        assert isinstance(result, WorkingCapitalAnalysis)
        assert result.current_working_capital == 40000  # 100000 - 60000
        assert result.trend_direction in ["increasing", "decreasing", "stable"]
        assert result.risk_level in ["low", "medium", "high", "critical"]
        assert len(result.recommendations) > 0
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_working_capital_data')
    async def test_working_capital_analysis_depletion_risk(self, mock_get_data, analyzer):
        """Test working capital analysis with depletion risk"""
        mock_get_data.return_value = {
            'current_assets': 50000,
            'current_liabilities': 45000,
            'historical_working_capital': [
                {'date': (datetime.utcnow() - timedelta(days=90)).isoformat(), 'working_capital': 15000},
                {'date': (datetime.utcnow() - timedelta(days=60)).isoformat(), 'working_capital': 10000},
                {'date': (datetime.utcnow() - timedelta(days=30)).isoformat(), 'working_capital': 5000},
            ]
        }
        
        result = await analyzer.calculate_working_capital_trend("test_business")
        
        assert isinstance(result, WorkingCapitalAnalysis)
        assert result.current_working_capital == 5000
        assert result.trend_direction == "decreasing"
        assert result.days_until_depletion is not None
        assert result.risk_level in ["medium", "high", "critical"]
    
    async def test_predict_time_series_with_arima(self, analyzer):
        """Test time series prediction using ARIMA"""
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        values = np.random.normal(1000, 100, 30) + np.arange(30) * 10  # Trending upward
        series = pd.Series(values, index=dates)
        
        prediction, confidence = await analyzer._predict_time_series(series, forecast_days=30)
        
        assert prediction > 0
        assert 0.0 <= confidence <= 1.0
    
    async def test_predict_time_series_insufficient_data(self, analyzer):
        """Test time series prediction with insufficient data"""
        # Create very small dataset
        dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
        values = [1000, 1100, 1200]
        series = pd.Series(values, index=dates)
        
        prediction, confidence = await analyzer._predict_time_series(series, forecast_days=30)
        
        assert prediction > 0
        assert confidence <= 0.5  # Should have low confidence
    
    async def test_identify_cash_flow_factors(self, analyzer, sample_transaction_data):
        """Test identification of cash flow factors"""
        df = pd.DataFrame(sample_transaction_data)
        df['date'] = pd.to_datetime(df['created_at'])
        df = df.set_index('date')
        
        factors = await analyzer._identify_cash_flow_factors(df)
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert all(isinstance(factor, str) for factor in factors)
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_transaction_history')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_customer_revenue_data')
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_working_capital_data')
    async def test_generate_comprehensive_insights(
        self, mock_wc_data, mock_customer_data, mock_transaction_data, 
        analyzer, sample_transaction_data, sample_customer_data
    ):
        """Test comprehensive insights generation"""
        # Setup mocks
        mock_transaction_data.return_value = sample_transaction_data
        mock_customer_data.return_value = sample_customer_data
        mock_wc_data.return_value = {
            'current_assets': 100000,
            'current_liabilities': 60000,
            'historical_working_capital': [
                {'date': (datetime.utcnow() - timedelta(days=30)).isoformat(), 'working_capital': 40000},
            ]
        }
        
        result = await analyzer.generate_insights("test_business")
        
        assert isinstance(result, BusinessInsightsResponse)
        assert result.success is True
        assert result.business_id == "test_business"
        assert len(result.insights) > 0
        assert result.next_update is not None
        
        # Check that different types of insights are generated
        insight_types = [insight.type for insight in result.insights]
        assert len(set(insight_types)) > 1  # Multiple types of insights
    
    async def test_create_cash_flow_insights_negative(self, analyzer):
        """Test creation of cash flow insights for negative cash flow"""
        prediction = CashFlowPrediction(
            predicted_inflow=50000,
            predicted_outflow=70000,
            net_cash_flow=-20000,
            confidence=0.8,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Seasonal decline"]
        )
        
        insights = await analyzer._create_cash_flow_insights("test_business", prediction)
        
        assert len(insights) > 0
        assert insights[0].type == InsightType.CASH_FLOW_PREDICTION
        assert "Cash Flow Warning" in insights[0].title
        assert insights[0].impact_score >= 0.8  # High impact for negative cash flow
        assert len(insights[0].recommendations) > 0
    
    async def test_create_cash_flow_insights_positive(self, analyzer):
        """Test creation of cash flow insights for positive cash flow"""
        prediction = CashFlowPrediction(
            predicted_inflow=100000,
            predicted_outflow=70000,
            net_cash_flow=30000,
            confidence=0.8,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Strong sales growth"]
        )
        
        insights = await analyzer._create_cash_flow_insights("test_business", prediction)
        
        assert len(insights) > 0
        assert insights[0].type == InsightType.CASH_FLOW_PREDICTION
        assert "Positive Cash Flow" in insights[0].title
        assert len(insights[0].recommendations) > 0
    
    async def test_create_customer_insights_concentration_risk(self, analyzer):
        """Test creation of customer insights for high concentration risk"""
        analysis = CustomerAnalysis(
            top_customers=[
                {'customer_name': 'Customer A', 'revenue': 70000, 'percentage': 70},
                {'customer_name': 'Customer B', 'revenue': 20000, 'percentage': 20},
                {'customer_name': 'Customer C', 'revenue': 10000, 'percentage': 10},
            ],
            revenue_concentration=0.1,  # High concentration (low diversity)
            pareto_analysis={'total_customers': 10},
            recommendations=["Diversify customer base"]
        )
        
        insights = await analyzer._create_customer_insights("test_business", analysis)
        
        assert len(insights) > 0
        assert insights[0].type == InsightType.CUSTOMER_ANALYSIS
        assert "Revenue Concentration Risk" in insights[0].title
        assert insights[0].impact_score >= 0.7  # High impact for concentration risk
    
    async def test_create_working_capital_insights_depletion(self, analyzer):
        """Test creation of working capital insights for depletion risk"""
        analysis = WorkingCapitalAnalysis(
            current_working_capital=50000,
            trend_direction="decreasing",
            days_until_depletion=45,
            recommendations=["Accelerate collections", "Delay payments"],
            risk_level="high"
        )
        
        insights = await analyzer._create_working_capital_insights("test_business", analysis)
        
        assert len(insights) > 0
        assert insights[0].type == InsightType.WORKING_CAPITAL
        assert "Working Capital Alert" in insights[0].title
        assert "45 days" in insights[0].description
        assert insights[0].impact_score >= 0.8  # High impact for depletion risk
    
    @patch('app.services.predictive_analytics.PredictiveAnalyzer._get_transaction_history')
    async def test_analyze_expense_trends(self, mock_get_data, analyzer):
        """Test expense trend analysis"""
        # Create expense data with increasing trend
        expense_data = [
            {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=150), 'category': 'office'},
            {'amount': -5500, 'created_at': datetime.utcnow() - timedelta(days=120), 'category': 'office'},
            {'amount': -6000, 'created_at': datetime.utcnow() - timedelta(days=90), 'category': 'office'},
            {'amount': -6500, 'created_at': datetime.utcnow() - timedelta(days=60), 'category': 'office'},
            {'amount': -7000, 'created_at': datetime.utcnow() - timedelta(days=30), 'category': 'office'},
        ]
        
        # Mock the database query
        with patch.object(analyzer.db, 'fetch_all', return_value=expense_data):
            insights = await analyzer._analyze_expense_trends("test_business")
        
        assert isinstance(insights, list)
        if len(insights) > 0:
            assert insights[0].type == InsightType.EXPENSE_TREND
            assert "expense" in insights[0].title.lower()
    
    async def test_generate_customer_recommendations(self, analyzer):
        """Test customer recommendation generation"""
        # Create sample customer revenue DataFrame
        customer_data = pd.DataFrame([
            {'customer_id': 'c1', 'amount': 50000, 'customer_name': 'Customer A'},
            {'customer_id': 'c2', 'amount': 30000, 'customer_name': 'Customer B'},
            {'customer_id': 'c3', 'amount': 20000, 'customer_name': 'Customer C'},
        ])
        
        # Test high concentration scenario
        recommendations = await analyzer._generate_customer_recommendations(
            customer_data, concentration=0.1, top_customer_count=2
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("diversify" in rec.lower() for rec in recommendations)
        
        # Test good diversification scenario
        recommendations = await analyzer._generate_customer_recommendations(
            customer_data, concentration=0.6, top_customer_count=10
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    async def test_assess_working_capital_risk(self, analyzer):
        """Test working capital risk assessment"""
        # Test critical risk (negative working capital)
        risk = await analyzer._assess_working_capital_risk(
            current_wc=-10000, slope=-100, days_until_depletion=None
        )
        assert risk == "critical"
        
        # Test high risk (depletion soon)
        risk = await analyzer._assess_working_capital_risk(
            current_wc=50000, slope=-1000, days_until_depletion=20
        )
        assert risk == "high"
        
        # Test medium risk (depletion in 60 days)
        risk = await analyzer._assess_working_capital_risk(
            current_wc=100000, slope=-500, days_until_depletion=60
        )
        assert risk == "medium"
        
        # Test low risk (stable)
        risk = await analyzer._assess_working_capital_risk(
            current_wc=100000, slope=100, days_until_depletion=None
        )
        assert risk == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])