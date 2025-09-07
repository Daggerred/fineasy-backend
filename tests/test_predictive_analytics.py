"""
Comprehensive tests for Predictive Analytics Service
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from app.services.predictive_analytics import PredictiveAnalyzer
from app.models.responses import (
    BusinessInsightsResponse, CashFlowPrediction, 
    CustomerAnalysis, WorkingCapitalAnalysis
)
from app.models.base import BusinessInsight, InsightType


@pytest.fixture
def predictive_analyzer():
    """Create PredictiveAnalyzer instance with mocked database"""
    with patch('app.services.predictive_analytics.DatabaseManager') as mock_db_class:
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        analyzer = PredictiveAnalyzer()
        return analyzer


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)
    
    data = []
    for date in dates:
        # Generate realistic transaction patterns
        num_transactions = np.random.poisson(5)  # Average 5 transactions per day
        for _ in range(num_transactions):
            data.append({
                'id': f'trans_{len(data)}',
                'business_id': 'business_123',
                'amount': np.random.lognormal(mean=6, sigma=1),  # Log-normal distribution
                'type': np.random.choice(['income', 'expense'], p=[0.6, 0.4]),
                'created_at': date + timedelta(hours=np.random.randint(8, 18)),
                'description': f'Transaction {len(data)}'
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    customers = []
    revenues = [50000, 30000, 25000, 20000, 15000, 10000, 8000, 5000, 3000, 2000]
    
    for i, revenue in enumerate(revenues):
        customers.append({
            'id': f'customer_{i}',
            'name': f'Customer {chr(65 + i)}',  # Customer A, B, C, etc.
            'total_revenue': revenue,
            'transaction_count': np.random.randint(10, 50),
            'last_transaction_date': datetime.now() - timedelta(days=np.random.randint(1, 30))
        })
    
    return customers


@pytest.fixture
def sample_invoice_data():
    """Sample invoice data for testing"""
    invoices = []
    for i in range(50):
        invoices.append({
            'id': f'invoice_{i}',
            'business_id': 'business_123',
            'total_amount': np.random.uniform(1000, 10000),
            'status': np.random.choice(['paid', 'pending', 'overdue'], p=[0.7, 0.2, 0.1]),
            'due_date': datetime.now() + timedelta(days=np.random.randint(-30, 60)),
            'created_at': datetime.now() - timedelta(days=np.random.randint(1, 90))
        })
    
    return invoices


class TestPredictiveAnalyzer:
    """Test cases for PredictiveAnalyzer class"""
    
    # =============================================================================
    # Test Cash Flow Prediction
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_predict_cash_flow_success(self, predictive_analyzer, sample_transaction_data):
        """Test successful cash flow prediction"""
        business_id = "business_123"
        months = 3
        
        # Mock database calls
        predictive_analyzer.db.get_transactions = AsyncMock(return_value=sample_transaction_data.to_dict('records'))
        
        # Test cash flow prediction
        prediction = await predictive_analyzer.predict_cash_flow(business_id, months)
        
        # Verify response structure
        assert isinstance(prediction, CashFlowPrediction)
        assert prediction.predicted_inflow > 0
        assert prediction.predicted_outflow > 0
        assert prediction.net_cash_flow == prediction.predicted_inflow - prediction.predicted_outflow
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.period_start is not None
        assert prediction.period_end is not None
        assert len(prediction.factors) > 0
    
    @pytest.mark.asyncio
    async def test_predict_cash_flow_insufficient_data(self, predictive_analyzer):
        """Test cash flow prediction with insufficient data"""
        business_id = "business_123"
        
        # Mock database to return minimal data
        minimal_data = [
            {'amount': 1000, 'type': 'income', 'created_at': datetime.now().isoformat()}
        ]
        predictive_analyzer.db.get_transactions = AsyncMock(return_value=minimal_data)
        
        # Test prediction with insufficient data
        prediction = await predictive_analyzer.predict_cash_flow(business_id, 3)
        
        # Should still return a prediction but with low confidence
        assert isinstance(prediction, CashFlowPrediction)
        assert prediction.confidence < 0.5  # Low confidence due to insufficient data
        assert "insufficient data" in " ".join(prediction.factors).lower()
    
    def test_calculate_seasonal_factors(self, predictive_analyzer, sample_transaction_data):
        """Test seasonal factor calculation"""
        # Test seasonal analysis
        seasonal_factors = predictive_analyzer._calculate_seasonal_factors(sample_transaction_data)
        
        assert isinstance(seasonal_factors, dict)
        assert len(seasonal_factors) == 12  # 12 months
        
        # All factors should be positive
        for month, factor in seasonal_factors.items():
            assert factor > 0
            assert 1 <= month <= 12
    
    def test_calculate_trend_factors(self, predictive_analyzer, sample_transaction_data):
        """Test trend factor calculation"""
        # Test trend analysis
        trend_factors = predictive_analyzer._calculate_trend_factors(sample_transaction_data)
        
        assert isinstance(trend_factors, dict)
        assert 'income_trend' in trend_factors
        assert 'expense_trend' in trend_factors
        assert 'growth_rate' in trend_factors
        
        # Trends should be numeric
        assert isinstance(trend_factors['income_trend'], (int, float))
        assert isinstance(trend_factors['expense_trend'], (int, float))
        assert isinstance(trend_factors['growth_rate'], (int, float))
    
    def test_apply_arima_forecasting(self, predictive_analyzer):
        """Test ARIMA forecasting application"""
        # Create sample time series data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        values = np.random.normal(1000, 200, 90) + np.sin(np.arange(90) * 2 * np.pi / 30) * 100
        
        time_series = pd.Series(values, index=dates)
        
        # Test ARIMA forecasting
        forecast = predictive_analyzer._apply_arima_forecasting(time_series, periods=30)
        
        assert len(forecast) == 30
        assert all(isinstance(val, (int, float)) for val in forecast)
        assert all(val > 0 for val in forecast)  # Should be positive values
    
    # =============================================================================
    # Test Customer Revenue Analysis
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_analyze_customer_revenue_success(self, predictive_analyzer, sample_customer_data):
        """Test successful customer revenue analysis"""
        business_id = "business_123"
        
        # Mock database calls
        predictive_analyzer.db.get_customer_revenue_data = AsyncMock(return_value=sample_customer_data)
        
        # Test customer analysis
        analysis = await predictive_analyzer.analyze_customer_revenue(business_id)
        
        # Verify response structure
        assert isinstance(analysis, CustomerAnalysis)
        assert len(analysis.top_customers) > 0
        assert 0.0 <= analysis.revenue_concentration <= 1.0
        assert isinstance(analysis.pareto_analysis, dict)
        assert len(analysis.recommendations) > 0
        
        # Verify top customers are sorted by revenue
        revenues = [customer['revenue'] for customer in analysis.top_customers]
        assert revenues == sorted(revenues, reverse=True)
    
    @pytest.mark.asyncio
    async def test_analyze_customer_revenue_empty_data(self, predictive_analyzer):
        """Test customer revenue analysis with no data"""
        business_id = "business_123"
        
        # Mock database to return empty data
        predictive_analyzer.db.get_customer_revenue_data = AsyncMock(return_value=[])
        
        # Test analysis with no data
        analysis = await predictive_analyzer.analyze_customer_revenue(business_id)
        
        # Should handle empty data gracefully
        assert isinstance(analysis, CustomerAnalysis)
        assert len(analysis.top_customers) == 0
        assert analysis.revenue_concentration == 0.0
        assert "no customer data" in " ".join(analysis.recommendations).lower()
    
    def test_calculate_pareto_analysis(self, predictive_analyzer, sample_customer_data):
        """Test Pareto principle analysis"""
        # Test Pareto analysis calculation
        pareto_result = predictive_analyzer._calculate_pareto_analysis(sample_customer_data)
        
        assert isinstance(pareto_result, dict)
        assert 'top_20_percent' in pareto_result
        assert 'top_10_percent' in pareto_result
        assert 'concentration_index' in pareto_result
        
        # Verify percentages are valid
        assert 0.0 <= pareto_result['top_20_percent'] <= 1.0
        assert 0.0 <= pareto_result['top_10_percent'] <= 1.0
        assert pareto_result['top_10_percent'] >= pareto_result['top_20_percent']
    
    def test_calculate_revenue_concentration(self, predictive_analyzer, sample_customer_data):
        """Test revenue concentration calculation"""
        # Test concentration calculation
        concentration = predictive_analyzer._calculate_revenue_concentration(sample_customer_data)
        
        assert 0.0 <= concentration <= 1.0
        
        # Test with uniform distribution (should have low concentration)
        uniform_data = [{'total_revenue': 1000} for _ in range(10)]
        uniform_concentration = predictive_analyzer._calculate_revenue_concentration(uniform_data)
        assert uniform_concentration < 0.5
        
        # Test with concentrated distribution (should have high concentration)
        concentrated_data = [{'total_revenue': 10000}] + [{'total_revenue': 100} for _ in range(9)]
        concentrated_concentration = predictive_analyzer._calculate_revenue_concentration(concentrated_data)
        assert concentrated_concentration > 0.5
    
    def test_generate_customer_recommendations(self, predictive_analyzer, sample_customer_data):
        """Test customer recommendation generation"""
        # Calculate analysis data
        pareto_analysis = predictive_analyzer._calculate_pareto_analysis(sample_customer_data)
        concentration = predictive_analyzer._calculate_revenue_concentration(sample_customer_data)
        
        # Test recommendation generation
        recommendations = predictive_analyzer._generate_customer_recommendations(
            sample_customer_data, pareto_analysis, concentration
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should contain relevant recommendations based on concentration
        if concentration > 0.7:
            assert any("diversify" in rec.lower() for rec in recommendations)
        
        # Should contain recommendations about top customers
        assert any("top customer" in rec.lower() for rec in recommendations)
    
    # =============================================================================
    # Test Working Capital Analysis
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_calculate_working_capital_trend_success(self, predictive_analyzer, sample_invoice_data):
        """Test successful working capital trend calculation"""
        business_id = "business_123"
        
        # Mock database calls
        predictive_analyzer.db.get_invoices = AsyncMock(return_value=sample_invoice_data)
        predictive_analyzer.db.get_current_cash_balance = AsyncMock(return_value=50000.0)
        
        # Test working capital analysis
        analysis = await predictive_analyzer.calculate_working_capital_trend(business_id)
        
        # Verify response structure
        assert isinstance(analysis, WorkingCapitalAnalysis)
        assert analysis.current_working_capital > 0
        assert analysis.trend_direction in ['increasing', 'decreasing', 'stable']
        assert analysis.days_until_depletion is None or analysis.days_until_depletion > 0
        assert len(analysis.recommendations) > 0
        assert analysis.risk_level in ['low', 'medium', 'high']
    
    @pytest.mark.asyncio
    async def test_calculate_working_capital_negative_trend(self, predictive_analyzer):
        """Test working capital analysis with negative trend"""
        business_id = "business_123"
        
        # Create data showing declining working capital
        declining_invoices = []
        for i in range(30):
            declining_invoices.append({
                'id': f'invoice_{i}',
                'total_amount': 1000 - (i * 20),  # Declining amounts
                'status': 'pending',
                'due_date': datetime.now() + timedelta(days=30),
                'created_at': datetime.now() - timedelta(days=i)
            })
        
        # Mock database calls
        predictive_analyzer.db.get_invoices = AsyncMock(return_value=declining_invoices)
        predictive_analyzer.db.get_current_cash_balance = AsyncMock(return_value=10000.0)  # Low balance
        
        # Test analysis
        analysis = await predictive_analyzer.calculate_working_capital_trend(business_id)
        
        # Should detect negative trend and high risk
        assert analysis.trend_direction == 'decreasing'
        assert analysis.risk_level in ['medium', 'high']
        assert analysis.days_until_depletion is not None
        assert analysis.days_until_depletion < 90  # Should be concerning
    
    def test_calculate_accounts_receivable(self, predictive_analyzer, sample_invoice_data):
        """Test accounts receivable calculation"""
        # Test AR calculation
        ar_total, ar_aging = predictive_analyzer._calculate_accounts_receivable(sample_invoice_data)
        
        assert ar_total >= 0
        assert isinstance(ar_aging, dict)
        assert 'current' in ar_aging
        assert '30_days' in ar_aging
        assert '60_days' in ar_aging
        assert '90_plus_days' in ar_aging
        
        # All aging buckets should be non-negative
        for bucket, amount in ar_aging.items():
            assert amount >= 0
    
    def test_calculate_cash_conversion_cycle(self, predictive_analyzer, sample_invoice_data):
        """Test cash conversion cycle calculation"""
        # Test CCC calculation
        ccc_days = predictive_analyzer._calculate_cash_conversion_cycle(sample_invoice_data)
        
        assert isinstance(ccc_days, (int, float))
        assert ccc_days >= 0  # Should be positive for most businesses
    
    def test_predict_working_capital_depletion(self, predictive_analyzer):
        """Test working capital depletion prediction"""
        # Test with declining trend
        current_balance = 50000.0
        monthly_burn_rate = 10000.0
        
        days_until_depletion = predictive_analyzer._predict_working_capital_depletion(
            current_balance, monthly_burn_rate
        )
        
        assert days_until_depletion is not None
        assert days_until_depletion > 0
        assert days_until_depletion < 200  # Should be reasonable timeframe
        
        # Test with positive cash flow (no depletion)
        days_no_depletion = predictive_analyzer._predict_working_capital_depletion(
            current_balance, -5000.0  # Negative burn rate (positive cash flow)
        )
        
        assert days_no_depletion is None  # No depletion expected
    
    # =============================================================================
    # Test Business Insights Generation
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_generate_insights_comprehensive(self, predictive_analyzer, sample_transaction_data, 
                                                 sample_customer_data, sample_invoice_data):
        """Test comprehensive business insights generation"""
        business_id = "business_123"
        
        # Mock all database calls
        predictive_analyzer.db.get_transactions = AsyncMock(return_value=sample_transaction_data.to_dict('records'))
        predictive_analyzer.db.get_customer_revenue_data = AsyncMock(return_value=sample_customer_data)
        predictive_analyzer.db.get_invoices = AsyncMock(return_value=sample_invoice_data)
        predictive_analyzer.db.get_current_cash_balance = AsyncMock(return_value=75000.0)
        
        # Test comprehensive insights generation
        response = await predictive_analyzer.generate_insights(business_id)
        
        # Verify response structure
        assert isinstance(response, BusinessInsightsResponse)
        assert response.business_id == business_id
        assert len(response.insights) > 0
        assert response.generated_at is not None
        assert response.next_update is not None
        
        # Should contain different types of insights
        insight_types = [insight.type for insight in response.insights]
        assert InsightType.CASH_FLOW in insight_types
        assert InsightType.CUSTOMER_ANALYSIS in insight_types
        assert InsightType.WORKING_CAPITAL in insight_types
    
    @pytest.mark.asyncio
    async def test_generate_insights_error_handling(self, predictive_analyzer):
        """Test insights generation error handling"""
        business_id = "business_123"
        
        # Mock database to raise exception
        predictive_analyzer.db.get_transactions = AsyncMock(side_effect=Exception("Database error"))
        
        # Test error handling
        response = await predictive_analyzer.generate_insights(business_id)
        
        # Should return response with error insights
        assert isinstance(response, BusinessInsightsResponse)
        assert response.business_id == business_id
        # May contain error insights or be empty depending on implementation
    
    def test_create_cash_flow_insight(self, predictive_analyzer):
        """Test cash flow insight creation"""
        # Create sample prediction
        prediction = CashFlowPrediction(
            predicted_inflow=60000.0,
            predicted_outflow=70000.0,
            net_cash_flow=-10000.0,
            confidence=0.8,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=90),
            factors=["Seasonal decline", "Increased expenses"]
        )
        
        # Test insight creation
        insight = predictive_analyzer._create_cash_flow_insight(prediction)
        
        assert isinstance(insight, BusinessInsight)
        assert insight.type == InsightType.CASH_FLOW
        assert insight.impact_score > 0.5  # Negative cash flow should have high impact
        assert "negative" in insight.description.lower() or "deficit" in insight.description.lower()
        assert len(insight.recommendations) > 0
    
    def test_create_customer_insight(self, predictive_analyzer, sample_customer_data):
        """Test customer insight creation"""
        # Create sample analysis
        analysis = CustomerAnalysis(
            top_customers=sample_customer_data[:3],
            revenue_concentration=0.8,
            pareto_analysis={'top_20_percent': 0.85},
            recommendations=["Diversify customer base", "Protect top customers"]
        )
        
        # Test insight creation
        insight = predictive_analyzer._create_customer_insight(analysis)
        
        assert isinstance(insight, BusinessInsight)
        assert insight.type == InsightType.CUSTOMER_ANALYSIS
        assert insight.impact_score > 0.0
        assert len(insight.recommendations) > 0
        assert "customer" in insight.description.lower()
    
    def test_create_working_capital_insight(self, predictive_analyzer):
        """Test working capital insight creation"""
        # Create sample analysis
        analysis = WorkingCapitalAnalysis(
            current_working_capital=25000.0,
            trend_direction="decreasing",
            days_until_depletion=45,
            recommendations=["Improve collections", "Extend payment terms"],
            risk_level="high"
        )
        
        # Test insight creation
        insight = predictive_analyzer._create_working_capital_insight(analysis)
        
        assert isinstance(insight, BusinessInsight)
        assert insight.type == InsightType.WORKING_CAPITAL
        assert insight.impact_score > 0.7  # High risk should have high impact
        assert "45 days" in insight.description
        assert len(insight.recommendations) > 0
    
    # =============================================================================
    # Test Utility Functions
    # =============================================================================
    
    def test_calculate_confidence_score(self, predictive_analyzer):
        """Test confidence score calculation"""
        # Test with sufficient data
        high_confidence = predictive_analyzer._calculate_confidence_score(
            data_points=1000,
            variance=0.1,
            trend_stability=0.9
        )
        assert 0.7 <= high_confidence <= 1.0
        
        # Test with insufficient data
        low_confidence = predictive_analyzer._calculate_confidence_score(
            data_points=10,
            variance=0.8,
            trend_stability=0.3
        )
        assert 0.0 <= low_confidence <= 0.5
    
    def test_normalize_time_series(self, predictive_analyzer):
        """Test time series normalization"""
        # Create sample time series with gaps
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        values = np.random.normal(1000, 100, len(dates))
        
        # Remove some dates to create gaps
        mask = np.random.choice([True, False], len(dates), p=[0.8, 0.2])
        sparse_dates = dates[mask]
        sparse_values = values[mask]
        
        time_series = pd.Series(sparse_values, index=sparse_dates)
        
        # Test normalization
        normalized = predictive_analyzer._normalize_time_series(time_series, freq='D')
        
        assert len(normalized) == len(dates)  # Should fill gaps
        assert normalized.index.freq is not None  # Should have regular frequency
    
    def test_detect_outliers(self, predictive_analyzer):
        """Test outlier detection"""
        # Create data with outliers
        normal_data = np.random.normal(1000, 100, 100)
        outliers = [5000, -2000, 10000]  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        # Test outlier detection
        outlier_indices = predictive_analyzer._detect_outliers(data_with_outliers)
        
        assert len(outlier_indices) > 0
        assert len(outlier_indices) <= len(outliers) + 5  # Should detect most outliers
        
        # Verify detected outliers are actually extreme values
        for idx in outlier_indices:
            value = data_with_outliers[idx]
            assert abs(value - np.mean(normal_data)) > 2 * np.std(normal_data)
    
    def test_calculate_moving_average(self, predictive_analyzer):
        """Test moving average calculation"""
        # Create sample data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test moving average
        ma_3 = predictive_analyzer._calculate_moving_average(data, window=3)
        ma_5 = predictive_analyzer._calculate_moving_average(data, window=5)
        
        assert len(ma_3) == len(data) - 2  # Window size 3
        assert len(ma_5) == len(data) - 4  # Window size 5
        
        # Verify calculation
        assert ma_3[0] == 2.0  # (1+2+3)/3
        assert ma_3[1] == 3.0  # (2+3+4)/3
    
    def test_calculate_growth_rate(self, predictive_analyzer):
        """Test growth rate calculation"""
        # Test positive growth
        current_value = 1200.0
        previous_value = 1000.0
        
        growth_rate = predictive_analyzer._calculate_growth_rate(current_value, previous_value)
        assert growth_rate == 0.2  # 20% growth
        
        # Test negative growth
        declining_rate = predictive_analyzer._calculate_growth_rate(800.0, 1000.0)
        assert declining_rate == -0.2  # 20% decline
        
        # Test zero previous value
        zero_base_rate = predictive_analyzer._calculate_growth_rate(1000.0, 0.0)
        assert zero_base_rate == 0.0  # Should handle gracefully


class TestPredictiveAnalyticsIntegration:
    """Integration tests for predictive analytics"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, predictive_analyzer, sample_transaction_data, 
                                        sample_customer_data, sample_invoice_data):
        """Test complete analysis pipeline"""
        business_id = "integration_test_business"
        
        # Mock all database calls
        predictive_analyzer.db.get_transactions = AsyncMock(return_value=sample_transaction_data.to_dict('records'))
        predictive_analyzer.db.get_customer_revenue_data = AsyncMock(return_value=sample_customer_data)
        predictive_analyzer.db.get_invoices = AsyncMock(return_value=sample_invoice_data)
        predictive_analyzer.db.get_current_cash_balance = AsyncMock(return_value=100000.0)
        
        # Run complete analysis
        cash_flow = await predictive_analyzer.predict_cash_flow(business_id, 3)
        customer_analysis = await predictive_analyzer.analyze_customer_revenue(business_id)
        working_capital = await predictive_analyzer.calculate_working_capital_trend(business_id)
        insights = await predictive_analyzer.generate_insights(business_id)
        
        # Verify all analyses completed successfully
        assert isinstance(cash_flow, CashFlowPrediction)
        assert isinstance(customer_analysis, CustomerAnalysis)
        assert isinstance(working_capital, WorkingCapitalAnalysis)
        assert isinstance(insights, BusinessInsightsResponse)
        
        # Verify insights contain all analysis types
        insight_types = [insight.type for insight in insights.insights]
        assert InsightType.CASH_FLOW in insight_types
        assert InsightType.CUSTOMER_ANALYSIS in insight_types
        assert InsightType.WORKING_CAPITAL in insight_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])