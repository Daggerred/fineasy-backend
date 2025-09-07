#!/usr/bin/env python3
"""
Standalone validation script for predictive analytics service
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.predictive_analytics import PredictiveAnalyzer
from app.models.responses import CashFlowPrediction, CustomerAnalysis, WorkingCapitalAnalysis


class MockDatabaseManager:
    """Mock database manager for testing"""
    
    async def fetch_all(self, query, params=None):
        """Mock fetch_all method"""
        return []
    
    async def fetch_one(self, query, params=None):
        """Mock fetch_one method"""
        return {'receivables': 0, 'payables': 0}


async def test_time_series_prediction():
    """Test time series prediction functionality"""
    print("Testing time series prediction...")
    
    # Create analyzer with mock database
    from unittest.mock import patch
    with patch('app.services.predictive_analytics.DatabaseManager', return_value=MockDatabaseManager()):
        analyzer = PredictiveAnalyzer()
    
    # Create sample time series data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    values = np.random.normal(1000, 100, 30) + np.arange(30) * 10  # Trending upward
    series = pd.Series(values, index=dates)
    
    try:
        prediction, confidence = await analyzer._predict_time_series(series, forecast_days=30)
        
        print(f"✓ Time series prediction successful")
        print(f"  - Prediction: {prediction:.2f}")
        print(f"  - Confidence: {confidence:.2f}")
        
        assert prediction > 0, "Prediction should be positive"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        
        return True
        
    except Exception as e:
        print(f"✗ Time series prediction failed: {e}")
        return False


async def test_cash_flow_factors():
    """Test cash flow factor identification"""
    print("Testing cash flow factor identification...")
    
    from unittest.mock import patch
    with patch('app.services.predictive_analytics.DatabaseManager', return_value=MockDatabaseManager()):
        analyzer = PredictiveAnalyzer()
    
    # Create sample transaction data
    sample_data = [
        {'amount': 10000, 'created_at': datetime.utcnow() - timedelta(days=30)},
        {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=29)},
        {'amount': 15000, 'created_at': datetime.utcnow() - timedelta(days=25)},
        {'amount': -7000, 'created_at': datetime.utcnow() - timedelta(days=20)},
        {'amount': 12000, 'created_at': datetime.utcnow() - timedelta(days=15)},
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        df['date'] = pd.to_datetime(df['created_at'])
        df = df.set_index('date')
        
        factors = await analyzer._identify_cash_flow_factors(df)
        
        print(f"✓ Cash flow factor identification successful")
        print(f"  - Factors identified: {len(factors)}")
        for factor in factors:
            print(f"    - {factor}")
        
        assert isinstance(factors, list), "Factors should be a list"
        assert len(factors) > 0, "Should identify at least one factor"
        
        return True
        
    except Exception as e:
        print(f"✗ Cash flow factor identification failed: {e}")
        return False


async def test_customer_analysis():
    """Test customer revenue analysis"""
    print("Testing customer revenue analysis...")
    
    analyzer = PredictiveAnalyzer()
    analyzer.db = MockDatabaseManager()
    
    # Mock the customer data retrieval
    sample_customer_data = [
        {'customer_id': 'cust_1', 'customer_name': 'Customer A', 'amount': 50000, 'transaction_count': 10},
        {'customer_id': 'cust_2', 'customer_name': 'Customer B', 'amount': 30000, 'transaction_count': 8},
        {'customer_id': 'cust_3', 'customer_name': 'Customer C', 'amount': 25000, 'transaction_count': 6},
        {'customer_id': 'cust_4', 'customer_name': 'Customer D', 'amount': 15000, 'transaction_count': 4},
        {'customer_id': 'cust_5', 'customer_name': 'Customer E', 'amount': 10000, 'transaction_count': 3},
    ]
    
    # Mock the database method
    analyzer._get_customer_revenue_data = AsyncMock(return_value=sample_customer_data)
    
    try:
        result = await analyzer.analyze_customer_revenue("test_business")
        
        print(f"✓ Customer revenue analysis successful")
        print(f"  - Top customers: {len(result.top_customers)}")
        print(f"  - Revenue concentration: {result.revenue_concentration:.2f}")
        print(f"  - Recommendations: {len(result.recommendations)}")
        
        assert isinstance(result, CustomerAnalysis), "Should return CustomerAnalysis"
        assert len(result.top_customers) > 0, "Should have top customers"
        assert 0.0 <= result.revenue_concentration <= 1.0, "Concentration should be between 0 and 1"
        
        # Verify Pareto analysis
        assert 'total_customers' in result.pareto_analysis, "Should have total customers"
        assert result.pareto_analysis['total_customers'] == 5, "Should have 5 customers"
        
        return True
        
    except Exception as e:
        print(f"✗ Customer revenue analysis failed: {e}")
        return False


async def test_working_capital_analysis():
    """Test working capital analysis"""
    print("Testing working capital analysis...")
    
    analyzer = PredictiveAnalyzer()
    analyzer.db = MockDatabaseManager()
    
    # Mock working capital data
    working_capital_data = {
        'current_assets': 100000,
        'current_liabilities': 60000,
        'historical_working_capital': [
            {'date': (datetime.utcnow() - timedelta(days=90)).isoformat(), 'working_capital': 35000},
            {'date': (datetime.utcnow() - timedelta(days=60)).isoformat(), 'working_capital': 38000},
            {'date': (datetime.utcnow() - timedelta(days=30)).isoformat(), 'working_capital': 40000},
        ]
    }
    
    # Mock the database method
    analyzer._get_working_capital_data = AsyncMock(return_value=working_capital_data)
    
    try:
        result = await analyzer.calculate_working_capital_trend("test_business")
        
        print(f"✓ Working capital analysis successful")
        print(f"  - Current working capital: ₹{result.current_working_capital:,.2f}")
        print(f"  - Trend direction: {result.trend_direction}")
        print(f"  - Risk level: {result.risk_level}")
        print(f"  - Recommendations: {len(result.recommendations)}")
        
        assert isinstance(result, WorkingCapitalAnalysis), "Should return WorkingCapitalAnalysis"
        assert result.current_working_capital == 40000, "Should calculate correct working capital"
        assert result.trend_direction in ["increasing", "decreasing", "stable"], "Should have valid trend"
        assert result.risk_level in ["low", "medium", "high", "critical"], "Should have valid risk level"
        
        return True
        
    except Exception as e:
        print(f"✗ Working capital analysis failed: {e}")
        return False


async def test_cash_flow_prediction():
    """Test cash flow prediction with mock data"""
    print("Testing cash flow prediction...")
    
    analyzer = PredictiveAnalyzer()
    analyzer.db = MockDatabaseManager()
    
    # Mock transaction data
    sample_transactions = [
        {'amount': 10000, 'created_at': datetime.utcnow() - timedelta(days=30), 'transaction_type': 'income'},
        {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=29), 'transaction_type': 'expense'},
        {'amount': 15000, 'created_at': datetime.utcnow() - timedelta(days=25), 'transaction_type': 'income'},
        {'amount': -7000, 'created_at': datetime.utcnow() - timedelta(days=20), 'transaction_type': 'expense'},
        {'amount': 12000, 'created_at': datetime.utcnow() - timedelta(days=15), 'transaction_type': 'income'},
        {'amount': -6000, 'created_at': datetime.utcnow() - timedelta(days=10), 'transaction_type': 'expense'},
        {'amount': 18000, 'created_at': datetime.utcnow() - timedelta(days=5), 'transaction_type': 'income'},
        {'amount': -8000, 'created_at': datetime.utcnow() - timedelta(days=2), 'transaction_type': 'expense'},
    ]
    
    # Mock the database method
    analyzer._get_transaction_history = AsyncMock(return_value=sample_transactions)
    
    try:
        result = await analyzer.predict_cash_flow("test_business", months=3)
        
        print(f"✓ Cash flow prediction successful")
        print(f"  - Predicted inflow: ₹{result.predicted_inflow:,.2f}")
        print(f"  - Predicted outflow: ₹{result.predicted_outflow:,.2f}")
        print(f"  - Net cash flow: ₹{result.net_cash_flow:,.2f}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Factors: {len(result.factors)}")
        
        assert isinstance(result, CashFlowPrediction), "Should return CashFlowPrediction"
        assert result.confidence > 0.0, "Should have some confidence"
        assert result.predicted_inflow > 0, "Should predict positive inflow"
        assert result.predicted_outflow > 0, "Should predict positive outflow"
        assert len(result.factors) > 0, "Should identify factors"
        
        return True
        
    except Exception as e:
        print(f"✗ Cash flow prediction failed: {e}")
        return False


async def test_insight_generation():
    """Test comprehensive insight generation"""
    print("Testing comprehensive insight generation...")
    
    analyzer = PredictiveAnalyzer()
    analyzer.db = MockDatabaseManager()
    
    # Mock all data retrieval methods
    analyzer._get_transaction_history = AsyncMock(return_value=[
        {'amount': 10000, 'created_at': datetime.utcnow() - timedelta(days=30), 'transaction_type': 'income'},
        {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=29), 'transaction_type': 'expense'},
    ])
    
    analyzer._get_customer_revenue_data = AsyncMock(return_value=[
        {'customer_id': 'cust_1', 'customer_name': 'Customer A', 'amount': 50000, 'transaction_count': 10},
    ])
    
    analyzer._get_working_capital_data = AsyncMock(return_value={
        'current_assets': 100000,
        'current_liabilities': 60000,
        'historical_working_capital': []
    })
    
    try:
        result = await analyzer.generate_insights("test_business")
        
        print(f"✓ Insight generation successful")
        print(f"  - Business ID: {result.business_id}")
        print(f"  - Success: {result.success}")
        print(f"  - Insights generated: {len(result.insights)}")
        print(f"  - Next update: {result.next_update}")
        
        assert result.success is True, "Should be successful"
        assert result.business_id == "test_business", "Should have correct business ID"
        assert result.next_update is not None, "Should have next update time"
        
        return True
        
    except Exception as e:
        print(f"✗ Insight generation failed: {e}")
        return False


async def main():
    """Run all validation tests"""
    print("🚀 Starting Predictive Analytics Service Validation")
    print("=" * 60)
    
    tests = [
        test_time_series_prediction,
        test_cash_flow_factors,
        test_customer_analysis,
        test_working_capital_analysis,
        test_cash_flow_prediction,
        test_insight_generation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print()
    print("=" * 60)
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All predictive analytics functionality is working correctly!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)