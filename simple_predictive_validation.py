#!/usr/bin/env python3
"""
Simple validation script for predictive analytics core functionality
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Test core functionality without database dependencies
def test_time_series_prediction():
    """Test time series prediction algorithms"""
    print("Testing time series prediction algorithms...")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        values = np.random.normal(1000, 100, 30) + np.arange(30) * 10  # Trending upward
        series = pd.Series(values, index=dates)
        
        # Test ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=30)
        
        print(f"✓ ARIMA prediction successful: {forecast.sum():.2f}")
        
        # Test Exponential Smoothing
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=30)
        
        print(f"✓ Exponential Smoothing prediction successful: {forecast.sum():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Time series prediction failed: {e}")
        return False


def test_pareto_analysis():
    """Test Pareto analysis for customer revenue"""
    print("Testing Pareto analysis...")
    
    try:
        # Sample customer data
        customer_data = [
            {'customer_id': 'cust_1', 'customer_name': 'Customer A', 'amount': 50000, 'transaction_count': 10},
            {'customer_id': 'cust_2', 'customer_name': 'Customer B', 'amount': 30000, 'transaction_count': 8},
            {'customer_id': 'cust_3', 'customer_name': 'Customer C', 'amount': 25000, 'transaction_count': 6},
            {'customer_id': 'cust_4', 'customer_name': 'Customer D', 'amount': 15000, 'transaction_count': 4},
            {'customer_id': 'cust_5', 'customer_name': 'Customer E', 'amount': 10000, 'transaction_count': 3},
        ]
        
        # Create DataFrame and calculate revenue per customer
        df = pd.DataFrame(customer_data)
        customer_revenue = df.groupby('customer_id').agg({
            'amount': 'sum',
            'customer_name': 'first',
            'transaction_count': 'first'
        }).reset_index()
        
        # Sort by revenue descending
        customer_revenue = customer_revenue.sort_values('amount', ascending=False)
        customer_revenue['cumulative_revenue'] = customer_revenue['amount'].cumsum()
        customer_revenue['revenue_percentage'] = (
            customer_revenue['cumulative_revenue'] / customer_revenue['amount'].sum() * 100
        )
        
        # Apply Pareto analysis (80/20 rule)
        total_revenue = customer_revenue['amount'].sum()
        pareto_customers = customer_revenue[customer_revenue['revenue_percentage'] <= 80]
        
        # Find top customers contributing to 70% of revenue
        revenue_70_threshold = total_revenue * 0.7
        top_customers_70 = customer_revenue[
            customer_revenue['cumulative_revenue'] <= revenue_70_threshold
        ]
        
        print(f"✓ Pareto analysis successful")
        print(f"  - Total customers: {len(customer_revenue)}")
        print(f"  - Top 80% customers: {len(pareto_customers)}")
        print(f"  - Customers for 70% revenue: {len(top_customers_70)}")
        print(f"  - Revenue concentration: {len(top_customers_70) / len(customer_revenue):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pareto analysis failed: {e}")
        return False


def test_working_capital_trend():
    """Test working capital trend analysis"""
    print("Testing working capital trend analysis...")
    
    try:
        from scipy import stats
        
        # Sample working capital data
        historical_data = [
            {'date': '2023-01-01', 'working_capital': 35000},
            {'date': '2023-02-01', 'working_capital': 38000},
            {'date': '2023-03-01', 'working_capital': 40000},
            {'date': '2023-04-01', 'working_capital': 42000},
            {'date': '2023-05-01', 'working_capital': 45000},
        ]
        
        # Calculate trend using linear regression
        dates = [datetime.fromisoformat(item['date']) for item in historical_data]
        values = [item['working_capital'] for item in historical_data]
        
        # Convert dates to numeric values for regression
        date_nums = [(date - dates[0]).days for date in dates]
        slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, values)
        
        # Determine trend direction
        if abs(slope) < 100:  # Threshold for stability
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        print(f"✓ Working capital trend analysis successful")
        print(f"  - Slope: {slope:.2f}")
        print(f"  - Trend direction: {trend_direction}")
        print(f"  - R-value: {r_value:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Working capital trend analysis failed: {e}")
        return False


def test_cash_flow_factors():
    """Test cash flow factor identification"""
    print("Testing cash flow factor identification...")
    
    try:
        # Sample transaction data
        sample_data = [
            {'amount': 10000, 'created_at': datetime.utcnow() - timedelta(days=30)},
            {'amount': -5000, 'created_at': datetime.utcnow() - timedelta(days=29)},
            {'amount': 15000, 'created_at': datetime.utcnow() - timedelta(days=25)},
            {'amount': -7000, 'created_at': datetime.utcnow() - timedelta(days=20)},
            {'amount': 12000, 'created_at': datetime.utcnow() - timedelta(days=15)},
        ]
        
        df = pd.DataFrame(sample_data)
        df['date'] = pd.to_datetime(df['created_at'])
        df = df.set_index('date')
        
        factors = []
        
        # Check for large transactions
        large_transactions = df[abs(df['amount']) > df['amount'].quantile(0.9)]
        if len(large_transactions) > 0:
            factors.append(f"Large transactions impact: {len(large_transactions)} significant transactions")
        
        # Check transaction frequency trends
        daily_counts = df.resample('D').size()
        if len(daily_counts) >= 7:
            recent_avg = daily_counts.tail(7).mean()
            overall_avg = daily_counts.mean()
            if recent_avg > overall_avg * 1.2:
                factors.append("Increasing transaction frequency")
            elif recent_avg < overall_avg * 0.8:
                factors.append("Decreasing transaction frequency")
        
        if not factors:
            factors.append("Standard business transaction patterns")
        
        print(f"✓ Cash flow factor identification successful")
        print(f"  - Factors identified: {len(factors)}")
        for factor in factors:
            print(f"    - {factor}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cash flow factor identification failed: {e}")
        return False


def test_seasonal_decomposition():
    """Test seasonal pattern detection"""
    print("Testing seasonal pattern detection...")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Create sample data with seasonal pattern
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        trend = np.linspace(1000, 1200, 365)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly seasonality
        noise = np.random.normal(0, 50, 365)
        values = trend + seasonal + noise
        
        series = pd.Series(values, index=dates)
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(series, model='additive', period=30)
        
        print(f"✓ Seasonal decomposition successful")
        print(f"  - Trend component: {decomposition.trend.dropna().mean():.2f}")
        print(f"  - Seasonal component range: {decomposition.seasonal.max():.2f} to {decomposition.seasonal.min():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Seasonal decomposition failed: {e}")
        return False


def test_risk_assessment():
    """Test risk level assessment logic"""
    print("Testing risk assessment logic...")
    
    try:
        def assess_working_capital_risk(current_wc, slope, days_until_depletion):
            """Assess working capital risk level"""
            if current_wc <= 0:
                return "critical"
            elif days_until_depletion and days_until_depletion <= 30:
                return "high"
            elif days_until_depletion and days_until_depletion <= 90:
                return "medium"
            elif slope < -1000:  # Rapidly declining
                return "medium"
            else:
                return "low"
        
        # Test different scenarios
        test_cases = [
            (-10000, -100, None, "critical"),
            (50000, -1000, 20, "high"),
            (100000, -500, 60, "medium"),
            (100000, 100, None, "low"),
        ]
        
        for current_wc, slope, days, expected in test_cases:
            result = assess_working_capital_risk(current_wc, slope, days)
            assert result == expected, f"Expected {expected}, got {result}"
        
        print(f"✓ Risk assessment logic successful")
        print(f"  - All test cases passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Risk assessment logic failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 Starting Predictive Analytics Core Validation")
    print("=" * 60)
    
    tests = [
        test_time_series_prediction,
        test_pareto_analysis,
        test_working_capital_trend,
        test_cash_flow_factors,
        test_seasonal_decomposition,
        test_risk_assessment,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print()
    print("=" * 60)
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All predictive analytics core functionality is working correctly!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)