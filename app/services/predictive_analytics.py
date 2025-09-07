"""
Predictive Analytics Service - AI-powered business intelligence
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

from ..models.base import BusinessInsight, InsightType
from ..models.responses import BusinessInsightsResponse, CashFlowPrediction, CustomerAnalysis, WorkingCapitalAnalysis
from ..database import get_db_manager

logger = logging.getLogger(__name__)


class PredictiveAnalyzer:
    """Predictive analytics service with ML-powered insights"""
    
    def __init__(self):
        self.db = get_db_manager()
        self.min_data_points = 10  # Minimum data points for reliable predictions
        self.confidence_threshold = 0.6  # Minimum confidence for predictions
    
    async def generate_insights(self, business_id: str) -> BusinessInsightsResponse:
        """Generate comprehensive business insights"""
        logger.info(f"Generating insights for business: {business_id}")
        
        try:
            insights = []
            
            # Generate cash flow insights
            cash_flow_prediction = await self.predict_cash_flow(business_id)
            if cash_flow_prediction.confidence >= self.confidence_threshold:
                insights.extend(await self._create_cash_flow_insights(business_id, cash_flow_prediction))
            
            # Generate customer revenue insights
            customer_analysis = await self.analyze_customer_revenue(business_id)
            if customer_analysis.top_customers:
                insights.extend(await self._create_customer_insights(business_id, customer_analysis))
            
            # Generate working capital insights
            working_capital_analysis = await self.calculate_working_capital_trend(business_id)
            if working_capital_analysis.current_working_capital != 0:
                insights.extend(await self._create_working_capital_insights(business_id, working_capital_analysis))
            
            # Generate expense trend insights
            expense_insights = await self._analyze_expense_trends(business_id)
            insights.extend(expense_insights)
            
            # Save insights to database and convert to response objects
            from ..models.responses import InsightResponse
            
            insight_responses = []
            for insight in insights:
                # Save to database
                insight_data = {
                    "id": insight.id,
                    "business_id": business_id,
                    "insight_type": insight.type.value,
                    "title": insight.title,
                    "description": insight.description,
                    "priority": insight.priority,
                    "category": insight.category,
                    "recommendations": insight.recommendations,
                    "impact_score": insight.impact_score,
                    "valid_until": insight.valid_until.isoformat() if insight.valid_until else None,

                    "created_at": datetime.utcnow().isoformat()
                }
                
                saved_id = await self.db.save_business_insight(insight_data)
                if saved_id:
                    logger.info(f"Saved insight {insight.id} to database")
                else:
                    logger.warning(f"Failed to save insight {insight.id} to database")
                
                # Create response object
                insight_response = InsightResponse(
                    success=True,
                    message="Insight generated",
                    insight_id=insight.id,
                    insight_type=insight.type.value,
                    title=insight.title,
                    description=insight.description,
                    priority=insight.priority,
                    category=insight.category,
                    recommendations=insight.recommendations,
                    data={}
                )
                insight_responses.append(insight_response)
            
            # Create proper response with all required fields
            categories = list(set([insight.category for insight in insights])) if insights else ["general"]
            priority_summary = {}
            for insight in insights:
                priority = insight.priority
                priority_summary[priority] = priority_summary.get(priority, 0) + 1
            
            return BusinessInsightsResponse(
                success=True,
                message="Insights generated successfully",
                insights=insight_responses,
                total_count=len(insights),
                categories=categories,
                priority_summary=priority_summary
            )
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return BusinessInsightsResponse(
                success=False,
                message=f"Insight generation failed: {str(e)}",
                insights=[],
                total_count=0,
                categories=[],
                priority_summary={}
            )
    
    async def predict_cash_flow(self, business_id: str, months: int = 3) -> CashFlowPrediction:
        """Predict cash flow using time series analysis"""
        logger.info(f"Predicting cash flow for business: {business_id}")
        
        try:
            # Get historical transaction data
            transactions_data = await self._get_transaction_history(business_id)
            
            if len(transactions_data) < self.min_data_points:
                logger.warning(f"Insufficient data for cash flow prediction: {len(transactions_data)} points")
                return CashFlowPrediction(
                    period=f"Next {months} months",
                    predicted_inflow=0.0,
                    predicted_outflow=0.0,
                    net_cash_flow=0.0,
                    confidence=0.0,
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow() + timedelta(days=30 * months),
                    factors=["Insufficient historical data for prediction"]
                )
            
            # Prepare time series data
            df = pd.DataFrame(transactions_data)
            df['date'] = pd.to_datetime(df['created_at'])
            df = df.set_index('date')
            
            # Aggregate daily cash flows
            daily_inflow = df[df['amount'] > 0].resample('D')['amount'].sum().fillna(0)
            daily_outflow = df[df['amount'] < 0].resample('D')['amount'].sum().fillna(0).abs()
            
            # Predict inflow and outflow separately
            inflow_prediction, inflow_confidence = await self._predict_time_series(daily_inflow, months * 30)
            outflow_prediction, outflow_confidence = await self._predict_time_series(daily_outflow, months * 30)
            
            # Calculate net cash flow
            net_cash_flow = inflow_prediction - outflow_prediction
            overall_confidence = (inflow_confidence + outflow_confidence) / 2
            
            # Identify key factors affecting cash flow
            factors = await self._identify_cash_flow_factors(df)
            
            period_start = datetime.utcnow()
            period_end = period_start + timedelta(days=30 * months)
            
            return CashFlowPrediction(
                period=f"Next {months} months",
                predicted_inflow=float(inflow_prediction),
                predicted_outflow=float(outflow_prediction),
                net_cash_flow=float(net_cash_flow),
                confidence=float(overall_confidence),
                period_start=period_start,
                period_end=period_end,
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Error predicting cash flow: {e}")
            return CashFlowPrediction(
                period=f"Next {months} months",
                predicted_inflow=0.0,
                predicted_outflow=0.0,
                net_cash_flow=0.0,
                confidence=0.0,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow() + timedelta(days=30 * months),
                factors=[f"Prediction error: {str(e)}"]
            )
    
    async def analyze_customer_revenue(self, business_id: str) -> CustomerAnalysis:
        """Analyze customer revenue using Pareto principle"""
        logger.info(f"Analyzing customer revenue for business: {business_id}")
        
        try:
            # Get customer transaction data
            customer_data = await self._get_customer_revenue_data(business_id)
            
            if not customer_data:
                return CustomerAnalysis(
                    recommendations=["No customer data available for analysis"]
                )
            
            # Create DataFrame and calculate revenue per customer
            df = pd.DataFrame(customer_data)
            customer_revenue = df.groupby('customer_id').agg({
                'amount': 'sum',
                'customer_name': 'first',
                'transaction_count': 'count'
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
            
            # Calculate concentration metrics
            revenue_concentration = len(top_customers_70) / len(customer_revenue) if len(customer_revenue) > 0 else 0
            
            # Prepare top customers data
            top_customers = []
            for _, customer in top_customers_70.head(10).iterrows():
                top_customers.append({
                    'customer_id': customer['customer_id'],
                    'customer_name': customer['customer_name'],
                    'revenue': float(customer['amount']),
                    'percentage': float(customer['amount'] / total_revenue * 100),
                    'transaction_count': int(customer['transaction_count'])
                })
            
            # Generate recommendations
            recommendations = await self._generate_customer_recommendations(
                customer_revenue, revenue_concentration, len(top_customers_70)
            )
            
            return CustomerAnalysis(
                top_customers=top_customers,
                revenue_concentration=float(revenue_concentration),
                pareto_analysis={
                    'total_customers': len(customer_revenue),
                    'top_20_percent_customers': len(pareto_customers),
                    'revenue_from_top_20_percent': float(pareto_customers['amount'].sum() / total_revenue * 100),
                    'customers_for_70_percent_revenue': len(top_customers_70)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing customer revenue: {e}")
            return CustomerAnalysis(
                recommendations=[f"Customer analysis error: {str(e)}"]
            )
    
    async def calculate_working_capital_trend(self, business_id: str) -> WorkingCapitalAnalysis:
        """Calculate working capital trend and depletion risk"""
        logger.info(f"Calculating working capital trend for business: {business_id}")
        
        try:
            # Get accounts receivable and payable data
            working_capital_data = await self._get_working_capital_data(business_id)
            
            if not working_capital_data:
                return WorkingCapitalAnalysis(
                    current_working_capital=0.0,
                    trend_direction="stable",
                    risk_level="low",
                    recommendations=["No working capital data available"]
                )
            
            # Calculate current working capital
            current_assets = working_capital_data.get('current_assets', 0)
            current_liabilities = working_capital_data.get('current_liabilities', 0)
            current_working_capital = current_assets - current_liabilities
            
            # Analyze trend using historical data
            historical_data = working_capital_data.get('historical_working_capital', [])
            
            if len(historical_data) >= 3:
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
                
                # Calculate days until depletion if trend is negative
                days_until_depletion = None
                if slope < 0 and current_working_capital > 0:
                    days_until_depletion = int(current_working_capital / abs(slope))
                
                # Determine risk level
                risk_level = await self._assess_working_capital_risk(
                    current_working_capital, slope, days_until_depletion
                )
            else:
                trend_direction = "stable"
                days_until_depletion = None
                risk_level = "low" if current_working_capital > 0 else "high"
            
            # Generate recommendations
            recommendations = await self._generate_working_capital_recommendations(
                current_working_capital, trend_direction, days_until_depletion, risk_level
            )
            
            return WorkingCapitalAnalysis(
                current_working_capital=float(current_working_capital),
                trend_direction=trend_direction,
                days_until_depletion=days_until_depletion,
                recommendations=recommendations,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating working capital trend: {e}")
            return WorkingCapitalAnalysis(
                current_working_capital=0.0,
                trend_direction="stable",
                risk_level="low",
                recommendations=[f"Working capital analysis error: {str(e)}"]
            )
    
    # Helper methods for data retrieval and analysis
    
    async def _get_transaction_history(self, business_id: str) -> List[Dict[str, Any]]:
        """Get transaction history for cash flow analysis"""
        try:
            # For now, return mock data since database might not have transactions table
            import random
            transactions = []
            base_date = datetime.now()
            
            for i in range(60):  # Generate 60 days of mock data
                date = base_date - timedelta(days=i)
                amount = random.uniform(-2000, 5000)  # Mix of positive and negative
                transactions.append({
                    'amount': amount,
                    'created_at': date,
                    'transaction_type': 'income' if amount > 0 else 'expense'
                })
            
            return transactions
        except Exception as e:
            logger.error(f"Error fetching transaction history: {e}")
            return []
    
    async def _get_customer_revenue_data(self, business_id: str) -> List[Dict[str, Any]]:
        """Get customer revenue data for Pareto analysis"""
        try:
            # Mock customer revenue data
            import random
            customers = [
                {'customer_id': 'cust_001', 'customer_name': 'ABC Corp', 'amount': random.uniform(50000, 200000), 'transaction_count': random.randint(5, 20)},
                {'customer_id': 'cust_002', 'customer_name': 'XYZ Ltd', 'amount': random.uniform(30000, 150000), 'transaction_count': random.randint(3, 15)},
                {'customer_id': 'cust_003', 'customer_name': 'Tech Solutions', 'amount': random.uniform(20000, 100000), 'transaction_count': random.randint(2, 10)},
                {'customer_id': 'cust_004', 'customer_name': 'Global Inc', 'amount': random.uniform(10000, 80000), 'transaction_count': random.randint(1, 8)},
                {'customer_id': 'cust_005', 'customer_name': 'Local Business', 'amount': random.uniform(5000, 50000), 'transaction_count': random.randint(1, 5)},
            ]
            return customers
        except Exception as e:
            logger.error(f"Error fetching customer revenue data: {e}")
            return []
    
    async def _get_working_capital_data(self, business_id: str) -> Dict[str, Any]:
        """Get working capital data"""
        try:
            # Mock working capital data
            import random
            
            current_assets = random.uniform(100000, 500000)
            current_liabilities = random.uniform(50000, 300000)
            
            # Generate historical data
            historical_data = []
            base_date = datetime.now()
            
            for i in range(30):  # 30 days of historical data
                date = base_date - timedelta(days=i)
                working_capital = random.uniform(50000, 200000) - (i * random.uniform(0, 1000))  # Slight downward trend
                historical_data.append({
                    'date': date.isoformat(),
                    'working_capital': working_capital
                })
            
            return {
                'current_assets': float(current_assets),
                'current_liabilities': float(current_liabilities),
                'historical_working_capital': historical_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching working capital data: {e}")
            return {}
    
    async def _predict_time_series(self, series: pd.Series, forecast_days: int) -> Tuple[float, float]:
        """Predict time series using ARIMA or exponential smoothing"""
        try:
            if len(series) < 10:
                # Use simple moving average for small datasets
                prediction = series.tail(7).mean() * forecast_days
                confidence = 0.3
                return prediction, confidence
            
            # Try ARIMA first
            try:
                # Auto-select ARIMA parameters
                model = ARIMA(series, order=(1, 1, 1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=forecast_days)
                prediction = forecast.sum()
                confidence = 0.7
                return prediction, confidence
            except:
                # Fallback to exponential smoothing
                try:
                    model = ExponentialSmoothing(series, trend='add', seasonal=None)
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=forecast_days)
                    prediction = forecast.sum()
                    confidence = 0.6
                    return prediction, confidence
                except:
                    # Final fallback to trend-based prediction
                    if len(series) >= 7:
                        recent_trend = series.tail(7).mean()
                        prediction = recent_trend * forecast_days
                        confidence = 0.4
                        return prediction, confidence
                    else:
                        prediction = series.mean() * forecast_days
                        confidence = 0.2
                        return prediction, confidence
                        
        except Exception as e:
            logger.error(f"Error in time series prediction: {e}")
            # Fallback to simple average
            prediction = series.mean() * forecast_days if len(series) > 0 else 0
            confidence = 0.1
            return prediction, confidence
    
    async def _identify_cash_flow_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify key factors affecting cash flow"""
        factors = []
        
        try:
            # Analyze transaction patterns
            if len(df) > 0:
                # Check for seasonal patterns
                if len(df) >= 30:
                    monthly_avg = df.resample('M')['amount'].sum()
                    if len(monthly_avg) >= 3:
                        cv = monthly_avg.std() / monthly_avg.mean()
                        if cv > 0.3:
                            factors.append("Seasonal revenue patterns detected")
                
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
        
        except Exception as e:
            logger.error(f"Error identifying cash flow factors: {e}")
            factors.append("Unable to analyze cash flow factors")
        
        return factors if factors else ["Standard business transaction patterns"]
    
    async def _create_cash_flow_insights(self, business_id: str, prediction: CashFlowPrediction) -> List[BusinessInsight]:
        """Create business insights from cash flow prediction"""
        insights = []
        
        try:
            # Cash flow warning insight
            if prediction.net_cash_flow < 0:
                insights.append(BusinessInsight(
                    type=InsightType.CASH_FLOW_PREDICTION,
                    title="Cash Flow Warning",
                    description=f"Predicted negative cash flow of ₹{abs(prediction.net_cash_flow):,.2f} over the next {(prediction.period_end - prediction.period_start).days} days",
                    recommendations=[
                        "Focus on collecting outstanding receivables",
                        "Consider delaying non-essential expenses",
                        "Explore short-term financing options",
                        "Accelerate sales activities"
                    ],
                    impact_score=0.9,
                    business_id=business_id,
                    category="cash_flow",
                    priority="high",
                    valid_until=prediction.period_end
                ))
            
            # Positive cash flow insight
            elif prediction.net_cash_flow > 0:
                insights.append(BusinessInsight(
                    type=InsightType.CASH_FLOW_PREDICTION,
                    title="Positive Cash Flow Forecast",
                    description=f"Predicted positive cash flow of ₹{prediction.net_cash_flow:,.2f} over the next {(prediction.period_end - prediction.period_start).days} days",
                    recommendations=[
                        "Consider investing surplus cash",
                        "Plan for business expansion opportunities",
                        "Build emergency cash reserves",
                        "Evaluate equipment or technology upgrades"
                    ],
                    impact_score=0.7,
                    business_id=business_id,
                    category="cash_flow",
                    priority="medium",
                    valid_until=prediction.period_end
                ))
        
        except Exception as e:
            logger.error(f"Error creating cash flow insights: {e}")
        
        return insights
    
    async def _create_customer_insights(self, business_id: str, analysis: CustomerAnalysis) -> List[BusinessInsight]:
        """Create business insights from customer analysis"""
        insights = []
        
        try:
            # High revenue concentration warning
            if analysis.revenue_concentration < 0.2 and len(analysis.top_customers) <= 3:
                top_customer_names = [c['customer_name'] for c in analysis.top_customers[:3]]
                insights.append(BusinessInsight(
                    type=InsightType.CUSTOMER_ANALYSIS,
                    title="Revenue Concentration Risk",
                    description=f"Top 3 customers ({', '.join(top_customer_names)}) contribute 70% of revenue — protect them",
                    recommendations=[
                        "Strengthen relationships with top customers",
                        "Diversify customer base to reduce dependency",
                        "Implement customer retention programs",
                        "Regular check-ins with key accounts"
                    ],
                    impact_score=0.8,
                    business_id=business_id,
                    category="customer_analysis",
                    priority="high"
                ))
            
            # Customer diversification opportunity
            elif analysis.revenue_concentration > 0.4:
                insights.append(BusinessInsight(
                    type=InsightType.CUSTOMER_ANALYSIS,
                    title="Well-Diversified Customer Base",
                    description=f"Revenue is well-distributed across {len(analysis.top_customers)} customers",
                    recommendations=[
                        "Continue maintaining diverse customer relationships",
                        "Identify patterns in successful customer acquisition",
                        "Scale successful customer engagement strategies"
                    ],
                    impact_score=0.6,
                    business_id=business_id,
                    category="customer_analysis",
                    priority="medium"
                ))
        
        except Exception as e:
            logger.error(f"Error creating customer insights: {e}")
        
        return insights
    
    async def _create_working_capital_insights(self, business_id: str, analysis: WorkingCapitalAnalysis) -> List[BusinessInsight]:
        """Create business insights from working capital analysis"""
        insights = []
        
        try:
            # Working capital depletion warning
            if analysis.days_until_depletion and analysis.days_until_depletion <= 90:
                insights.append(BusinessInsight(
                    type=InsightType.WORKING_CAPITAL,
                    title="Working Capital Alert",
                    description=f"Your working capital will run out in {analysis.days_until_depletion} days unless receivables improve",
                    recommendations=analysis.recommendations,
                    impact_score=0.9,
                    business_id=business_id,
                    category="working_capital",
                    priority="high"
                ))
            
            # Positive working capital trend
            elif analysis.trend_direction == "increasing":
                insights.append(BusinessInsight(
                    type=InsightType.WORKING_CAPITAL,
                    title="Improving Working Capital",
                    description=f"Working capital is trending upward (₹{analysis.current_working_capital:,.2f})",
                    recommendations=analysis.recommendations,
                    impact_score=0.6,
                    business_id=business_id,
                    category="working_capital",
                    priority="medium"
                ))
        
        except Exception as e:
            logger.error(f"Error creating working capital insights: {e}")
        
        return insights
    
    async def _analyze_expense_trends(self, business_id: str) -> List[BusinessInsight]:
        """Analyze expense trends and generate insights"""
        insights = []
        
        try:
            # Mock expense data for testing
            import random
            
            # Generate mock expense data
            expense_data = []
            base_date = datetime.now()
            
            for i in range(60):  # 60 days of expense data
                date = base_date - timedelta(days=i)
                amount = random.uniform(1000, 10000)  # Random expense amounts
                category = random.choice(['utilities', 'supplies', 'rent', 'marketing', 'travel'])
                expense_data.append({
                    'amount': amount,
                    'created_at': date,
                    'category': category
                })
            
            if len(expense_data) < 10:
                return insights
            
            df = pd.DataFrame(expense_data)
            df['date'] = pd.to_datetime(df['created_at'])
            
            # Analyze monthly expense trends
            monthly_expenses = df.set_index('date').resample('M')['amount'].sum()
            
            if len(monthly_expenses) >= 3:
                # Calculate trend
                recent_avg = monthly_expenses.tail(2).mean()
                overall_avg = monthly_expenses.mean()
                
                if recent_avg > overall_avg * 1.2:
                    insights.append(BusinessInsight(
                        type=InsightType.EXPENSE_TREND,
                        title="Rising Expense Trend",
                        description=f"Monthly expenses have increased by {((recent_avg / overall_avg - 1) * 100):.1f}% recently",
                        recommendations=[
                            "Review recent expense categories for optimization",
                            "Identify non-essential expenses to reduce",
                            "Negotiate better rates with suppliers",
                            "Implement expense approval processes"
                        ],
                        impact_score=0.7,
                        business_id=business_id
                    ))
        
        except Exception as e:
            logger.error(f"Error analyzing expense trends: {e}")
        
        return insights
    
    async def _generate_customer_recommendations(self, customer_revenue: pd.DataFrame, 
                                               concentration: float, top_customer_count: int) -> List[str]:
        """Generate customer-specific recommendations"""
        recommendations = []
        
        if concentration < 0.2:
            recommendations.extend([
                "High revenue concentration detected - diversify customer base",
                "Implement customer retention programs for top clients",
                "Develop backup plans for key customer relationships"
            ])
        elif concentration > 0.5:
            recommendations.extend([
                "Good customer diversification - maintain current strategies",
                "Focus on scaling successful customer acquisition methods"
            ])
        
        if top_customer_count <= 5:
            recommendations.append("Consider expanding customer base to reduce dependency risk")
        
        return recommendations if recommendations else ["Continue monitoring customer revenue patterns"]
    
    async def _generate_working_capital_recommendations(self, current_wc: float, trend: str, 
                                                      days_until_depletion: Optional[int], 
                                                      risk_level: str) -> List[str]:
        """Generate working capital recommendations"""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Urgent: Accelerate accounts receivable collection",
                "Consider factoring or invoice financing",
                "Delay non-critical payments where possible",
                "Explore emergency credit facilities"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor cash flow closely",
                "Improve invoice collection processes",
                "Negotiate better payment terms with customers"
            ])
        else:
            recommendations.extend([
                "Maintain current working capital management",
                "Consider investing surplus in growth opportunities"
            ])
        
        if trend == "decreasing":
            recommendations.append("Address declining working capital trend immediately")
        
        return recommendations
    
    async def _assess_working_capital_risk(self, current_wc: float, slope: float, 
                                         days_until_depletion: Optional[int]) -> str:
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