"""
Response models for API endpoints
"""
from pydantic import BaseModel
from typing import Any, Optional, Dict, List
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = datetime.utcnow()


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DataResponse(BaseResponse):
    """Response with data payload"""
    data: Any


class ListResponse(BaseResponse):
    """Response with list data"""
    data: List[Any]
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    checks: Dict[str, Any]


# Fraud Detection Response Models
class FraudAnalysisResponse(BaseResponse):
    """Fraud analysis response"""
    risk_score: float
    risk_level: str
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_id: str


class FraudAlertResponse(BaseResponse):
    """Fraud alert response"""
    alert_id: str
    alert_type: str
    severity: str
    description: str
    created_at: datetime
    status: str


# Insights Response Models
class InsightResponse(BaseResponse):
    """Business insight response"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    priority: str
    category: str
    recommendations: List[str]
    data: Dict[str, Any]


class BusinessInsightsResponse(BaseResponse):
    """Business insights collection response"""
    insights: List[InsightResponse]
    total_count: int
    categories: List[str]
    priority_summary: Dict[str, int]


class PredictiveAnalysisResponse(BaseResponse):
    """Predictive analysis response"""
    analysis_id: str
    predictions: List[Dict[str, Any]]
    confidence_score: float
    time_horizon: str
    model_version: str


# Compliance Response Models
class ComplianceResponse(BaseResponse):
    """General compliance response"""
    compliance_status: str
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    score: float

class ComplianceCheckResponse(BaseResponse):
    """Compliance check response"""
    check_id: str
    compliance_status: str
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    score: float


class GSTValidationResponse(BaseResponse):
    """GST validation response"""
    gstin: str
    is_valid: bool
    business_name: Optional[str] = None
    status: str
    registration_date: Optional[str] = None

class GSTValidationResult(BaseModel):
    """GST validation result model"""
    gstin: str
    is_valid: bool
    business_name: Optional[str] = None
    status: str
    registration_date: Optional[str] = None
    errors: List[str] = []

class TaxVerificationResult(BaseModel):
    """Tax verification result model"""
    tax_id: str
    is_valid: bool
    tax_type: str
    status: str
    verification_date: datetime
    errors: List[str] = []


# Invoice Response Models
class InvoiceAnalysisResponse(BaseResponse):
    """Invoice analysis response"""
    analysis_id: str
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    validation_results: Dict[str, Any]
    suggestions: List[str]


class NLPProcessingResponse(BaseResponse):
    """NLP processing response"""
    processing_id: str
    extracted_entities: List[Dict[str, Any]]
    sentiment_score: Optional[float] = None
    language: str
    confidence: float


# ML Engine Response Models
class ModelPredictionResponse(BaseResponse):
    """ML model prediction response"""
    prediction_id: str
    model_name: str
    model_version: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time_ms: float


class ModelStatusResponse(BaseResponse):
    """ML model status response"""
    model_name: str
    model_version: str
    status: str
    last_trained: Optional[datetime] = None
    accuracy_metrics: Optional[Dict[str, float]] = None


# Notification Response Models
class NotificationResponse(BaseResponse):
    """Notification response"""
    notification_id: str
    notification_type: str
    title: str
    message: str
    priority: str
    created_at: datetime
    read_at: Optional[datetime] = None


class NotificationPreferencesResponse(BaseResponse):
    """Notification preferences response"""
    user_id: str
    preferences: List[Dict[str, Any]]
    updated_at: datetime


# Cache Response Models
class CacheStatsResponse(BaseResponse):
    """Cache statistics response"""
    hit_rate: float
    total_keys: int
    memory_usage_mb: float
    uptime_seconds: int


# Feature Flag Response Models
class FeatureFlagResponse(BaseResponse):
    """Feature flag response"""
    flag_name: str
    enabled: bool


class CustomerAnalysis(BaseModel):
    """Customer analysis response"""
    top_customers: List[Dict[str, Any]] = []
    revenue_concentration: float = 0.0
    pareto_analysis: Dict[str, Any] = {}
    recommendations: List[str] = []


class WorkingCapitalAnalysis(BaseModel):
    """Working capital analysis response"""
    current_working_capital: float
    trend_direction: str
    days_until_depletion: Optional[int] = None
    risk_level: str
    recommendations: List[str] = []
    description: Optional[str] = None
    rollout_percentage: Optional[float] = None

# Additional Models for Insights
class CashFlowPrediction(BaseModel):
    """Cash flow prediction model"""
    period: str
    predicted_inflow: float
    predicted_outflow: float
    net_cash_flow: float
    confidence: float
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    factors: List[str] = []


class RevenueProjection(BaseModel):
    """Revenue projection model"""
    period: str
    projected_revenue: float
    growth_rate: float
    confidence: float


class ExpenseAnalysis(BaseModel):
    """Expense analysis model"""
    category: str
    current_amount: float
    projected_amount: float
    variance_percentage: float


class CustomerSegment(BaseModel):
    """Customer segment model"""
    segment_name: str
    customer_count: int
    revenue_contribution: float
    growth_potential: str


class ProductPerformance(BaseModel):
    """Product performance model"""
    product_name: str
    revenue: float
    units_sold: int
    profit_margin: float
    trend: str


class MarketTrend(BaseModel):
    """Market trend model"""
    trend_name: str
    impact_score: float
    description: str
    recommendation: str


class RiskAssessment(BaseModel):
    """Risk assessment model"""
    risk_type: str
    probability: float
    impact: str
    mitigation_strategy: str


class OpportunityAnalysis(BaseModel):
    """Opportunity analysis model"""
    opportunity_type: str
    potential_value: float
    effort_required: str
    timeline: str


class CompetitorInsight(BaseModel):
    """Competitor insight model"""
    competitor_name: str
    market_share: float
    strengths: List[str]
    weaknesses: List[str]


class SeasonalPattern(BaseModel):
    """Seasonal pattern model"""
    pattern_name: str
    peak_months: List[str]
    impact_percentage: float
    recommendation: str





class MLModelMetadata(BaseModel):
    """ML model metadata model"""
    model_name: str
    model_version: str
    model_type: str
    accuracy: Optional[float] = None
    last_trained: Optional[datetime] = None
    status: str = "active"
    parameters: Optional[Dict[str, Any]] = None
# Invoice Response Models
class InvoiceGenerationResponse(BaseResponse):
    """Invoice generation response"""
    invoice_id: Optional[str] = None
    invoice_data: Optional[Dict[str, Any]] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


class ParseTextResponse(BaseResponse):
    """Text parsing response"""
    parsed_data: Dict[str, Any]
    entities: Dict[str, Any]
    confidence: float


class EntityResolutionResponse(BaseResponse):
    """Entity resolution response"""
    resolved_entities: Dict[str, Any]
    confidence_scores: Dict[str, float]


class InvoicePreviewResponse(BaseResponse):
    """Invoice preview response"""
    preview_data: Dict[str, Any]
    validation_results: Dict[str, Any]