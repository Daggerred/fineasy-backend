
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class FraudType(str, Enum):

    DUPLICATE_INVOICE = "duplicate_invoice"
    PAYMENT_MISMATCH = "payment_mismatch"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    SUPPLIER_DUPLICATE = "supplier_duplicate"


class InsightType(str, Enum):

    CASH_FLOW_PREDICTION = "cash_flow_prediction"
    CUSTOMER_ANALYSIS = "customer_analysis"
    EXPENSE_TREND = "expense_trend"
    REVENUE_FORECAST = "revenue_forecast"
    WORKING_CAPITAL = "working_capital"


class ComplianceType(str, Enum):

    GST_VALIDATION = "gst_validation"
    TAX_CALCULATION = "tax_calculation"
    MISSING_FIELDS = "missing_fields"
    DEADLINE_WARNING = "deadline_warning"


class ComplianceSeverity(str, Enum):

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):

    COMPLIANT = "compliant"
    ISSUES_FOUND = "issues_found"
    CRITICAL_ISSUES = "critical_issues"


class BaseResponse(BaseModel):

    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class FraudAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: FraudType
    message: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    business_id: str
    entity_id: Optional[str] = None


class BusinessInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: InsightType
    title: str
    description: str
    recommendations: List[str] = Field(default_factory=list)
    impact_score: float = Field(ge=0.0, le=1.0)
    valid_until: Optional[datetime] = None
    business_id: str
    category: str = "general"  # Add category attribute
    priority: str = "medium"  # Add priority attribute


class ComplianceIssue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ComplianceType
    description: str
    plain_language_explanation: str
    suggested_fixes: List[str] = Field(default_factory=list)
    severity: ComplianceSeverity
    field_name: Optional[str] = None
    current_value: Optional[str] = None
    expected_value: Optional[str] = None


class InvoiceItem(BaseModel):
    """Invoice item model"""
    name: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total_price: float = Field(ge=0)
    tax_rate: Optional[float] = Field(default=0.0, ge=0, le=100)
    description: Optional[str] = None


class InvoiceRequest(BaseModel):
    """Atta uthana hain"""
    raw_input: str
    business_id: str
    customer_name: Optional[str] = None
    items: List[InvoiceItem] = Field(default_factory=list)
    payment_preference: Optional[str] = None
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)


class AuthToken(BaseModel):
    token: str
    user_id: str
    business_id: Optional[str] = None