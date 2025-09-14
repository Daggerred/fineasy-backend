"""
Fraud Detection API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

from ..models.responses import FraudAnalysisResponse
from ..models.base import FraudAlert, FraudType
from ..services.fraud_detection import FraudDetector
from ..utils.auth import verify_token
from ..utils.security_middleware import security_middleware
from ..utils.audit_logger import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/fraud", tags=["fraud-detection"])
security = HTTPBearer()



class FraudAnalysisRequest(BaseModel):
    """Request model for fraud analysis"""
    business_id: str = Field(..., description="Business ID to analyze")
    analysis_types: Optional[List[str]] = Field(
        default=["duplicates", "mismatches", "patterns"],
        description="Types of fraud analysis to perform"
    )
    date_range_days: Optional[int] = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to analyze (1-365)"
    )
    include_resolved: Optional[bool] = Field(
        default=False,
        description="Include previously resolved alerts"
    )
    
    @field_validator('analysis_types')
    @classmethod
    def validate_analysis_types(cls, v):
        valid_types = ["duplicates", "mismatches", "patterns", "velocity"]
        if v:
            invalid_types = [t for t in v if t not in valid_types]
            if invalid_types:
                raise ValueError(f"Invalid analysis types: {invalid_types}. Valid types: {valid_types}")
        return v


class AlertUpdateRequest(BaseModel):
    
    status: str = Field(..., description="New alert status")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = ["active", "resolved", "false_positive", "investigating"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Valid statuses: {valid_statuses}")
        return v


class BulkAnalysisRequest(BaseModel):
    
    business_ids: List[str] = Field(..., description="List of business IDs to analyze")
    analysis_types: Optional[List[str]] = Field(
        default=["duplicates", "mismatches"],
        description="Types of fraud analysis to perform"
    )
    
    @field_validator('business_ids')
    @classmethod
    def validate_business_ids(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 business IDs allowed per bulk request")
        return v


class FraudAlertSummary(BaseModel):
    """Summary of fraud alerts"""
    total_alerts: int
    high_risk_alerts: int
    medium_risk_alerts: int
    low_risk_alerts: int
    alert_types: Dict[str, int]
    latest_alert_time: Optional[datetime]


class FraudAnalysisDetailResponse(FraudAnalysisResponse):
    alert_summary: FraudAlertSummary
    recommendations: List[str] = Field(default_factory=list)
    next_analysis_recommended: Optional[datetime] = None


# API Endpoints
@router.post("/analyze", response_model=FraudAnalysisDetailResponse)
async def analyze_fraud(
    request: FraudAnalysisRequest,
    token: str = Depends(security),
    fastapi_request: Request = None
):
   
    try:
        
        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
           
            logger.warning(f"Authentication bypassed for development - business_id: {request.business_id}")
            user_id = "dev_user"
        
       
        if isinstance(user_id, dict):
            user_id = user_id.get('user_id', 'dev_user')
        
      
        security_context = await security_middleware.validate_request(
            fastapi_request, request.business_id, "fraud_detection"
        )
        
        
        audit_logger.log_ai_operation(
            event_type=AuditEventType.FRAUD_DETECTION,
            business_id=request.business_id,
            user_id=user_id,
            operation_details={
                "analysis_types": request.analysis_types,
                "date_range_days": request.date_range_days
            },
            severity=AuditSeverity.MEDIUM,
            ip_address=security_context.get("client_ip"),
            user_agent=security_context.get("user_agent")
        )
        
        logger.info(f"Starting fraud analysis for business: {request.business_id} by user: {user_id}")
        
        
        detector = FraudDetector()
        
       
        result = await detector.analyze_fraud(request.business_id)
        
        if not result.success:
            logger.error(f"Fraud analysis failed for business {request.business_id}: {result.message}")
            raise HTTPException(status_code=500, detail=result.message)
        
       
        alert_summary = _generate_alert_summary(result.alerts)
        
       
        recommendations = _generate_recommendations(result.alerts, result.risk_score)
        
        
        next_analysis = _calculate_next_analysis_time(result.risk_score)
        
      
        logger.info(f"Fraud analysis completed for business {request.business_id}: "
                   f"{len(result.alerts)} alerts, risk score: {result.risk_score:.2f}")
        
        return FraudAnalysisDetailResponse(
            **result.dict(),
            alert_summary=alert_summary,
            recommendations=recommendations,
            next_analysis_recommended=next_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fraud analysis error for business {request.business_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Fraud analysis failed: {str(e)}"
        )


@router.get("/alerts/{business_id}", response_model=FraudAnalysisResponse)
async def get_fraud_alerts(
    business_id: str = Path(..., description="Business ID"),
    include_resolved: bool = Query(False, description="Include resolved alerts"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    token: str = Depends(security)
):
    
    try:
        # Verify authentication token
        user_info = await verify_token(token.credentials)
        logger.info(f"Fetching fraud alerts for business: {business_id} by user: {user_info.get('user_id')}")
        
        # Validate alert type filter
        if alert_type and alert_type not in [t.value for t in FraudType]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid alert type: {alert_type}. Valid types: {[t.value for t in FraudType]}"
            )
        
        
        detector = FraudDetector()
        
        
        alerts = await detector.get_fraud_alerts(
            business_id=business_id,
            include_resolved=include_resolved,
            alert_type=alert_type,
            limit=limit,
            offset=offset
        )
        
        
        risk_score = detector._calculate_risk_score(alerts)
        
        logger.info(f"Retrieved {len(alerts)} fraud alerts for business {business_id}")
        
        return FraudAnalysisResponse(
            business_id=business_id,
            alerts=alerts,
            risk_score=risk_score,
            analysis_metadata={
                "total_alerts": len(alerts),
                "include_resolved": include_resolved,
                "alert_type_filter": alert_type,
                "limit": limit,
                "offset": offset,
                "retrieved_at": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fraud alerts for business {business_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch fraud alerts: {str(e)}"
        )


@router.put("/alerts/{alert_id}")
async def update_fraud_alert(
    alert_id: str = Path(..., description="Alert ID"),
    request: AlertUpdateRequest = Body(...),
    token: str = Depends(security)
):
    
    try:
       
        user_info = await verify_token(token.credentials)
        logger.info(f"Updating fraud alert {alert_id} by user: {user_info.get('user_id')}")
        
      
        detector = FraudDetector()
        
       
        updated_alert = await detector.update_fraud_alert(
            alert_id=alert_id,
            status=request.status,
            resolution_notes=request.resolution_notes,
            updated_by=user_info.get('user_id')
        )
        
        if not updated_alert:
            raise HTTPException(status_code=404, detail="Fraud alert not found")
        
        logger.info(f"Fraud alert {alert_id} updated to status: {request.status}")
        
        return {
            "success": True,
            "message": f"Alert {alert_id} updated successfully",
            "alert": updated_alert,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating fraud alert {alert_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update fraud alert: {str(e)}"
        )


@router.post("/analyze/bulk")
async def bulk_fraud_analysis(
    request: BulkAnalysisRequest,
    token: str = Depends(security)
):
    
    try:
        # Verify authentication token
        user_info = await verify_token(token.credentials)
        logger.info(f"Starting bulk fraud analysis for {len(request.business_ids)} businesses by user: {user_info.get('user_id')}")
        
        # Initialize fraud detector
        detector = FraudDetector()
        
        # Perform analysis for each business
        results = {}
        for business_id in request.business_ids:
            try:
                result = await detector.analyze_fraud(business_id)
                results[business_id] = {
                    "success": result.success,
                    "alert_count": len(result.alerts),
                    "risk_score": result.risk_score,
                    "high_risk_alerts": len([a for a in result.alerts if a.confidence_score >= 0.8])
                }
            except Exception as e:
                logger.error(f"Bulk analysis failed for business {business_id}: {e}")
                results[business_id] = {
                    "success": False,
                    "error": str(e),
                    "alert_count": 0,
                    "risk_score": 0.0
                }
        
        # Calculate summary statistics
        successful_analyses = sum(1 for r in results.values() if r.get("success", False))
        total_alerts = sum(r.get("alert_count", 0) for r in results.values())
        avg_risk_score = sum(r.get("risk_score", 0) for r in results.values()) / len(results)
        
        logger.info(f"Bulk fraud analysis completed: {successful_analyses}/{len(request.business_ids)} successful, "
                   f"{total_alerts} total alerts")
        
        return {
            "success": True,
            "message": f"Bulk analysis completed for {len(request.business_ids)} businesses",
            "summary": {
                "successful_analyses": successful_analyses,
                "failed_analyses": len(request.business_ids) - successful_analyses,
                "total_alerts": total_alerts,
                "average_risk_score": round(avg_risk_score, 2)
            },
            "results": results,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk fraud analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Bulk fraud analysis failed: {str(e)}"
        )


@router.get("/stats/{business_id}")
async def get_fraud_statistics(
    business_id: str = Path(..., description="Business ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics"),
    token: str = Depends(security)
):
    try:
        # Verify authentication token
        user_info = await verify_token(token.credentials)
        logger.info(f"Fetching fraud statistics for business: {business_id} by user: {user_info.get('user_id')}")
        
        # Initialize fraud detector
        detector = FraudDetector()
        
        # Iske baad khana khaaaa
        stats = await detector.get_fraud_statistics(business_id, days)
        
        logger.info(f"Retrieved fraud statistics for business {business_id} over {days} days")
        
        return {
            "success": True,
            "business_id": business_id,
            "period_days": days,
            "statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fraud statistics for business {business_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch fraud statistics: {str(e)}"
        )



def _generate_alert_summary(alerts: List[FraudAlert]) -> FraudAlertSummary:
    """Generate summary statistics for fraud alerts"""
    if not alerts:
        return FraudAlertSummary(
            total_alerts=0,
            high_risk_alerts=0,
            medium_risk_alerts=0,
            low_risk_alerts=0,
            alert_types={},
            latest_alert_time=None
        )
    
    high_risk = len([a for a in alerts if a.confidence_score >= 0.8])
    medium_risk = len([a for a in alerts if 0.5 <= a.confidence_score < 0.8])
    low_risk = len([a for a in alerts if a.confidence_score < 0.5])
    
    alert_types = {}
    for alert in alerts:
        alert_type = alert.type.value
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    latest_alert_time = max(alert.detected_at for alert in alerts) if alerts else None
    
    return FraudAlertSummary(
        total_alerts=len(alerts),
        high_risk_alerts=high_risk,
        medium_risk_alerts=medium_risk,
        low_risk_alerts=low_risk,
        alert_types=alert_types,
        latest_alert_time=latest_alert_time
    )


def _generate_recommendations(alerts: List[FraudAlert], risk_score: float) -> List[str]:
    recommendations = []
    
    if risk_score >= 0.8:
        recommendations.append("High fraud risk detected. Review all flagged transactions immediately.")
    elif risk_score >= 0.5:
        recommendations.append("Medium fraud risk detected. Schedule review of flagged transactions.")
    
    alert_types = [alert.type for alert in alerts]
    
    if FraudType.DUPLICATE_INVOICE in alert_types:
        recommendations.append("Implement invoice number validation to prevent duplicate entries.")
    
    if FraudType.PAYMENT_MISMATCH in alert_types:
        recommendations.append("Reconcile payment records with invoice amounts regularly.")
    
    if FraudType.SUSPICIOUS_PATTERN in alert_types:
        recommendations.append("Review transaction patterns and consider implementing additional controls.")
    
    if FraudType.SUPPLIER_DUPLICATE in alert_types:
        recommendations.append("Verify supplier invoices against purchase orders before payment.")
    
    if len(alerts) > 10:
        recommendations.append("Consider implementing automated fraud prevention controls.")
    
    return recommendations


def _calculate_next_analysis_time(risk_score: float) -> datetime:
    base_time = datetime.utcnow()
    
    if risk_score >= 0.8:
        
        return base_time + timedelta(days=1)
    elif risk_score >= 0.5:
        return base_time + timedelta(days=3)
    else:
       
        return base_time + timedelta(days=7)