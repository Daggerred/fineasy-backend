"""
Compliance API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ..models.responses import ComplianceResponse, GSTValidationResult, TaxVerificationResult
from ..models.base import ComplianceIssue, ComplianceStatus, ComplianceSeverity
from ..services.compliance import ComplianceChecker
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])
security = HTTPBearer()


class ComplianceCheckRequest(BaseModel):
    invoice_id: str
    business_id: str


class GSTValidationRequest(BaseModel):
    
    gstin: str


class ComplianceIssueResolutionRequest(BaseModel):
    
    issue_id: str
    resolution_action: str
    resolution_notes: Optional[str] = None
    resolved_by: str


class ComplianceReminderRequest(BaseModel):
    
    business_id: str
    reminder_type: str = Field(..., description="Type of reminder (gst_filing, tax_payment, etc.)")
    due_date: datetime
    description: str
    priority: str = Field(default="medium", description="Priority level: low, medium, high")
    recurring: bool = Field(default=False, description="Whether this is a recurring reminder")
    recurring_interval_days: Optional[int] = Field(default=None, description="Days between recurring reminders")


class ComplianceTrackingResponse(BaseModel):
  
    business_id: str
    total_issues: int
    resolved_issues: int
    pending_issues: int
    critical_issues: int
    compliance_score: float
    last_updated: datetime
    issues_by_type: Dict[str, int]
    resolution_rate: float


class ComplianceReminderResponse(BaseModel):
    
    reminder_id: str
    business_id: str
    reminder_type: str
    due_date: datetime
    description: str
    priority: str
    status: str
    created_at: datetime
    next_reminder: Optional[datetime] = None


@router.post("/check", response_model=ComplianceResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        result = await checker.check_compliance(request.invoice_id)
        return result
    except Exception as e:
        logger.error(f"Compliance check error for invoice {request.invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


@router.post("/gst/validate", response_model=GSTValidationResult)
async def validate_gst(
    request: GSTValidationRequest,
    token: str = Depends(security)
):
    
    try:
        checker = ComplianceChecker()
        result = await checker.validate_gst_number(request.gstin)
        return result
    except Exception as e:
        logger.error(f"GST validation error for {request.gstin}: {e}")
        raise HTTPException(status_code=500, detail=f"GST validation failed: {str(e)}")


@router.post("/tax/verify", response_model=TaxVerificationResult)
async def verify_tax_calculations(
    invoice_id: str = Query(..., description="Invoice ID to verify tax calculations"),
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        result = await checker.verify_tax_calculations(invoice_id)
        return result
    except Exception as e:
        logger.error(f"Tax verification error for invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Tax verification failed: {str(e)}")


@router.get("/issues/{invoice_id}", response_model=List[ComplianceIssue])
async def get_compliance_issues(
    invoice_id: str,
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        issues = await checker.check_invoice_compliance(invoice_id)
        return issues
    except Exception as e:
        logger.error(f"Error getting compliance issues for invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance issues: {str(e)}")


@router.post("/issues/{issue_id}/resolve", response_model=Dict[str, Any])
async def resolve_compliance_issue(
    issue_id: str,
    request: ComplianceIssueResolutionRequest,
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        result = await checker.resolve_compliance_issue(
            issue_id=issue_id,
            resolution_action=request.resolution_action,
            resolution_notes=request.resolution_notes,
            resolved_by=request.resolved_by
        )
        return result
    except Exception as e:
        logger.error(f"Error resolving compliance issue {issue_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve issue: {str(e)}")


@router.get("/tracking/{business_id}", response_model=ComplianceTrackingResponse)
async def get_compliance_tracking(
    business_id: str,
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        tracking_data = await checker.get_compliance_tracking(business_id)
        return tracking_data
    except Exception as e:
        logger.error(f"Error getting compliance tracking for business {business_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance tracking: {str(e)}")


@router.post("/reminders", response_model=ComplianceReminderResponse)
async def create_compliance_reminder(
    request: ComplianceReminderRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
   
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        reminder = await checker.create_compliance_reminder(
            business_id=request.business_id,
            reminder_type=request.reminder_type,
            due_date=request.due_date,
            description=request.description,
            priority=request.priority,
            recurring=request.recurring,
            recurring_interval_days=request.recurring_interval_days
        )
        
       
        background_tasks.add_task(
            checker.schedule_reminder_notifications,
            reminder["reminder_id"]
        )
        
        return ComplianceReminderResponse(**reminder)
    except Exception as e:
        logger.error(f"Error creating compliance reminder: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create reminder: {str(e)}")


@router.get("/reminders/{business_id}", response_model=List[ComplianceReminderResponse])
async def get_compliance_reminders(
    business_id: str,
    status: Optional[str] = Query(None, description="Filter by status: active, completed, overdue"),
    reminder_type: Optional[str] = Query(None, description="Filter by reminder type"),
    token: str = Depends(security)
):
    
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        reminders = await checker.get_compliance_reminders(
            business_id=business_id,
            status=status,
            reminder_type=reminder_type
        )
        return [ComplianceReminderResponse(**reminder) for reminder in reminders]
    except Exception as e:
        logger.error(f"Error getting compliance reminders for business {business_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reminders: {str(e)}")


@router.put("/reminders/{reminder_id}/complete")
async def complete_compliance_reminder(
    reminder_id: str,
    completion_notes: Optional[str] = None,
    token: str = Depends(security)
):
    
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        result = await checker.complete_compliance_reminder(
            reminder_id=reminder_id,
            completion_notes=completion_notes
        )
        return result
    except Exception as e:
        logger.error(f"Error completing compliance reminder {reminder_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete reminder: {str(e)}")


@router.get("/explanations/{issue_type}")
async def get_plain_language_explanation(
    issue_type: str,
    context: Optional[Dict[str, Any]] = None,
    token: str = Depends(security)
):
    try:
        checker = ComplianceChecker()
        explanation = await checker.generate_plain_language_explanation(
            issue_type=issue_type,
            context=context or {}
        )
        return {
            "issue_type": issue_type,
            "explanation": explanation["explanation"],
            "suggested_actions": explanation["suggested_actions"],
            "resources": explanation.get("resources", []),
            "severity_explanation": explanation.get("severity_explanation", "")
        }
    except Exception as e:
        logger.error(f"Error generating explanation for {issue_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@router.get("/deadlines/{business_id}")
async def get_upcoming_deadlines(
    business_id: str,
    days_ahead: int = Query(30, description="Number of days to look ahead for deadlines"),
    token: str = Depends(security)
):
    
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        deadlines = await checker.get_upcoming_deadlines(
            business_id=business_id,
            days_ahead=days_ahead
        )
        return {
            "business_id": business_id,
            "deadlines": deadlines,
            "total_deadlines": len(deadlines),
            "critical_deadlines": len([d for d in deadlines if d.get("priority") == "high"]),
            "generated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting deadlines for business {business_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get deadlines: {str(e)}")


@router.post("/bulk-check")
async def bulk_compliance_check(
    business_id: str,
    invoice_ids: List[str],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    if not settings.COMPLIANCE_CHECKING_ENABLED:
        raise HTTPException(status_code=503, detail="Compliance checking is currently disabled")
    
    try:
        checker = ComplianceChecker()
        job_id = await checker.start_bulk_compliance_check(
            business_id=business_id,
            invoice_ids=invoice_ids
        )
        
        # Schedule background processing
        background_tasks.add_task(
            checker.process_bulk_compliance_check,
            job_id,
            business_id,
            invoice_ids
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "total_invoices": len(invoice_ids),
            "estimated_completion": datetime.utcnow() + timedelta(minutes=len(invoice_ids) * 2)
        }
    except Exception as e:
        logger.error(f"Error starting bulk compliance check: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start bulk check: {str(e)}")


@router.get("/bulk-check/{job_id}/status")
async def get_bulk_check_status(
    job_id: str,
    token: str = Depends(security)
):
  
    try:
        checker = ComplianceChecker()
        status = await checker.get_bulk_check_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting bulk check status for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/health")
async def compliance_health_check():
   
    return {
        "service": "compliance",
        "status": "healthy" if settings.COMPLIANCE_CHECKING_ENABLED else "disabled",
        "gst_api_configured": bool(settings.GST_API_KEY),
        "features": {
            "gst_validation": True,
            "tax_verification": True,
            "field_validation": True,
            "place_of_supply_validation": True,
            "issue_tracking": True,
            "automated_reminders": True,
            "plain_language_explanations": True,
            "bulk_processing": True
        }
    }