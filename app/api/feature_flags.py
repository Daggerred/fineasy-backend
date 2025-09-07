"""
Feature flags API endpoints for managing AI capabilities rollout.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.utils.feature_flags import get_feature_flag_manager, FeatureFlagStatus, ABTestVariant
from app.utils.auth import get_current_user, require_admin
from app.models.responses import APIResponse

router = APIRouter(prefix="/api/v1/feature-flags", tags=["feature-flags"])


class FeatureFlagResponse(BaseModel):
    """Feature flag response model."""
    name: str
    status: str
    description: str
    rollout_percentage: float
    ab_test_enabled: bool
    ab_test_variants: Dict[str, float]
    target_users: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class FeatureFlagUpdateRequest(BaseModel):
    """Feature flag update request model."""
    status: Optional[str] = None
    rollout_percentage: Optional[float] = Field(None, ge=0, le=100)
    ab_test_enabled: Optional[bool] = None
    ab_test_variants: Optional[Dict[str, float]] = None
    target_users: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeatureCheckRequest(BaseModel):
    """Feature check request model."""
    feature_name: str
    user_id: str
    business_id: Optional[str] = None


class FeatureCheckResponse(BaseModel):
    """Feature check response model."""
    feature_name: str
    enabled: bool
    variant: Optional[str] = None
    metadata: Dict[str, Any] = {}


class InteractionTrackingRequest(BaseModel):
    """Interaction tracking request model."""
    feature_name: str
    user_id: str
    interaction_type: str = "view"


class ConversionTrackingRequest(BaseModel):
    """Conversion tracking request model."""
    feature_name: str
    user_id: str
    conversion_value: float = 1.0


class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    feature_name: str
    period_days: int
    total_interactions: int
    total_conversions: int
    conversion_rate: float
    ab_test_results: List[Dict[str, Any]]
    generated_at: datetime


@router.get("/", response_model=List[FeatureFlagResponse])
async def list_feature_flags(
    current_user: dict = Depends(require_admin)
):
    """List all feature flags (admin only)."""
    try:
        # This would typically fetch from database
        # For now, return the default flags
        flags = [
            {
                "name": "fraud_detection",
                "status": "enabled",
                "description": "AI-powered fraud detection system",
                "rollout_percentage": 100.0,
                "ab_test_enabled": False,
                "ab_test_variants": {},
                "target_users": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {}
            },
            {
                "name": "smart_notifications",
                "status": "testing",
                "description": "Intelligent notification prioritization",
                "rollout_percentage": 30.0,
                "ab_test_enabled": True,
                "ab_test_variants": {"control": 40.0, "variant_a": 30.0, "variant_b": 30.0},
                "target_users": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {}
            }
        ]
        
        return [FeatureFlagResponse(**flag) for flag in flags]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feature flags: {str(e)}")


@router.get("/{feature_name}", response_model=FeatureFlagResponse)
async def get_feature_flag(
    feature_name: str,
    current_user: dict = Depends(require_admin)
):
    """Get specific feature flag (admin only)."""
    try:
        flag = await get_feature_flag_manager().get_feature_flag(feature_name)
        if not flag:
            raise HTTPException(status_code=404, detail="Feature flag not found")
        
        return FeatureFlagResponse(
            name=flag.name,
            status=flag.status.value,
            description=flag.description,
            rollout_percentage=flag.rollout_percentage,
            ab_test_enabled=flag.ab_test_enabled,
            ab_test_variants=flag.ab_test_variants,
            target_users=flag.target_users,
            created_at=flag.created_at,
            updated_at=flag.updated_at,
            metadata=flag.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feature flag: {str(e)}")


@router.put("/{feature_name}", response_model=APIResponse)
async def update_feature_flag(
    feature_name: str,
    request: FeatureFlagUpdateRequest,
    current_user: dict = Depends(require_admin)
):
    """Update feature flag configuration (admin only)."""
    try:
        updates = {}
        
        if request.status is not None:
            # Validate status
            try:
                FeatureFlagStatus(request.status)
                updates["status"] = request.status
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status value")
        
        if request.rollout_percentage is not None:
            updates["rollout_percentage"] = request.rollout_percentage
        
        if request.ab_test_enabled is not None:
            updates["ab_test_enabled"] = request.ab_test_enabled
        
        if request.ab_test_variants is not None:
            # Validate variant weights sum to 100
            total_weight = sum(request.ab_test_variants.values())
            if abs(total_weight - 100.0) > 0.01:
                raise HTTPException(status_code=400, detail="A/B test variant weights must sum to 100")
            updates["ab_test_variants"] = request.ab_test_variants
        
        if request.target_users is not None:
            updates["target_users"] = request.target_users
        
        if request.metadata is not None:
            updates["metadata"] = request.metadata
        
        success = await get_feature_flag_manager().update_feature_flag(feature_name, updates)
        
        if success:
            return APIResponse(success=True, message="Feature flag updated successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to update feature flag")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating feature flag: {str(e)}")


@router.post("/check", response_model=FeatureCheckResponse)
async def check_feature_enabled(
    request: FeatureCheckRequest,
    current_user: dict = Depends(get_current_user)
):
    """Check if a feature is enabled for a specific user."""
    try:
        enabled = await get_feature_flag_manager().is_feature_enabled(
            request.feature_name, 
            request.user_id, 
            request.business_id
        )
        
        variant = None
        if enabled:
            ab_variant = await get_feature_flag_manager().get_ab_test_variant(
                request.feature_name, 
                request.user_id
            )
            variant = ab_variant.value
        
        return FeatureCheckResponse(
            feature_name=request.feature_name,
            enabled=enabled,
            variant=variant
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking feature: {str(e)}")


@router.post("/track/interaction", response_model=APIResponse)
async def track_interaction(
    request: InteractionTrackingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Track user interaction with a feature."""
    try:
        await get_feature_flag_manager().track_interaction(
            request.feature_name,
            request.user_id,
            request.interaction_type
        )
        
        return APIResponse(success=True, message="Interaction tracked successfully")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking interaction: {str(e)}")


@router.post("/track/conversion", response_model=APIResponse)
async def track_conversion(
    request: ConversionTrackingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Track conversion event for A/B testing."""
    try:
        await feature_flag_manager.track_conversion(
            request.feature_name,
            request.user_id,
            request.conversion_value
        )
        
        return APIResponse(success=True, message="Conversion tracked successfully")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking conversion: {str(e)}")


@router.get("/{feature_name}/analytics", response_model=AnalyticsResponse)
async def get_feature_analytics(
    feature_name: str,
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(require_admin)
):
    """Get feature usage analytics (admin only)."""
    try:
        analytics = await feature_flag_manager.get_feature_analytics(feature_name, days)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Analytics not found")
        
        return AnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")


@router.post("/{feature_name}/emergency-disable", response_model=APIResponse)
async def emergency_disable_feature(
    feature_name: str,
    reason: str,
    current_user: dict = Depends(require_admin)
):
    """Emergency disable a feature flag (admin only)."""
    try:
        success = await feature_flag_manager.emergency_disable_feature(feature_name, reason)
        
        if success:
            return APIResponse(success=True, message=f"Feature {feature_name} emergency disabled")
        else:
            raise HTTPException(status_code=500, detail="Failed to disable feature")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error disabling feature: {str(e)}")


@router.get("/{feature_name}/performance", response_model=Dict[str, Any])
async def get_feature_performance(
    feature_name: str,
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(require_admin)
):
    """Get feature performance metrics (admin only)."""
    try:
        # This would fetch performance data from the database
        # For now, return mock data
        performance_data = {
            "feature_name": feature_name,
            "period_days": days,
            "metrics": {
                "response_time_ms": 150.5,
                "error_rate": 0.02,
                "success_rate": 0.98,
                "throughput_per_minute": 45.2
            },
            "trends": {
                "response_time_trend": "stable",
                "error_rate_trend": "decreasing",
                "usage_trend": "increasing"
            },
            "alerts": [],
            "generated_at": datetime.utcnow()
        }
        
        return performance_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching performance data: {str(e)}")