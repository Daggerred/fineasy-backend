"""
Smart Notifications API endpoints

Provides REST API for managing intelligent notifications including:
- Sending notifications with smart prioritization
- Managing user preferences
- Getting notification analytics
- Acknowledging notifications
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..services.smart_notifications import (
    SmartNotificationService,
    NotificationPreference,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from ..models.responses import BaseResponse
from ..utils.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

# Pydantic models for API requests/responses
from pydantic import BaseModel

class SendNotificationRequest(BaseModel):
    business_id: str
    notification_type: str
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    priority: Optional[str] = "medium"
    consolidation_key: Optional[str] = None

class NotificationPreferenceRequest(BaseModel):
    channel: str
    notification_type: str
    enabled: bool
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    frequency_limit: Optional[int] = None

class UpdatePreferencesRequest(BaseModel):
    preferences: List[NotificationPreferenceRequest]

class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    type: str
    priority: str
    data: Dict[str, Any]
    created_at: str
    read: bool

class NotificationAnalyticsResponse(BaseModel):
    total_notifications_sent: int
    total_notifications_acknowledged: int
    acknowledgment_rate: float
    average_response_time_minutes: float
    most_effective_channels: List[str]
    optimal_send_times: List[str]
    notification_type_preferences: Dict[str, Any]

class SendNotificationResponse(BaseResponse):
    notification_id: Optional[str] = None

# Lazy initialization of service
notification_service = None

def get_notification_service():
    global notification_service
    if notification_service is None:
        notification_service = SmartNotificationService()
    return notification_service

@router.post("/send", response_model=SendNotificationResponse)
async def send_notification(
    request: SendNotificationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Send a smart notification with intelligent prioritization and scheduling.
    """
    try:
        # Validate notification type and priority
        try:
            notification_type = NotificationType(request.notification_type)
            priority = NotificationPriority(request.priority)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid notification type or priority: {str(e)}")
        
        # Send notification
        success = await get_notification_service().send_notification(
            user_id=current_user['id'],
            business_id=request.business_id,
            notification_type=notification_type,
            title=request.title,
            message=request.message,
            data=request.data,
            priority=priority,
            consolidation_key=request.consolidation_key
        )
        
        if success:
            return SendNotificationResponse(
                success=True,
                message="Notification sent successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[NotificationResponse])
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    unread_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's in-app notifications.
    """
    try:
        from ..database import get_supabase
        supabase = get_supabase()
        
        query = supabase.table('in_app_notifications').select('*').eq('user_id', current_user['id'])
        
        if unread_only:
            query = query.eq('read', False)
        
        result = query.order('created_at', desc=True).range(offset, offset + limit - 1).execute()
        
        notifications = []
        for row in result.data:
            notifications.append(NotificationResponse(
                id=row['notification_id'],
                title=row['title'],
                message=row['message'],
                type=row.get('type', 'general'),
                priority=row['priority'],
                data=row['data'],
                created_at=row['created_at'],
                read=row['read']
            ))
        
        return notifications
        
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notification_id}/acknowledge")
async def acknowledge_notification(
    notification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Acknowledge a notification (mark as read/handled).
    """
    try:
        success = await get_notification_service().acknowledge_notification(
            notification_id=notification_id,
            user_id=current_user['id']
        )
        
        if success:
            # Also mark in-app notification as read
            from ..database import get_supabase
            supabase = get_supabase()
            
            supabase.table('in_app_notifications').update({
                'read': True,
                'read_at': datetime.utcnow().isoformat()
            }).eq('notification_id', notification_id).eq('user_id', current_user['id']).execute()
            
            return {"success": True, "message": "Notification acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Notification not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/preferences")
async def get_notification_preferences(
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's notification preferences.
    """
    try:
        preferences = await get_notification_service()._get_user_preferences(current_user['id'])
        
        preference_data = []
        for pref in preferences:
            preference_data.append({
                'channel': pref.channel.value,
                'notification_type': pref.notification_type.value,
                'enabled': pref.enabled,
                'quiet_hours_start': pref.quiet_hours_start,
                'quiet_hours_end': pref.quiet_hours_end,
                'frequency_limit': pref.frequency_limit
            })
        
        return {"preferences": preference_data}
        
    except Exception as e:
        logger.error(f"Error getting notification preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/preferences")
async def update_notification_preferences(
    request: UpdatePreferencesRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user's notification preferences.
    """
    try:
        # Convert request to NotificationPreference objects
        preferences = []
        for pref_req in request.preferences:
            try:
                channel = NotificationChannel(pref_req.channel)
                notification_type = NotificationType(pref_req.notification_type)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid channel or notification type: {str(e)}")
            
            preferences.append(NotificationPreference(
                user_id=current_user['id'],
                channel=channel,
                notification_type=notification_type,
                enabled=pref_req.enabled,
                quiet_hours_start=pref_req.quiet_hours_start,
                quiet_hours_end=pref_req.quiet_hours_end,
                frequency_limit=pref_req.frequency_limit
            ))
        
        success = await get_notification_service().update_user_preferences(
            user_id=current_user['id'],
            preferences=preferences
        )
        
        if success:
            return {"success": True, "message": "Preferences updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating notification preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analytics", response_model=NotificationAnalyticsResponse)
async def get_notification_analytics(
    current_user: dict = Depends(get_current_user)
):
    """
    Get notification analytics to help users optimize their preferences.
    """
    try:
        analytics = await get_notification_service().get_notification_analytics(current_user['id'])
        
        return NotificationAnalyticsResponse(
            total_notifications_sent=analytics.get('total_notifications_sent', 0),
            total_notifications_acknowledged=analytics.get('total_notifications_acknowledged', 0),
            acknowledgment_rate=analytics.get('acknowledgment_rate', 0.0),
            average_response_time_minutes=analytics.get('average_response_time_minutes', 0.0),
            most_effective_channels=analytics.get('most_effective_channels', []),
            optimal_send_times=analytics.get('optimal_send_times', []),
            notification_type_preferences=analytics.get('notification_type_preferences', {})
        )
        
    except Exception as e:
        logger.error(f"Error getting notification analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/test")
async def send_test_notification(
    current_user: dict = Depends(get_current_user)
):
    """
    Send a test notification for debugging purposes.
    """
    try:
        success = await get_notification_service().send_notification(
            user_id=current_user['id'],
            business_id="test_business",
            notification_type=NotificationType.SYSTEM_UPDATE,
            title="Test Notification",
            message="This is a test notification from the smart notification system.",
            data={"test": True},
            priority=NotificationPriority.LOW
        )
        
        if success:
            return {"success": True, "message": "Test notification sent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test notification")
            
    except Exception as e:
        logger.error(f"Error sending test notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a notification.
    """
    try:
        from ..database import get_supabase
        supabase = get_supabase()
        
        # Delete from in-app notifications
        result = supabase.table('in_app_notifications').delete().eq('notification_id', notification_id).eq('user_id', current_user['id']).execute()
        
        if result.data:
            return {"success": True, "message": "Notification deleted"}
        else:
            raise HTTPException(status_code=404, detail="Notification not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/mark-all-read")
async def mark_all_notifications_read(
    current_user: dict = Depends(get_current_user)
):
    """
    Mark all notifications as read for the current user.
    """
    try:
        from ..database import get_supabase
        supabase = get_supabase()
        
        supabase.table('in_app_notifications').update({
            'read': True,
            'read_at': datetime.utcnow().isoformat()
        }).eq('user_id', current_user['id']).eq('read', False).execute()
        
        return {"success": True, "message": "All notifications marked as read"}
        
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper endpoints for integration with other AI services

@router.post("/fraud-alert")
async def send_fraud_alert(
    business_id: str,
    alert_details: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Send a fraud alert notification (used by fraud detection service).
    """
    try:
        success = await get_notification_service().send_notification(
            user_id=current_user['id'],
            business_id=business_id,
            notification_type=NotificationType.FRAUD_ALERT,
            title="Fraud Alert",
            message=alert_details.get('message', 'Suspicious activity detected'),
            data=alert_details,
            priority=NotificationPriority.CRITICAL,
            consolidation_key=f"fraud_{business_id}"
        )
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error sending fraud alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/compliance-warning")
async def send_compliance_warning(
    business_id: str,
    warning_details: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Send a compliance warning notification (used by compliance service).
    """
    try:
        success = await get_notification_service().send_notification(
            user_id=current_user['id'],
            business_id=business_id,
            notification_type=NotificationType.COMPLIANCE_WARNING,
            title="Compliance Warning",
            message=warning_details.get('message', 'Compliance issue detected'),
            data=warning_details,
            priority=NotificationPriority.HIGH,
            consolidation_key=f"compliance_{business_id}"
        )
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error sending compliance warning: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/business-insight")
async def send_business_insight(
    business_id: str,
    insight_details: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Send a business insight notification (used by predictive analytics service).
    """
    try:
        success = await get_notification_service().send_notification(
            user_id=current_user['id'],
            business_id=business_id,
            notification_type=NotificationType.BUSINESS_INSIGHT,
            title="Business Insight",
            message=insight_details.get('message', 'New business insight available'),
            data=insight_details,
            priority=NotificationPriority.MEDIUM,
            consolidation_key=f"insights_{business_id}"
        )
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error sending business insight: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")