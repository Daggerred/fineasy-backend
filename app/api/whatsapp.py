"""
WhatsApp API endpoints for automated messaging
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Optional
import logging
import requests
import json
from datetime import datetime

from ..utils.auth import verify_token
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/whatsapp", tags=["whatsapp"])
security = HTTPBearer()


class WhatsAppMessage(BaseModel):
    """WhatsApp message request model"""
    phone_number: str = Field(..., description="Phone number with country code")
    message: str = Field(..., description="Message content")
    template_type: str = Field(..., description="Message template type")
    attachment_url: Optional[str] = Field(None, description="Optional attachment URL")


class WhatsAppResponse(BaseModel):
    """WhatsApp API response model"""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@router.post("/send", response_model=WhatsAppResponse)
async def send_whatsapp_message(
    message_data: WhatsAppMessage,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Send WhatsApp message"""
    try:
        # Verify authentication
        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Validate phone number
        if not _is_valid_phone_number(message_data.phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format")
        
        # Send message in background
        background_tasks.add_task(
            _send_whatsapp_message_async,
            message_data.phone_number,
            message_data.message,
            message_data.template_type,
            message_data.attachment_url,
            user_id
        )
        
        return WhatsAppResponse(
            success=True,
            message_id=f"msg_{datetime.utcnow().timestamp()}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WhatsApp message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_whatsapp_status():
    """Check WhatsApp service status"""
    try:
        # Check if WhatsApp API credentials are configured
        if not settings.WHATSAPP_API_TOKEN:
            return {
                "available": False,
                "error": "WhatsApp API not configured",
                "timestamp": datetime.utcnow()
            }
        
        # Test API connection (if needed)
        return {
            "available": True,
            "status": "operational",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"WhatsApp status check error: {e}")
        return {
            "available": False,
            "error": str(e),
            "timestamp": datetime.utcnow()
        }


async def _send_whatsapp_message_async(
    phone_number: str,
    message: str,
    template_type: str,
    attachment_url: Optional[str],
    user_id: str
):
    """Send WhatsApp message asynchronously"""
    try:
        # Choose WhatsApp provider based on configuration
        if settings.WHATSAPP_PROVIDER == "twilio":
            success = await _send_via_twilio(phone_number, message, attachment_url)
        elif settings.WHATSAPP_PROVIDER == "whatsapp_business":
            success = await _send_via_whatsapp_business(phone_number, message, attachment_url)
        else:
            # Default to webhook/custom provider
            success = await _send_via_webhook(phone_number, message, attachment_url)
        
        # Log the message attempt
        await _log_whatsapp_message(
            phone_number=phone_number,
            message=message,
            template_type=template_type,
            success=success,
            user_id=user_id
        )
        
        logger.info(f"WhatsApp message sent to {phone_number}: {success}")
        
    except Exception as e:
        logger.error(f"WhatsApp async send error: {e}")
        await _log_whatsapp_message(
            phone_number=phone_number,
            message=message,
            template_type=template_type,
            success=False,
            user_id=user_id,
            error=str(e)
        )


async def _send_via_twilio(phone_number: str, message: str, attachment_url: Optional[str]) -> bool:
    """Send message via Twilio WhatsApp API"""
    try:
        from twilio.rest import Client
        
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        message_data = {
            'from_': f'whatsapp:{settings.TWILIO_WHATSAPP_NUMBER}',
            'body': message,
            'to': f'whatsapp:+{phone_number}'
        }
        
        if attachment_url:
            message_data['media_url'] = [attachment_url]
        
        message = client.messages.create(**message_data)
        return message.sid is not None
        
    except Exception as e:
        logger.error(f"Twilio WhatsApp error: {e}")
        return False


async def _send_via_whatsapp_business(phone_number: str, message: str, attachment_url: Optional[str]) -> bool:
    """Send message via WhatsApp Business API"""
    try:
        url = f"https://graph.facebook.com/v17.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
        
        headers = {
            'Authorization': f'Bearer {settings.WHATSAPP_API_TOKEN}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {"body": message}
        }
        
        if attachment_url:
            payload["type"] = "document"
            payload["document"] = {
                "link": attachment_url,
                "caption": message
            }
        
        response = requests.post(url, headers=headers, json=payload)
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"WhatsApp Business API error: {e}")
        return False


async def _send_via_webhook(phone_number: str, message: str, attachment_url: Optional[str]) -> bool:
    """Send message via custom webhook"""
    try:
        if not settings.WHATSAPP_WEBHOOK_URL:
            logger.warning("WhatsApp webhook URL not configured")
            return False
        
        payload = {
            "phone_number": phone_number,
            "message": message,
            "attachment_url": attachment_url,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = requests.post(
            settings.WHATSAPP_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}")
        return False


async def _log_whatsapp_message(
    phone_number: str,
    message: str,
    template_type: str,
    success: bool,
    user_id: str,
    error: Optional[str] = None
):
    """Log WhatsApp message attempt to database"""
    try:
        from ..database import get_db_manager
        
        db = get_db_manager()
        
        log_data = {
            "phone_number": phone_number,
            "message": message,
            "template_type": template_type,
            "success": success,
            "user_id": user_id,
            "error": error,
            "sent_at": datetime.utcnow().isoformat()
        }
        
        await db.log_whatsapp_message(log_data)
        
    except Exception as e:
        logger.error(f"WhatsApp logging error: {e}")


def _is_valid_phone_number(phone_number: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters
    cleaned = ''.join(filter(str.isdigit, phone_number))
    
    # Check if it's a valid length (10-15 digits)
    return 10 <= len(cleaned) <= 15