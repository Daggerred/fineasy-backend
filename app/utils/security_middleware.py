"""
Security middleware for AI operations
"""
from fastapi import Request, HTTPException
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Security middleware for validating AI operations"""
    
    def __init__(self):
        self.rate_limits = {}
        self.blocked_ips = set()
    
    async def validate_request(self, request: Request, business_id: str, operation_type: str) -> Dict[str, Any]:
        """Validate security for AI operation requests"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                raise HTTPException(status_code=403, detail="IP address blocked")
            
            # Basic rate limiting (simplified)
            current_time = time.time()
            rate_key = f"{client_ip}:{operation_type}"
            
            if rate_key in self.rate_limits:
                last_request, count = self.rate_limits[rate_key]
                if current_time - last_request < 60:  # 1 minute window
                    if count > 10:  # Max 10 requests per minute
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                    self.rate_limits[rate_key] = (last_request, count + 1)
                else:
                    self.rate_limits[rate_key] = (current_time, 1)
            else:
                self.rate_limits[rate_key] = (current_time, 1)
            
            # Return security context
            return {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "security_score": 0.8,  # Mock security score
                "validated_at": datetime.utcnow().isoformat(),
                "business_id": business_id,
                "operation_type": operation_type
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            raise HTTPException(status_code=500, detail="Security validation failed")
    
    def block_ip(self, ip_address: str):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP address: {ip_address}")


# Global security middleware instance
security_middleware = SecurityMiddleware()