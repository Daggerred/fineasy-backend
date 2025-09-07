"""
Authentication utilities and middleware
"""
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Dict, Any
import logging
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoints
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            response = await call_next(request)
            return response
        
        # For now, just pass through - implement actual auth later
        response = await call_next(request)
        return response


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    try:
        # For now, return a mock user - implement actual JWT validation later
        return {
            "id": "mock_user_id",
            "email": "user@example.com",
            "business_id": "mock_business_id"
        }
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


async def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return user ID"""
    try:
        # For development, return a mock user ID
        # In production, implement proper JWT verification
        if token and len(token) > 10:  # Basic token validation
            return "mock_user_id"
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


async def verify_api_key(api_key: str) -> bool:
    """Verify API key"""
    # Mock implementation - replace with actual verification
    return api_key == "test_api_key"


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    # Use a mock secret for now
    encoded_jwt = jwt.encode(to_encode, "mock_secret", algorithm="HS256")
    return encoded_jwt