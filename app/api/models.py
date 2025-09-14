"""
AI Models API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from typing import List, Dict, Any
import logging
from datetime import datetime

from ..utils.auth import verify_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])
security = HTTPBearer()


@router.get("/performance")
async def get_models_performance(
    token: str = Depends(security)
):

    try:

        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
            # TO:DO: Bypass auth for development only baad mei nikal dena
            logger.warning("Authentication bypassed for development - models performance")
            user_id = "dev_user"
        
        models_performance = [
            {
                "model_name": "fraud_detection",
                "model_type": "classification",
                "accuracy_score": 0.85,
                "last_updated": datetime.utcnow().isoformat(),
                "total_predictions": 1250,
                "correct_predictions": 1062,
                "response_time_ms": 45.2
            },
            {
                "model_name": "cash_flow_prediction",
                "model_type": "regression",
                "accuracy_score": 0.78,
                "last_updated": datetime.utcnow().isoformat(),
                "total_predictions": 890,
                "correct_predictions": 694,
                "response_time_ms": 32.1
            },
            {
                "model_name": "customer_analysis",
                "model_type": "clustering",
                "accuracy_score": 0.92,
                "last_updated": datetime.utcnow().isoformat(),
                "total_predictions": 567,
                "correct_predictions": 521,
                "response_time_ms": 28.7
            }
        ]
        
        return {
            "success": True,
            "message": "Model performance metrics retrieved successfully",
            "models": models_performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Models performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_models_status(
    token: str = Depends(security)
):
    """Get status of all AI models"""
    try:
        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
            logger.warning("Authentication bypassed for development - models status")
            user_id = "dev_user"
        
        models_status = [
            {
                "model_name": "fraud_detection",
                "status": "active",
                "version": "1.2.0",
                "last_trained": datetime.utcnow().isoformat(),
                "health": "healthy"
            },
            {
                "model_name": "cash_flow_prediction",
                "status": "active",
                "version": "1.1.0",
                "last_trained": datetime.utcnow().isoformat(),
                "health": "healthy"
            },
            {
                "model_name": "customer_analysis",
                "status": "active",
                "version": "1.0.0",
                "last_trained": datetime.utcnow().isoformat(),
                "health": "healthy"
            }
        ]
        
        return {
            "success": True,
            "message": "Model status retrieved successfully",
            "models": models_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Models status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))