"""
Business Insights API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from typing import Optional, List
import logging
import json
from datetime import datetime, timedelta


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from ..models.responses import (
    BusinessInsightsResponse, CashFlowPrediction, 
    CustomerAnalysis, WorkingCapitalAnalysis
)
from ..services.predictive_analytics import PredictiveAnalyzer
from ..utils.cache import cache
from ..utils.auth import verify_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/insights", tags=["insights"])
security = HTTPBearer()


async def background_generate_insights(business_id: str):
    """Background task to generate and cache insights"""
    try:
        analyzer = PredictiveAnalyzer()
        result = await analyzer.generate_insights(business_id)
        
        # Cache the result
        cache_key = f"insights:{business_id}"
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=86400)  # 24 hours in seconds
        
        logger.info(f"Background insights generation completed for business: {business_id}")
    except Exception as e:
        logger.error(f"Background insights generation failed: {e}")


@router.get("/{business_id}", response_model=BusinessInsightsResponse)
async def get_business_insights(
    business_id: str,
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Force refresh insights"),
    token: str = Depends(security)
):
    """Get business insights for a business with caching"""
    try:
        # Verify authentication (bypass for development)
        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
            # For development, allow access with a warning
            logger.warning(f"Authentication bypassed for development - business_id: {business_id}")
            user_id = "dev_user"
        
        cache_key = f"insights:{business_id}"
        
        # Check cache first unless force refresh is requested
        if not force_refresh:
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached insights for business: {business_id}")
                try:
                    # Debug: check what type of data we got from cache
                    logger.debug(f"Cached result type: {type(cached_result)}")
                    if hasattr(cached_result, '__name__'):
                        logger.debug(f"Cached result name: {cached_result.__name__}")
                    
                    # Ensure cached_result is a string
                    if not isinstance(cached_result, str):
                        logger.warning(f"Cached result is not a string, got {type(cached_result)}, clearing cache")
                        await cache.delete(cache_key)
                    else:
                        return BusinessInsightsResponse(**json.loads(cached_result))
                except Exception as e:
                    logger.error(f"Error parsing cached result: {e}, clearing cache")
                    await cache.delete(cache_key)
        
        # Generate insights
        analyzer = PredictiveAnalyzer()
        result = await analyzer.generate_insights(business_id)
        
        # Cache the result
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=86400)  # 24 hours in seconds
        
        # Schedule background refresh for next time
        background_tasks.add_task(background_generate_insights, business_id)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Business insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=BusinessInsightsResponse)
async def generate_insights(
    business_id: str,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Generate new business insights and invalidate cache"""
    try:
        # Verify authentication (bypass for development)
        user_id = await verify_token(token.credentials if hasattr(token, 'credentials') else str(token))
        if not user_id:
            # For development, allow access with a warning
            logger.warning(f"Authentication bypassed for development - business_id: {business_id}")
            user_id = "dev_user"
        
        # Clear existing cache
        cache_key = f"insights:{business_id}"
        await cache.delete(cache_key)
        
        # Generate fresh insights
        analyzer = PredictiveAnalyzer()
        result = await analyzer.generate_insights(business_id)
        
        # Cache the new result
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=86400)  # 24 hours in seconds
        
        # Schedule background refresh
        background_tasks.add_task(background_generate_insights, business_id)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/cash-flow", response_model=CashFlowPrediction)
async def get_cash_flow_prediction(
    business_id: str,
    months: int = Query(3, ge=1, le=12, description="Number of months to predict"),
    token: str = Depends(security)
):
    """Get cash flow prediction with caching"""
    try:
        # Verify authentication
        user_id = await verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        cache_key = f"cash_flow:{business_id}:{months}"
        
        # Check cache first
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached cash flow prediction for business: {business_id}")
            return CashFlowPrediction(**json.loads(cached_result))
        
        # Generate prediction
        analyzer = PredictiveAnalyzer()
        result = await analyzer.predict_cash_flow(business_id, months)
        
        # Cache the result for 6 hours (shorter TTL for predictions)
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=21600)  # 6 hours in seconds
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cash flow prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/customer-analysis", response_model=CustomerAnalysis)
async def get_customer_analysis(
    business_id: str,
    token: str = Depends(security)
):
    """Get customer revenue analysis with caching"""
    try:
        # Verify authentication
        user_id = await verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        cache_key = f"customer_analysis:{business_id}"
        
        # Check cache first
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached customer analysis for business: {business_id}")
            return CustomerAnalysis(**json.loads(cached_result))
        
        # Generate analysis
        analyzer = PredictiveAnalyzer()
        result = await analyzer.analyze_customer_revenue(business_id)
        
        # Cache the result for 12 hours
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=43200)  # 12 hours in seconds
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Customer analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/working-capital", response_model=WorkingCapitalAnalysis)
async def get_working_capital_analysis(
    business_id: str,
    token: str = Depends(security)
):
    """Get working capital analysis with caching"""
    try:
        # Verify authentication
        user_id = await verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        cache_key = f"working_capital:{business_id}"
        
        # Check cache first
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached working capital analysis for business: {business_id}")
            return WorkingCapitalAnalysis(**json.loads(cached_result))
        
        # Generate analysis
        analyzer = PredictiveAnalyzer()
        result = await analyzer.calculate_working_capital_trend(business_id)
        
        # Cache the result for 8 hours
        await cache.set(cache_key, json.dumps(result.dict(), cls=DateTimeEncoder), ex=28800)  # 8 hours in seconds
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Working capital analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{business_id}/cache")
async def clear_insights_cache(
    business_id: str,
    token: str = Depends(security)
):
    """Clear all cached insights for a business"""
    try:
        # Verify authentication
        user_id = await verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Clear all insight-related cache entries
        patterns = [
            f"insights:{business_id}",
            f"cash_flow:{business_id}:*",
            f"customer_analysis:{business_id}",
            f"working_capital:{business_id}"
        ]
        
        cleared_count = 0
        for pattern in patterns:
            cleared_count += await cache.clear_pattern(pattern)
        
        logger.info(f"Cleared {cleared_count} cache entries for business: {business_id}")
        
        return {
            "success": True,
            "message": f"Cleared {cleared_count} cache entries",
            "business_id": business_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate_insights(
    business_ids: List[str],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Generate insights for multiple businesses in background"""
    try:
        # Verify authentication
        user_id = await verify_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Limit batch size
        if len(business_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 businesses per batch")
        
        # Schedule background tasks for each business
        for business_id in business_ids:
            background_tasks.add_task(background_generate_insights, business_id)
        
        logger.info(f"Scheduled background insight generation for {len(business_ids)} businesses")
        
        return {
            "success": True,
            "message": f"Scheduled insight generation for {len(business_ids)} businesses",
            "business_ids": business_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch insight generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))