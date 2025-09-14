import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer

from ..services.ml_engine import ml_engine, TrainingConfig, ModelType
from ..models.responses import (
    ModelTrainingRequest, ModelTrainingResponse, ModelDeploymentResponse,
    ModelPerformanceResponse, FeedbackResponse, MLModelMetadata,
    ModelPerformanceMetrics, FeedbackData
)
from ..utils.auth import verify_token, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ml", tags=["ML Engine"])
security = HTTPBearer()


@router.post("/generate-invoice")
async def generate_invoice_from_text(request: dict):
    try:
        from ..services.gemini_service import gemini_service
        
        text = request.get("text", "")
        business_id = request.get("business_id", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        business_context = {}
        if business_id:
            from ..database import DatabaseManager
            db = DatabaseManager()
            business_context = {
                "customers": await db.get_customers(business_id),
                "products": await db.get_products(business_id)
            }
        result = await gemini_service.generate_invoice_from_text(text, business_context)
        
        return result
        
    except Exception as e:
        logger.error(f"Invoice generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse-cv")
async def parse_cv_content(request: dict):
    try:
        from ..services.gemini_service import gemini_service
        
        cv_text = request.get("cv_text", "")
        business_id = request.get("business_id", "")
        
        if not cv_text:
            raise HTTPException(status_code=400, detail="CV text is required")
        result = await gemini_service.parse_cv_content(cv_text)
        
        return result
        
    except Exception as e:
        logger.error(f"CV parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def check_ml_health():
    try:
        from ..services.gemini_service import gemini_service
        
        health_status = await gemini_service.check_service_health()
        
        return {
            "status": "healthy",
            "gemini_service": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        end_time = datetime.utcnow()
        
        training_duration = (end_time - start_time).total_seconds()
        
        # Get performance metrics
        performance_metrics = ModelPerformanceMetrics(**metadata.accuracy_metrics)
        
        return ModelTrainingResponse(
            success=True,
            message=f"Model {metadata.model_name} trained successfully",
            model_metadata=metadata,
            training_duration_seconds=training_duration,
            performance_metrics=performance_metrics
        )
        
    except ValueError as e:
        logger.error(f"Training validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/deploy/{model_name}/{model_version}", response_model=ModelDeploymentResponse)
async def deploy_model(
    model_name: str,
    model_version: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Deploying model: {model_name} v{model_version}")
        
        success = await ml_engine.deploy_model(model_name, model_version)
        
        if success:
            return ModelDeploymentResponse(
                success=True,
                message=f"Model {model_name} v{model_version} deployed successfully",
                model_name=model_name,
                model_version=model_version,
                deployment_status="deployed"
            )
        else:
            raise HTTPException(status_code=500, detail="Deployment failed")
            
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@router.get("/models/{model_name}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Getting performance metrics for model: {model_name}")
        
        metrics = await ml_engine.monitor_model_performance(model_name)
        trend = "stable"
        if metrics.accuracy > 0.8:
            trend = "good"
        elif metrics.accuracy < 0.6:
            trend = "degrading"
        
        recommendations = []
        if metrics.accuracy < 0.7:
            recommendations.append("Consider retraining the model with more recent data")
        if metrics.sample_size < 100:
            recommendations.append("Collect more feedback data for better performance evaluation")
        
        return ModelPerformanceResponse(
            success=True,
            message=f"Performance metrics retrieved for {model_name}",
            model_name=model_name,
            current_metrics=metrics,
            performance_trend=trend,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(
    model_name: str,
    prediction_id: str,
    actual_outcome: bool,
    user_feedback: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Recording feedback for model: {model_name}, prediction: {prediction_id}")
        
        success = await ml_engine.record_feedback(
            model_name=model_name,
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            user_feedback=user_feedback
        )
        
        if success:
     
            retraining_triggered = await ml_engine.retrain_model_if_needed(model_name)
            
            return FeedbackResponse(
                success=True,
                message="Feedback recorded successfully",
                feedback_id=f"{model_name}_{prediction_id}_{datetime.utcnow().timestamp()}",
                model_name=model_name,
                retraining_triggered=retraining_triggered
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
            
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/models/{model_name}/metadata", response_model=MLModelMetadata)
async def get_model_metadata(
    model_name: str,
    model_version: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Getting metadata for model: {model_name}")
        
        if model_version:
            metadata = await ml_engine._get_model_metadata(model_name, model_version)
        else:
            metadata = await ml_engine._get_active_model_metadata(model_name)
        
        if metadata:
            return metadata
        else:
            raise HTTPException(status_code=404, detail="Model not found")
            
    except Exception as e:
        logger.error(f"Failed to get model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@router.post("/models/{model_name}/retrain")
async def trigger_retraining(
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Manually trigger model retraining"""
    try:
        logger.info(f"Manually triggering retraining for model: {model_name}")

        background_tasks.add_task(ml_engine.retrain_model_if_needed, model_name)
        
        return {
            "success": True,
            "message": f"Retraining triggered for {model_name}",
            "model_name": model_name,
            "triggered_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


@router.get("/models", response_model=List[MLModelMetadata])
async def list_models(
    business_id: Optional[str] = None,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info("Listing models")
        
        # Build query
        supabase = ml_engine._get_supabase()
        if not supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        query = supabase.table("ml_models").select("*")
        
        if business_id:
            query = query.eq("business_id", business_id)
        if model_type:
            query = query.eq("model_type", model_type)
        if status:
            query = query.eq("status", status)
        
        response = await query.order("trained_at", desc=True).execute()
        
        models = []
        for data in response.data or []:
            models.append(MLModelMetadata(**data))
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.delete("/models/{model_name}/{model_version}")
async def delete_model(
    model_name: str,
    model_version: str,
    current_user: dict = Depends(get_current_user)
):
  
    try:
        logger.info(f"Deleting model: {model_name} v{model_version}")
   
        metadata = await ml_engine._get_model_metadata(model_name, model_version)
        if not metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        
    
        if metadata.status == "deployed":
            raise HTTPException(status_code=400, detail="Cannot delete deployed model")
        
      
        if metadata.model_path:
            model_path = Path(metadata.model_path)
            if model_path.exists():
                model_path.unlink()
            
            
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
        
        supabase = ml_engine._get_supabase()
        if not supabase:
            raise HTTPException(status_code=503, detail="Database not available")
        await supabase.table("ml_models").delete().eq(
            "model_name", model_name
        ).eq("model_version", model_version).execute()
        return {
            "success": True,
            "message": f"Model {model_name} v{model_version} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
@router.get("/health")
async def health_check():
    """Health check endpoint for ML engine"""
    try:
        active_models_count = len(ml_engine.active_models)
        return {
            "status": "healthy",
            "active_models": active_models_count,
            "timestamp": datetime.utcnow(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="ML Engine unhealthy")