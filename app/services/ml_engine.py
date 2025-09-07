"""
AI Analytics Engine Core
Implements machine learning model training pipeline, versioning, and performance monitoring
"""
import asyncio
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from ..database import get_supabase
from ..config import settings
from ..models.responses import MLModelMetadata, ModelPerformanceMetrics, FeedbackData
from ..utils.cache import ml_model_cache, cache_performance_monitor
from ..utils.resource_manager import resource_manager


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models supported"""
    FRAUD_DETECTION = "fraud_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    PREDICTION = "prediction"


class ModelStatus(str, Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: ModelType
    training_data_query: str
    feature_columns: List[str]
    target_column: Optional[str] = None
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    min_samples: int = 100
    max_training_time_minutes: int = 30


class MLModelManager:
    """Manages ML model lifecycle including training, versioning, and deployment"""
    
    def __init__(self):
        self.supabase = None  # Will be initialized when needed
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(exist_ok=True)
        self.active_models: Dict[str, Any] = {}
        self.performance_cache: Dict[str, ModelPerformanceMetrics] = {}
        
    def _get_supabase(self):
        """Get Supabase client, initializing if needed"""
        if self.supabase is None:
            try:
                self.supabase = get_supabase()
            except RuntimeError:
                # Database not initialized yet, return None
                return None
        return self.supabase
        
    async def train_model(self, config: TrainingConfig, business_id: str) -> MLModelMetadata:
        """Train a new ML model with the given configuration"""
        logger.info(f"Starting model training for {config.model_type} - Business: {business_id}")
        
        try:
            # Load training data
            training_data = await self._load_training_data(config, business_id)
            if len(training_data) < config.min_samples:
                raise ValueError(f"Insufficient training data: {len(training_data)} < {config.min_samples}")
            
            # Prepare features and target
            X, y = self._prepare_features(training_data, config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=42, stratify=y if y is not None else None
            )
            
            # Create and train model
            model = self._create_model(config.model_type)
            trained_model = await self._train_model_async(model, X_train, y_train, config)
            
            # Evaluate model
            performance = await self._evaluate_model(trained_model, X_test, y_test, config)
            
            # Generate model metadata
            model_version = self._generate_model_version(config.model_type, business_id)
            data_hash = self._calculate_data_hash(training_data)
            
            metadata = MLModelMetadata(
                model_name=f"{config.model_type}_{business_id}",
                model_version=model_version,
                model_type=config.model_type,
                business_id=business_id,
                training_data_hash=data_hash,
                accuracy_metrics=performance.dict(),
                training_config=config.__dict__,
                trained_at=datetime.utcnow(),
                status=ModelStatus.VALIDATING
            )
            
            # Save model
            model_path = await self._save_model(trained_model, metadata)
            metadata.model_path = str(model_path)
            
            # Store metadata in database
            await self._store_model_metadata(metadata)
            
            logger.info(f"Model training completed: {metadata.model_name} v{metadata.model_version}")
            return metadata
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    async def deploy_model(self, model_name: str, model_version: str) -> bool:
        """Deploy a trained model to production with caching optimization"""
        try:
            # Load model metadata
            metadata = await self._get_model_metadata(model_name, model_version)
            if not metadata:
                raise ValueError(f"Model not found: {model_name} v{model_version}")
            
            # Load and validate model
            model = await self._load_model(metadata.model_path)
            if not model:
                raise ValueError(f"Failed to load model from {metadata.model_path}")
            
            # Deprecate old active model and invalidate cache
            await self._deprecate_active_model(model_name)
            await ml_model_cache.invalidate_model_cache(model_name)
            
            # Activate new model
            self.active_models[model_name] = model
            metadata.status = ModelStatus.DEPLOYED
            metadata.deployed_at = datetime.utcnow()
            
            # Cache the new model
            await ml_model_cache.cache_model(
                model_name, model_version, model, metadata.dict()
            )
            
            # Update database
            await self._update_model_status(metadata)
            
            # Track deployment
            await resource_manager.track_model_usage(
                model_name, model_version, "deploy"
            )
            
            logger.info(f"Model deployed successfully: {model_name} v{model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return False
    
    async def get_active_model(self, model_name: str) -> Optional[Any]:
        """Get the currently active model with caching optimization"""
        start_time = time.time()
        
        # First check cache
        cached_model = await ml_model_cache.get_cached_model(model_name, "latest")
        if cached_model:
            await cache_performance_monitor.record_cache_hit("ml_model")
            await resource_manager.track_model_usage(
                model_name, "latest", "cache_hit", 
                duration_ms=(time.time() - start_time) * 1000
            )
            return cached_model
        
        await cache_performance_monitor.record_cache_miss("ml_model")
        
        # Check in-memory cache
        if model_name in self.active_models:
            model = self.active_models[model_name]
            
            # Cache the model for future use
            metadata = await self._get_active_model_metadata(model_name)
            if metadata:
                await ml_model_cache.cache_model(
                    model_name, metadata.model_version, model, 
                    metadata.dict() if metadata else None
                )
            
            await resource_manager.track_model_usage(
                model_name, "latest", "memory_hit",
                duration_ms=(time.time() - start_time) * 1000
            )
            return model
        
        # Load from database
        metadata = await self._get_active_model_metadata(model_name)
        if metadata and metadata.model_path:
            model = await self._load_model(metadata.model_path)
            if model:
                self.active_models[model_name] = model
                
                # Cache the loaded model
                await ml_model_cache.cache_model(
                    model_name, metadata.model_version, model, metadata.dict()
                )
                
                await resource_manager.track_model_usage(
                    model_name, metadata.model_version, "load",
                    duration_ms=(time.time() - start_time) * 1000
                )
                return model
        
        return None
    
    async def record_feedback(self, model_name: str, prediction_id: str, 
                            actual_outcome: bool, user_feedback: Optional[str] = None) -> bool:
        """Record feedback for model predictions to improve accuracy"""
        try:
            feedback = FeedbackData(
                model_name=model_name,
                prediction_id=prediction_id,
                actual_outcome=actual_outcome,
                user_feedback=user_feedback,
                recorded_at=datetime.utcnow()
            )
            
            # Store feedback in database
            await self._store_feedback(feedback)
            
            # Check if retraining is needed
            await self._check_retraining_trigger(model_name)
            
            logger.info(f"Feedback recorded for {model_name}: {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {str(e)}")
            return False
    
    async def monitor_model_performance(self, model_name: str) -> ModelPerformanceMetrics:
        """Monitor and evaluate model performance over time"""
        try:
            # Get recent predictions and feedback
            recent_data = await self._get_recent_performance_data(model_name)
            
            if not recent_data:
                # Return cached performance if no recent data
                return self.performance_cache.get(model_name, ModelPerformanceMetrics())
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(recent_data)
            
            # Update cache
            self.performance_cache[model_name] = metrics
            
            # Store in database
            await self._store_performance_metrics(model_name, metrics)
            
            # Check for performance degradation
            await self._check_performance_degradation(model_name, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed for {model_name}: {str(e)}")
            return ModelPerformanceMetrics()
    
    async def retrain_model_if_needed(self, model_name: str) -> bool:
        """Check if model needs retraining and trigger if necessary"""
        try:
            # Get current performance
            current_performance = await self.monitor_model_performance(model_name)
            
            # Get model metadata
            metadata = await self._get_active_model_metadata(model_name)
            if not metadata:
                return False
            
            # Check retraining criteria
            needs_retraining = await self._evaluate_retraining_criteria(
                model_name, current_performance, metadata
            )
            
            if needs_retraining:
                logger.info(f"Triggering retraining for {model_name}")
                
                # Get original training config
                config = TrainingConfig(**metadata.training_config)
                
                # Retrain model
                new_metadata = await self.train_model(config, metadata.business_id)
                
                # Auto-deploy if performance improved
                if await self._should_auto_deploy(metadata, new_metadata):
                    await self.deploy_model(new_metadata.model_name, new_metadata.model_version)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Retraining check failed for {model_name}: {str(e)}")
            return False
    
    # Private helper methods
    
    async def _load_training_data(self, config: TrainingConfig, business_id: str) -> pd.DataFrame:
        """Load training data from database"""
        try:
            # Execute query with business_id parameter
            query = config.training_data_query.replace("{business_id}", business_id)
            
            supabase = self._get_supabase()
            if not supabase:
                raise RuntimeError("Database not available")
            response = await supabase.rpc("execute_query", {"query": query}).execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def _prepare_features(self, data: pd.DataFrame, config: TrainingConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features and target variables"""
        # Select feature columns
        X = data[config.feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Prepare target if supervised learning
        y = None
        if config.target_column and config.target_column in data.columns:
            y = data[config.target_column].values
        
        return X, y
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create ML model based on type"""
        if model_type == ModelType.FRAUD_DETECTION:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        elif model_type == ModelType.ANOMALY_DETECTION:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('detector', IsolationForest(contamination=0.1, random_state=42))
            ])
        else:
            # Default to anomaly detection
            return Pipeline([
                ('scaler', StandardScaler()),
                ('detector', IsolationForest(contamination=0.1, random_state=42))
            ])
    
    async def _train_model_async(self, model: Any, X_train: np.ndarray, 
                               y_train: Optional[np.ndarray], config: TrainingConfig) -> Any:
        """Train model asynchronously"""
        def train():
            if y_train is not None:
                # Supervised learning
                model.fit(X_train, y_train)
            else:
                # Unsupervised learning
                model.fit(X_train)
            return model
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        trained_model = await loop.run_in_executor(None, train)
        
        return trained_model
    
    async def _evaluate_model(self, model: Any, X_test: np.ndarray, 
                            y_test: Optional[np.ndarray], config: TrainingConfig) -> ModelPerformanceMetrics:
        """Evaluate model performance"""
        try:
            if y_test is not None:
                # Supervised learning evaluation
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_test, y_test, cv=min(config.cross_validation_folds, len(X_test)))
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                return ModelPerformanceMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    cross_val_mean=cv_mean,
                    cross_val_std=cv_std,
                    sample_size=len(X_test),
                    evaluated_at=datetime.utcnow()
                )
            else:
                # Unsupervised learning evaluation
                predictions = model.predict(X_test)
                anomaly_ratio = (predictions == -1).sum() / len(predictions)
                
                return ModelPerformanceMetrics(
                    anomaly_ratio=anomaly_ratio,
                    sample_size=len(X_test),
                    evaluated_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return ModelPerformanceMetrics()
    
    def _generate_model_version(self, model_type: ModelType, business_id: str) -> str:
        """Generate unique model version"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{model_type.value}_{business_id}_{timestamp}"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for versioning"""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    async def _save_model(self, model: Any, metadata: MLModelMetadata) -> Path:
        """Save trained model to disk"""
        model_filename = f"{metadata.model_name}_{metadata.model_version}.joblib"
        model_path = self.models_dir / model_filename
        
        # Save model using joblib
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.dict(), f, indent=2, default=str)
        
        return model_path
    
    async def _load_model(self, model_path: str) -> Optional[Any]:
        """Load model from disk"""
        try:
            path = Path(model_path)
            if path.exists():
                return joblib.load(path)
            return None
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return None
    
    async def _store_model_metadata(self, metadata: MLModelMetadata) -> bool:
        """Store model metadata in database"""
        try:
            data = {
                "model_name": metadata.model_name,
                "model_version": metadata.model_version,
                "model_type": metadata.model_type,
                "business_id": metadata.business_id,
                "training_data_hash": metadata.training_data_hash,
                "accuracy_metrics": metadata.accuracy_metrics,
                "training_config": metadata.training_config,
                "model_path": metadata.model_path,
                "status": metadata.status,
                "trained_at": metadata.trained_at.isoformat(),
                "deployed_at": metadata.deployed_at.isoformat() if metadata.deployed_at else None
            }
            
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("ml_models").insert(data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model metadata: {str(e)}")
            return False
    
    async def _get_model_metadata(self, model_name: str, model_version: str) -> Optional[MLModelMetadata]:
        """Get model metadata from database"""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return None
            response = await supabase.table("ml_models").select("*").eq(
                "model_name", model_name
            ).eq("model_version", model_version).execute()
            
            if response.data:
                data = response.data[0]
                return MLModelMetadata(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {str(e)}")
            return None
    
    async def _get_active_model_metadata(self, model_name: str) -> Optional[MLModelMetadata]:
        """Get active model metadata"""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return None
            response = await supabase.table("ml_models").select("*").eq(
                "model_name", model_name
            ).eq("status", ModelStatus.DEPLOYED).order("deployed_at", desc=True).limit(1).execute()
            
            if response.data:
                data = response.data[0]
                return MLModelMetadata(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get active model metadata: {str(e)}")
            return None
    
    async def _update_model_status(self, metadata: MLModelMetadata) -> bool:
        """Update model status in database"""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("ml_models").update({
                "status": metadata.status,
                "deployed_at": metadata.deployed_at.isoformat() if metadata.deployed_at else None
            }).eq("model_name", metadata.model_name).eq("model_version", metadata.model_version).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model status: {str(e)}")
            return False
    
    async def _deprecate_active_model(self, model_name: str) -> bool:
        """Deprecate currently active model"""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("ml_models").update({
                "status": ModelStatus.DEPRECATED
            }).eq("model_name", model_name).eq("status", ModelStatus.DEPLOYED).execute()
            
            # Remove from active models cache
            if model_name in self.active_models:
                del self.active_models[model_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deprecate active model: {str(e)}")
            return False
    
    async def _store_feedback(self, feedback: FeedbackData) -> bool:
        """Store feedback data in database"""
        try:
            data = {
                "model_name": feedback.model_name,
                "prediction_id": feedback.prediction_id,
                "actual_outcome": feedback.actual_outcome,
                "user_feedback": feedback.user_feedback,
                "recorded_at": feedback.recorded_at.isoformat()
            }
            
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("model_feedback").insert(data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            return False
    
    async def _get_recent_performance_data(self, model_name: str, days: int = 7) -> List[Dict]:
        """Get recent performance data for model"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            supabase = self._get_supabase()
            if not supabase:
                return []
            response = await supabase.table("model_feedback").select("*").eq(
                "model_name", model_name
            ).gte("recorded_at", cutoff_date.isoformat()).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent performance data: {str(e)}")
            return []
    
    def _calculate_performance_metrics(self, feedback_data: List[Dict]) -> ModelPerformanceMetrics:
        """Calculate performance metrics from feedback data"""
        if not feedback_data:
            return ModelPerformanceMetrics()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(feedback_data)
        
        # Calculate accuracy from feedback
        total_predictions = len(df)
        correct_predictions = df['actual_outcome'].sum() if 'actual_outcome' in df.columns else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return ModelPerformanceMetrics(
            accuracy=accuracy,
            sample_size=total_predictions,
            evaluated_at=datetime.utcnow()
        )
    
    async def _store_performance_metrics(self, model_name: str, metrics: ModelPerformanceMetrics) -> bool:
        """Store performance metrics in database"""
        try:
            data = {
                "model_name": model_name,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "sample_size": metrics.sample_size,
                "evaluated_at": metrics.evaluated_at.isoformat()
            }
            
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("model_performance").insert(data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {str(e)}")
            return False
    
    async def _check_performance_degradation(self, model_name: str, current_metrics: ModelPerformanceMetrics) -> bool:
        """Check for performance degradation and alert if needed"""
        try:
            # Get historical performance
            supabase = self._get_supabase()
            if not supabase:
                return False
            response = await supabase.table("model_performance").select("*").eq(
                "model_name", model_name
            ).order("evaluated_at", desc=True).limit(10).execute()
            
            if not response.data or len(response.data) < 2:
                return False
            
            # Calculate average historical accuracy
            historical_accuracies = [item['accuracy'] for item in response.data[1:]]  # Exclude current
            avg_historical_accuracy = sum(historical_accuracies) / len(historical_accuracies)
            
            # Check for significant degradation (>10% drop)
            degradation_threshold = 0.1
            if current_metrics.accuracy < (avg_historical_accuracy - degradation_threshold):
                logger.warning(f"Performance degradation detected for {model_name}: "
                             f"Current: {current_metrics.accuracy:.3f}, "
                             f"Historical: {avg_historical_accuracy:.3f}")
                
                # Store alert
                await self._store_performance_alert(model_name, current_metrics, avg_historical_accuracy)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check performance degradation: {str(e)}")
            return False
    
    async def _store_performance_alert(self, model_name: str, current_metrics: ModelPerformanceMetrics, 
                                     historical_accuracy: float) -> bool:
        """Store performance degradation alert"""
        try:
            alert_data = {
                "model_name": model_name,
                "alert_type": "performance_degradation",
                "current_accuracy": current_metrics.accuracy,
                "historical_accuracy": historical_accuracy,
                "degradation_amount": historical_accuracy - current_metrics.accuracy,
                "created_at": datetime.utcnow().isoformat()
            }
            
            supabase = self._get_supabase()
            if not supabase:
                return False
            await supabase.table("model_alerts").insert(alert_data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store performance alert: {str(e)}")
            return False
    
    async def _check_retraining_trigger(self, model_name: str) -> bool:
        """Check if model should be retrained based on feedback volume"""
        try:
            # Count recent feedback
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            supabase = self._get_supabase()
            if not supabase:
                return False
            response = await supabase.table("model_feedback").select("*", count="exact").eq(
                "model_name", model_name
            ).gte("recorded_at", cutoff_date.isoformat()).execute()
            
            feedback_count = response.count or 0
            
            # Trigger retraining if we have enough new feedback (e.g., 50+ samples)
            retraining_threshold = 50
            if feedback_count >= retraining_threshold:
                logger.info(f"Retraining trigger activated for {model_name}: {feedback_count} new feedback samples")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check retraining trigger: {str(e)}")
            return False
    
    async def _evaluate_retraining_criteria(self, model_name: str, current_performance: ModelPerformanceMetrics, 
                                          metadata: MLModelMetadata) -> bool:
        """Evaluate if model needs retraining"""
        try:
            # Check multiple criteria
            criteria_met = 0
            
            # 1. Performance degradation
            if current_performance.accuracy < 0.7:  # Below acceptable threshold
                criteria_met += 1
            
            # 2. Model age (older than 30 days)
            if metadata.trained_at < datetime.utcnow() - timedelta(days=30):
                criteria_met += 1
            
            # 3. Sufficient new feedback
            if await self._check_retraining_trigger(model_name):
                criteria_met += 1
            
            # Retrain if 2 or more criteria are met
            return criteria_met >= 2
            
        except Exception as e:
            logger.error(f"Failed to evaluate retraining criteria: {str(e)}")
            return False
    
    async def _should_auto_deploy(self, old_metadata: MLModelMetadata, new_metadata: MLModelMetadata) -> bool:
        """Determine if new model should be auto-deployed"""
        try:
            old_accuracy = old_metadata.accuracy_metrics.get('accuracy', 0.0)
            new_accuracy = new_metadata.accuracy_metrics.get('accuracy', 0.0)
            
            # Auto-deploy if new model is significantly better (>5% improvement)
            improvement_threshold = 0.05
            return new_accuracy > (old_accuracy + improvement_threshold)
            
        except Exception as e:
            logger.error(f"Failed to evaluate auto-deployment: {str(e)}")
            return False

    async def get_model_health(self) -> Dict[str, Any]:
        """Get health status of ML models and engine"""
        try:
            health_status = {
                "status": "healthy",
                "loaded_models": len(self.active_models),
                "models": {},
                "performance_cache_size": len(self.performance_cache),
                "models_directory_exists": self.models_dir.exists(),
                "last_check": datetime.utcnow().isoformat()
            }
            
            # Check each active model
            for model_name, model in self.active_models.items():
                try:
                    # Basic model health check
                    model_health = {
                        "status": "healthy",
                        "type": type(model).__name__,
                        "loaded_at": getattr(model, '_loaded_at', 'unknown'),
                        "memory_usage_mb": self._estimate_model_memory(model)
                    }
                    
                    # Check if model can make predictions
                    if hasattr(model, 'predict'):
                        # Create dummy data for prediction test
                        dummy_data = np.array([[0.5, 0.5, 0.5]])  # Simple test data
                        try:
                            _ = model.predict(dummy_data)
                            model_health["prediction_test"] = "passed"
                        except Exception as e:
                            model_health["prediction_test"] = f"failed: {str(e)}"
                            model_health["status"] = "degraded"
                    
                    health_status["models"][model_name] = model_health
                    
                except Exception as e:
                    health_status["models"][model_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            # Check models directory
            if not self.models_dir.exists():
                health_status["status"] = "degraded"
                health_status["models_directory_error"] = "Models directory does not exist"
            
            # Check database connection
            try:
                supabase = self._get_supabase()
                if supabase is None:
                    health_status["database_status"] = "not_initialized"
                else:
                    health_status["database_status"] = "connected"
            except Exception as e:
                health_status["database_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get model health: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "loaded_models": 0,
                "last_check": datetime.utcnow().isoformat()
            }
    
    def _estimate_model_memory(self, model) -> float:
        """Estimate memory usage of a model in MB"""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0


# Global ML engine instance
ml_engine = MLModelManager()