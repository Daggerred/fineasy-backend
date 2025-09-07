"""
Tests for ML Engine functionality
"""
import pytest
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from app.services.ml_engine import (
    MLModelManager, TrainingConfig, ModelType, ModelStatus,
    ml_engine
)
from app.models.responses import (
    MLModelMetadata, ModelPerformanceMetrics, FeedbackData
)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client"""
    mock = AsyncMock()
    mock.table.return_value = mock
    mock.select.return_value = mock
    mock.insert.return_value = mock
    mock.update.return_value = mock
    mock.delete.return_value = mock
    mock.eq.return_value = mock
    mock.gte.return_value = mock
    mock.order.return_value = mock
    mock.limit.return_value = mock
    mock.execute.return_value = AsyncMock(data=[], count=0)
    return mock


@pytest.fixture
def ml_manager(mock_supabase):
    """ML Model Manager instance with mocked dependencies"""
    manager = MLModelManager()
    manager.supabase = mock_supabase
    return manager


@pytest.fixture
def sample_training_config():
    """Sample training configuration"""
    return TrainingConfig(
        model_type=ModelType.FRAUD_DETECTION,
        training_data_query="SELECT * FROM transactions WHERE business_id = '{business_id}'",
        feature_columns=["amount", "frequency", "time_diff"],
        target_column="is_fraud",
        test_size=0.2,
        min_samples=100
    )


@pytest.fixture
def sample_training_data():
    """Sample training data"""
    np.random.seed(42)
    data = {
        'amount': np.random.uniform(10, 1000, 200),
        'frequency': np.random.randint(1, 10, 200),
        'time_diff': np.random.uniform(0, 24, 200),
        'is_fraud': np.random.choice([0, 1], 200, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)


class TestMLModelManager:
    """Test cases for ML Model Manager"""
    
    @pytest.mark.asyncio
    async def test_train_model_success(self, ml_manager, sample_training_config, sample_training_data):
        """Test successful model training"""
        business_id = "test-business-123"
        
        # Mock data loading
        with patch.object(ml_manager, '_load_training_data', return_value=sample_training_data):
            with patch.object(ml_manager, '_save_model', return_value=Path("test_model.joblib")):
                with patch.object(ml_manager, '_store_model_metadata', return_value=True):
                    
                    metadata = await ml_manager.train_model(sample_training_config, business_id)
                    
                    assert metadata is not None
                    assert metadata.model_name == f"{sample_training_config.model_type}_{business_id}"
                    assert metadata.business_id == business_id
                    assert metadata.model_type == sample_training_config.model_type
                    assert metadata.status == ModelStatus.VALIDATING
                    assert metadata.accuracy_metrics is not None
    
    @pytest.mark.asyncio
    async def test_train_model_insufficient_data(self, ml_manager, sample_training_config):
        """Test training with insufficient data"""
        business_id = "test-business-123"
        insufficient_data = pd.DataFrame({'amount': [1, 2], 'frequency': [1, 1], 'time_diff': [1, 2], 'is_fraud': [0, 1]})
        
        with patch.object(ml_manager, '_load_training_data', return_value=insufficient_data):
            with pytest.raises(ValueError, match="Insufficient training data"):
                await ml_manager.train_model(sample_training_config, business_id)
    
    @pytest.mark.asyncio
    async def test_deploy_model_success(self, ml_manager):
        """Test successful model deployment"""
        model_name = "fraud_detection_test-business-123"
        model_version = "v1.0"
        
        # Mock metadata retrieval
        metadata = MLModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=ModelType.FRAUD_DETECTION,
            business_id="test-business-123",
            training_data_hash="test-hash",
            model_path="test_model.joblib",
            status=ModelStatus.VALIDATING
        )
        
        with patch.object(ml_manager, '_get_model_metadata', return_value=metadata):
            with patch.object(ml_manager, '_load_model', return_value=Mock()):
                with patch.object(ml_manager, '_deprecate_active_model', return_value=True):
                    with patch.object(ml_manager, '_update_model_status', return_value=True):
                        
                        result = await ml_manager.deploy_model(model_name, model_version)
                        
                        assert result is True
                        assert model_name in ml_manager.active_models
    
    @pytest.mark.asyncio
    async def test_deploy_model_not_found(self, ml_manager):
        """Test deployment of non-existent model"""
        model_name = "non-existent-model"
        model_version = "v1.0"
        
        with patch.object(ml_manager, '_get_model_metadata', return_value=None):
            result = await ml_manager.deploy_model(model_name, model_version)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_active_model(self, ml_manager):
        """Test getting active model"""
        model_name = "test-model"
        mock_model = Mock()
        
        # Test cached model
        ml_manager.active_models[model_name] = mock_model
        result = await ml_manager.get_active_model(model_name)
        assert result == mock_model
        
        # Test loading from database
        ml_manager.active_models.clear()
        metadata = MLModelMetadata(
            model_name=model_name,
            model_version="v1.0",
            model_type=ModelType.FRAUD_DETECTION,
            business_id="test-business",
            training_data_hash="test-hash",
            model_path="test_model.joblib",
            status=ModelStatus.DEPLOYED
        )
        
        with patch.object(ml_manager, '_get_active_model_metadata', return_value=metadata):
            with patch.object(ml_manager, '_load_model', return_value=mock_model):
                result = await ml_manager.get_active_model(model_name)
                assert result == mock_model
                assert model_name in ml_manager.active_models
    
    @pytest.mark.asyncio
    async def test_record_feedback(self, ml_manager):
        """Test recording model feedback"""
        model_name = "test-model"
        prediction_id = "pred-123"
        actual_outcome = True
        user_feedback = "This was correctly identified as fraud"
        
        with patch.object(ml_manager, '_store_feedback', return_value=True):
            with patch.object(ml_manager, '_check_retraining_trigger', return_value=False):
                
                result = await ml_manager.record_feedback(
                    model_name, prediction_id, actual_outcome, user_feedback
                )
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_monitor_model_performance(self, ml_manager):
        """Test model performance monitoring"""
        model_name = "test-model"
        
        # Mock recent performance data
        recent_data = [
            {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            {"actual_outcome": False, "recorded_at": datetime.utcnow().isoformat()},
            {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
        ]
        
        with patch.object(ml_manager, '_get_recent_performance_data', return_value=recent_data):
            with patch.object(ml_manager, '_store_performance_metrics', return_value=True):
                with patch.object(ml_manager, '_check_performance_degradation', return_value=False):
                    
                    metrics = await ml_manager.monitor_model_performance(model_name)
                    
                    assert isinstance(metrics, ModelPerformanceMetrics)
                    assert metrics.sample_size == len(recent_data)
                    assert 0 <= metrics.accuracy <= 1
    
    @pytest.mark.asyncio
    async def test_retrain_model_if_needed(self, ml_manager, sample_training_config):
        """Test conditional model retraining"""
        model_name = "test-model"
        
        # Mock current performance (poor performance to trigger retraining)
        poor_performance = ModelPerformanceMetrics(accuracy=0.5, sample_size=100)
        
        # Mock metadata
        metadata = MLModelMetadata(
            model_name=model_name,
            model_version="v1.0",
            model_type=ModelType.FRAUD_DETECTION,
            business_id="test-business",
            training_data_hash="test-hash",
            training_config=sample_training_config.__dict__,
            trained_at=datetime.utcnow() - timedelta(days=35)  # Old model
        )
        
        with patch.object(ml_manager, 'monitor_model_performance', return_value=poor_performance):
            with patch.object(ml_manager, '_get_active_model_metadata', return_value=metadata):
                with patch.object(ml_manager, '_evaluate_retraining_criteria', return_value=True):
                    with patch.object(ml_manager, 'train_model', return_value=metadata):
                        with patch.object(ml_manager, '_should_auto_deploy', return_value=True):
                            with patch.object(ml_manager, 'deploy_model', return_value=True):
                                
                                result = await ml_manager.retrain_model_if_needed(model_name)
                                
                                assert result is True
    
    def test_prepare_features(self, ml_manager, sample_training_data, sample_training_config):
        """Test feature preparation"""
        X, y = ml_manager._prepare_features(sample_training_data, sample_training_config)
        
        assert X.shape[0] == len(sample_training_data)
        assert X.shape[1] == len(sample_training_config.feature_columns)
        assert y is not None
        assert len(y) == len(sample_training_data)
    
    def test_create_model_fraud_detection(self, ml_manager):
        """Test fraud detection model creation"""
        model = ml_manager._create_model(ModelType.FRAUD_DETECTION)
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_model_anomaly_detection(self, ml_manager):
        """Test anomaly detection model creation"""
        model = ml_manager._create_model(ModelType.ANOMALY_DETECTION)
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_generate_model_version(self, ml_manager):
        """Test model version generation"""
        model_type = ModelType.FRAUD_DETECTION
        business_id = "test-business-123"
        
        version = ml_manager._generate_model_version(model_type, business_id)
        
        assert model_type.value in version
        assert business_id in version
        assert len(version) > 20  # Should include timestamp
    
    def test_calculate_data_hash(self, ml_manager, sample_training_data):
        """Test data hash calculation"""
        hash1 = ml_manager._calculate_data_hash(sample_training_data)
        hash2 = ml_manager._calculate_data_hash(sample_training_data)
        
        assert hash1 == hash2  # Same data should produce same hash
        assert len(hash1) == 64  # SHA256 hash length
        
        # Different data should produce different hash
        modified_data = sample_training_data.copy()
        modified_data.iloc[0, 0] = 999999
        hash3 = ml_manager._calculate_data_hash(modified_data)
        assert hash1 != hash3
    
    def test_calculate_performance_metrics_empty_data(self, ml_manager):
        """Test performance metrics calculation with empty data"""
        metrics = ml_manager._calculate_performance_metrics([])
        
        assert isinstance(metrics, ModelPerformanceMetrics)
        assert metrics.accuracy == 0.0
        assert metrics.sample_size == 0
    
    def test_calculate_performance_metrics_with_data(self, ml_manager):
        """Test performance metrics calculation with data"""
        feedback_data = [
            {"actual_outcome": True},
            {"actual_outcome": False},
            {"actual_outcome": True},
            {"actual_outcome": True},
        ]
        
        metrics = ml_manager._calculate_performance_metrics(feedback_data)
        
        assert isinstance(metrics, ModelPerformanceMetrics)
        assert metrics.sample_size == 4
        assert metrics.accuracy == 0.75  # 3 out of 4 correct


class TestTrainingConfig:
    """Test cases for TrainingConfig"""
    
    def test_training_config_creation(self):
        """Test training configuration creation"""
        config = TrainingConfig(
            model_type=ModelType.FRAUD_DETECTION,
            training_data_query="SELECT * FROM transactions",
            feature_columns=["amount", "frequency"],
            target_column="is_fraud"
        )
        
        assert config.model_type == ModelType.FRAUD_DETECTION
        assert config.training_data_query == "SELECT * FROM transactions"
        assert config.feature_columns == ["amount", "frequency"]
        assert config.target_column == "is_fraud"
        assert config.test_size == 0.2  # Default value
        assert config.min_samples == 100  # Default value


class TestModelTypes:
    """Test cases for model types and enums"""
    
    def test_model_type_enum(self):
        """Test ModelType enum values"""
        assert ModelType.FRAUD_DETECTION == "fraud_detection"
        assert ModelType.ANOMALY_DETECTION == "anomaly_detection"
        assert ModelType.PATTERN_RECOGNITION == "pattern_recognition"
        assert ModelType.PREDICTION == "prediction"
    
    def test_model_status_enum(self):
        """Test ModelStatus enum values"""
        assert ModelStatus.TRAINING == "training"
        assert ModelStatus.VALIDATING == "validating"
        assert ModelStatus.DEPLOYED == "deployed"
        assert ModelStatus.DEPRECATED == "deprecated"
        assert ModelStatus.FAILED == "failed"


class TestMLEngineIntegration:
    """Integration tests for ML Engine"""
    
    @pytest.mark.asyncio
    async def test_full_training_pipeline(self, ml_manager, sample_training_config, sample_training_data):
        """Test complete training pipeline"""
        business_id = "integration-test-business"
        
        # Mock all external dependencies
        with patch.object(ml_manager, '_load_training_data', return_value=sample_training_data):
            with patch.object(ml_manager, '_save_model', return_value=Path("test_model.joblib")):
                with patch.object(ml_manager, '_store_model_metadata', return_value=True):
                    
                    # Train model
                    metadata = await ml_manager.train_model(sample_training_config, business_id)
                    assert metadata.status == ModelStatus.VALIDATING
                    
                    # Deploy model
                    with patch.object(ml_manager, '_get_model_metadata', return_value=metadata):
                        with patch.object(ml_manager, '_load_model', return_value=Mock()):
                            with patch.object(ml_manager, '_deprecate_active_model', return_value=True):
                                with patch.object(ml_manager, '_update_model_status', return_value=True):
                                    
                                    deploy_result = await ml_manager.deploy_model(
                                        metadata.model_name, metadata.model_version
                                    )
                                    assert deploy_result is True
                    
                    # Get active model
                    active_model = await ml_manager.get_active_model(metadata.model_name)
                    assert active_model is not None
                    
                    # Record feedback
                    with patch.object(ml_manager, '_store_feedback', return_value=True):
                        with patch.object(ml_manager, '_check_retraining_trigger', return_value=False):
                            feedback_result = await ml_manager.record_feedback(
                                metadata.model_name, "test-prediction-1", True
                            )
                            assert feedback_result is True
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self, ml_manager):
        """Test performance monitoring workflow"""
        model_name = "performance-test-model"
        
        # Mock performance data
        performance_data = [
            {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            {"actual_outcome": False, "recorded_at": datetime.utcnow().isoformat()},
        ]
        
        with patch.object(ml_manager, '_get_recent_performance_data', return_value=performance_data):
            with patch.object(ml_manager, '_store_performance_metrics', return_value=True):
                with patch.object(ml_manager, '_check_performance_degradation', return_value=False):
                    
                    metrics = await ml_manager.monitor_model_performance(model_name)
                    
                    assert metrics.sample_size == 3
                    assert metrics.accuracy > 0.5  # Should be around 0.67
                    assert model_name in ml_manager.performance_cache


@pytest.mark.asyncio
async def test_global_ml_engine_instance():
    """Test global ML engine instance"""
    assert ml_engine is not None
    assert isinstance(ml_engine, MLModelManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])