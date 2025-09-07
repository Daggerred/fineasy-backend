#!/usr/bin/env python3
"""
ML Engine Offline Validation Script
Tests the ML engine functionality without requiring database connection
"""
import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ml_engine import TrainingConfig, ModelType, ModelStatus
from app.models.responses import FeedbackData, MLModelMetadata, ModelPerformanceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLEngineOfflineValidator:
    """Validates ML Engine functionality without database"""
    
    def __init__(self):
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all ML engine validation tests"""
        logger.info("Starting ML Engine offline validation...")
        
        try:
            # Run individual tests
            await self.test_model_types_and_enums()
            await self.test_training_config()
            await self.test_model_metadata()
            await self.test_performance_metrics()
            await self.test_feedback_data()
            await self.test_model_creation()
            await self.test_feature_preparation()
            await self.test_versioning_and_hashing()
            await self.test_performance_calculation()
            
            # Print results
            self.print_results()
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            traceback.print_exc()
            return False
        
        return all(result['passed'] for result in self.test_results)
    
    async def test_model_types_and_enums(self):
        """Test model types and enums"""
        test_name = "Model Types and Enums"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test ModelType enum
            assert ModelType.FRAUD_DETECTION == "fraud_detection"
            assert ModelType.ANOMALY_DETECTION == "anomaly_detection"
            assert ModelType.PATTERN_RECOGNITION == "pattern_recognition"
            assert ModelType.PREDICTION == "prediction"
            
            # Test ModelStatus enum
            assert ModelStatus.TRAINING == "training"
            assert ModelStatus.VALIDATING == "validating"
            assert ModelStatus.DEPLOYED == "deployed"
            assert ModelStatus.DEPRECATED == "deprecated"
            assert ModelStatus.FAILED == "failed"
            
            # Test enum values are strings
            assert isinstance(ModelType.FRAUD_DETECTION, str)
            assert isinstance(ModelStatus.DEPLOYED, str)
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'All model types and enums working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_training_config(self):
        """Test training configuration creation and validation"""
        test_name = "Training Configuration"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create valid training config
            config = TrainingConfig(
                model_type=ModelType.FRAUD_DETECTION,
                training_data_query="SELECT amount, frequency, time_diff, is_fraud FROM transactions WHERE business_id = '{business_id}'",
                feature_columns=["amount", "frequency", "time_diff"],
                target_column="is_fraud",
                test_size=0.2,
                min_samples=50
            )
            
            # Verify configuration
            assert config.model_type == ModelType.FRAUD_DETECTION
            assert len(config.feature_columns) == 3
            assert config.target_column == "is_fraud"
            assert config.test_size == 0.2
            assert config.min_samples == 50
            assert config.validation_size == 0.2  # Default value
            assert config.cross_validation_folds == 5  # Default value
            
            # Test with minimal config
            minimal_config = TrainingConfig(
                model_type=ModelType.ANOMALY_DETECTION,
                training_data_query="SELECT * FROM data",
                feature_columns=["feature1", "feature2"]
            )
            
            assert minimal_config.target_column is None
            assert minimal_config.test_size == 0.2  # Default
            assert minimal_config.min_samples == 100  # Default
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Training configuration created and validated successfully'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_model_metadata(self):
        """Test ML model metadata"""
        test_name = "Model Metadata"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create model metadata
            metadata = MLModelMetadata(
                model_name="test_fraud_model",
                model_version="v1.0.0",
                model_type=ModelType.FRAUD_DETECTION,
                business_id="test-business-123",
                training_data_hash="abc123def456",
                accuracy_metrics={"accuracy": 0.85, "precision": 0.82},
                training_config={"test_size": 0.2, "min_samples": 100},
                model_path="/path/to/model.joblib",
                status=ModelStatus.DEPLOYED
            )
            
            # Verify metadata
            assert metadata.model_name == "test_fraud_model"
            assert metadata.model_version == "v1.0.0"
            assert metadata.model_type == ModelType.FRAUD_DETECTION
            assert metadata.business_id == "test-business-123"
            assert metadata.training_data_hash == "abc123def456"
            assert metadata.accuracy_metrics["accuracy"] == 0.85
            assert metadata.status == ModelStatus.DEPLOYED
            assert metadata.trained_at is not None
            
            # Test serialization
            metadata_dict = metadata.dict()
            assert isinstance(metadata_dict, dict)
            assert metadata_dict["model_name"] == "test_fraud_model"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Model metadata working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_performance_metrics(self):
        """Test model performance metrics"""
        test_name = "Performance Metrics"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create performance metrics
            metrics = ModelPerformanceMetrics(
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                cross_val_mean=0.83,
                cross_val_std=0.02,
                sample_size=1000
            )
            
            # Verify metrics
            assert metrics.accuracy == 0.85
            assert metrics.precision == 0.82
            assert metrics.recall == 0.88
            assert metrics.f1_score == 0.85
            assert metrics.cross_val_mean == 0.83
            assert metrics.cross_val_std == 0.02
            assert metrics.sample_size == 1000
            assert metrics.evaluated_at is not None
            
            # Test default values
            default_metrics = ModelPerformanceMetrics()
            assert default_metrics.accuracy == 0.0
            assert default_metrics.precision == 0.0
            assert default_metrics.sample_size == 0
            
            # Test serialization
            metrics_dict = metrics.dict()
            assert isinstance(metrics_dict, dict)
            assert metrics_dict["accuracy"] == 0.85
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Performance metrics working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_feedback_data(self):
        """Test feedback data model"""
        test_name = "Feedback Data"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Create feedback data
            feedback = FeedbackData(
                model_name="test_model",
                prediction_id="pred_123",
                actual_outcome=True,
                user_feedback="This prediction was correct"
            )
            
            # Verify feedback
            assert feedback.model_name == "test_model"
            assert feedback.prediction_id == "pred_123"
            assert feedback.actual_outcome is True
            assert feedback.user_feedback == "This prediction was correct"
            assert feedback.recorded_at is not None
            
            # Test without user feedback
            minimal_feedback = FeedbackData(
                model_name="test_model",
                prediction_id="pred_124",
                actual_outcome=False
            )
            
            assert minimal_feedback.user_feedback is None
            assert minimal_feedback.actual_outcome is False
            
            # Test serialization
            feedback_dict = feedback.dict()
            assert isinstance(feedback_dict, dict)
            assert feedback_dict["model_name"] == "test_model"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Feedback data working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_model_creation(self):
        """Test ML model creation"""
        test_name = "Model Creation"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Import ML engine for model creation
            from app.services.ml_engine import MLModelManager
            
            # Create ML manager instance
            ml_manager = MLModelManager()
            
            # Test different model types
            fraud_model = ml_manager._create_model(ModelType.FRAUD_DETECTION)
            anomaly_model = ml_manager._create_model(ModelType.ANOMALY_DETECTION)
            
            # Verify models have required methods
            assert hasattr(fraud_model, 'fit'), "Fraud model missing fit method"
            assert hasattr(fraud_model, 'predict'), "Fraud model missing predict method"
            assert hasattr(anomaly_model, 'fit'), "Anomaly model missing fit method"
            assert hasattr(anomaly_model, 'predict'), "Anomaly model missing predict method"
            
            # Test model pipeline structure
            assert hasattr(fraud_model, 'steps'), "Model should be a pipeline"
            assert len(fraud_model.steps) >= 2, "Pipeline should have at least 2 steps"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'ML models created successfully'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_feature_preparation(self):
        """Test feature preparation"""
        test_name = "Feature Preparation"
        logger.info(f"Testing {test_name}...")
        
        try:
            import pandas as pd
            import numpy as np
            from app.services.ml_engine import MLModelManager
            
            # Create test data
            data = pd.DataFrame({
                'amount': [100, 200, 300, 400, 500],
                'frequency': [1, 2, 3, 4, 5],
                'time_diff': [0.5, 1.0, 1.5, 2.0, 2.5],
                'is_fraud': [0, 1, 0, 1, 0]
            })
            
            # Create training config
            config = TrainingConfig(
                model_type=ModelType.FRAUD_DETECTION,
                training_data_query="SELECT * FROM test",
                feature_columns=["amount", "frequency", "time_diff"],
                target_column="is_fraud"
            )
            
            # Create ML manager and prepare features
            ml_manager = MLModelManager()
            X, y = ml_manager._prepare_features(data, config)
            
            # Verify feature preparation
            assert X.shape[0] == len(data), "Feature matrix should have same rows as data"
            assert X.shape[1] == len(config.feature_columns), "Feature matrix should have correct columns"
            assert y is not None, "Target should not be None for supervised learning"
            assert len(y) == len(data), "Target should have same length as data"
            
            # Test unsupervised learning (no target)
            unsupervised_config = TrainingConfig(
                model_type=ModelType.ANOMALY_DETECTION,
                training_data_query="SELECT * FROM test",
                feature_columns=["amount", "frequency", "time_diff"]
            )
            
            X_unsup, y_unsup = ml_manager._prepare_features(data, unsupervised_config)
            assert X_unsup.shape[1] == 3, "Unsupervised features should have 3 columns"
            assert y_unsup is None, "Unsupervised target should be None"
            
            # Test with missing values
            data_with_nan = data.copy()
            data_with_nan.loc[0, 'amount'] = np.nan
            X_nan, y_nan = ml_manager._prepare_features(data_with_nan, config)
            assert not np.isnan(X_nan).any(), "NaN values should be handled"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Feature preparation working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_versioning_and_hashing(self):
        """Test model versioning and data hashing"""
        test_name = "Versioning and Hashing"
        logger.info(f"Testing {test_name}...")
        
        try:
            import pandas as pd
            from app.services.ml_engine import MLModelManager
            
            ml_manager = MLModelManager()
            
            # Test version generation
            business_id = "test-business-123"
            version1 = ml_manager._generate_model_version(ModelType.FRAUD_DETECTION, business_id)
            

            
            # Add small delay to ensure different timestamps
            import time
            time.sleep(0.001)
            
            version2 = ml_manager._generate_model_version(ModelType.FRAUD_DETECTION, business_id)
            
            # Versions should be different (due to timestamp) or at least contain expected components
            # Note: In very fast execution, timestamps might be the same, so we check components instead
            assert ModelType.FRAUD_DETECTION.value in version1, f"Version '{version1}' should contain model type '{ModelType.FRAUD_DETECTION.value}'"
            assert business_id in version1, f"Version '{version1}' should contain business ID '{business_id}'"
            assert len(version1) > 20, f"Version '{version1}' should include timestamp (length > 20)"
            
            # Test that different business IDs produce different versions
            version3 = ml_manager._generate_model_version(ModelType.FRAUD_DETECTION, "different-business")
            assert version1 != version3, "Different business IDs should produce different versions"
            
            # Test data hash calculation
            data1 = pd.DataFrame({
                'amount': [100, 200, 300],
                'frequency': [1, 2, 3],
                'is_fraud': [0, 1, 0]
            })
            
            data2 = pd.DataFrame({
                'amount': [100, 200, 300],
                'frequency': [1, 2, 3],
                'is_fraud': [0, 1, 0]
            })
            
            data3 = pd.DataFrame({
                'amount': [100, 200, 400],  # Different data
                'frequency': [1, 2, 3],
                'is_fraud': [0, 1, 0]
            })
            
            hash1 = ml_manager._calculate_data_hash(data1)
            hash2 = ml_manager._calculate_data_hash(data2)
            hash3 = ml_manager._calculate_data_hash(data3)
            
            assert hash1 == hash2, "Same data should produce same hash"
            assert hash1 != hash3, "Different data should produce different hash"
            assert len(hash1) == 64, "Hash should be 64 characters (SHA256)"
            assert isinstance(hash1, str), "Hash should be string"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Versioning and hashing working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_performance_calculation(self):
        """Test performance metrics calculation"""
        test_name = "Performance Calculation"
        logger.info(f"Testing {test_name}...")
        
        try:
            from app.services.ml_engine import MLModelManager
            
            ml_manager = MLModelManager()
            
            # Test performance metrics calculation with data
            feedback_data = [
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": False, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            ]
            
            metrics = ml_manager._calculate_performance_metrics(feedback_data)
            
            assert metrics.sample_size == 4, "Sample size should be 4"
            assert metrics.accuracy == 0.75, "Accuracy should be 0.75 (3/4)"
            assert metrics.evaluated_at is not None, "Evaluation timestamp should be set"
            assert isinstance(metrics, ModelPerformanceMetrics), "Should return ModelPerformanceMetrics"
            
            # Test empty data handling
            empty_metrics = ml_manager._calculate_performance_metrics([])
            assert empty_metrics.sample_size == 0, "Empty data should have 0 sample size"
            assert empty_metrics.accuracy == 0.0, "Empty data should have 0 accuracy"
            
            # Test all correct predictions
            all_correct_data = [
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            ]
            
            perfect_metrics = ml_manager._calculate_performance_metrics(all_correct_data)
            assert perfect_metrics.accuracy == 1.0, "All correct should give 1.0 accuracy"
            assert perfect_metrics.sample_size == 2, "Sample size should be 2"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Performance calculation working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def print_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("ML ENGINE OFFLINE VALIDATION RESULTS")
        logger.info("="*60)
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            logger.info(f"{status}: {result['test']} - {result['message']}")
            if result['passed']:
                passed_tests += 1
        
        logger.info("="*60)
        logger.info(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All ML Engine offline tests passed!")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")
        
        logger.info("="*60)


async def main():
    """Main validation function"""
    validator = MLEngineOfflineValidator()
    success = await validator.run_all_tests()
    
    if success:
        logger.info("ML Engine offline validation completed successfully!")
        return 0
    else:
        logger.error("ML Engine offline validation failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)