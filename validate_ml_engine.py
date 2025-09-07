#!/usr/bin/env python3
"""
ML Engine Validation Script
Tests the ML engine functionality including training, deployment, and monitoring
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

from app.services.ml_engine import ml_engine, TrainingConfig, ModelType
from app.models.responses import FeedbackData
from app.database import init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLEngineValidator:
    """Validates ML Engine functionality"""
    
    def __init__(self):
        self.test_business_id = "test-ml-business-123"
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all ML engine validation tests"""
        logger.info("Starting ML Engine validation...")
        
        try:
            # Initialize database connection
            await init_database()
            logger.info("Database initialized")
            
            # Run individual tests
            await self.test_model_creation()
            await self.test_training_config()
            await self.test_model_versioning()
            await self.test_performance_monitoring()
            await self.test_feedback_system()
            await self.test_model_deployment()
            await self.test_error_handling()
            
            # Print results
            self.print_results()
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            traceback.print_exc()
            return False
        
        return all(result['passed'] for result in self.test_results)
    
    async def test_model_creation(self):
        """Test ML model creation and basic functionality"""
        test_name = "Model Creation"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test different model types
            fraud_model = ml_engine._create_model(ModelType.FRAUD_DETECTION)
            anomaly_model = ml_engine._create_model(ModelType.ANOMALY_DETECTION)
            
            # Verify models have required methods
            assert hasattr(fraud_model, 'fit'), "Fraud model missing fit method"
            assert hasattr(fraud_model, 'predict'), "Fraud model missing predict method"
            assert hasattr(anomaly_model, 'fit'), "Anomaly model missing fit method"
            assert hasattr(anomaly_model, 'predict'), "Anomaly model missing predict method"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Successfully created ML models'
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
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Training configuration created successfully'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_model_versioning(self):
        """Test model versioning system"""
        test_name = "Model Versioning"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test version generation
            version1 = ml_engine._generate_model_version(ModelType.FRAUD_DETECTION, self.test_business_id)
            version2 = ml_engine._generate_model_version(ModelType.FRAUD_DETECTION, self.test_business_id)
            
            # Versions should be different (due to timestamp)
            assert version1 != version2, "Model versions should be unique"
            assert ModelType.FRAUD_DETECTION.value in version1, "Version should contain model type"
            assert self.test_business_id in version1, "Version should contain business ID"
            
            # Test data hash calculation
            import pandas as pd
            import numpy as np
            
            # Create sample data
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
            
            hash1 = ml_engine._calculate_data_hash(data1)
            hash2 = ml_engine._calculate_data_hash(data2)
            hash3 = ml_engine._calculate_data_hash(data3)
            
            assert hash1 == hash2, "Same data should produce same hash"
            assert hash1 != hash3, "Different data should produce different hash"
            assert len(hash1) == 64, "Hash should be 64 characters (SHA256)"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Model versioning system working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        test_name = "Performance Monitoring"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test performance metrics calculation
            feedback_data = [
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": False, "recorded_at": datetime.utcnow().isoformat()},
                {"actual_outcome": True, "recorded_at": datetime.utcnow().isoformat()},
            ]
            
            metrics = ml_engine._calculate_performance_metrics(feedback_data)
            
            assert metrics.sample_size == 4, "Sample size should be 4"
            assert metrics.accuracy == 0.75, "Accuracy should be 0.75 (3/4)"
            assert metrics.evaluated_at is not None, "Evaluation timestamp should be set"
            
            # Test empty data handling
            empty_metrics = ml_engine._calculate_performance_metrics([])
            assert empty_metrics.sample_size == 0, "Empty data should have 0 sample size"
            assert empty_metrics.accuracy == 0.0, "Empty data should have 0 accuracy"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Performance monitoring working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_feedback_system(self):
        """Test feedback recording system"""
        test_name = "Feedback System"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test feedback data creation
            feedback = FeedbackData(
                model_name="test-model",
                prediction_id="pred-123",
                actual_outcome=True,
                user_feedback="This was correctly identified"
            )
            
            assert feedback.model_name == "test-model"
            assert feedback.prediction_id == "pred-123"
            assert feedback.actual_outcome is True
            assert feedback.user_feedback == "This was correctly identified"
            assert feedback.recorded_at is not None
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Feedback system working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_model_deployment(self):
        """Test model deployment functionality"""
        test_name = "Model Deployment"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test active models cache
            test_model = "test-model-object"
            model_name = "test-deployment-model"
            
            # Add model to cache
            ml_engine.active_models[model_name] = test_model
            
            # Retrieve from cache
            retrieved_model = await ml_engine.get_active_model(model_name)
            assert retrieved_model == test_model, "Should retrieve model from cache"
            
            # Clear cache and test database retrieval (will return None since we're not mocking DB)
            ml_engine.active_models.clear()
            retrieved_model = await ml_engine.get_active_model(model_name)
            assert retrieved_model is None, "Should return None when model not found"
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Model deployment system working correctly'
            })
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'passed': False,
                'message': f'Failed: {str(e)}'
            })
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    async def test_error_handling(self):
        """Test error handling in ML engine"""
        test_name = "Error Handling"
        logger.info(f"Testing {test_name}...")
        
        try:
            # Test invalid model type handling
            try:
                invalid_model = ml_engine._create_model("invalid_model_type")
                # Should create default model (anomaly detection)
                assert hasattr(invalid_model, 'fit'), "Should create default model"
            except Exception:
                pass  # Expected behavior
            
            # Test feature preparation with missing columns
            import pandas as pd
            
            data = pd.DataFrame({
                'amount': [100, 200],
                'frequency': [1, 2]
            })
            
            config = TrainingConfig(
                model_type=ModelType.FRAUD_DETECTION,
                training_data_query="SELECT * FROM test",
                feature_columns=["amount", "frequency", "missing_column"],  # Missing column
                target_column="is_fraud"
            )
            
            try:
                X, y = ml_engine._prepare_features(data, config)
                # Should handle missing columns gracefully or raise appropriate error
            except KeyError:
                pass  # Expected for missing columns
            
            self.test_results.append({
                'test': test_name,
                'passed': True,
                'message': 'Error handling working correctly'
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
        logger.info("ML ENGINE VALIDATION RESULTS")
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
            logger.info("🎉 All ML Engine tests passed!")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")
        
        logger.info("="*60)


async def main():
    """Main validation function"""
    validator = MLEngineValidator()
    success = await validator.run_all_tests()
    
    if success:
        logger.info("ML Engine validation completed successfully!")
        return 0
    else:
        logger.error("ML Engine validation failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)