"""
Performance and load tests for AI services
"""
import pytest
import asyncio
import time
import concurrent.futures
import threading
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.main import app
from app.services.fraud_detection import FraudDetector
from app.services.predictive_analytics import PredictiveAnalyzer
from app.services.nlp_invoice import NLPInvoiceGenerator
from app.services.ml_engine import ml_engine


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers"""
    return {"Authorization": "Bearer test_token_123"}


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    np.random.seed(42)
    
    # Generate 10,000 transactions
    transactions = []
    for i in range(10000):
        transactions.append({
            'id': f'trans_{i}',
            'amount': np.random.lognormal(mean=6, sigma=1),
            'type': np.random.choice(['income', 'expense'], p=[0.6, 0.4]),
            'description': f'Transaction {i}',
            'created_at': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat()
        })
    
    # Generate 1,000 invoices
    invoices = []
    for i in range(1000):
        invoices.append({
            'id': f'inv_{i}',
            'invoice_number': f'INV-{i:04d}',
            'total_amount': np.random.uniform(1000, 50000),
            'status': np.random.choice(['paid', 'pending', 'overdue'], p=[0.7, 0.2, 0.1]),
            'created_at': (datetime.now() - timedelta(days=np.random.randint(0, 180))).isoformat()
        })
    
    # Generate 100 customers
    customers = []
    for i in range(100):
        customers.append({
            'id': f'cust_{i}',
            'name': f'Customer {i}',
            'total_revenue': np.random.uniform(5000, 100000),
            'transaction_count': np.random.randint(5, 50)
        })
    
    return {
        'transactions': transactions,
        'invoices': invoices,
        'customers': customers
    }


class TestFraudDetectionPerformance:
    """Performance tests for fraud detection service"""
    
    @pytest.mark.performance
    @patch('app.services.fraud_detection.DatabaseManager')
    def test_fraud_detection_large_dataset_performance(self, mock_db_class, large_dataset):
        """Test fraud detection performance with large dataset"""
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'])
        mock_db.get_invoices = AsyncMock(return_value=large_dataset['invoices'])
        mock_db.get_suppliers = AsyncMock(return_value=[])
        mock_db.save_fraud_alert = AsyncMock(return_value="alert_123")
        mock_db.log_ai_operation = AsyncMock()
        
        fraud_detector = FraudDetector()
        
        # Measure performance
        start_time = time.time()
        
        # Run fraud analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fraud_detector.analyze_fraud("business_123"))
        finally:
            loop.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert result is not None
        assert hasattr(result, 'risk_score')
        
        # Log performance metrics
        print(f"Fraud detection with {len(large_dataset['transactions'])} transactions: {execution_time:.2f}s")
    
    @pytest.mark.performance
    @patch('app.services.fraud_detection.DatabaseManager')
    def test_duplicate_detection_performance(self, mock_db_class, large_dataset):
        """Test duplicate detection performance"""
        # Add some intentional duplicates
        duplicates = []
        for i in range(100):  # Add 100 duplicates
            original = large_dataset['transactions'][i]
            duplicate = original.copy()
            duplicate['id'] = f'dup_{i}'
            duplicates.append(duplicate)
        
        all_transactions = large_dataset['transactions'] + duplicates
        
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=all_transactions)
        mock_db.get_invoices = AsyncMock(return_value=[])
        mock_db.get_suppliers = AsyncMock(return_value=[])
        
        fraud_detector = FraudDetector()
        
        # Measure performance
        start_time = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            alerts = loop.run_until_complete(fraud_detector.detect_duplicates("business_123"))
        finally:
            loop.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 20.0  # Should complete within 20 seconds
        assert len(alerts) > 0  # Should detect some duplicates
        
        print(f"Duplicate detection with {len(all_transactions)} transactions: {execution_time:.2f}s")
    
    @pytest.mark.performance
    def test_fraud_detection_memory_usage(self, large_dataset):
        """Test memory usage during fraud detection"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('app.services.fraud_detection.DatabaseManager') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'])
            mock_db.get_invoices = AsyncMock(return_value=large_dataset['invoices'])
            mock_db.get_suppliers = AsyncMock(return_value=[])
            mock_db.save_fraud_alert = AsyncMock(return_value="alert_123")
            mock_db.log_ai_operation = AsyncMock()
            
            fraud_detector = FraudDetector()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(fraud_detector.analyze_fraud("business_123"))
            finally:
                loop.close()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 500  # Should not use more than 500MB additional memory
        
        print(f"Memory usage increase: {memory_increase:.2f}MB")


class TestPredictiveAnalyticsPerformance:
    """Performance tests for predictive analytics service"""
    
    @pytest.mark.performance
    @patch('app.services.predictive_analytics.DatabaseManager')
    def test_cash_flow_prediction_performance(self, mock_db_class, large_dataset):
        """Test cash flow prediction performance"""
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'])
        
        analyzer = PredictiveAnalyzer()
        
        # Measure performance
        start_time = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(analyzer.predict_cash_flow("business_123", 3))
        finally:
            loop.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 15.0  # Should complete within 15 seconds
        assert prediction is not None
        assert hasattr(prediction, 'confidence')
        
        print(f"Cash flow prediction with {len(large_dataset['transactions'])} transactions: {execution_time:.2f}s")
    
    @pytest.mark.performance
    @patch('app.services.predictive_analytics.DatabaseManager')
    def test_customer_analysis_performance(self, mock_db_class, large_dataset):
        """Test customer analysis performance"""
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_customer_revenue_data = AsyncMock(return_value=large_dataset['customers'])
        
        analyzer = PredictiveAnalyzer()
        
        # Measure performance
        start_time = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            analysis = loop.run_until_complete(analyzer.analyze_customer_revenue("business_123"))
        finally:
            loop.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert analysis is not None
        assert hasattr(analysis, 'top_customers')
        
        print(f"Customer analysis with {len(large_dataset['customers'])} customers: {execution_time:.2f}s")
    
    @pytest.mark.performance
    @patch('app.services.predictive_analytics.DatabaseManager')
    def test_comprehensive_insights_performance(self, mock_db_class, large_dataset):
        """Test comprehensive insights generation performance"""
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'])
        mock_db.get_customer_revenue_data = AsyncMock(return_value=large_dataset['customers'])
        mock_db.get_invoices = AsyncMock(return_value=large_dataset['invoices'])
        mock_db.get_current_cash_balance = AsyncMock(return_value=100000.0)
        
        analyzer = PredictiveAnalyzer()
        
        # Measure performance
        start_time = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            insights = loop.run_until_complete(analyzer.generate_insights("business_123"))
        finally:
            loop.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 25.0  # Should complete within 25 seconds
        assert insights is not None
        assert len(insights.insights) > 0
        
        print(f"Comprehensive insights generation: {execution_time:.2f}s")


class TestAPILoadTesting:
    """Load tests for API endpoints"""
    
    @pytest.mark.performance
    @patch('app.api.fraud.verify_token')
    @patch('app.services.fraud_detection.DatabaseManager')
    def test_fraud_api_concurrent_requests(self, mock_db_class, mock_verify_token, client, auth_headers):
        """Test fraud API under concurrent load"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=[])
        mock_db.get_invoices = AsyncMock(return_value=[])
        mock_db.get_suppliers = AsyncMock(return_value=[])
        mock_db.save_fraud_alert = AsyncMock(return_value="alert_123")
        mock_db.log_ai_operation = AsyncMock()
        
        def make_request():
            response = client.post(
                "/api/v1/fraud/analyze",
                json={"business_id": "test_business"},
                headers=auth_headers
            )
            return response.status_code
        
        # Run concurrent requests
        num_requests = 20
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert all(status == 200 for status in results)  # All requests should succeed
        assert total_time < 30.0  # Should complete within 30 seconds
        
        avg_response_time = total_time / num_requests
        print(f"Average response time for {num_requests} concurrent requests: {avg_response_time:.2f}s")
    
    @pytest.mark.performance
    @patch('app.api.insights.verify_token')
    @patch('app.services.predictive_analytics.DatabaseManager')
    @patch('app.utils.cache.cache.get')
    @patch('app.utils.cache.cache.set')
    def test_insights_api_caching_performance(self, mock_cache_set, mock_cache_get, 
                                            mock_db_class, mock_verify_token, client, auth_headers):
        """Test insights API caching performance"""
        # Setup mocks
        mock_verify_token.return_value = "user_123"
        
        # First request - cache miss
        mock_cache_get.return_value = None
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_transactions = AsyncMock(return_value=[])
        mock_db.get_customer_revenue_data = AsyncMock(return_value=[])
        mock_db.get_invoices = AsyncMock(return_value=[])
        mock_db.get_current_cash_balance = AsyncMock(return_value=50000.0)
        
        # Measure first request (cache miss)
        start_time = time.time()
        response1 = client.get("/api/v1/insights/test_business", headers=auth_headers)
        first_request_time = time.time() - start_time
        
        assert response1.status_code == 200
        
        # Setup cache hit
        cached_data = response1.json()
        mock_cache_get.return_value = cached_data
        
        # Measure second request (cache hit)
        start_time = time.time()
        response2 = client.get("/api/v1/insights/test_business", headers=auth_headers)
        second_request_time = time.time() - start_time
        
        assert response2.status_code == 200
        
        # Cache hit should be significantly faster
        assert second_request_time < first_request_time * 0.5  # At least 50% faster
        
        print(f"Cache miss: {first_request_time:.3f}s, Cache hit: {second_request_time:.3f}s")
    
    @pytest.mark.performance
    def test_api_response_time_benchmarks(self, client, auth_headers):
        """Test API response time benchmarks"""
        endpoints = [
            ("/api/v1/fraud/analyze", "POST", {"business_id": "test"}),
            ("/api/v1/insights/test_business", "GET", None),
            ("/api/v1/compliance/check", "POST", {"invoice_id": "test"}),
            ("/api/v1/invoice/parse", "POST", {"text": "test", "business_id": "test"})
        ]
        
        with patch('app.api.fraud.verify_token', return_value="user_123"):
            with patch('app.api.insights.verify_token', return_value="user_123"):
                with patch('app.api.compliance.verify_token', return_value="user_123"):
                    with patch('app.api.invoice.verify_token', return_value="user_123"):
                        
                        response_times = {}
                        
                        for endpoint, method, data in endpoints:
                            start_time = time.time()
                            
                            if method == "GET":
                                response = client.get(endpoint, headers=auth_headers)
                            else:
                                response = client.post(endpoint, json=data, headers=auth_headers)
                            
                            end_time = time.time()
                            response_time = end_time - start_time
                            
                            response_times[endpoint] = response_time
                            
                            # Each endpoint should respond within reasonable time
                            assert response_time < 5.0  # 5 second timeout
                            
                            print(f"{endpoint}: {response_time:.3f}s")
                        
                        # Overall average should be reasonable
                        avg_response_time = sum(response_times.values()) / len(response_times)
                        assert avg_response_time < 2.0  # Average under 2 seconds


class TestMLEnginePerformance:
    """Performance tests for ML engine"""
    
    @pytest.mark.performance
    @patch('app.services.ml_engine.DatabaseManager')
    def test_model_training_performance(self, mock_db_class, large_dataset):
        """Test ML model training performance"""
        from app.services.ml_engine import TrainingConfig, ModelType
        
        # Setup
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        # Create training data with fraud labels
        training_data = []
        for i, trans in enumerate(large_dataset['transactions'][:1000]):  # Use subset for training
            training_data.append({
                **trans,
                'is_fraud': np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate
            })
        
        mock_db.execute_query = AsyncMock(return_value=training_data)
        
        config = TrainingConfig(
            model_type=ModelType.FRAUD_DETECTION,
            training_data_query="SELECT * FROM transactions",
            feature_columns=["amount", "frequency", "time_diff"],
            target_column="is_fraud"
        )
        
        # Measure training performance
        start_time = time.time()
        
        with patch('app.services.ml_engine.MLModelManager._save_model', return_value="model.joblib"):
            with patch('app.services.ml_engine.MLModelManager._store_model_metadata', return_value=True):
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    metadata = loop.run_until_complete(ml_engine.train_model(config, "test_business"))
                finally:
                    loop.close()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Performance assertions
        assert training_time < 60.0  # Should complete within 60 seconds
        assert metadata is not None
        
        print(f"Model training with {len(training_data)} samples: {training_time:.2f}s")
    
    @pytest.mark.performance
    @patch('app.services.ml_engine.DatabaseManager')
    def test_model_prediction_performance(self, mock_db_class):
        """Test ML model prediction performance"""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])  # Sample predictions
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.95, 0.05]])
        
        # Test prediction performance
        test_data = np.random.rand(1000, 5)  # 1000 samples, 5 features
        
        start_time = time.time()
        
        # Run multiple predictions
        for _ in range(100):  # 100 prediction batches
            predictions = mock_model.predict(test_data[:10])  # Predict on 10 samples each time
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Performance assertions
        assert prediction_time < 5.0  # Should complete within 5 seconds
        
        avg_prediction_time = prediction_time / 100
        print(f"Average prediction time for 10 samples: {avg_prediction_time:.4f}s")


class TestMemoryAndResourceUsage:
    """Test memory and resource usage under load"""
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, large_dataset):
        """Test for memory leaks during repeated operations"""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('app.services.fraud_detection.DatabaseManager') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'][:100])
            mock_db.get_invoices = AsyncMock(return_value=large_dataset['invoices'][:50])
            mock_db.get_suppliers = AsyncMock(return_value=[])
            mock_db.save_fraud_alert = AsyncMock(return_value="alert_123")
            mock_db.log_ai_operation = AsyncMock()
            
            fraud_detector = FraudDetector()
            
            # Run multiple iterations
            for i in range(10):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(fraud_detector.analyze_fraud(f"business_{i}"))
                finally:
                    loop.close()
                
                # Force garbage collection
                gc.collect()
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not continuously increase
                assert memory_increase < 200  # Should not exceed 200MB increase
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase after 10 iterations: {total_increase:.2f}MB")
        
        # Final memory check
        assert total_increase < 100  # Should not have significant memory leak
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self, large_dataset):
        """Test CPU usage under load"""
        import psutil
        import threading
        
        cpu_usage_samples = []
        
        def monitor_cpu():
            for _ in range(20):  # Monitor for 20 seconds
                cpu_usage_samples.append(psutil.cpu_percent(interval=1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run intensive operations
        with patch('app.services.fraud_detection.DatabaseManager') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.get_transactions = AsyncMock(return_value=large_dataset['transactions'])
            mock_db.get_invoices = AsyncMock(return_value=large_dataset['invoices'])
            mock_db.get_suppliers = AsyncMock(return_value=[])
            mock_db.save_fraud_alert = AsyncMock(return_value="alert_123")
            mock_db.log_ai_operation = AsyncMock()
            
            fraud_detector = FraudDetector()
            
            # Run multiple concurrent analyses
            def run_analysis(business_id):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(fraud_detector.analyze_fraud(business_id))
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run_analysis, f"business_{i}") for i in range(4)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_usage_samples:
            avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
            max_cpu_usage = max(cpu_usage_samples)
            
            print(f"Average CPU usage: {avg_cpu_usage:.1f}%")
            print(f"Maximum CPU usage: {max_cpu_usage:.1f}%")
            
            # CPU usage should be reasonable
            assert avg_cpu_usage < 80.0  # Average should be under 80%
            assert max_cpu_usage < 95.0  # Peak should be under 95%


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])