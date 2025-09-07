"""
Pytest configuration and shared fixtures for AI backend tests
"""
import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
import numpy as np

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing"""
    mock_client = AsyncMock()
    
    # Mock table operations
    mock_table = AsyncMock()
    mock_client.table.return_value = mock_table
    
    # Mock query operations
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.delete.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.gte.return_value = mock_table
    mock_table.lte.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.limit.return_value = mock_table
    
    # Mock execute with default empty response
    mock_table.execute.return_value = AsyncMock(data=[], count=0)
    
    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_redis = Mock()
    
    # Mock Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.expire.return_value = True
    mock_redis.flushdb.return_value = True
    
    return mock_redis


@pytest.fixture
def sample_business_data():
    """Sample business data for testing"""
    return {
        "business_id": "test_business_123",
        "business_name": "Test Business Ltd",
        "owner_id": "user_123",
        "created_at": datetime.utcnow().isoformat(),
        "settings": {
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "gst_number": "27AAPFU0939F1ZV"
        }
    }


@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing"""
    base_date = datetime.utcnow() - timedelta(days=30)
    transactions = []
    
    for i in range(20):
        transactions.append({
            "id": f"trans_{i}",
            "business_id": "test_business_123",
            "amount": np.random.uniform(100, 10000),
            "type": np.random.choice(["income", "expense"]),
            "description": f"Transaction {i}",
            "category": np.random.choice(["sales", "supplies", "services", "utilities"]),
            "created_at": (base_date + timedelta(days=i)).isoformat(),
            "updated_at": (base_date + timedelta(days=i)).isoformat()
        })
    
    return transactions


@pytest.fixture
def sample_invoices():
    """Sample invoice data for testing"""
    base_date = datetime.utcnow() - timedelta(days=60)
    invoices = []
    
    for i in range(15):
        invoices.append({
            "id": f"inv_{i}",
            "business_id": "test_business_123",
            "invoice_number": f"INV-{i:03d}",
            "customer_id": f"cust_{i % 5}",
            "customer_name": f"Customer {chr(65 + (i % 5))}",
            "total_amount": np.random.uniform(1000, 50000),
            "tax_amount": np.random.uniform(100, 5000),
            "status": np.random.choice(["draft", "sent", "paid", "overdue"]),
            "due_date": (base_date + timedelta(days=i + 30)).isoformat(),
            "created_at": (base_date + timedelta(days=i)).isoformat(),
            "items": [
                {
                    "name": f"Product {i}",
                    "quantity": np.random.randint(1, 10),
                    "unit_price": np.random.uniform(100, 1000),
                    "total": np.random.uniform(100, 10000)
                }
            ]
        })
    
    return invoices


@pytest.fixture
def sample_customers():
    """Sample customer data for testing"""
    customers = []
    
    for i in range(10):
        customers.append({
            "id": f"cust_{i}",
            "business_id": "test_business_123",
            "name": f"Customer {chr(65 + i)}",
            "email": f"customer{i}@example.com",
            "phone": f"+91-98765-4321{i}",
            "address": f"Address {i}, City, State",
            "gst_number": f"27AAPFU0939F1Z{i}" if i % 3 == 0 else None,
            "total_revenue": np.random.uniform(5000, 100000),
            "transaction_count": np.random.randint(5, 50),
            "last_transaction_date": (datetime.utcnow() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            "created_at": (datetime.utcnow() - timedelta(days=np.random.randint(30, 365))).isoformat()
        })
    
    return customers


@pytest.fixture
def sample_products():
    """Sample product data for testing"""
    products = []
    
    product_names = [
        "Laptop Computer", "Office Chair", "Software License", "Consulting Hours",
        "Mobile Phone", "Printer", "Desk", "Monitor", "Keyboard", "Mouse"
    ]
    
    for i, name in enumerate(product_names):
        products.append({
            "id": f"prod_{i}",
            "business_id": "test_business_123",
            "name": name,
            "description": f"Description for {name}",
            "price": np.random.uniform(500, 50000),
            "unit": np.random.choice(["piece", "hour", "license", "kg", "meter"]),
            "category": np.random.choice(["electronics", "furniture", "software", "services"]),
            "stock_quantity": np.random.randint(0, 100),
            "created_at": (datetime.utcnow() - timedelta(days=np.random.randint(1, 180))).isoformat()
        })
    
    return products


@pytest.fixture
def sample_fraud_alerts():
    """Sample fraud alert data for testing"""
    from app.models.base import FraudType
    
    alerts = []
    fraud_types = list(FraudType)
    
    for i in range(5):
        alerts.append({
            "id": f"alert_{i}",
            "business_id": "test_business_123",
            "type": fraud_types[i % len(fraud_types)].value,
            "message": f"Fraud alert {i}: Suspicious activity detected",
            "confidence_score": np.random.uniform(0.7, 1.0),
            "evidence": {
                "transaction_ids": [f"trans_{i}", f"trans_{i+1}"],
                "similarity_score": np.random.uniform(0.8, 1.0),
                "risk_factors": ["duplicate_amount", "similar_description"]
            },
            "status": "active",
            "created_at": (datetime.utcnow() - timedelta(hours=np.random.randint(1, 24))).isoformat(),
            "acknowledged_at": None,
            "acknowledged_by": None
        })
    
    return alerts


@pytest.fixture
def sample_ml_models():
    """Sample ML model metadata for testing"""
    from app.services.ml_engine import ModelType, ModelStatus
    
    models = []
    model_types = list(ModelType)
    
    for i, model_type in enumerate(model_types):
        models.append({
            "id": f"model_{i}",
            "model_name": f"{model_type.value}_test_business_123",
            "model_version": f"v1.{i}",
            "model_type": model_type.value,
            "business_id": "test_business_123",
            "training_data_hash": f"hash_{i}",
            "model_path": f"models/{model_type.value}_test_business_123_v1_{i}.joblib",
            "status": ModelStatus.DEPLOYED.value if i % 2 == 0 else ModelStatus.TRAINING.value,
            "accuracy_metrics": {
                "accuracy": np.random.uniform(0.8, 0.95),
                "precision": np.random.uniform(0.75, 0.9),
                "recall": np.random.uniform(0.7, 0.85),
                "f1_score": np.random.uniform(0.75, 0.88)
            },
            "training_config": {
                "feature_columns": ["amount", "frequency", "time_diff"],
                "target_column": "is_fraud",
                "test_size": 0.2,
                "min_samples": 100
            },
            "trained_at": (datetime.utcnow() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            "deployed_at": (datetime.utcnow() - timedelta(days=np.random.randint(1, 7))).isoformat() if i % 2 == 0 else None,
            "is_active": i % 2 == 0
        })
    
    return models


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    env_vars = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test_service_key",
        "SUPABASE_ANON_KEY": "test_anon_key",
        "REDIS_URL": "redis://localhost:6379",
        "OPENAI_API_KEY": "test_openai_key",
        "GST_API_URL": "https://api.gst.gov.in",
        "GST_API_KEY": "test_gst_key",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "API_VERSION": "v1"
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    yield env_vars
    
    # Clean up environment variables
    for key in env_vars.keys():
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP model for testing"""
    mock_nlp = Mock()
    
    # Mock document processing
    def mock_process(text):
        mock_doc = Mock()
        mock_doc.text = text
        
        # Mock entities based on common patterns
        entities = []
        if "customer" in text.lower() or "client" in text.lower():
            entities.append(Mock(text="Test Customer", label_="ORG"))
        if any(digit in text for digit in "0123456789"):
            entities.append(Mock(text="5", label_="CARDINAL"))
        if "laptop" in text.lower() or "computer" in text.lower():
            entities.append(Mock(text="laptop", label_="PRODUCT"))
        if "upi" in text.lower():
            entities.append(Mock(text="UPI", label_="PAYMENT"))
        
        mock_doc.ents = entities
        return mock_doc
    
    mock_nlp.side_effect = mock_process
    return mock_nlp


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    import logging
    
    # Set up test logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some noisy loggers during tests
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "accuracy: mark test as an accuracy validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file names"""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
        elif "test_integration" in item.fspath.basename or "test_api" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "test_end_to_end" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "accuracy" in item.fspath.basename:
            item.add_marker(pytest.mark.accuracy)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def cleanup_test_data():
    """Cleanup test data after tests"""
    yield
    
    # Cleanup logic here if needed
    # For example, clearing test databases, removing test files, etc.
    pass


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.start_cpu = psutil.cpu_percent()
        
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            return {
                "execution_time": end_time - self.start_time,
                "memory_usage": end_memory - self.start_memory,
                "cpu_usage": end_cpu
            }
    
    return PerformanceMonitor()


# Database testing utilities
@pytest.fixture
def mock_database_operations():
    """Mock common database operations"""
    class MockDatabaseOperations:
        def __init__(self):
            self.data_store = {}
        
        async def insert(self, table, data):
            if table not in self.data_store:
                self.data_store[table] = []
            self.data_store[table].append(data)
            return data
        
        async def select(self, table, filters=None):
            if table not in self.data_store:
                return []
            
            data = self.data_store[table]
            if filters:
                # Simple filtering logic
                filtered_data = []
                for item in data:
                    match = True
                    for key, value in filters.items():
                        if item.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_data.append(item)
                return filtered_data
            
            return data
        
        async def update(self, table, data, filters):
            if table not in self.data_store:
                return False
            
            updated = False
            for item in self.data_store[table]:
                match = True
                for key, value in filters.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    item.update(data)
                    updated = True
            
            return updated
        
        async def delete(self, table, filters):
            if table not in self.data_store:
                return 0
            
            original_count = len(self.data_store[table])
            self.data_store[table] = [
                item for item in self.data_store[table]
                if not all(item.get(key) == value for key, value in filters.items())
            ]
            
            return original_count - len(self.data_store[table])
    
    return MockDatabaseOperations()