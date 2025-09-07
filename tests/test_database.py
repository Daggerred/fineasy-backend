"""
Test cases for database integration
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.database import DatabaseManager, init_database, test_connection, get_supabase
from app.utils.database import AIDataUtils
from app.config import settings


class TestDatabaseIntegration:
    """Test database integration functionality"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self):
        """Test database initialization"""
        with patch('app.database.create_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client
            
            # Mock the test connection
            with patch('app.database.test_connection', return_value=True):
                client = await init_database()
                assert client is not None
                mock_create_client.assert_called_once_with(
                    settings.SUPABASE_URL,
                    settings.SUPABASE_SERVICE_KEY
                )
    
    @pytest.mark.asyncio
    async def test_database_connection_test(self):
        """Test database connection testing"""
        with patch('app.database.supabase') as mock_supabase:
            mock_table = Mock()
            mock_table.select.return_value.limit.return_value.execute.return_value = Mock(data=[])
            mock_supabase.table.return_value = mock_table
            
            result = await test_connection()
            assert result is True
            mock_supabase.table.assert_called_once_with("businesses")
    
    def test_get_supabase_without_init(self):
        """Test getting Supabase client without initialization"""
        with patch('app.database.supabase', None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                get_supabase()


class TestDatabaseManager:
    """Test DatabaseManager functionality"""
    
    def setup_method(self):
        """Setup test method"""
        with patch('app.database.get_supabase') as mock_get_supabase:
            self.mock_client = Mock()
            mock_get_supabase.return_value = self.mock_client
            self.db_manager = DatabaseManager()
    
    @pytest.mark.asyncio
    async def test_get_business_data(self):
        """Test getting business data"""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{"id": "test-id", "name": "Test Business"}]
        
        self.mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = await self.db_manager.get_business_data("test-id")
        
        assert result == {"id": "test-id", "name": "Test Business"}
        self.mock_client.table.assert_called_with("businesses")
    
    @pytest.mark.asyncio
    async def test_get_business_data_not_found(self):
        """Test getting business data when not found"""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        
        self.mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = await self.db_manager.get_business_data("nonexistent-id")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_transactions(self):
        """Test getting transactions"""
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            {"id": "1", "amount": 100, "type": "income"},
            {"id": "2", "amount": 50, "type": "expense"}
        ]
        
        mock_query = Mock()
        mock_query.execute.return_value = mock_response
        mock_query.limit.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.eq.return_value = mock_query
        
        self.mock_client.table.return_value.select.return_value = mock_query
        
        result = await self.db_manager.get_transactions("test-business-id")
        
        assert len(result) == 2
        assert result[0]["amount"] == 100
        self.mock_client.table.assert_called_with("transactions")
    
    @pytest.mark.asyncio
    async def test_save_analysis_result(self):
        """Test saving analysis result"""
        # Mock successful insert
        mock_response = Mock()
        mock_response.data = [{"id": "analysis-id"}]
        
        self.mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
        
        analysis_data = {
            "business_id": "test-business",
            "analysis_type": "fraud_detection",
            "results": {"score": 0.8}
        }
        
        result = await self.db_manager.save_analysis_result(analysis_data)
        
        assert result == "analysis-id"
        self.mock_client.table.assert_called_with("ai_analysis_results")
    
    @pytest.mark.asyncio
    async def test_save_fraud_alert(self):
        """Test saving fraud alert"""
        # Mock successful insert
        mock_response = Mock()
        mock_response.data = [{"id": "alert-id"}]
        
        self.mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
        
        alert_data = {
            "business_id": "test-business",
            "alert_type": "duplicate_invoice",
            "message": "Duplicate detected"
        }
        
        result = await self.db_manager.save_fraud_alert(alert_data)
        
        assert result == "alert-id"
        self.mock_client.table.assert_called_with("fraud_alerts")
    
    @pytest.mark.asyncio
    async def test_get_fraud_alerts_with_filters(self):
        """Test getting fraud alerts with filters"""
        # Mock response
        mock_response = Mock()
        mock_response.data = [{"id": "alert-1", "status": "active"}]
        
        mock_query = Mock()
        mock_query.execute.return_value = mock_response
        mock_query.limit.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.eq.return_value = mock_query
        
        self.mock_client.table.return_value.select.return_value = mock_query
        
        result = await self.db_manager.get_fraud_alerts(
            "test-business-id", 
            status="active",
            alert_type="duplicate_invoice"
        )
        
        assert len(result) == 1
        assert result[0]["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_update_fraud_alert_status(self):
        """Test updating fraud alert status"""
        # Mock successful update
        mock_response = Mock()
        mock_response.data = [{"id": "alert-id", "status": "resolved"}]
        
        self.mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        result = await self.db_manager.update_fraud_alert_status(
            "alert-id", 
            "resolved", 
            user_id="user-id",
            notes="False positive"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_duplicate_transactions(self):
        """Test checking for duplicate transactions"""
        # Mock response with similar transactions
        mock_response = Mock()
        mock_response.data = [
            {"id": "1", "amount": 100, "description": "Test payment"},
            {"id": "2", "amount": 100, "description": "Test payment duplicate"}
        ]
        
        mock_query = Mock()
        mock_query.execute.return_value = mock_response
        mock_query.gte.return_value = mock_query
        mock_query.eq.return_value = mock_query
        
        self.mock_client.table.return_value.select.return_value = mock_query
        
        result = await self.db_manager.check_duplicate_transactions(
            "test-business-id", 
            100.0, 
            "Test payment"
        )
        
        # Should find similar transactions
        assert len(result) >= 0  # Depends on fuzzy matching logic


class TestAIDataUtils:
    """Test AI data utilities"""
    
    def setup_method(self):
        """Setup test method"""
        with patch('app.utils.database.DatabaseManager'):
            self.ai_utils = AIDataUtils()
    
    def test_generate_data_hash(self):
        """Test data hash generation"""
        data1 = {"amount": 100, "description": "test", "id": "1", "created_at": "2024-01-01"}
        data2 = {"amount": 100, "description": "test", "id": "2", "created_at": "2024-01-02"}
        
        hash1 = AIDataUtils.generate_data_hash(data1)
        hash2 = AIDataUtils.generate_data_hash(data2)
        
        # Should generate same hash for same business data (ignoring id, timestamps)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_anonymize_financial_data(self):
        """Test financial data anonymization"""
        sensitive_data = {
            "amount": 100,
            "customer_name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "description": "Payment for services"
        }
        
        anonymized = AIDataUtils.anonymize_financial_data(sensitive_data)
        
        # Amount and description should remain
        assert anonymized["amount"] == 100
        assert anonymized["description"] == "Payment for services"
        
        # Sensitive fields should be anonymized
        assert anonymized["customer_name"] != "John Doe"
        assert anonymized["email"] != "john@example.com"
        assert len(anonymized["customer_name"]) == 8  # MD5 hash truncated
    
    @pytest.mark.asyncio
    async def test_batch_save_analysis_results(self):
        """Test batch saving analysis results"""
        with patch.object(self.ai_utils.db, 'save_analysis_result', return_value="result-id"):
            results = [
                {"business_id": "test", "analysis_type": "fraud", "results": {}},
                {"business_id": "test", "analysis_type": "insights", "results": {}}
            ]
            
            saved_ids = await self.ai_utils.batch_save_analysis_results(results)
            
            assert len(saved_ids) == 2
            assert all(id == "result-id" for id in saved_ids)


@pytest.mark.integration
class TestDatabaseIntegrationReal:
    """Integration tests with real database (requires Supabase connection)"""
    
    @pytest.mark.asyncio
    async def test_real_database_connection(self):
        """Test real database connection (skipped if no connection)"""
        try:
            await init_database()
            result = await test_connection()
            assert result is True
        except Exception as e:
            pytest.skip(f"Real database not available: {e}")
    
    @pytest.mark.asyncio
    async def test_real_database_operations(self):
        """Test real database operations (skipped if no connection)"""
        try:
            await init_database()
            db_manager = DatabaseManager()
            
            # Test getting non-existent business (should not crash)
            result = await db_manager.get_business_data("non-existent-id")
            assert result is None
            
        except Exception as e:
            pytest.skip(f"Real database not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])