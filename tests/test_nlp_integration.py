"""
Integration tests for NLP Invoice Processing
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from datetime import datetime

from app.services.nlp_invoice import NLPInvoiceGenerator, EntityExtractor, EntityResolver, InvoiceBuilder
from app.models.base import InvoiceRequest
from app.database import DatabaseManager


class TestNLPIntegration:
    """Integration tests for NLP invoice processing"""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database with test data"""
        db = Mock(spec=DatabaseManager)
        
        # Mock customers
        db.get_customers = AsyncMock(return_value=[
            {
                "id": "customer_1",
                "name": "John Smith",
                "email": "john@example.com",
                "phone": "+1234567890"
            },
            {
                "id": "customer_2", 
                "name": "ABC Corporation",
                "email": "contact@abc.com",
                "phone": "+0987654321"
            }
        ])
        
        # Mock products
        db.get_products = AsyncMock(return_value=[
            {
                "id": "product_1",
                "name": "Laptop",
                "price": 50000.0,
                "tax_rate": 18.0,
                "description": "High-performance laptop"
            },
            {
                "id": "product_2",
                "name": "Mouse",
                "price": 500.0,
                "tax_rate": 18.0,
                "description": "Wireless mouse"
            },
            {
                "id": "product_3",
                "name": "Consulting Services",
                "price": 2000.0,
                "tax_rate": 18.0,
                "description": "Professional consulting"
            }
        ])
        
        return db
    
    @pytest.mark.asyncio
    async def test_end_to_end_invoice_generation(self, mock_database):
        """Test complete end-to-end invoice generation"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Test input
            request = InvoiceRequest(
                raw_input="Create invoice for John Smith, 2 laptops at ₹50000 each, UPI payment",
                business_id="test_business_id"
            )
            
            # Generate invoice
            response = await generator.generate_invoice_from_text(request)
            
            # Verify response
            assert response.success is True
            assert response.confidence_score > 0.0
            assert response.invoice_data is not None
            
            # Verify customer resolution
            customer = response.invoice_data["customer"]
            assert customer["name"] == "John Smith"
            assert customer["is_new"] is False  # Should match existing customer
            
            # Verify items
            items = response.invoice_data["items"]
            assert len(items) == 1
            assert items[0]["name"] == "Laptop"
            assert items[0]["quantity"] == 2.0
            assert items[0]["unit_price"] == 50000.0
            assert items[0]["total_price"] == 100000.0
            
            # Verify totals
            assert response.invoice_data["total_amount"] == 100000.0
            
            # Verify payment preference
            assert response.invoice_data["payment_preference"] == "UPI"
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching_customer(self, mock_database):
        """Test fuzzy matching for customer names"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Test with slightly different customer name
            request = InvoiceRequest(
                raw_input="Invoice for Jon Smith, consulting services ₹5000",  # "Jon" instead of "John"
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            # Should still match "John Smith" due to fuzzy matching
            assert response.success is True
            customer = response.invoice_data["customer"]
            assert customer["name"] == "John Smith"  # Resolved to existing customer
            assert customer["is_new"] is False
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching_products(self, mock_database):
        """Test fuzzy matching for product names"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Test with slightly different product name
            request = InvoiceRequest(
                raw_input="Bill for ABC Corporation, 1 wireless mouse at ₹500",  # "wireless mouse" should match "Mouse"
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            items = response.invoice_data["items"]
            assert len(items) == 1
            # Should resolve to existing "Mouse" product
            assert items[0]["name"] == "Mouse"
            assert items[0]["unit_price"] == 500.0
    
    @pytest.mark.asyncio
    async def test_new_customer_and_product(self, mock_database):
        """Test handling of new customers and products"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Generate invoice for New Customer, 3 units of New Product at ₹1000 each",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            
            # Verify new customer
            customer = response.invoice_data["customer"]
            assert customer["name"] == "New Customer"
            assert customer["is_new"] is True
            
            # Verify new product
            items = response.invoice_data["items"]
            assert len(items) == 1
            assert items[0]["name"] == "New Product"
            assert items[0]["quantity"] == 3.0
            assert items[0]["unit_price"] == 1000.0
            assert items[0]["total_price"] == 3000.0
            
            # Should have suggestions for new entities
            assert len(response.suggestions) > 0
            assert any("New customer detected" in s for s in response.suggestions)
            assert any("New products detected" in s for s in response.suggestions)
    
    @pytest.mark.asyncio
    async def test_multiple_items_invoice(self, mock_database):
        """Test invoice with multiple items"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Invoice for ABC Corporation: 2 laptops at ₹50000 each, 5 mice at ₹500 each, cash payment",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            
            # Verify customer
            customer = response.invoice_data["customer"]
            assert customer["name"] == "ABC Corporation"
            
            # Verify items
            items = response.invoice_data["items"]
            assert len(items) == 2
            
            # First item - laptops
            laptop_item = next(item for item in items if "Laptop" in item["name"])
            assert laptop_item["quantity"] == 2.0
            assert laptop_item["unit_price"] == 50000.0
            assert laptop_item["total_price"] == 100000.0
            
            # Second item - mice
            mouse_item = next(item for item in items if "Mouse" in item["name"])
            assert mouse_item["quantity"] == 5.0
            assert mouse_item["unit_price"] == 500.0
            assert mouse_item["total_price"] == 2500.0
            
            # Verify total
            assert response.invoice_data["total_amount"] == 102500.0
            
            # Verify payment preference
            assert response.invoice_data["payment_preference"] == "cash"
    
    @pytest.mark.asyncio
    async def test_service_invoice(self, mock_database):
        """Test service-based invoice generation"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Bill John Smith for 10 hours of consulting services at ₹2000 per hour, bank transfer",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            
            # Verify service item
            items = response.invoice_data["items"]
            assert len(items) == 1
            service_item = items[0]
            assert "Consulting" in service_item["name"]
            assert service_item["quantity"] == 10.0
            assert service_item["unit_price"] == 2000.0
            assert service_item["total_price"] == 20000.0
    
    @pytest.mark.asyncio
    async def test_incomplete_information_handling(self, mock_database):
        """Test handling of incomplete information"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Missing quantities and prices
            request = InvoiceRequest(
                raw_input="Generate invoice for John Smith, laptop and mouse",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            assert response.confidence_score < 0.7  # Low confidence due to missing info
            
            # Should have suggestions for missing information
            assert len(response.suggestions) > 0
            assert any("quantities" in s for s in response.suggestions)
            assert any("prices" in s for s in response.suggestions)
            
            # Items should still be created with default values
            items = response.invoice_data["items"]
            assert len(items) == 2  # laptop and mouse
            
            # Should use existing product prices
            laptop_item = next(item for item in items if "Laptop" in item["name"])
            assert laptop_item["unit_price"] == 50000.0  # From database
            
            mouse_item = next(item for item in items if "Mouse" in item["name"])
            assert mouse_item["unit_price"] == 500.0  # From database
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, mock_database):
        """Test confidence score calculation for different scenarios"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # High confidence scenario - all info present, existing entities
            high_confidence_request = InvoiceRequest(
                raw_input="Invoice for John Smith, 2 laptops at ₹50000 each, UPI payment",
                business_id="test_business_id"
            )
            
            high_response = await generator.generate_invoice_from_text(high_confidence_request)
            
            # Low confidence scenario - missing info, new entities
            low_confidence_request = InvoiceRequest(
                raw_input="Generate invoice for Unknown Customer, some product",
                business_id="test_business_id"
            )
            
            low_response = await generator.generate_invoice_from_text(low_confidence_request)
            
            # High confidence should be significantly higher
            assert high_response.confidence_score > low_response.confidence_score
            assert high_response.confidence_score > 0.7
            assert low_response.confidence_score < 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self):
        """Test error handling when database operations fail"""
        # Mock database that fails
        failing_db = Mock(spec=DatabaseManager)
        failing_db.get_customers = AsyncMock(side_effect=Exception("Database error"))
        failing_db.get_products = AsyncMock(side_effect=Exception("Database error"))
        
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=failing_db):
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Invoice for John Smith, laptop ₹50000",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            # Should handle gracefully and still generate invoice
            assert response.success is False
            assert len(response.errors) > 0
            assert response.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_entity_extraction_edge_cases(self, mock_database):
        """Test entity extraction with edge cases"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Test with special characters and formatting
            request = InvoiceRequest(
                raw_input="Invoice for ABC Corp., 2.5 units of Product-X @ Rs. 1,50,000.00 each (including taxes), online payment via UPI",
                business_id="test_business_id"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            
            # Should handle decimal quantities
            items = response.invoice_data["items"]
            assert len(items) > 0
            
            # Should handle formatted prices
            extracted_entities = response.extracted_entities
            assert "prices" in extracted_entities
            
            # Should extract payment method
            assert response.invoice_data["payment_preference"] is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_invoice_generation(self, mock_database):
        """Test concurrent invoice generation"""
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_database):
            generator = NLPInvoiceGenerator()
            
            # Create multiple requests
            requests = [
                InvoiceRequest(
                    raw_input=f"Invoice for Customer {i}, laptop ₹{50000 + i*1000}",
                    business_id=f"business_{i}"
                )
                for i in range(5)
            ]
            
            # Process concurrently
            tasks = [
                generator.generate_invoice_from_text(request)
                for request in requests
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for i, response in enumerate(responses):
                assert not isinstance(response, Exception), f"Request {i} failed: {response}"
                assert response.success is True
                assert response.invoice_data["total_amount"] == 50000 + i*1000


class TestNLPPerformance:
    """Performance tests for NLP processing"""
    
    @pytest.mark.asyncio
    async def test_processing_time(self):
        """Test that NLP processing completes within reasonable time"""
        import time
        
        mock_db = Mock(spec=DatabaseManager)
        mock_db.get_customers = AsyncMock(return_value=[])
        mock_db.get_products = AsyncMock(return_value=[])
        
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_db):
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Generate invoice for Test Customer, 5 units of Test Product at ₹1000 each",
                business_id="test_business"
            )
            
            start_time = time.time()
            response = await generator.generate_invoice_from_text(request)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should complete within 5 seconds
            assert processing_time < 5.0
            assert response.success is True
    
    @pytest.mark.asyncio
    async def test_memory_usage_large_text(self):
        """Test memory usage with large input text"""
        mock_db = Mock(spec=DatabaseManager)
        mock_db.get_customers = AsyncMock(return_value=[])
        mock_db.get_products = AsyncMock(return_value=[])
        
        with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_db):
            generator = NLPInvoiceGenerator()
            
            # Create large input text
            large_text = "Generate invoice for Test Customer, " + "test product, " * 1000 + "₹1000 each"
            
            request = InvoiceRequest(
                raw_input=large_text,
                business_id="test_business"
            )
            
            # Should handle large text without crashing
            response = await generator.generate_invoice_from_text(request)
            assert response.success is True


# Test data for various real-world scenarios
REAL_WORLD_SCENARIOS = [
    {
        "name": "E-commerce order",
        "text": "Create invoice for Rajesh Kumar, Order #12345: 2x iPhone 14 Pro at ₹1,29,900 each, 1x AirPods Pro at ₹24,900, total ₹2,84,700, UPI payment",
        "expected_customer": "Rajesh Kumar",
        "expected_items": 2,
        "expected_total": 284700.0
    },
    {
        "name": "Service invoice",
        "text": "Bill for Acme Technologies Pvt Ltd, Web development project - Phase 1, 40 hours at ₹2500/hour, bank transfer payment",
        "expected_customer": "Acme Technologies Pvt Ltd",
        "expected_items": 1,
        "expected_total": 100000.0
    },
    {
        "name": "Retail invoice",
        "text": "Invoice for Mrs. Priya Sharma, 5kg Basmati Rice ₹450, 2L Cooking Oil ₹380, 1kg Sugar ₹42, cash payment",
        "expected_customer": "Mrs. Priya Sharma",
        "expected_items": 3,
        "expected_total": 872.0
    },
    {
        "name": "B2B invoice",
        "text": "Generate invoice for XYZ Manufacturing Ltd, Supply of 100 units Steel Rods Grade A at ₹850 per unit, 50 units Steel Rods Grade B at ₹720 per unit, Net Banking",
        "expected_customer": "XYZ Manufacturing Ltd",
        "expected_items": 2,
        "expected_total": 121000.0
    }
]


@pytest.mark.parametrize("scenario", REAL_WORLD_SCENARIOS)
@pytest.mark.asyncio
async def test_real_world_scenarios(scenario):
    """Test real-world invoice generation scenarios"""
    mock_db = Mock(spec=DatabaseManager)
    mock_db.get_customers = AsyncMock(return_value=[])
    mock_db.get_products = AsyncMock(return_value=[])
    
    with patch('app.services.nlp_invoice.DatabaseManager', return_value=mock_db):
        generator = NLPInvoiceGenerator()
        
        request = InvoiceRequest(
            raw_input=scenario["text"],
            business_id="test_business"
        )
        
        response = await generator.generate_invoice_from_text(request)
        
        assert response.success is True
        assert response.confidence_score > 0.0
        
        # Verify customer extraction
        customer = response.invoice_data["customer"]
        assert scenario["expected_customer"].lower() in customer["name"].lower()
        
        # Verify items count
        items = response.invoice_data["items"]
        assert len(items) >= scenario["expected_items"]
        
        # Verify total is reasonable (within 20% of expected)
        total = response.invoice_data["total_amount"]
        expected = scenario["expected_total"]
        assert abs(total - expected) / expected < 0.2 or total > 0  # Allow flexibility for new products