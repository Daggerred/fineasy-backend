"""
Tests for NLP Invoice Generation Service
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.services.nlp_invoice import (
    NLPInvoiceGenerator, EntityExtractor, EntityResolver, InvoiceBuilder
)
from app.models.base import InvoiceRequest, InvoiceItem
from app.models.responses import InvoiceGenerationResponse


class TestEntityExtractor:
    """Test entity extraction functionality"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
    
    def test_extract_with_regex_fallback(self):
        """Test entity extraction using regex patterns when spaCy is not available"""
        # Mock spaCy not being available
        self.extractor.nlp = None
        
        text = "Generate invoice for Rajesh Traders, 10 units of Widget A at ₹500 each, UPI payment"
        entities = self.extractor.extract_entities(text)
        
        assert "money" in entities
        assert "quantities" in entities
        assert "payment_methods" in entities
        assert any("500" in money for money in entities["money"])
        assert "10" in entities["quantities"]
        assert "UPI" in entities["payment_methods"]
    
    def test_extract_custom_patterns(self):
        """Test custom pattern extraction"""
        text = "Invoice for ABC Corp, 5 pieces of Product X at Rs.200 per unit, cash payment"
        patterns = self.extractor._extract_custom_patterns(text)
        
        assert "5" in patterns["quantities"]
        assert "200" in patterns["prices"]
        assert "cash" in patterns["payment_methods"]
    
    def test_extract_multiple_items(self):
        """Test extraction of multiple items and quantities"""
        text = "Bill for 3 units of Item A at ₹100, 2 pieces of Item B at ₹250"
        entities = self.extractor.extract_entities(text)
        
        assert len(entities["quantities"]) >= 2
        assert len(entities["prices"]) >= 2
    
    @patch('spacy.load')
    def test_extract_with_spacy(self, mock_spacy_load):
        """Test entity extraction with spaCy"""
        # Mock spaCy NLP pipeline
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.text = "John Doe"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 8
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        text = "Invoice for John Doe"
        entities = extractor.extract_entities(text)
        
        assert "customer_names" in entities
        assert "John Doe" in entities["customer_names"]


class TestEntityResolver:
    """Test entity resolution functionality"""
    
    def setup_method(self):
        self.mock_db = Mock()
        self.resolver = EntityResolver(self.mock_db)
    
    @pytest.mark.asyncio
    async def test_resolve_customer_existing(self):
        """Test resolving existing customer with fuzzy matching"""
        # Mock existing customers
        self.mock_db.get_customers = AsyncMock(return_value=[
            {"id": "1", "name": "John Smith", "email": "john@example.com"},
            {"id": "2", "name": "Jane Doe", "email": "jane@example.com"}
        ])
        
        # Test fuzzy matching
        result = await self.resolver.resolve_customer(["Jon Smith"], "business_id")
        
        assert result is not None
        assert result["name"] == "John Smith"
        assert result["is_new"] is False
        assert result["match_confidence"] >= 80
    
    @pytest.mark.asyncio
    async def test_resolve_customer_new(self):
        """Test resolving new customer"""
        self.mock_db.get_customers = AsyncMock(return_value=[])
        
        result = await self.resolver.resolve_customer(["New Customer"], "business_id")
        
        assert result is not None
        assert result["name"] == "New Customer"
        assert result["is_new"] is True
    
    @pytest.mark.asyncio
    async def test_resolve_products_existing(self):
        """Test resolving existing products"""
        self.mock_db.get_products = AsyncMock(return_value=[
            {"id": "1", "name": "Widget A", "price": 100.0},
            {"id": "2", "name": "Widget B", "price": 200.0}
        ])
        
        result = await self.resolver.resolve_products(["Widget A"], "business_id")
        
        assert len(result) == 1
        assert result[0]["name"] == "Widget A"
        assert result[0]["is_new"] is False
    
    @pytest.mark.asyncio
    async def test_resolve_products_new(self):
        """Test resolving new products"""
        self.mock_db.get_products = AsyncMock(return_value=[])
        
        result = await self.resolver.resolve_products(["New Product"], "business_id")
        
        assert len(result) == 1
        assert result[0]["name"] == "New Product"
        assert result[0]["is_new"] is True


class TestInvoiceBuilder:
    """Test invoice building functionality"""
    
    def setup_method(self):
        self.builder = InvoiceBuilder()
    
    def test_build_invoice_items(self):
        """Test building invoice items from entities"""
        entities = {
            "quantities": ["2", "3"],
            "prices": ["100", "200"]
        }
        resolved_products = [
            {"name": "Product A", "is_new": False},
            {"name": "Product B", "is_new": False}
        ]
        
        items = self.builder.build_invoice_items(entities, resolved_products)
        
        assert len(items) == 2
        assert items[0].name == "Product A"
        assert items[0].quantity == 2.0
        assert items[0].unit_price == 100.0
        assert items[0].total_price == 200.0
    
    def test_build_invoice_items_missing_data(self):
        """Test building items with missing quantities or prices"""
        entities = {"quantities": ["1"]}  # Missing prices
        resolved_products = [{"name": "Product A", "is_new": True}]
        
        items = self.builder.build_invoice_items(entities, resolved_products)
        
        assert len(items) == 1
        assert items[0].quantity == 1.0
        assert items[0].unit_price == 0.0
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        entities = {
            "customer_names": ["John Doe"],
            "items": ["Product A"],
            "quantities": ["2"]
        }
        resolved_customer = {"name": "John Doe", "is_new": False, "match_confidence": 90}
        resolved_products = [{"name": "Product A", "is_new": False, "match_confidence": 85}]
        
        score = self.builder.calculate_confidence_score(entities, resolved_customer, resolved_products)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high confidence


class TestNLPInvoiceGenerator:
    """Test main NLP invoice generator"""
    
    def setup_method(self):
        with patch('app.services.nlp_invoice.DatabaseManager'):
            self.generator = NLPInvoiceGenerator()
            self.generator.db = Mock()
    
    @pytest.mark.asyncio
    async def test_parse_invoice_request(self):
        """Test parsing invoice request from text"""
        text = "Generate invoice for John Doe, 5 units of Product A, UPI payment"
        
        with patch.object(self.generator.entity_extractor, 'extract_entities') as mock_extract:
            mock_extract.return_value = {
                "customer_names": ["John Doe"],
                "items": ["Product A"],
                "quantities": ["5"],
                "payment_methods": ["UPI"]
            }
            
            request = await self.generator.parse_invoice_request(text, "business_id")
            
            assert request.raw_input == text
            assert request.business_id == "business_id"
            assert request.customer_name == "John Doe"
            assert request.payment_preference == "UPI"
    
    @pytest.mark.asyncio
    async def test_resolve_entities(self):
        """Test entity resolution"""
        entities = {
            "customer_names": ["John Doe"],
            "items": ["Product A"]
        }
        
        with patch.object(self.generator.entity_resolver, 'resolve_customer') as mock_resolve_customer, \
             patch.object(self.generator.entity_resolver, 'resolve_products') as mock_resolve_products:
            
            mock_resolve_customer.return_value = {"name": "John Doe", "is_new": False}
            mock_resolve_products.return_value = [{"name": "Product A", "is_new": False}]
            
            resolved = await self.generator.resolve_entities(entities, "business_id")
            
            assert "customer" in resolved
            assert "products" in resolved
            assert resolved["customer"]["name"] == "John Doe"
            assert len(resolved["products"]) == 1
    
    @pytest.mark.asyncio
    async def test_generate_invoice_from_text_success(self):
        """Test successful invoice generation from text"""
        request = InvoiceRequest(
            raw_input="Invoice for John Doe, 2 units of Widget A at ₹100 each",
            business_id="business_id"
        )
        
        # Mock all the dependencies
        with patch.object(self.generator, 'parse_invoice_request') as mock_parse, \
             patch.object(self.generator.entity_extractor, 'extract_entities') as mock_extract, \
             patch.object(self.generator, 'resolve_entities') as mock_resolve, \
             patch.object(self.generator.invoice_builder, 'build_invoice_items') as mock_build, \
             patch.object(self.generator.invoice_builder, 'calculate_confidence_score') as mock_confidence:
            
            mock_parse.return_value = request
            mock_extract.return_value = {
                "customer_names": ["John Doe"],
                "items": ["Widget A"],
                "quantities": ["2"],
                "prices": ["100"],
                "payment_methods": ["UPI"]
            }
            mock_resolve.return_value = {
                "customer": {"name": "John Doe", "is_new": False},
                "products": [{"name": "Widget A", "is_new": False}]
            }
            mock_build.return_value = [
                InvoiceItem(name="Widget A", quantity=2.0, unit_price=100.0, total_price=200.0)
            ]
            mock_confidence.return_value = 0.85
            
            response = await self.generator.generate_invoice_from_text(request)
            
            assert response.success is True
            assert response.confidence_score == 0.85
            assert response.invoice_data is not None
            assert response.invoice_data["total_amount"] == 200.0
    
    @pytest.mark.asyncio
    async def test_generate_invoice_from_text_error(self):
        """Test error handling in invoice generation"""
        request = InvoiceRequest(
            raw_input="Invalid input",
            business_id="business_id"
        )
        
        # Mock an exception
        with patch.object(self.generator, 'parse_invoice_request', side_effect=Exception("Test error")):
            response = await self.generator.generate_invoice_from_text(request)
            
            assert response.success is False
            assert "Test error" in response.errors
            assert response.confidence_score == 0.0
    
    def test_generate_suggestions(self):
        """Test suggestion generation"""
        entities = {"customer_names": [], "quantities": [], "prices": []}
        resolved_entities = {
            "customer": {"is_new": True},
            "products": [{"name": "New Product", "is_new": True}]
        }
        confidence_score = 0.5
        
        suggestions = self.generator._generate_suggestions(entities, resolved_entities, confidence_score)
        
        assert len(suggestions) > 0
        assert any("customer name" in s for s in suggestions)
        assert any("quantities" in s for s in suggestions)
        assert any("New customer detected" in s for s in suggestions)


class TestIntegration:
    """Integration tests for NLP invoice generation"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_invoice_generation(self):
        """Test complete invoice generation workflow"""
        with patch('app.services.nlp_invoice.DatabaseManager') as mock_db_class:
            # Setup mock database
            mock_db = Mock()
            mock_db.get_customers = AsyncMock(return_value=[
                {"id": "1", "name": "John Smith", "email": "john@example.com"}
            ])
            mock_db.get_products = AsyncMock(return_value=[
                {"id": "1", "name": "Widget A", "price": 100.0}
            ])
            mock_db_class.return_value = mock_db
            
            generator = NLPInvoiceGenerator()
            
            request = InvoiceRequest(
                raw_input="Create invoice for John Smith, 3 units of Widget A at ₹100 each, UPI payment",
                business_id="test_business"
            )
            
            response = await generator.generate_invoice_from_text(request)
            
            assert response.success is True
            assert response.confidence_score > 0.0
            assert response.invoice_data is not None
            assert response.invoice_data["customer"]["name"] == "John Smith"
            assert len(response.invoice_data["items"]) == 1
            assert response.invoice_data["items"][0]["name"] == "Widget A"
            assert response.invoice_data["total_amount"] == 300.0


# Test data for various invoice generation scenarios
TEST_INVOICE_TEXTS = [
    "Generate invoice for Rajesh Traders, 10 units of Item A at ₹500 each, UPI payment",
    "Bill ABC Corp for 5 pieces of Product X, Rs.200 per unit, cash payment",
    "Create invoice for John Doe, 2 widgets at ₹150 each",
    "Invoice for 3 units of Service A, ₹1000 total, bank transfer",
    "Generate bill for Mary Johnson, consulting services ₹5000, online payment"
]


@pytest.mark.parametrize("text", TEST_INVOICE_TEXTS)
def test_entity_extraction_various_formats(text):
    """Test entity extraction with various text formats"""
    extractor = EntityExtractor()
    entities = extractor.extract_entities(text)
    
    # Should extract at least some entities
    assert any(entities.values()), f"No entities extracted from: {text}"