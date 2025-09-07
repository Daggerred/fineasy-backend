"""
Comprehensive tests for NLP Invoice Generation Service
"""
import pytest
import asyncio
import spacy
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from app.services.nlp_invoice import NLPInvoiceGenerator
from app.models.responses import InvoiceGenerationResponse
from app.models.base import InvoiceRequest, InvoiceItem, ResolvedEntities


@pytest.fixture
def nlp_generator():
    """Create NLPInvoiceGenerator instance with mocked dependencies"""
    with patch('app.services.nlp_invoice.DatabaseManager') as mock_db_class:
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        # Mock spaCy model
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_spacy_load.return_value = mock_nlp
            
            generator = NLPInvoiceGenerator()
            return generator


@pytest.fixture
def sample_customers():
    """Sample customer data for entity resolution"""
    return [
        {'id': 'cust1', 'name': 'Rajesh Traders', 'email': 'rajesh@traders.com'},
        {'id': 'cust2', 'name': 'Priya Enterprises', 'email': 'priya@enterprises.com'},
        {'id': 'cust3', 'name': 'Mumbai Tech Solutions', 'email': 'info@mumbaitech.com'},
        {'id': 'cust4', 'name': 'Delhi Suppliers', 'email': 'contact@delhisuppliers.com'}
    ]


@pytest.fixture
def sample_products():
    """Sample product data for entity resolution"""
    return [
        {'id': 'prod1', 'name': 'Laptop Computer', 'price': 50000.0, 'unit': 'piece'},
        {'id': 'prod2', 'name': 'Office Chair', 'price': 8000.0, 'unit': 'piece'},
        {'id': 'prod3', 'name': 'Software License', 'price': 15000.0, 'unit': 'license'},
        {'id': 'prod4', 'name': 'Consulting Hours', 'price': 2000.0, 'unit': 'hour'}
    ]


class TestNLPInvoiceGenerator:
    """Test cases for NLP Invoice Generator"""
    
    # =============================================================================
    # Test Text Parsing and Entity Extraction
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_parse_invoice_request_simple(self, nlp_generator):
        """Test parsing simple invoice request"""
        text = "Generate an invoice for Rajesh Traders, 10 units of Laptop Computer"
        business_id = "business_123"
        
        # Mock spaCy processing
        mock_doc = Mock()
        mock_entities = [
            Mock(text="Rajesh Traders", label_="PERSON"),
            Mock(text="10", label_="CARDINAL"),
            Mock(text="Laptop Computer", label_="PRODUCT")
        ]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Test parsing
        request = await nlp_generator.parse_invoice_request(text, business_id)
        
        # Verify request structure
        assert isinstance(request, InvoiceRequest)
        assert request.raw_input == text
        assert request.business_id == business_id
        assert request.customer_name == "Rajesh Traders"
        assert len(request.items) == 1
        assert request.items[0].name == "Laptop Computer"
        assert request.items[0].quantity == 10
    
    @pytest.mark.asyncio
    async def test_parse_invoice_request_complex(self, nlp_generator):
        """Test parsing complex invoice request with multiple items"""
        text = "Create invoice for Mumbai Tech Solutions: 5 laptops at 50000 each, 10 office chairs at 8000 each, UPI payment preferred"
        business_id = "business_123"
        
        # Mock spaCy processing with complex entities
        mock_doc = Mock()
        mock_entities = [
            Mock(text="Mumbai Tech Solutions", label_="ORG"),
            Mock(text="5", label_="CARDINAL"),
            Mock(text="laptops", label_="PRODUCT"),
            Mock(text="50000", label_="MONEY"),
            Mock(text="10", label_="CARDINAL"),
            Mock(text="office chairs", label_="PRODUCT"),
            Mock(text="8000", label_="MONEY"),
            Mock(text="UPI", label_="PAYMENT")
        ]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Test parsing
        request = await nlp_generator.parse_invoice_request(text, business_id)
        
        # Verify complex parsing
        assert request.customer_name == "Mumbai Tech Solutions"
        assert len(request.items) == 2
        assert request.payment_preference == "UPI"
        
        # Verify first item
        assert request.items[0].name == "laptops"
        assert request.items[0].quantity == 5
        assert request.items[0].unit_price == 50000.0
        
        # Verify second item
        assert request.items[1].name == "office chairs"
        assert request.items[1].quantity == 10
        assert request.items[1].unit_price == 8000.0
    
    def test_extract_customer_entity(self, nlp_generator):
        """Test customer entity extraction"""
        # Test various customer name patterns
        test_cases = [
            ("Generate invoice for Rajesh Traders", "Rajesh Traders"),
            ("Bill to Priya Enterprises Ltd", "Priya Enterprises Ltd"),
            ("Customer: Mumbai Tech Solutions", "Mumbai Tech Solutions"),
            ("Invoice ABC Company", "ABC Company"),
            ("No customer mentioned", None)
        ]
        
        for text, expected in test_cases:
            # Mock spaCy entities
            if expected:
                mock_entities = [Mock(text=expected, label_="ORG")]
            else:
                mock_entities = []
            
            mock_doc = Mock()
            mock_doc.ents = mock_entities
            nlp_generator.nlp.return_value = mock_doc
            
            result = nlp_generator._extract_customer_entity(text)
            assert result == expected
    
    def test_extract_item_entities(self, nlp_generator):
        """Test item entity extraction"""
        text = "5 laptops, 10 chairs at 8000 each, 3 software licenses"
        
        # Mock spaCy processing
        mock_doc = Mock()
        mock_entities = [
            Mock(text="5", label_="CARDINAL"),
            Mock(text="laptops", label_="PRODUCT"),
            Mock(text="10", label_="CARDINAL"),
            Mock(text="chairs", label_="PRODUCT"),
            Mock(text="8000", label_="MONEY"),
            Mock(text="3", label_="CARDINAL"),
            Mock(text="software licenses", label_="PRODUCT")
        ]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Test item extraction
        items = nlp_generator._extract_item_entities(text)
        
        assert len(items) == 3
        
        # Verify first item
        assert items[0].name == "laptops"
        assert items[0].quantity == 5
        
        # Verify second item
        assert items[1].name == "chairs"
        assert items[1].quantity == 10
        assert items[1].unit_price == 8000.0
        
        # Verify third item
        assert items[2].name == "software licenses"
        assert items[2].quantity == 3
    
    def test_extract_payment_preference(self, nlp_generator):
        """Test payment preference extraction"""
        test_cases = [
            ("UPI payment preferred", "UPI"),
            ("Bank transfer required", "bank_transfer"),
            ("Cash payment", "cash"),
            ("Credit card payment", "credit_card"),
            ("Net banking", "net_banking"),
            ("No payment method mentioned", None)
        ]
        
        for text, expected in test_cases:
            result = nlp_generator._extract_payment_preference(text)
            assert result == expected
    
    def test_extract_quantities_and_prices(self, nlp_generator):
        """Test quantity and price extraction"""
        # Test various quantity/price patterns
        test_cases = [
            ("10 units at 500 each", [(10, 500.0)]),
            ("5 pieces for 2000 total", [(5, 400.0)]),  # 2000/5 = 400
            ("3 items", [(3, None)]),
            ("Price 1500", [(None, 1500.0)]),
            ("No numbers", [])
        ]
        
        for text, expected in test_cases:
            # Mock spaCy processing
            mock_doc = Mock()
            if "10 units at 500" in text:
                mock_entities = [
                    Mock(text="10", label_="CARDINAL"),
                    Mock(text="500", label_="MONEY")
                ]
            elif "5 pieces for 2000" in text:
                mock_entities = [
                    Mock(text="5", label_="CARDINAL"),
                    Mock(text="2000", label_="MONEY")
                ]
            elif "3 items" in text:
                mock_entities = [Mock(text="3", label_="CARDINAL")]
            elif "Price 1500" in text:
                mock_entities = [Mock(text="1500", label_="MONEY")]
            else:
                mock_entities = []
            
            mock_doc.ents = mock_entities
            nlp_generator.nlp.return_value = mock_doc
            
            result = nlp_generator._extract_quantities_and_prices(text)
            
            if expected:
                assert len(result) == len(expected)
                for i, (exp_qty, exp_price) in enumerate(expected):
                    if exp_qty is not None:
                        assert result[i][0] == exp_qty
                    if exp_price is not None:
                        assert abs(result[i][1] - exp_price) < 0.01
    
    # =============================================================================
    # Test Entity Resolution and Fuzzy Matching
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_resolve_entities_exact_match(self, nlp_generator, sample_customers, sample_products):
        """Test entity resolution with exact matches"""
        entities = {
            'customer_name': 'Rajesh Traders',
            'items': [
                InvoiceItem(name='Laptop Computer', quantity=5),
                InvoiceItem(name='Office Chair', quantity=2)
            ]
        }
        business_id = "business_123"
        
        # Mock database calls
        nlp_generator.db.get_customers = AsyncMock(return_value=sample_customers)
        nlp_generator.db.get_products = AsyncMock(return_value=sample_products)
        
        # Test entity resolution
        resolved = await nlp_generator.resolve_entities(entities, business_id)
        
        # Verify resolution
        assert isinstance(resolved, ResolvedEntities)
        assert resolved.customer_id == 'cust1'  # Rajesh Traders
        assert len(resolved.resolved_items) == 2
        
        # Verify first item resolution
        assert resolved.resolved_items[0].product_id == 'prod1'  # Laptop Computer
        assert resolved.resolved_items[0].unit_price == 50000.0
        
        # Verify second item resolution
        assert resolved.resolved_items[1].product_id == 'prod2'  # Office Chair
        assert resolved.resolved_items[1].unit_price == 8000.0
    
    @pytest.mark.asyncio
    async def test_resolve_entities_fuzzy_match(self, nlp_generator, sample_customers, sample_products):
        """Test entity resolution with fuzzy matching"""
        entities = {
            'customer_name': 'Rajesh Trader',  # Missing 's'
            'items': [
                InvoiceItem(name='Laptop', quantity=3),  # Partial match
                InvoiceItem(name='Chair Office', quantity=1)  # Word order different
            ]
        }
        business_id = "business_123"
        
        # Mock database calls
        nlp_generator.db.get_customers = AsyncMock(return_value=sample_customers)
        nlp_generator.db.get_products = AsyncMock(return_value=sample_products)
        
        # Test fuzzy resolution
        resolved = await nlp_generator.resolve_entities(entities, business_id)
        
        # Should resolve with fuzzy matching
        assert resolved.customer_id == 'cust1'  # Should match Rajesh Traders
        assert len(resolved.resolved_items) == 2
        
        # Should match products with fuzzy logic
        assert resolved.resolved_items[0].product_id == 'prod1'  # Laptop -> Laptop Computer
        assert resolved.resolved_items[1].product_id == 'prod2'  # Chair Office -> Office Chair
    
    @pytest.mark.asyncio
    async def test_resolve_entities_no_match(self, nlp_generator, sample_customers, sample_products):
        """Test entity resolution when no matches found"""
        entities = {
            'customer_name': 'Unknown Customer',
            'items': [
                InvoiceItem(name='Nonexistent Product', quantity=1)
            ]
        }
        business_id = "business_123"
        
        # Mock database calls
        nlp_generator.db.get_customers = AsyncMock(return_value=sample_customers)
        nlp_generator.db.get_products = AsyncMock(return_value=sample_products)
        
        # Test resolution with no matches
        resolved = await nlp_generator.resolve_entities(entities, business_id)
        
        # Should handle no matches gracefully
        assert resolved.customer_id is None
        assert len(resolved.unresolved_customers) == 1
        assert resolved.unresolved_customers[0] == 'Unknown Customer'
        
        assert len(resolved.unresolved_items) == 1
        assert resolved.unresolved_items[0].name == 'Nonexistent Product'
    
    def test_fuzzy_match_customer(self, nlp_generator, sample_customers):
        """Test fuzzy customer matching"""
        # Test exact match
        exact_match = nlp_generator._fuzzy_match_customer('Rajesh Traders', sample_customers)
        assert exact_match['id'] == 'cust1'
        
        # Test fuzzy match
        fuzzy_match = nlp_generator._fuzzy_match_customer('Rajesh Trader', sample_customers)
        assert fuzzy_match['id'] == 'cust1'  # Should match despite missing 's'
        
        # Test partial match
        partial_match = nlp_generator._fuzzy_match_customer('Mumbai Tech', sample_customers)
        assert partial_match['id'] == 'cust3'  # Should match Mumbai Tech Solutions
        
        # Test no match
        no_match = nlp_generator._fuzzy_match_customer('Completely Different Name', sample_customers)
        assert no_match is None
    
    def test_fuzzy_match_product(self, nlp_generator, sample_products):
        """Test fuzzy product matching"""
        # Test exact match
        exact_match = nlp_generator._fuzzy_match_product('Laptop Computer', sample_products)
        assert exact_match['id'] == 'prod1'
        
        # Test partial match
        partial_match = nlp_generator._fuzzy_match_product('Laptop', sample_products)
        assert partial_match['id'] == 'prod1'
        
        # Test word order different
        order_match = nlp_generator._fuzzy_match_product('Chair Office', sample_products)
        assert order_match['id'] == 'prod2'  # Should match Office Chair
        
        # Test no match
        no_match = nlp_generator._fuzzy_match_product('Nonexistent Item', sample_products)
        assert no_match is None
    
    def test_calculate_similarity_score(self, nlp_generator):
        """Test similarity score calculation"""
        # Test identical strings
        identical_score = nlp_generator._calculate_similarity_score('Laptop Computer', 'Laptop Computer')
        assert identical_score == 1.0
        
        # Test similar strings
        similar_score = nlp_generator._calculate_similarity_score('Laptop Computer', 'Laptop')
        assert 0.5 < similar_score < 1.0
        
        # Test different strings
        different_score = nlp_generator._calculate_similarity_score('Laptop', 'Chair')
        assert different_score < 0.5
        
        # Test empty strings
        empty_score = nlp_generator._calculate_similarity_score('', 'Laptop')
        assert empty_score == 0.0
    
    # =============================================================================
    # Test Invoice Construction and Validation
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_generate_invoice_success(self, nlp_generator):
        """Test successful invoice generation"""
        request = InvoiceRequest(
            raw_input="Generate invoice for Rajesh Traders, 5 laptops",
            business_id="business_123",
            customer_name="Rajesh Traders",
            items=[
                InvoiceItem(name="Laptop Computer", quantity=5, unit_price=50000.0)
            ],
            payment_preference="UPI"
        )
        
        # Mock successful invoice creation
        invoice_data = {
            'id': 'inv_123',
            'invoice_number': 'INV-001',
            'customer_name': 'Rajesh Traders',
            'items': [
                {
                    'name': 'Laptop Computer',
                    'quantity': 5,
                    'unit_price': 50000.0,
                    'total': 250000.0
                }
            ],
            'subtotal': 250000.0,
            'tax_amount': 45000.0,
            'total_amount': 295000.0,
            'payment_preference': 'UPI'
        }
        
        nlp_generator.db.create_invoice = AsyncMock(return_value=invoice_data)
        
        # Test invoice generation
        response = await nlp_generator.generate_invoice(request)
        
        # Verify response
        assert isinstance(response, InvoiceGenerationResponse)
        assert response.success is True
        assert response.invoice_id == 'inv_123'
        assert response.invoice_data == invoice_data
        assert len(response.errors) == 0
    
    @pytest.mark.asyncio
    async def test_generate_invoice_validation_errors(self, nlp_generator):
        """Test invoice generation with validation errors"""
        request = InvoiceRequest(
            raw_input="Generate invoice",
            business_id="business_123",
            customer_name=None,  # Missing customer
            items=[],  # No items
            payment_preference=None
        )
        
        # Test validation
        response = await nlp_generator.generate_invoice(request)
        
        # Should have validation errors
        assert response.success is False
        assert len(response.errors) > 0
        assert any("customer" in error.lower() for error in response.errors)
        assert any("item" in error.lower() for error in response.errors)
    
    def test_validate_invoice_request(self, nlp_generator):
        """Test invoice request validation"""
        # Test valid request
        valid_request = InvoiceRequest(
            raw_input="Valid request",
            business_id="business_123",
            customer_name="Test Customer",
            items=[
                InvoiceItem(name="Test Product", quantity=1, unit_price=100.0)
            ]
        )
        
        errors = nlp_generator._validate_invoice_request(valid_request)
        assert len(errors) == 0
        
        # Test invalid request
        invalid_request = InvoiceRequest(
            raw_input="Invalid request",
            business_id="business_123",
            customer_name=None,
            items=[]
        )
        
        errors = nlp_generator._validate_invoice_request(invalid_request)
        assert len(errors) > 0
        assert any("customer" in error.lower() for error in errors)
        assert any("item" in error.lower() for error in errors)
    
    def test_calculate_invoice_totals(self, nlp_generator):
        """Test invoice total calculations"""
        items = [
            InvoiceItem(name="Item 1", quantity=2, unit_price=1000.0),
            InvoiceItem(name="Item 2", quantity=3, unit_price=500.0)
        ]
        
        # Test calculation
        subtotal, tax_amount, total = nlp_generator._calculate_invoice_totals(items)
        
        assert subtotal == 3500.0  # (2*1000) + (3*500)
        assert tax_amount == 630.0  # 18% GST
        assert total == 4130.0  # subtotal + tax
    
    def test_generate_invoice_number(self, nlp_generator):
        """Test invoice number generation"""
        business_id = "business_123"
        
        # Mock database call
        nlp_generator.db.get_next_invoice_number = AsyncMock(return_value="INV-001")
        
        # Test generation
        invoice_number = nlp_generator._generate_invoice_number(business_id)
        
        assert invoice_number.startswith("INV-")
        assert len(invoice_number) > 4
    
    def test_create_upi_payment_link(self, nlp_generator):
        """Test UPI payment link creation"""
        invoice_data = {
            'total_amount': 1000.0,
            'invoice_number': 'INV-001',
            'customer_name': 'Test Customer'
        }
        business_upi = "business@upi"
        
        # Test UPI link creation
        upi_link = nlp_generator._create_upi_payment_link(invoice_data, business_upi)
        
        assert upi_link.startswith("upi://pay")
        assert "pa=business@upi" in upi_link
        assert "am=1000.0" in upi_link
        assert "tn=INV-001" in upi_link
    
    # =============================================================================
    # Test Error Handling and Edge Cases
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_parse_empty_input(self, nlp_generator):
        """Test parsing empty or invalid input"""
        empty_inputs = ["", "   ", None]
        
        for empty_input in empty_inputs:
            if empty_input is None:
                continue
                
            request = await nlp_generator.parse_invoice_request(empty_input, "business_123")
            
            assert request.raw_input == empty_input
            assert request.customer_name is None
            assert len(request.items) == 0
    
    @pytest.mark.asyncio
    async def test_resolve_entities_database_error(self, nlp_generator):
        """Test entity resolution with database errors"""
        entities = {
            'customer_name': 'Test Customer',
            'items': [InvoiceItem(name='Test Product', quantity=1)]
        }
        
        # Mock database error
        nlp_generator.db.get_customers = AsyncMock(side_effect=Exception("Database error"))
        nlp_generator.db.get_products = AsyncMock(side_effect=Exception("Database error"))
        
        # Test error handling
        resolved = await nlp_generator.resolve_entities(entities, "business_123")
        
        # Should handle errors gracefully
        assert resolved.customer_id is None
        assert len(resolved.unresolved_customers) == 1
        assert len(resolved.unresolved_items) == 1
    
    @pytest.mark.asyncio
    async def test_generate_invoice_database_error(self, nlp_generator):
        """Test invoice generation with database error"""
        request = InvoiceRequest(
            raw_input="Test request",
            business_id="business_123",
            customer_name="Test Customer",
            items=[InvoiceItem(name="Test Product", quantity=1, unit_price=100.0)]
        )
        
        # Mock database error
        nlp_generator.db.create_invoice = AsyncMock(side_effect=Exception("Database error"))
        
        # Test error handling
        response = await nlp_generator.generate_invoice(request)
        
        # Should return error response
        assert response.success is False
        assert len(response.errors) > 0
        assert any("database" in error.lower() for error in response.errors)
    
    def test_handle_special_characters(self, nlp_generator):
        """Test handling of special characters in input"""
        text_with_special_chars = "Generate invoice for Rajesh & Co., 5 items @ ₹1,000 each"
        
        # Mock spaCy processing
        mock_doc = Mock()
        mock_entities = [
            Mock(text="Rajesh & Co.", label_="ORG"),
            Mock(text="5", label_="CARDINAL"),
            Mock(text="1,000", label_="MONEY")
        ]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Should handle special characters without errors
        customer = nlp_generator._extract_customer_entity(text_with_special_chars)
        assert customer == "Rajesh & Co."
        
        # Should parse currency symbols
        quantities_prices = nlp_generator._extract_quantities_and_prices(text_with_special_chars)
        assert len(quantities_prices) > 0
    
    def test_handle_multilingual_input(self, nlp_generator):
        """Test handling of multilingual input"""
        hindi_text = "राजेश ट्रेडर्स के लिए इनवॉइस बनाएं"
        
        # Mock spaCy processing for Hindi text
        mock_doc = Mock()
        mock_entities = [Mock(text="राजेश ट्रेडर्स", label_="PERSON")]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Should handle non-English text gracefully
        customer = nlp_generator._extract_customer_entity(hindi_text)
        assert customer == "राजेश ट्रेडर्स"


class TestNLPInvoiceIntegration:
    """Integration tests for NLP Invoice Generation"""
    
    @pytest.mark.asyncio
    async def test_full_invoice_generation_pipeline(self, nlp_generator, sample_customers, sample_products):
        """Test complete invoice generation pipeline"""
        text = "Generate invoice for Rajesh Traders, 2 laptops at 50000 each, UPI payment"
        business_id = "business_123"
        
        # Mock spaCy processing
        mock_doc = Mock()
        mock_entities = [
            Mock(text="Rajesh Traders", label_="ORG"),
            Mock(text="2", label_="CARDINAL"),
            Mock(text="laptops", label_="PRODUCT"),
            Mock(text="50000", label_="MONEY"),
            Mock(text="UPI", label_="PAYMENT")
        ]
        mock_doc.ents = mock_entities
        nlp_generator.nlp.return_value = mock_doc
        
        # Mock database calls
        nlp_generator.db.get_customers = AsyncMock(return_value=sample_customers)
        nlp_generator.db.get_products = AsyncMock(return_value=sample_products)
        nlp_generator.db.create_invoice = AsyncMock(return_value={
            'id': 'inv_123',
            'invoice_number': 'INV-001',
            'total_amount': 118000.0  # 100000 + 18% tax
        })
        
        # Run complete pipeline
        request = await nlp_generator.parse_invoice_request(text, business_id)
        resolved = await nlp_generator.resolve_entities(request.__dict__, business_id)
        
        # Update request with resolved entities
        request.customer_id = resolved.customer_id
        request.items = resolved.resolved_items
        
        response = await nlp_generator.generate_invoice(request)
        
        # Verify complete pipeline
        assert response.success is True
        assert response.invoice_id == 'inv_123'
        assert response.invoice_data['total_amount'] == 118000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])