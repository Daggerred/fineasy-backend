#!/usr/bin/env python3
"""
Simple test script for NLP invoice generation service
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.nlp_invoice import EntityExtractor, NLPInvoiceGenerator
from app.models.base import InvoiceRequest


async def test_entity_extraction():
    """Test entity extraction functionality"""
    print("Testing Entity Extraction...")
    
    extractor = EntityExtractor()
    
    test_texts = [
        "Generate invoice for Rajesh Traders, 10 units of Widget A at ₹500 each, UPI payment",
        "Bill ABC Corp for 5 pieces of Product X, Rs.200 per unit, cash payment",
        "Create invoice for John Doe, 2 widgets at ₹150 each"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        entities = extractor.extract_entities(text)
        
        print(f"  Customer names: {entities.get('customer_names', [])}")
        print(f"  Items: {entities.get('items', [])}")
        print(f"  Quantities: {entities.get('quantities', [])}")
        print(f"  Prices: {entities.get('prices', [])}")
        print(f"  Payment methods: {entities.get('payment_methods', [])}")
        print(f"  Money: {entities.get('money', [])}")


async def test_nlp_service():
    """Test NLP invoice generation service"""
    print("\n" + "="*50)
    print("Testing NLP Invoice Generation Service...")
    
    # Mock the database manager to avoid connection issues
    from unittest.mock import Mock, AsyncMock
    
    # Create a mock generator with mocked database
    generator = NLPInvoiceGenerator()
    generator.db = Mock()
    generator.db.get_customers = AsyncMock(return_value=[
        {"id": "1", "name": "Rajesh Traders", "email": "rajesh@example.com"}
    ])
    generator.db.get_products = AsyncMock(return_value=[
        {"id": "1", "name": "Widget A", "price": 500.0}
    ])
    
    # Test invoice generation
    request = InvoiceRequest(
        raw_input="Generate invoice for Rajesh Traders, 10 units of Widget A at ₹500 each, UPI payment",
        business_id="test_business"
    )
    
    try:
        response = await generator.generate_invoice_from_text(request)
        
        print(f"Success: {response.success}")
        print(f"Message: {response.message}")
        print(f"Confidence Score: {response.confidence_score}")
        
        if response.invoice_data:
            print(f"Customer: {response.invoice_data.get('customer', {}).get('name', 'Unknown')}")
            print(f"Items: {len(response.invoice_data.get('items', []))}")
            print(f"Total Amount: ₹{response.invoice_data.get('total_amount', 0)}")
            print(f"Payment Method: {response.invoice_data.get('payment_preference', 'Not specified')}")
        
        if response.suggestions:
            print(f"Suggestions: {response.suggestions}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all tests"""
    print("NLP Invoice Generation Service Test")
    print("=" * 50)
    
    await test_entity_extraction()
    await test_nlp_service()
    
    print("\n" + "="*50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())