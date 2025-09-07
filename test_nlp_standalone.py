#!/usr/bin/env python3
"""
Standalone test for NLP invoice generation service components
"""
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.nlp_invoice import EntityExtractor, EntityResolver, InvoiceBuilder
from app.models.base import InvoiceItem


def test_entity_extraction():
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


async def test_entity_resolver():
    """Test entity resolution functionality"""
    print("\n" + "="*50)
    print("Testing Entity Resolution...")
    
    # Mock database manager
    mock_db = Mock()
    mock_db.get_customers = AsyncMock(return_value=[
        {"id": "1", "name": "Rajesh Traders", "email": "rajesh@example.com"},
        {"id": "2", "name": "ABC Corp", "email": "abc@example.com"}
    ])
    mock_db.get_products = AsyncMock(return_value=[
        {"id": "1", "name": "Widget A", "price": 500.0},
        {"id": "2", "name": "Product X", "price": 200.0}
    ])
    
    resolver = EntityResolver(mock_db)
    
    # Test customer resolution
    customer_result = await resolver.resolve_customer(["Rajesh Traders"], "business_id")
    print(f"Customer resolution: {customer_result}")
    
    # Test fuzzy customer matching
    fuzzy_customer = await resolver.resolve_customer(["Rajesh Trading"], "business_id")
    print(f"Fuzzy customer match: {fuzzy_customer}")
    
    # Test product resolution
    product_result = await resolver.resolve_products(["Widget A", "Product X"], "business_id")
    print(f"Product resolution: {product_result}")


def test_invoice_builder():
    """Test invoice building functionality"""
    print("\n" + "="*50)
    print("Testing Invoice Builder...")
    
    builder = InvoiceBuilder()
    
    # Test data
    entities = {
        "quantities": ["10", "5"],
        "prices": ["500", "200"]
    }
    resolved_products = [
        {"name": "Widget A", "is_new": False, "price": 500.0},
        {"name": "Product X", "is_new": False, "price": 200.0}
    ]
    
    # Build invoice items
    items = builder.build_invoice_items(entities, resolved_products)
    
    print(f"Generated {len(items)} invoice items:")
    for i, item in enumerate(items, 1):
        print(f"  Item {i}: {item.name} - Qty: {item.quantity}, Price: ₹{item.unit_price}, Total: ₹{item.total_price}")
    
    # Test confidence calculation
    resolved_customer = {"name": "Rajesh Traders", "is_new": False, "match_confidence": 95}
    confidence = builder.calculate_confidence_score(entities, resolved_customer, resolved_products)
    print(f"Confidence Score: {confidence:.2f}")


def test_spacy_integration():
    """Test spaCy integration"""
    print("\n" + "="*50)
    print("Testing spaCy Integration...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        text = "Generate invoice for John Smith, 5 units of Product A at $100 each"
        doc = nlp(text)
        
        print(f"Text: {text}")
        print("Named Entities:")
        for ent in doc.ents:
            print(f"  {ent.text} ({ent.label_})")
        
        print("spaCy integration working correctly!")
        
    except Exception as e:
        print(f"spaCy integration error: {e}")


async def main():
    """Run all tests"""
    print("NLP Invoice Generation Service - Standalone Tests")
    print("=" * 60)
    
    test_entity_extraction()
    await test_entity_resolver()
    test_invoice_builder()
    test_spacy_integration()
    
    print("\n" + "="*60)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())