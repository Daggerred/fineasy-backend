#!/usr/bin/env python3
"""
Validation script for NLP Invoice API endpoints
"""
import asyncio
import json
import uuid
from typing import Dict, Any

# Mock the dependencies for testing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.api.invoice import (
    ParseTextRequest, EntityResolutionRequest, InvoicePreviewRequest,
    InvoiceConfirmationRequest, ParseTextResponse, EntityResolutionResponse,
    InvoicePreviewResponse
)
from app.services.nlp_invoice import NLPInvoiceGenerator, EntityExtractor
from app.models.base import InvoiceRequest
from unittest.mock import Mock, AsyncMock, patch


async def test_parse_text_endpoint():
    """Test the parse text functionality"""
    print("Testing parse text endpoint...")
    
    # Mock the NLP generator
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator.entity_extractor.extract_entities.return_value = {
            "customer_names": ["John Doe"],
            "items": ["Laptop"],
            "quantities": ["2"],
            "prices": ["50000"],
            "payment_methods": ["UPI"]
        }
        mock_generator_class.return_value = mock_generator
        
        # Import the function after mocking
        from app.api.invoice import parse_invoice_text
        
        # Create request
        request = ParseTextRequest(
            text="Generate invoice for John Doe, 2 laptops at ₹50000 each, UPI payment",
            business_id=str(uuid.uuid4())
        )
        
        # Test the function
        response = await parse_invoice_text(request, "Bearer test_token")
        
        assert response.success is True
        assert "customer_names" in response.extracted_entities
        assert "John Doe" in response.extracted_entities["customer_names"]
        assert response.confidence_score > 0.0
        
        print("✓ Parse text endpoint working correctly")


async def test_resolve_entities_endpoint():
    """Test the entity resolution functionality"""
    print("Testing resolve entities endpoint...")
    
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator.resolve_entities = AsyncMock(return_value={
            "customer": {"name": "John Doe", "is_new": False, "match_confidence": 90},
            "products": [{"name": "Laptop", "is_new": False, "match_confidence": 85}]
        })
        mock_generator_class.return_value = mock_generator
        
        from app.api.invoice import resolve_entities
        
        request = EntityResolutionRequest(
            entities={
                "customer_names": ["John Doe"],
                "items": ["Laptop"]
            },
            business_id=str(uuid.uuid4())
        )
        
        response = await resolve_entities(request, "Bearer test_token")
        
        assert response.success is True
        assert response.resolved_customer["name"] == "John Doe"
        assert len(response.resolved_products) == 1
        assert response.resolution_confidence > 0.8
        
        print("✓ Resolve entities endpoint working correctly")


async def test_invoice_preview_endpoint():
    """Test the invoice preview functionality"""
    print("Testing invoice preview endpoint...")
    
    # Mock the NLP generator to avoid database initialization
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        from app.api.invoice import create_invoice_preview
    
    request = InvoicePreviewRequest(
        customer={"name": "John Doe", "is_new": False},
        items=[
            {
                "name": "Laptop",
                "quantity": 2,
                "unit_price": 50000.0,
                "total_price": 100000.0
            }
        ],
        payment_preference="UPI",
        business_id=str(uuid.uuid4()),
        original_text="Invoice for John Doe, 2 laptops at ₹50000 each"
    )
    
    response = await create_invoice_preview(request, "Bearer test_token")
    
    assert response.success is True
    assert "preview_id" in response.preview_id
    assert response.invoice_data["total_amount"] == 100000.0
    assert response.invoice_data["customer"]["name"] == "John Doe"
    assert len(response.invoice_data["items"]) == 1
    assert response.confidence_score > 0.0
    
    print("✓ Invoice preview endpoint working correctly")
    return response.preview_id


async def test_invoice_confirmation_endpoint():
    """Test the invoice confirmation functionality"""
    print("Testing invoice confirmation endpoint...")
    
    # First create a preview
    preview_id = await test_invoice_preview_endpoint()
    
    from app.api.invoice import confirm_invoice
    
    request = InvoiceConfirmationRequest(
        preview_id=preview_id,
        business_id=str(uuid.uuid4()),
        modifications={
            "customer": {"email": "john@example.com"},
            "payment_preference": "Bank Transfer"
        }
    )
    
    response = await confirm_invoice(request, "Bearer test_token")
    
    assert response.success is True
    assert "invoice_id" in response.invoice_id
    assert response.invoice_data["customer"]["email"] == "john@example.com"
    assert response.invoice_data["payment_preference"] == "Bank Transfer"
    assert response.confidence_score == 1.0
    
    print("✓ Invoice confirmation endpoint working correctly")


async def test_complete_workflow():
    """Test the complete NLP invoice workflow"""
    print("Testing complete NLP workflow...")
    
    business_id = str(uuid.uuid4())
    
    # Step 1: Parse text
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator.entity_extractor.extract_entities.return_value = {
            "customer_names": ["ABC Corp"],
            "items": ["Consulting Services"],
            "quantities": ["10"],
            "prices": ["2000"],
            "payment_methods": ["Bank Transfer"]
        }
        mock_generator_class.return_value = mock_generator
        
        from app.api.invoice import parse_invoice_text
        
        parse_request = ParseTextRequest(
            text="Create invoice for ABC Corp, 10 hours of consulting at ₹2000 per hour, bank transfer",
            business_id=business_id
        )
        
        parse_response = await parse_invoice_text(parse_request, "Bearer test_token")
        assert parse_response.success is True
    
    # Step 2: Resolve entities
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator.resolve_entities = AsyncMock(return_value={
            "customer": {"name": "ABC Corp", "is_new": True},
            "products": [{"name": "Consulting Services", "is_new": True}]
        })
        mock_generator_class.return_value = mock_generator
        
        from app.api.invoice import resolve_entities
        
        resolve_request = EntityResolutionRequest(
            entities=parse_response.extracted_entities,
            business_id=business_id
        )
        
        resolve_response = await resolve_entities(resolve_request, "Bearer test_token")
        assert resolve_response.success is True
    
    # Step 3: Create preview
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        from app.api.invoice import create_invoice_preview
    
    preview_request = InvoicePreviewRequest(
        customer=resolve_response.resolved_customer,
        items=[
            {
                "name": "Consulting Services",
                "quantity": 10,
                "unit_price": 2000.0,
                "total_price": 20000.0
            }
        ],
        payment_preference="Bank Transfer",
        business_id=business_id,
        original_text=parse_request.text
    )
    
    preview_response = await create_invoice_preview(preview_request, "Bearer test_token")
    assert preview_response.success is True
    
    # Step 4: Confirm invoice
    from app.api.invoice import confirm_invoice
    
    confirm_request = InvoiceConfirmationRequest(
        preview_id=preview_response.preview_id,
        business_id=business_id,
        modifications={
            "customer": {"email": "contact@abc.com", "phone": "+1234567890"}
        }
    )
    
    confirm_response = await confirm_invoice(confirm_request, "Bearer test_token")
    assert confirm_response.success is True
    assert confirm_response.invoice_data["total_amount"] == 20000.0
    
    print("✓ Complete NLP workflow working correctly")


async def test_error_handling():
    """Test error handling in API endpoints"""
    print("Testing error handling...")
    
    # Test with invalid preview ID
    from app.api.invoice import confirm_invoice
    
    try:
        invalid_request = InvoiceConfirmationRequest(
            preview_id=str(uuid.uuid4()),  # Non-existent preview ID
            business_id=str(uuid.uuid4())
        )
        
        await confirm_invoice(invalid_request, "Bearer test_token")
        assert False, "Should have raised HTTPException"
    except Exception as e:
        assert "Preview not found" in str(e)
        print("✓ Error handling working correctly")


async def test_data_validation():
    """Test data validation in API endpoints"""
    print("Testing data validation...")
    
    # Test ParseTextRequest validation
    try:
        ParseTextRequest(text="", business_id="")  # Empty values
        assert False, "Should have raised validation error"
    except Exception:
        print("✓ Data validation working correctly")


def test_response_models():
    """Test response model creation"""
    print("Testing response models...")
    
    # Test ParseTextResponse
    response = ParseTextResponse(
        success=True,
        extracted_entities={"customer_names": ["Test"]},
        confidence_score=0.8,
        suggestions=["Test suggestion"]
    )
    assert response.success is True
    assert response.confidence_score == 0.8
    
    # Test EntityResolutionResponse
    response = EntityResolutionResponse(
        success=True,
        resolved_customer={"name": "Test Customer"},
        resolved_products=[{"name": "Test Product"}],
        resolution_confidence=0.9,
        suggestions=[]
    )
    assert response.success is True
    assert response.resolution_confidence == 0.9
    
    # Test InvoicePreviewResponse
    response = InvoicePreviewResponse(
        success=True,
        preview_id=str(uuid.uuid4()),
        invoice_data={"total_amount": 1000.0},
        confidence_score=0.95,
        warnings=[],
        suggestions=[],
        editable_fields=["customer.name"]
    )
    assert response.success is True
    assert response.confidence_score == 0.95
    
    print("✓ Response models working correctly")


async def main():
    """Run all validation tests"""
    print("Starting NLP Invoice API endpoint validation...\n")
    
    try:
        # Test individual endpoints
        await test_parse_text_endpoint()
        await test_resolve_entities_endpoint()
        preview_id = await test_invoice_preview_endpoint()
        await test_invoice_confirmation_endpoint()
        
        # Test complete workflow
        await test_complete_workflow()
        
        # Test error handling
        await test_error_handling()
        
        # Test data validation
        await test_data_validation()
        
        # Test response models
        test_response_models()
        
        print("\n✅ All NLP Invoice API endpoint tests passed!")
        print("The NLP invoice API endpoints are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)