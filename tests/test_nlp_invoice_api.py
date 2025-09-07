"""
Tests for NLP Invoice API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json
import uuid

from app.main import app
from app.models.base import InvoiceRequest, InvoiceItem
from app.models.responses import InvoiceGenerationResponse


class TestNLPInvoiceAPI:
    """Test NLP Invoice API endpoints"""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.test_token = "Bearer test_token"
        self.test_business_id = str(uuid.uuid4())
    
    def test_parse_invoice_text_success(self):
        """Test successful text parsing"""
        request_data = {
            "text": "Generate invoice for John Doe, 5 units of Widget A at ₹100 each, UPI payment",
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.entity_extractor.extract_entities.return_value = {
                "customer_names": ["John Doe"],
                "items": ["Widget A"],
                "quantities": ["5"],
                "prices": ["100"],
                "payment_methods": ["UPI"]
            }
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/parse",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "extracted_entities" in data
            assert "confidence_score" in data
            assert "suggestions" in data
            assert data["extracted_entities"]["customer_names"] == ["John Doe"]
    
    def test_parse_invoice_text_missing_entities(self):
        """Test text parsing with missing entities"""
        request_data = {
            "text": "Generate invoice",
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.entity_extractor.extract_entities.return_value = {
                "customer_names": [],
                "items": [],
                "quantities": [],
                "prices": [],
                "payment_methods": []
            }
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/parse",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["confidence_score"] == 0.0  # No required entities found
            assert len(data["suggestions"]) > 0
    
    def test_resolve_entities_success(self):
        """Test successful entity resolution"""
        request_data = {
            "entities": {
                "customer_names": ["John Doe"],
                "items": ["Widget A"]
            },
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.resolve_entities = AsyncMock(return_value={
                "customer": {"name": "John Doe", "is_new": False, "match_confidence": 90},
                "products": [{"name": "Widget A", "is_new": False, "match_confidence": 85}]
            })
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/resolve-entities",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["resolved_customer"]["name"] == "John Doe"
            assert len(data["resolved_products"]) == 1
            assert data["resolution_confidence"] > 0.8
    
    def test_resolve_entities_new_customer_and_products(self):
        """Test entity resolution with new customer and products"""
        request_data = {
            "entities": {
                "customer_names": ["New Customer"],
                "items": ["New Product"]
            },
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.resolve_entities = AsyncMock(return_value={
                "customer": {"name": "New Customer", "is_new": True},
                "products": [{"name": "New Product", "is_new": True}]
            })
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/resolve-entities",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["resolved_customer"]["is_new"] is True
            assert data["resolved_products"][0]["is_new"] is True
            assert "New customer detected" in " ".join(data["suggestions"])
    
    def test_create_invoice_preview_success(self):
        """Test successful invoice preview creation"""
        request_data = {
            "customer": {"name": "John Doe", "is_new": False},
            "items": [
                {
                    "name": "Widget A",
                    "quantity": 2,
                    "unit_price": 100.0,
                    "total_price": 200.0
                }
            ],
            "payment_preference": "UPI",
            "business_id": self.test_business_id,
            "original_text": "Invoice for John Doe, 2 Widget A at ₹100 each"
        }
        
        response = self.client.post(
            "/invoice/preview",
            json=request_data,
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preview_id" in data
        assert data["invoice_data"]["total_amount"] == 200.0
        assert data["invoice_data"]["customer"]["name"] == "John Doe"
        assert len(data["invoice_data"]["items"]) == 1
        assert data["confidence_score"] > 0.0
    
    def test_create_invoice_preview_missing_data(self):
        """Test invoice preview with missing data"""
        request_data = {
            "items": [
                {
                    "name": "Widget A",
                    "quantity": 2,
                    "unit_price": 0.0  # Missing price
                }
            ],
            "business_id": self.test_business_id
        }
        
        response = self.client.post(
            "/invoice/preview",
            json=request_data,
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["warnings"]) > 0
        assert any("No price specified" in warning for warning in data["warnings"])
        assert data["confidence_score"] < 0.7
    
    def test_confirm_invoice_success(self):
        """Test successful invoice confirmation"""
        # First create a preview
        preview_request = {
            "customer": {"name": "John Doe"},
            "items": [{"name": "Widget A", "quantity": 2, "unit_price": 100.0}],
            "business_id": self.test_business_id
        }
        
        preview_response = self.client.post(
            "/invoice/preview",
            json=preview_request,
            headers={"Authorization": self.test_token}
        )
        
        preview_id = preview_response.json()["preview_id"]
        
        # Now confirm the invoice
        confirm_request = {
            "preview_id": preview_id,
            "business_id": self.test_business_id,
            "modifications": {
                "customer": {"email": "john@example.com"},
                "payment_preference": "Bank Transfer"
            }
        }
        
        response = self.client.post(
            "/invoice/confirm",
            json=confirm_request,
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "invoice_id" in data
        assert data["invoice_data"]["customer"]["email"] == "john@example.com"
        assert data["invoice_data"]["payment_preference"] == "Bank Transfer"
        assert data["confidence_score"] == 1.0
    
    def test_confirm_invoice_not_found(self):
        """Test invoice confirmation with invalid preview ID"""
        confirm_request = {
            "preview_id": str(uuid.uuid4()),
            "business_id": self.test_business_id
        }
        
        response = self.client.post(
            "/invoice/confirm",
            json=confirm_request,
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 404
        assert "Preview not found" in response.json()["detail"]
    
    def test_get_invoice_preview_success(self):
        """Test retrieving invoice preview"""
        # First create a preview
        preview_request = {
            "customer": {"name": "John Doe"},
            "items": [{"name": "Widget A", "quantity": 1, "unit_price": 100.0}],
            "business_id": self.test_business_id
        }
        
        preview_response = self.client.post(
            "/invoice/preview",
            json=preview_request,
            headers={"Authorization": self.test_token}
        )
        
        preview_id = preview_response.json()["preview_id"]
        
        # Now retrieve the preview
        response = self.client.get(
            f"/invoice/preview/{preview_id}",
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["preview_id"] == preview_id
        assert data["invoice_data"]["customer"]["name"] == "John Doe"
    
    def test_get_invoice_preview_not_found(self):
        """Test retrieving non-existent preview"""
        response = self.client.get(
            f"/invoice/preview/{uuid.uuid4()}",
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 404
        assert "Preview not found" in response.json()["detail"]
    
    def test_cancel_invoice_preview_success(self):
        """Test cancelling invoice preview"""
        # First create a preview
        preview_request = {
            "customer": {"name": "John Doe"},
            "items": [{"name": "Widget A", "quantity": 1, "unit_price": 100.0}],
            "business_id": self.test_business_id
        }
        
        preview_response = self.client.post(
            "/invoice/preview",
            json=preview_request,
            headers={"Authorization": self.test_token}
        )
        
        preview_id = preview_response.json()["preview_id"]
        
        # Now cancel the preview
        response = self.client.delete(
            f"/invoice/preview/{preview_id}",
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cancelled successfully" in data["message"]
        
        # Verify preview is deleted
        get_response = self.client.get(
            f"/invoice/preview/{preview_id}",
            headers={"Authorization": self.test_token}
        )
        assert get_response.status_code == 404
    
    def test_generate_invoice_from_text_complete_workflow(self):
        """Test complete invoice generation workflow"""
        request_data = {
            "raw_input": "Generate invoice for ABC Corp, 3 units of Product X at ₹500 each, UPI payment",
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_response = InvoiceGenerationResponse(
                success=True,
                message="Invoice generated successfully",
                invoice_data={
                    "customer": {"name": "ABC Corp"},
                    "items": [{"name": "Product X", "quantity": 3, "unit_price": 500.0, "total_price": 1500.0}],
                    "total_amount": 1500.0,
                    "payment_preference": "UPI"
                },
                confidence_score=0.9,
                suggestions=["Invoice generated successfully"]
            )
            mock_generator.generate_invoice_from_text = AsyncMock(return_value=mock_response)
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/generate",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["confidence_score"] == 0.9
            assert data["invoice_data"]["total_amount"] == 1500.0
    
    def test_api_error_handling(self):
        """Test API error handling"""
        request_data = {
            "text": "Generate invoice",
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.entity_extractor.extract_entities.side_effect = Exception("Test error")
            mock_generator_class.return_value = mock_generator
            
            response = self.client.post(
                "/invoice/parse",
                json=request_data,
                headers={"Authorization": self.test_token}
            )
            
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]
    
    def test_authentication_required(self):
        """Test that authentication is required for all endpoints"""
        request_data = {
            "text": "Generate invoice",
            "business_id": self.test_business_id
        }
        
        # Test without authorization header
        response = self.client.post("/invoice/parse", json=request_data)
        assert response.status_code == 403
        
        # Test with invalid token format
        response = self.client.post(
            "/invoice/parse",
            json=request_data,
            headers={"Authorization": "invalid_token"}
        )
        assert response.status_code == 422


class TestNLPInvoiceWorkflows:
    """Test complete NLP invoice workflows"""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.test_token = "Bearer test_token"
        self.test_business_id = str(uuid.uuid4())
    
    def test_complete_nlp_workflow(self):
        """Test complete NLP invoice generation workflow"""
        # Step 1: Parse text
        parse_request = {
            "text": "Create invoice for John Smith, 2 laptops at ₹50000 each, bank transfer",
            "business_id": self.test_business_id
        }
        
        with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
            mock_generator = Mock()
            
            # Mock entity extraction
            mock_generator.entity_extractor.extract_entities.return_value = {
                "customer_names": ["John Smith"],
                "items": ["laptops"],
                "quantities": ["2"],
                "prices": ["50000"],
                "payment_methods": ["bank transfer"]
            }
            
            # Mock entity resolution
            mock_generator.resolve_entities = AsyncMock(return_value={
                "customer": {"name": "John Smith", "is_new": True},
                "products": [{"name": "laptops", "is_new": True}]
            })
            
            mock_generator_class.return_value = mock_generator
            
            # Parse text
            parse_response = self.client.post(
                "/invoice/parse",
                json=parse_request,
                headers={"Authorization": self.test_token}
            )
            
            assert parse_response.status_code == 200
            parse_data = parse_response.json()
            
            # Step 2: Resolve entities
            resolve_request = {
                "entities": parse_data["extracted_entities"],
                "business_id": self.test_business_id
            }
            
            resolve_response = self.client.post(
                "/invoice/resolve-entities",
                json=resolve_request,
                headers={"Authorization": self.test_token}
            )
            
            assert resolve_response.status_code == 200
            resolve_data = resolve_response.json()
            
            # Step 3: Create preview
            preview_request = {
                "customer": resolve_data["resolved_customer"],
                "items": [
                    {
                        "name": "laptops",
                        "quantity": 2,
                        "unit_price": 50000.0,
                        "total_price": 100000.0
                    }
                ],
                "payment_preference": "bank transfer",
                "business_id": self.test_business_id,
                "original_text": parse_request["text"]
            }
            
            preview_response = self.client.post(
                "/invoice/preview",
                json=preview_request,
                headers={"Authorization": self.test_token}
            )
            
            assert preview_response.status_code == 200
            preview_data = preview_response.json()
            
            # Step 4: Confirm invoice
            confirm_request = {
                "preview_id": preview_data["preview_id"],
                "business_id": self.test_business_id,
                "modifications": {
                    "customer": {"email": "john@example.com", "phone": "+1234567890"}
                }
            }
            
            confirm_response = self.client.post(
                "/invoice/confirm",
                json=confirm_request,
                headers={"Authorization": self.test_token}
            )
            
            assert confirm_response.status_code == 200
            confirm_data = confirm_response.json()
            
            # Verify final invoice
            assert confirm_data["success"] is True
            assert "invoice_id" in confirm_data
            assert confirm_data["invoice_data"]["customer"]["email"] == "john@example.com"
            assert confirm_data["invoice_data"]["total_amount"] == 100000.0
    
    def test_workflow_with_corrections(self):
        """Test workflow with user corrections at preview stage"""
        # Create preview with incorrect data
        preview_request = {
            "customer": {"name": "John Doe"},
            "items": [
                {
                    "name": "Widget",
                    "quantity": 1,
                    "unit_price": 100.0,
                    "total_price": 100.0
                }
            ],
            "business_id": self.test_business_id
        }
        
        preview_response = self.client.post(
            "/invoice/preview",
            json=preview_request,
            headers={"Authorization": self.test_token}
        )
        
        preview_id = preview_response.json()["preview_id"]
        
        # Confirm with corrections
        confirm_request = {
            "preview_id": preview_id,
            "business_id": self.test_business_id,
            "modifications": {
                "customer": {
                    "name": "John Smith",  # Corrected name
                    "email": "john.smith@example.com"
                },
                "items": [
                    {
                        "name": "Premium Widget",  # Corrected name
                        "quantity": 2,  # Corrected quantity
                        "unit_price": 150.0,  # Corrected price
                        "total_price": 300.0
                    }
                ],
                "payment_preference": "Credit Card"
            }
        }
        
        confirm_response = self.client.post(
            "/invoice/confirm",
            json=confirm_request,
            headers={"Authorization": self.test_token}
        )
        
        assert confirm_response.status_code == 200
        data = confirm_response.json()
        
        # Verify corrections were applied
        assert data["invoice_data"]["customer"]["name"] == "John Smith"
        assert data["invoice_data"]["customer"]["email"] == "john.smith@example.com"
        assert data["invoice_data"]["items"][0]["name"] == "Premium Widget"
        assert data["invoice_data"]["items"][0]["quantity"] == 2
        assert data["invoice_data"]["total_amount"] == 300.0
        assert data["invoice_data"]["payment_preference"] == "Credit Card"


# Test data for various scenarios
TEST_INVOICE_SCENARIOS = [
    {
        "name": "Simple invoice",
        "text": "Invoice for John Doe, 1 widget at ₹100",
        "expected_entities": ["customer_names", "items", "quantities", "prices"]
    },
    {
        "name": "Multiple items",
        "text": "Bill for ABC Corp: 5 laptops at ₹50000 each, 3 mice at ₹500 each",
        "expected_entities": ["customer_names", "items", "quantities", "prices"]
    },
    {
        "name": "With payment method",
        "text": "Generate invoice for XYZ Ltd, consulting services ₹10000, UPI payment",
        "expected_entities": ["customer_names", "items", "prices", "payment_methods"]
    },
    {
        "name": "Service invoice",
        "text": "Create bill for Mary Johnson, web development ₹25000, bank transfer",
        "expected_entities": ["customer_names", "items", "prices", "payment_methods"]
    }
]


@pytest.mark.parametrize("scenario", TEST_INVOICE_SCENARIOS)
def test_parse_various_invoice_formats(scenario):
    """Test parsing various invoice text formats"""
    client = TestClient(app)
    test_token = "Bearer test_token"
    test_business_id = str(uuid.uuid4())
    
    request_data = {
        "text": scenario["text"],
        "business_id": test_business_id
    }
    
    with patch('app.api.invoice.NLPInvoiceGenerator') as mock_generator_class:
        mock_generator = Mock()
        
        # Mock entity extraction to return expected entities
        mock_entities = {}
        for entity_type in scenario["expected_entities"]:
            mock_entities[entity_type] = ["mock_value"]
        
        mock_generator.entity_extractor.extract_entities.return_value = mock_entities
        mock_generator_class.return_value = mock_generator
        
        response = client.post(
            "/invoice/parse",
            json=request_data,
            headers={"Authorization": test_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify expected entities are present
        for entity_type in scenario["expected_entities"]:
            assert entity_type in data["extracted_entities"]