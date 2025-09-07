"""
NLP Invoice Generation API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import uuid

from ..models.base import InvoiceRequest, InvoiceItem
from ..models.responses import InvoiceGenerationResponse
from ..services.nlp_invoice import NLPInvoiceGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/invoice", tags=["invoice"])
security = HTTPBearer()


class ParseTextRequest(BaseModel):
    """Request model for parsing invoice text"""
    text: str = Field(..., description="Natural language text to parse")
    business_id: str = Field(..., description="Business ID for context")


class EntityResolutionRequest(BaseModel):
    """Request model for entity resolution"""
    entities: Dict[str, Any] = Field(..., description="Extracted entities to resolve")
    business_id: str = Field(..., description="Business ID for context")


class InvoicePreviewRequest(BaseModel):
    """Request model for invoice preview"""
    customer: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    payment_preference: Optional[str] = None
    business_id: str = Field(..., description="Business ID")
    original_text: Optional[str] = None


class InvoiceConfirmationRequest(BaseModel):
    """Request model for invoice confirmation"""
    preview_id: str = Field(..., description="Preview ID to confirm")
    modifications: Optional[Dict[str, Any]] = None
    business_id: str = Field(..., description="Business ID")


class ParseTextResponse(BaseModel):
    """Response model for text parsing"""
    success: bool = True
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class EntityResolutionResponse(BaseModel):
    """Response model for entity resolution"""
    success: bool = True
    resolved_customer: Optional[Dict[str, Any]] = None
    resolved_products: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_confidence: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class InvoicePreviewResponse(BaseModel):
    """Response model for invoice preview"""
    success: bool = True
    preview_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invoice_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    editable_fields: List[str] = Field(default_factory=list)
    message: Optional[str] = None


# Store preview data temporarily (in production, use Redis or database)
_preview_cache: Dict[str, Dict[str, Any]] = {}


@router.post("/generate", response_model=InvoiceGenerationResponse)
async def generate_invoice_from_text(
    request: InvoiceRequest,
    token: str = Depends(security)
):
    """Generate invoice from natural language text (complete workflow)"""
    try:
        logger.info(f"Generating invoice from text for business: {request.business_id}")
        generator = NLPInvoiceGenerator()
        result = await generator.generate_invoice_from_text(request)
        return result
    except Exception as e:
        logger.error(f"Invoice generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse", response_model=ParseTextResponse)
async def parse_invoice_text(
    request: ParseTextRequest,
    token: str = Depends(security)
):
    """Parse natural language text for invoice entities"""
    try:
        logger.info(f"Parsing invoice text for business: {request.business_id}")
        generator = NLPInvoiceGenerator()
        
        # Extract entities from text
        entities = generator.entity_extractor.extract_entities(request.text)
        
        # Calculate confidence based on entity completeness
        required_entities = ["customer_names", "items"]
        found_entities = sum(1 for entity in required_entities if entities.get(entity))
        confidence_score = found_entities / len(required_entities)
        
        # Generate suggestions for improvement
        suggestions = []
        if not entities.get("customer_names"):
            suggestions.append("Consider specifying the customer name more clearly")
        if not entities.get("items"):
            suggestions.append("Please specify the items or services to include")
        if not entities.get("quantities"):
            suggestions.append("Consider adding quantities for better accuracy")
        if not entities.get("prices"):
            suggestions.append("Including prices will improve invoice accuracy")
        
        return ParseTextResponse(
            success=True,
            extracted_entities=entities,
            confidence_score=confidence_score,
            suggestions=suggestions,
            message="Text parsed successfully"
        )
        
    except Exception as e:
        logger.error(f"Invoice parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve-entities", response_model=EntityResolutionResponse)
async def resolve_entities(
    request: EntityResolutionRequest,
    token: str = Depends(security)
):
    """Resolve extracted entities with existing business data"""
    try:
        logger.info(f"Resolving entities for business: {request.business_id}")
        generator = NLPInvoiceGenerator()
        
        # Resolve entities with existing data
        resolved_entities = await generator.resolve_entities(request.entities, request.business_id)
        
        # Calculate resolution confidence
        customer_confidence = 0.0
        if resolved_entities.get("customer"):
            customer = resolved_entities["customer"]
            if not customer.get("is_new", True):
                customer_confidence = customer.get("match_confidence", 0) / 100
            else:
                customer_confidence = 0.5  # New customer gets medium confidence
        
        product_confidence = 0.0
        if resolved_entities.get("products"):
            products = resolved_entities["products"]
            product_scores = [
                p.get("match_confidence", 50) / 100 if not p.get("is_new", True) else 0.5
                for p in products
            ]
            product_confidence = sum(product_scores) / len(product_scores) if product_scores else 0.0
        
        resolution_confidence = (customer_confidence + product_confidence) / 2
        
        # Generate suggestions
        suggestions = []
        if resolved_entities.get("customer", {}).get("is_new"):
            suggestions.append("New customer detected - consider adding contact details")
        
        new_products = [p for p in resolved_entities.get("products", []) if p.get("is_new")]
        if new_products:
            suggestions.append(f"New products detected: {', '.join(p['name'] for p in new_products)}")
        
        return EntityResolutionResponse(
            success=True,
            resolved_customer=resolved_entities.get("customer"),
            resolved_products=resolved_entities.get("products", []),
            resolution_confidence=resolution_confidence,
            suggestions=suggestions,
            message="Entities resolved successfully"
        )
        
    except Exception as e:
        logger.error(f"Entity resolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview", response_model=InvoicePreviewResponse)
async def create_invoice_preview(
    request: InvoicePreviewRequest,
    token: str = Depends(security)
):
    """Create invoice preview for user confirmation"""
    try:
        logger.info(f"Creating invoice preview for business: {request.business_id}")
        generator = NLPInvoiceGenerator()
        
        # Build invoice items from request data
        invoice_items = []
        total_amount = 0.0
        warnings = []
        
        for item_data in request.items:
            try:
                item = InvoiceItem(
                    name=item_data.get("name", "Unknown Item"),
                    quantity=float(item_data.get("quantity", 1.0)),
                    unit_price=float(item_data.get("unit_price", 0.0)),
                    total_price=float(item_data.get("total_price", 0.0)),
                    tax_rate=item_data.get("tax_rate", 0.0),
                    description=item_data.get("description")
                )
                
                # Recalculate total if not provided
                if item.total_price == 0.0:
                    item.total_price = item.quantity * item.unit_price
                
                invoice_items.append(item)
                total_amount += item.total_price
                
                # Add warnings for missing data
                if item.unit_price == 0.0:
                    warnings.append(f"No price specified for {item.name}")
                
            except (ValueError, TypeError) as e:
                warnings.append(f"Invalid item data: {item_data.get('name', 'Unknown')}")
        
        # Calculate confidence score
        confidence_factors = []
        
        # Customer confidence
        if request.customer:
            if not request.customer.get("is_new", True):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            warnings.append("No customer specified")
        
        # Items confidence
        if invoice_items:
            item_confidence = sum(1.0 if item.unit_price > 0 else 0.5 for item in invoice_items) / len(invoice_items)
            confidence_factors.append(item_confidence)
        else:
            confidence_factors.append(0.0)
            warnings.append("No items specified")
        
        confidence_score = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
        
        # Create preview data
        preview_id = str(uuid.uuid4())
        invoice_data = {
            "customer": request.customer,
            "items": [item.dict() for item in invoice_items],
            "payment_preference": request.payment_preference,
            "total_amount": total_amount,
            "subtotal": total_amount,  # Before tax
            "tax_amount": 0.0,  # Calculate based on items
            "generated_at": "preview",
            "business_id": request.business_id
        }
        
        # Calculate tax if applicable
        tax_amount = sum(
            item.total_price * (item.tax_rate or 0.0) / 100 
            for item in invoice_items
        )
        invoice_data["tax_amount"] = tax_amount
        invoice_data["total_amount"] = total_amount + tax_amount
        
        # Store preview in cache
        _preview_cache[preview_id] = {
            "invoice_data": invoice_data,
            "original_text": request.original_text,
            "created_at": "now"  # In production, use actual timestamp
        }
        
        # Generate suggestions
        suggestions = []
        if confidence_score < 0.7:
            suggestions.append("Consider reviewing and completing missing information")
        if not request.payment_preference:
            suggestions.append("Consider specifying payment method")
        if total_amount == 0.0:
            suggestions.append("Please add item prices for accurate invoice")
        
        # Define editable fields
        editable_fields = [
            "customer.name", "customer.email", "customer.phone",
            "items[].name", "items[].quantity", "items[].unit_price",
            "payment_preference"
        ]
        
        return InvoicePreviewResponse(
            success=True,
            preview_id=preview_id,
            invoice_data=invoice_data,
            confidence_score=confidence_score,
            warnings=warnings,
            suggestions=suggestions,
            editable_fields=editable_fields,
            message="Invoice preview created successfully"
        )
        
    except Exception as e:
        logger.error(f"Invoice preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confirm", response_model=InvoiceGenerationResponse)
async def confirm_invoice(
    request: InvoiceConfirmationRequest,
    token: str = Depends(security)
):
    """Confirm and finalize invoice from preview"""
    try:
        logger.info(f"Confirming invoice preview: {request.preview_id}")
        
        # Retrieve preview data
        if request.preview_id not in _preview_cache:
            raise HTTPException(status_code=404, detail="Preview not found or expired")
        
        preview_data = _preview_cache[request.preview_id]
        invoice_data = preview_data["invoice_data"].copy()
        
        # Apply modifications if provided
        if request.modifications:
            # Apply customer modifications
            if "customer" in request.modifications:
                invoice_data["customer"].update(request.modifications["customer"])
            
            # Apply item modifications
            if "items" in request.modifications:
                for i, item_mod in enumerate(request.modifications["items"]):
                    if i < len(invoice_data["items"]):
                        invoice_data["items"][i].update(item_mod)
                        # Recalculate total price
                        item = invoice_data["items"][i]
                        item["total_price"] = item["quantity"] * item["unit_price"]
            
            # Apply other modifications
            for key, value in request.modifications.items():
                if key not in ["customer", "items"]:
                    invoice_data[key] = value
            
            # Recalculate totals
            subtotal = sum(item["total_price"] for item in invoice_data["items"])
            tax_amount = sum(
                item["total_price"] * (item.get("tax_rate", 0.0) / 100)
                for item in invoice_data["items"]
            )
            invoice_data["subtotal"] = subtotal
            invoice_data["tax_amount"] = tax_amount
            invoice_data["total_amount"] = subtotal + tax_amount
        
        # Generate final invoice ID
        invoice_id = str(uuid.uuid4())
        invoice_data["invoice_id"] = invoice_id
        invoice_data["generated_at"] = "now"  # In production, use actual timestamp
        invoice_data["status"] = "confirmed"
        
        # Clean up preview cache
        del _preview_cache[request.preview_id]
        
        # In production, save to database here
        # await generator.db.save_invoice(invoice_data)
        
        return InvoiceGenerationResponse(
            success=True,
            message="Invoice confirmed and generated successfully",
            invoice_id=invoice_id,
            invoice_data=invoice_data,
            confidence_score=1.0,  # Confirmed invoices have full confidence
            suggestions=["Invoice has been successfully generated and can be sent to customer"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Invoice confirmation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/invoice/preview/{preview_id}")
async def get_invoice_preview(
    preview_id: str,
    token: str = Depends(security)
):
    """Retrieve invoice preview by ID"""
    try:
        if preview_id not in _preview_cache:
            raise HTTPException(status_code=404, detail="Preview not found or expired")
        
        preview_data = _preview_cache[preview_id]
        return {
            "success": True,
            "preview_id": preview_id,
            "invoice_data": preview_data["invoice_data"],
            "original_text": preview_data.get("original_text"),
            "created_at": preview_data.get("created_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invoice/preview/{preview_id}")
async def cancel_invoice_preview(
    preview_id: str,
    token: str = Depends(security)
):
    """Cancel and delete invoice preview"""
    try:
        if preview_id not in _preview_cache:
            raise HTTPException(status_code=404, detail="Preview not found or expired")
        
        del _preview_cache[preview_id]
        
        return {
            "success": True,
            "message": "Invoice preview cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))