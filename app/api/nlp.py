"""
NLP processing endpoints for invoice generation and document processing
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import re
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

router = APIRouter()

class NLPInvoiceRequest(BaseModel):
    text: str
    business_id: str

class NLPInvoiceResponse(BaseModel):
    success: bool
    message: str
    invoice_data: Optional[Dict[str, Any]] = None

@router.post("/process-invoice", response_model=NLPInvoiceResponse)
async def process_invoice_nlp(request: NLPInvoiceRequest):
    """Process natural language input to generate invoice data"""
    try:
        logger.info(f"Processing NLP invoice request for business: {request.business_id}")
        
        # Parse the natural language input
        invoice_data = parse_invoice_text(request.text)
        
        if not invoice_data:
            return NLPInvoiceResponse(
                success=False,
                message="Could not extract invoice information from the provided text"
            )
        
        # Add business_id to invoice data
        invoice_data["business_id"] = request.business_id
        
        return NLPInvoiceResponse(
            success=True,
            message="Invoice data extracted successfully",
            invoice_data=invoice_data
        )
        
    except Exception as e:
        logger.error(f"Error processing NLP invoice: {e}")
        return NLPInvoiceResponse(
            success=False,
            message=f"Error processing invoice: {str(e)}"
        )

@router.post("/create-invoice")
async def create_invoice_from_nlp(request: NLPInvoiceRequest):
    """Create and save invoice from NLP processing"""
    try:
        from ..database import get_db_manager
        
        logger.info(f"Creating invoice from NLP for business: {request.business_id}")
        
        # Parse the natural language input
        invoice_data = parse_invoice_text(request.text)
        
        if not invoice_data:
            return {
                "success": False,
                "message": "Could not extract invoice information from the provided text"
            }
        
        # Add business_id and additional fields for database
        invoice_data["business_id"] = request.business_id
        invoice_data["status"] = "draft"
        invoice_data["created_at"] = datetime.now().isoformat()
        invoice_data["updated_at"] = datetime.now().isoformat()
        
        # Get database manager
        db = get_db_manager()
        
        # Save invoice to database
        saved_invoice = await db.insert("invoices", invoice_data)
        
        if saved_invoice:
            logger.info(f"Invoice created successfully with ID: {saved_invoice.get('id')}")
            return {
                "success": True,
                "message": "Invoice created successfully",
                "invoice_id": saved_invoice.get("id"),
                "invoice_data": saved_invoice
            }
        else:
            return {
                "success": False,
                "message": "Failed to save invoice to database"
            }
        
    except Exception as e:
        logger.error(f"Error creating invoice from NLP: {e}")
        return {
            "success": False,
            "message": f"Error creating invoice: {str(e)}"
        }

def parse_invoice_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse natural language text to extract invoice information"""
    try:
        text = text.lower().strip()
        
        # Initialize invoice data
        invoice_data = {
            "customer_name": "",
            "customer_email": "",
            "customer_phone": "",
            "items": [],
            "subtotal": 0.0,
            "tax_rate": 0.0,
            "tax_amount": 0.0,
            "discount_amount": 0.0,
            "total_amount": 0.0,
            "notes": "",
            "date": datetime.now().isoformat(),
            "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "invoice_number": f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        # Extract customer name - enhanced patterns
        customer_patterns = [
            r"(?:for|to)\s+([a-zA-Z\s]+?)(?:\s+for|\s+\d|\s*$)",
            r"invoice\s+for\s+([a-zA-Z\s]+?)(?:\s+for|\s+\d|\s*$)",
            r"bill\s+(?:for\s+)?([a-zA-Z\s]+?)(?:\s+for|\s+\d|\s*$)",
            r"create\s+invoice\s+for\s+([a-zA-Z\s]+?)(?:\s+for|\s+\d|\s*$)",
            # Handle company names like "nitiayog textiles"
            r"for\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+for|\s+\d)"
        ]
        
        for pattern in customer_patterns:
            match = re.search(pattern, text)
            if match:
                invoice_data["customer_name"] = match.group(1).strip().title()
                break
        
        # Extract email if present
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            invoice_data["customer_email"] = email_match.group()
        
        # Extract phone if present
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phone_match:
            invoice_data["customer_phone"] = phone_match.group()
        
        # Extract items with quantities and prices
        items = extract_items_from_text(text)
        invoice_data["items"] = items
        
        # Calculate totals
        subtotal = sum(item["total_price"] for item in items)
        invoice_data["subtotal"] = subtotal
        
        # Extract discount if mentioned - enhanced patterns
        discount_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*discount',
            r'with\s+(\d+(?:\.\d+)?)\s*%\s*discount',
            r'discount\s+of\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*percent\s+discount'
        ]
        
        for pattern in discount_patterns:
            discount_match = re.search(pattern, text, re.IGNORECASE)
            if discount_match:
                discount_percent = float(discount_match.group(1))
                if discount_percent > 1:  # Assume percentage
                    discount_percent = discount_percent / 100
                invoice_data["discount_amount"] = subtotal * discount_percent
                break
        
        # Extract tax if mentioned, otherwise default to GST 18%
        tax_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:tax|gst)', text, re.IGNORECASE)
        if tax_match:
            tax_percent = float(tax_match.group(1))
            if tax_percent > 1:  # Assume percentage
                tax_percent = tax_percent / 100
            invoice_data["tax_rate"] = tax_percent
        else:
            # Default to 18% GST for Indian businesses
            invoice_data["tax_rate"] = 0.18
        
        # Calculate tax amount
        taxable_amount = subtotal - invoice_data["discount_amount"]
        invoice_data["tax_amount"] = taxable_amount * invoice_data["tax_rate"]
        
        # Calculate total
        invoice_data["total_amount"] = subtotal - invoice_data["discount_amount"] + invoice_data["tax_amount"]
        
        # Add GST breakdown for Indian businesses
        invoice_data["gst_breakdown"] = {
            "cgst": invoice_data["tax_amount"] / 2,
            "sgst": invoice_data["tax_amount"] / 2,
            "igst": 0.0  # For inter-state transactions
        }
        
        return invoice_data
        
    except Exception as e:
        logger.error(f"Error parsing invoice text: {e}")
        return None

def extract_items_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract items with quantities and prices from text"""
    items = []
    
    # Enhanced patterns for better extraction including Indian currency
    patterns = [
        # Pattern: "8 drums for 5000 rupees each"
        r'(\d+)\s+([a-zA-Z\s]+?)\s+(?:for|at|@)\s+(\d+(?:\.\d+)?)\s+(?:rupees?|rs\.?|₹|dollars?|\$)\s+each',
        # Pattern: "5 laptops at $1000 each"
        r'(\d+)\s*(?:x\s*)?([a-zA-Z\s]+?)\s*(?:at|@|for)\s*(?:\$|₹|rs\.?\s*)?(\d+(?:\.\d+)?)\s*(?:each|per|total)?',
        # Pattern: "drums 8 for 5000"
        r'([a-zA-Z\s]+?)\s+(\d+)\s+(?:for|at|@)\s+(?:\$|₹|rs\.?\s*)?(\d+(?:\.\d+)?)',
        # Pattern: "8 drums 5000 rupees"
        r'(\d+)\s+([a-zA-Z\s]+?)\s+(\d+(?:\.\d+)?)\s+(?:rupees?|rs\.?|₹|dollars?|\$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:  # If we found matches with this pattern, use them and break
            for match in matches:
                try:
                    if len(match) == 3:
                        # Try to determine which is quantity, description, and price
                        if match[0].isdigit():
                            quantity = int(match[0])
                            description = match[1].strip().title()
                            unit_price = float(match[2])
                        elif match[1].isdigit():
                            description = match[0].strip().title()
                            quantity = int(match[1])
                            unit_price = float(match[2])
                        else:
                            continue
                        
                        # Check if price is total or per unit
                        if 'total' in text.lower():
                            total_price = unit_price
                            unit_price = total_price / quantity if quantity > 0 else 0
                        else:
                            total_price = quantity * unit_price
                    
                    items.append({
                        "description": description,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total_price": total_price
                    })
                except (ValueError, IndexError):
                    continue
            break  # Break after finding matches with this pattern
    
    # If no items found with patterns, try to extract simple item mentions
    if not items:
        # Look for product names with prices
        simple_patterns = [
            r'([a-zA-Z\s]+?)\s*\$(\d+(?:\.\d+)?)',
            r'\$(\d+(?:\.\d+)?)\s*([a-zA-Z\s]+)',
        ]
        
        for pattern in simple_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    if pattern.startswith('([a-zA-Z'):
                        description = match[0].strip().title()
                        price = float(match[1])
                    else:
                        price = float(match[0])
                        description = match[1].strip().title()
                    
                    items.append({
                        "description": description,
                        "quantity": 1,
                        "unit_price": price,
                        "total_price": price
                    })
                except (ValueError, IndexError):
                    continue
    
    return items

@router.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    business_id: str = Form(...)
):
    """Process uploaded document with OCR/NLP"""
    try:
        logger.info(f"Processing document: {file.filename} of type: {document_type}")
        
        # For now, return a mock response
        # In a real implementation, you would:
        # 1. Save the uploaded file
        # 2. Use OCR to extract text (e.g., Tesseract, Google Vision API)
        # 3. Process the extracted text based on document type
        # 4. Return structured data
        
        return {
            "success": True,
            "message": "Document processed successfully",
            "extracted_data": {
                "document_type": document_type,
                "filename": file.filename,
                "text": "Sample extracted text from document",
                "structured_data": {
                    "amount": 1000.0,
                    "date": datetime.now().isoformat(),
                    "vendor": "Sample Vendor",
                    "items": [
                        {
                            "description": "Sample Item",
                            "quantity": 1,
                            "unit_price": 1000.0,
                            "total_price": 1000.0
                        }
                    ]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/extract-text")
async def extract_text_from_image(
    file: UploadFile = File(...),
    business_id: str = Form(...)
):
    """Extract text from image using OCR"""
    try:
        logger.info(f"Extracting text from image: {file.filename}")
        
        # Mock OCR response
        # In a real implementation, use OCR service like Tesseract or Google Vision API
        
        return {
            "success": True,
            "message": "Text extracted successfully",
            "extracted_text": "Sample extracted text from the uploaded image",
            "confidence": 0.95
        }
        
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

@router.post("/analyze-receipt")
async def analyze_receipt(
    file: UploadFile = File(...),
    business_id: str = Form(...)
):
    """Analyze receipt and extract structured data"""
    try:
        logger.info(f"Analyzing receipt: {file.filename}")
        
        # Mock receipt analysis
        # In a real implementation:
        # 1. Use OCR to extract text
        # 2. Parse receipt format
        # 3. Extract vendor, items, amounts, dates, etc.
        
        return {
            "success": True,
            "message": "Receipt analyzed successfully",
            "receipt_data": {
                "vendor": "Sample Store",
                "date": datetime.now().isoformat(),
                "total_amount": 125.50,
                "tax_amount": 10.50,
                "subtotal": 115.00,
                "items": [
                    {
                        "description": "Item 1",
                        "quantity": 2,
                        "unit_price": 25.00,
                        "total_price": 50.00
                    },
                    {
                        "description": "Item 2",
                        "quantity": 1,
                        "unit_price": 65.00,
                        "total_price": 65.00
                    }
                ],
                "payment_method": "Credit Card",
                "receipt_number": "RCP-123456"
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing receipt: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing receipt: {str(e)}")