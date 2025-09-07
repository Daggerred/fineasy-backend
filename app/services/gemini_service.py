"""
Google Gemini AI Service for NLP and CV processing
"""
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import json
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for Google Gemini AI operations"""
    
    def __init__(self):
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.enabled = True
        else:
            self.model = None
            self.enabled = False
            logger.warning("Gemini API key not provided. AI features will be disabled.")
    
    async def generate_invoice_from_text(self, text: str, business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate invoice data from natural language text using Gemini"""
        if not self.enabled:
            return {
                "success": False,
                "error": "Gemini AI service is not configured",
                "suggestions": ["Please configure GEMINI_API_KEY in environment variables"]
            }
        
        try:
            # Build context-aware prompt
            prompt = self._build_invoice_prompt(text, business_context)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse response
            result = self._parse_invoice_response(response.text)
            
            return {
                "success": True,
                "invoice_data": result.get("invoice_data"),
                "suggestions": result.get("suggestions", []),
                "confidence": result.get("confidence", 0.8),
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Gemini invoice generation error: {e}")
            return {
                "success": False,
                "error": f"AI processing failed: {str(e)}",
                "suggestions": ["Please try rephrasing your input", "Ensure all required information is included"]
            }
    
    async def parse_cv_content(self, cv_text: str, business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse CV content and generate service invoice data using Gemini"""
        if not self.enabled:
            return {
                "success": False,
                "error": "Gemini AI service is not configured",
                "suggestions": ["Please configure GEMINI_API_KEY in environment variables"]
            }
        
        try:
            # Build CV analysis prompt
            prompt = self._build_cv_prompt(cv_text, business_context)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse response
            result = self._parse_cv_response(response.text)
            
            return {
                "success": True,
                "extracted_text": cv_text,
                "invoice_data": result.get("invoice_data"),
                "suggestions": result.get("suggestions", []),
                "confidence": result.get("confidence", 0.7),
                "personal_info": result.get("personal_info", {}),
                "services": result.get("services", [])
            }
            
        except Exception as e:
            logger.error(f"Gemini CV parsing error: {e}")
            return {
                "success": False,
                "error": f"CV parsing failed: {str(e)}",
                "suggestions": ["Please ensure the CV contains clear professional information"]
            }
    
    def _build_invoice_prompt(self, text: str, business_context: Dict[str, Any] = None) -> str:
        """Build prompt for invoice generation"""
        context_info = ""
        if business_context:
            customers = business_context.get("customers", [])
            products = business_context.get("products", [])
            
            if customers:
                context_info += f"\nKnown customers: {', '.join([c.get('name', '') for c in customers[:5]])}"
            if products:
                context_info += f"\nAvailable products: {', '.join([p.get('name', '') for p in products[:5]])}"
        
        return f"""
You are an AI assistant that extracts invoice information from natural language text.

Context: This is for an Indian business management app that needs to generate GST-compliant invoices.
{context_info}

Input text: "{text}"

Please analyze the text and extract invoice information. Return a JSON response with this exact structure:

{{
    "invoice_data": {{
        "customer": {{
            "name": "Customer Name",
            "is_new": true/false,
            "phone": "phone number if mentioned",
            "email": "email if mentioned",
            "gst_number": "GST number if mentioned"
        }},
        "items": [
            {{
                "name": "Item/Service Name",
                "quantity": number,
                "unit": "unit type (pieces, hours, etc.)",
                "unit_price": number,
                "total_price": number,
                "hsn_code": "HSN code if applicable"
            }}
        ],
        "payment_method": "cash/upi/card/bank_transfer/cheque",
        "due_date": "YYYY-MM-DD format if mentioned",
        "tax_rate": 18.0,
        "subtotal": number,
        "tax_amount": number,
        "total_amount": number,
        "notes": "any additional notes"
    }},
    "suggestions": [
        "List of helpful suggestions for the user"
    ],
    "confidence": 0.0-1.0
}}

Rules:
1. Extract customer name from the text
2. Identify items/services with quantities and prices
3. Calculate totals including 18% GST
4. Use Indian currency format (₹)
5. If information is missing, mark customer as "is_new": true
6. Provide helpful suggestions for missing information
7. Return valid JSON only, no additional text
"""
    
    def _build_cv_prompt(self, cv_text: str, business_context: Dict[str, Any] = None) -> str:
        """Build prompt for CV analysis"""
        return f"""
You are an AI assistant that analyzes CVs/Resumes to extract information for creating professional service invoices.

CV Content:
{cv_text}

Please analyze this CV and extract information that can be used to create invoices for professional services. Return a JSON response with this exact structure:

{{
    "invoice_data": {{
        "customer": {{
            "name": "Full Name from CV",
            "email": "email@example.com",
            "phone": "+91XXXXXXXXXX",
            "is_new": true,
            "customer_type": "service_provider"
        }},
        "items": [
            {{
                "name": "Service Description",
                "category": "Service Category",
                "quantity": 1,
                "unit": "hour/project/day",
                "unit_price": suggested_rate,
                "total_price": suggested_rate,
                "hsn_code": "998314"
            }}
        ],
        "payment_terms": "Net 30 days",
        "notes": "Professional services based on CV analysis"
    }},
    "personal_info": {{
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+91XXXXXXXXXX",
        "experience_level": "junior/mid/senior",
        "top_skills": ["skill1", "skill2", "skill3"]
    }},
    "services": [
        {{
            "category": "Service Category",
            "description": "Detailed service description",
            "suggested_rate": hourly_rate_in_rupees,
            "unit": "hour/project/day"
        }}
    ],
    "suggestions": [
        "List of suggestions based on CV analysis"
    ],
    "confidence": 0.0-1.0
}}

Rules:
1. Extract personal information (name, contact details)
2. Identify professional skills that can be billed as services
3. Suggest appropriate hourly rates in Indian Rupees (₹500-₹5000/hour based on experience)
4. Categorize services (Development, Consulting, Design, etc.)
5. Determine experience level from CV content
6. Use HSN code 998314 for professional services
7. Return valid JSON only, no additional text
"""
    
    def _parse_invoice_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response for invoice generation"""
        try:
            # Clean response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # Parse JSON
            result = json.loads(cleaned_text)
            
            # Validate and enhance result
            if "invoice_data" in result:
                invoice_data = result["invoice_data"]
                
                # Calculate totals if not present
                if "items" in invoice_data:
                    subtotal = sum(item.get("total_price", 0) for item in invoice_data["items"])
                    tax_rate = invoice_data.get("tax_rate", 18.0)
                    tax_amount = subtotal * (tax_rate / 100)
                    total_amount = subtotal + tax_amount
                    
                    invoice_data["subtotal"] = subtotal
                    invoice_data["tax_amount"] = tax_amount
                    invoice_data["total_amount"] = total_amount
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            return {
                "invoice_data": None,
                "suggestions": ["AI response format error. Please try again."],
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return {
                "invoice_data": None,
                "suggestions": ["Error processing AI response. Please try again."],
                "confidence": 0.0
            }
    
    def _parse_cv_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response for CV analysis"""
        try:
            # Clean response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # Parse JSON
            result = json.loads(cleaned_text)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini CV JSON response: {e}")
            return {
                "invoice_data": None,
                "personal_info": {},
                "services": [],
                "suggestions": ["AI response format error. Please try again."],
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Error parsing Gemini CV response: {e}")
            return {
                "invoice_data": None,
                "personal_info": {},
                "services": [],
                "suggestions": ["Error processing AI response. Please try again."],
                "confidence": 0.0
            }
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Check if Gemini service is healthy"""
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Gemini API key not configured"
            }
        
        try:
            # Simple test query
            response = self.model.generate_content("Hello, respond with 'OK' if you're working.")
            
            return {
                "status": "healthy",
                "message": "Gemini service is operational",
                "model": settings.GEMINI_MODEL
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Gemini service error: {str(e)}"
            }


# Global instance
gemini_service = GeminiService()