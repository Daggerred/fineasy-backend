"""
NLP Invoice Generation Service using Google Gemini
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime

from ..models.base import InvoiceRequest, InvoiceItem
from ..models.responses import InvoiceGenerationResponse
from ..database import DatabaseManager
from .gemini_service import gemini_service

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from natural language text"""
    
    def __init__(self):
        try:
            # Load spaCy model - using small English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using spaCy NER and custom patterns"""
        if not self.nlp:
            return self._extract_with_regex(text)
        
        doc = self.nlp(text)
        entities = {
            "customer_names": [],
            "items": [],
            "quantities": [],
            "prices": [],
            "payment_methods": [],
            "organizations": [],
            "money": [],
            "raw_entities": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            entities["raw_entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
            if ent.label_ in ["PERSON", "ORG"]:
                entities["customer_names"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
        
        # Extract custom patterns
        entities.update(self._extract_custom_patterns(text))
        
        return entities
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns"""
        entities = {
            "customer_names": [],
            "items": [],
            "quantities": [],
            "prices": [],
            "payment_methods": [],
            "organizations": [],
            "money": []
        }
        
        # Extract money amounts
        money_pattern = r'(?:₹|Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        entities["money"] = [f"₹{amount}" for amount in money_matches]
        
        # Extract quantities
        qty_pattern = r'(\d+)\s*(?:units?|pieces?|pcs?|items?|qty)'
        qty_matches = re.findall(qty_pattern, text, re.IGNORECASE)
        entities["quantities"] = qty_matches
        
        # Extract payment methods
        payment_pattern = r'\b(UPI|cash|card|cheque|bank transfer|online|gpay|paytm|phonepe)\b'
        payment_matches = re.findall(payment_pattern, text, re.IGNORECASE)
        entities["payment_methods"] = payment_matches
        
        return entities
    
    def _extract_custom_patterns(self, text: str) -> Dict[str, Any]:
        """Extract custom patterns specific to invoice generation"""
        patterns = {}
        
        # Extract quantities with units
        qty_pattern = r'(\d+(?:\.\d+)?)\s*(?:units?|pieces?|pcs?|items?|qty|nos?)'
        quantities = re.findall(qty_pattern, text, re.IGNORECASE)
        patterns["quantities"] = quantities
        
        # Extract prices
        price_pattern = r'(?:₹|Rs\.?|INR|@|at)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        prices = re.findall(price_pattern, text, re.IGNORECASE)
        patterns["prices"] = prices
        
        # Extract payment methods
        payment_pattern = r'\b(UPI|cash|card|cheque|bank transfer|online|gpay|paytm|phonepe)\b'
        payment_methods = re.findall(payment_pattern, text, re.IGNORECASE)
        patterns["payment_methods"] = payment_methods
        
        # Extract item names (words between "for" and quantity/price)
        item_pattern = r'(?:for|of)\s+([a-zA-Z\s]+?)(?:\s+\d+|\s+₹|\s+Rs)'
        items = re.findall(item_pattern, text, re.IGNORECASE)
        patterns["items"] = [item.strip() for item in items]
        
        return patterns


class EntityResolver:
    """Resolve extracted entities with existing business data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def resolve_customer(self, customer_names: List[str], business_id: str) -> Optional[Dict[str, Any]]:
        """Resolve customer name using fuzzy matching"""
        if not customer_names:
            return None
        
        try:
            # Get existing customers for the business
            existing_customers = await self.db.get_customers(business_id)
            
            if not existing_customers:
                return {"name": customer_names[0], "is_new": True}
            
            # Find best match using fuzzy matching
            customer_names_list = [customer["name"] for customer in existing_customers]
            
            best_match = None
            best_score = 0
            
            for input_name in customer_names:
                match = process.extractOne(input_name, customer_names_list, scorer=fuzz.ratio)
                if match and match[1] > best_score:
                    best_score = match[1]
                    best_match = next(c for c in existing_customers if c["name"] == match[0])
            
            # Use match if confidence is high enough
            if best_score >= 80:
                return {**best_match, "is_new": False, "match_confidence": best_score}
            else:
                return {"name": customer_names[0], "is_new": True}
                
        except Exception as e:
            logger.error(f"Error resolving customer: {e}")
            return {"name": customer_names[0] if customer_names else "Unknown", "is_new": True}
    
    async def resolve_products(self, item_names: List[str], business_id: str) -> List[Dict[str, Any]]:
        """Resolve product names using fuzzy matching"""
        if not item_names:
            return []
        
        try:
            # Get existing products for the business
            existing_products = await self.db.get_products(business_id)
            
            resolved_items = []
            
            for item_name in item_names:
                if not existing_products:
                    resolved_items.append({"name": item_name, "is_new": True})
                    continue
                
                # Find best match
                product_names = [product["name"] for product in existing_products]
                match = process.extractOne(item_name, product_names, scorer=fuzz.ratio)
                
                if match and match[1] >= 70:
                    matched_product = next(p for p in existing_products if p["name"] == match[0])
                    resolved_items.append({
                        **matched_product,
                        "is_new": False,
                        "match_confidence": match[1]
                    })
                else:
                    resolved_items.append({"name": item_name, "is_new": True})
            
            return resolved_items
            
        except Exception as e:
            logger.error(f"Error resolving products: {e}")
            return [{"name": name, "is_new": True} for name in item_names]


class InvoiceBuilder:
    """Build invoice structure from resolved entities"""
    
    def build_invoice_items(self, entities: Dict[str, Any], resolved_products: List[Dict[str, Any]]) -> List[InvoiceItem]:
        """Build invoice items from entities and resolved products"""
        items = []
        
        # Extract quantities and prices
        quantities = [float(q) for q in entities.get("quantities", [])]
        prices = [float(p.replace(",", "")) for p in entities.get("prices", [])]
        
        # Match items with quantities and prices
        for i, product in enumerate(resolved_products):
            quantity = quantities[i] if i < len(quantities) else 1.0
            unit_price = prices[i] if i < len(prices) else 0.0
            
            # If no price found, use product price if available
            if unit_price == 0.0 and not product.get("is_new", True):
                unit_price = float(product.get("price", 0.0))
            
            total_price = quantity * unit_price
            
            item = InvoiceItem(
                name=product["name"],
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                tax_rate=product.get("tax_rate", 0.0),
                description=product.get("description")
            )
            items.append(item)
        
        return items
    
    def calculate_confidence_score(self, entities: Dict[str, Any], resolved_customer: Optional[Dict], resolved_products: List[Dict]) -> float:
        """Calculate confidence score for the invoice generation"""
        score = 0.0
        total_factors = 0
        
        # Customer resolution confidence
        if resolved_customer:
            if not resolved_customer.get("is_new", True):
                score += resolved_customer.get("match_confidence", 0) / 100
            else:
                score += 0.5  # New customer gets medium confidence
            total_factors += 1
        
        # Product resolution confidence
        if resolved_products:
            product_confidence = sum(
                p.get("match_confidence", 50) / 100 if not p.get("is_new", True) else 0.5
                for p in resolved_products
            ) / len(resolved_products)
            score += product_confidence
            total_factors += 1
        
        # Entity extraction completeness
        required_entities = ["customer_names", "items", "quantities"]
        found_entities = sum(1 for entity in required_entities if entities.get(entity))
        entity_score = found_entities / len(required_entities)
        score += entity_score
        total_factors += 1
        
        return score / total_factors if total_factors > 0 else 0.0


class NLPInvoiceGenerator:
    """Natural language invoice generation service"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.entity_extractor = EntityExtractor()
        self.entity_resolver = EntityResolver(self.db)
        self.invoice_builder = InvoiceBuilder()
    
    async def generate_invoice_from_text(self, request: InvoiceRequest) -> InvoiceGenerationResponse:
        """Generate invoice from natural language input"""
        logger.info(f"Generating invoice from text for business: {request.business_id}")
        
        try:
            # Step 1: Parse the invoice request
            parsed_request = await self.parse_invoice_request(request.raw_input, request.business_id)
            
            # Step 2: Extract entities
            entities = self.entity_extractor.extract_entities(request.raw_input)
            
            # Step 3: Resolve entities with existing data
            resolved_entities = await self.resolve_entities(entities, request.business_id)
            
            # Step 4: Build invoice structure
            invoice_items = self.invoice_builder.build_invoice_items(
                entities, resolved_entities["products"]
            )
            
            # Step 5: Calculate confidence score
            confidence_score = self.invoice_builder.calculate_confidence_score(
                entities, resolved_entities["customer"], resolved_entities["products"]
            )
            
            # Step 6: Build invoice data
            invoice_data = {
                "customer": resolved_entities["customer"],
                "items": [item.dict() for item in invoice_items],
                "payment_preference": entities.get("payment_methods", [None])[0],
                "total_amount": sum(item.total_price for item in invoice_items),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Generate suggestions for improvement
            suggestions = self._generate_suggestions(entities, resolved_entities, confidence_score)
            
            return InvoiceGenerationResponse(
                success=True,
                message="Invoice generated successfully from natural language input",
                invoice_data=invoice_data,
                extracted_entities=entities,
                confidence_score=confidence_score,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error generating invoice from text: {e}")
            return InvoiceGenerationResponse(
                success=False,
                message=f"Invoice generation failed: {str(e)}",
                confidence_score=0.0,
                errors=[str(e)]
            )
    
    async def parse_invoice_request(self, text: str, business_id: str) -> InvoiceRequest:
        """Parse natural language invoice request"""
        entities = self.entity_extractor.extract_entities(text)
        
        return InvoiceRequest(
            raw_input=text,
            business_id=business_id,
            customer_name=entities.get("customer_names", [None])[0],
            payment_preference=entities.get("payment_methods", [None])[0],
            extracted_entities=entities
        )
    
    async def resolve_entities(self, entities: Dict[str, Any], business_id: str) -> Dict[str, Any]:
        """Resolve extracted entities with existing business data"""
        resolved_customer = await self.entity_resolver.resolve_customer(
            entities.get("customer_names", []), business_id
        )
        
        resolved_products = await self.entity_resolver.resolve_products(
            entities.get("items", []), business_id
        )
        
        return {
            "customer": resolved_customer,
            "products": resolved_products
        }
    
    def _generate_suggestions(self, entities: Dict[str, Any], resolved_entities: Dict[str, Any], confidence_score: float) -> List[str]:
        """Generate suggestions for improving invoice accuracy"""
        suggestions = []
        
        if confidence_score < 0.7:
            suggestions.append("Consider providing more specific details for better accuracy")
        
        if not entities.get("customer_names"):
            suggestions.append("Please specify the customer name")
        
        if not entities.get("quantities"):
            suggestions.append("Consider specifying quantities for items")
        
        if not entities.get("prices"):
            suggestions.append("Consider including prices for better invoice accuracy")
        
        if resolved_entities.get("customer", {}).get("is_new"):
            suggestions.append("New customer detected - you may want to add their contact details")
        
        new_products = [p for p in resolved_entities.get("products", []) if p.get("is_new")]
        if new_products:
            suggestions.append(f"New products detected: {', '.join(p['name'] for p in new_products)}")
        
        return suggestions