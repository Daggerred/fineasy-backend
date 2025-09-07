"""
Data anonymization utilities for AI processing
Removes or masks PII while preserving data patterns for analysis
"""
import re
import hashlib
import random
import string
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataAnonymizer:
    """Anonymizes sensitive data while preserving analytical value"""
    
    def __init__(self, salt: Optional[str] = None):
        """Initialize with optional salt for consistent hashing"""
        self.salt = salt or "ai_anonymization_salt_2024"
        self.fake_names = [
            "Customer A", "Customer B", "Customer C", "Supplier X", "Supplier Y", "Supplier Z",
            "Business Partner", "Service Provider", "Vendor", "Client"
        ]
        self.fake_companies = [
            "ABC Corp", "XYZ Ltd", "Tech Solutions", "Global Services", "Business Hub",
            "Trade Center", "Commerce Co", "Enterprise Ltd", "Solutions Inc", "Services Group"
        ]
    
    def anonymize_financial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize financial data for AI processing"""
        try:
            anonymized = data.copy()
            
            # Anonymize personal identifiers
            if 'customer_name' in anonymized:
                anonymized['customer_name'] = self._anonymize_name(anonymized['customer_name'])
            
            if 'supplier_name' in anonymized:
                anonymized['supplier_name'] = self._anonymize_name(anonymized['supplier_name'])
            
            if 'business_name' in anonymized:
                anonymized['business_name'] = self._anonymize_company(anonymized['business_name'])
            
            # Anonymize contact information
            if 'phone' in anonymized:
                anonymized['phone'] = self._anonymize_phone(anonymized['phone'])
            
            if 'email' in anonymized:
                anonymized['email'] = self._anonymize_email(anonymized['email'])
            
            if 'address' in anonymized:
                anonymized['address'] = self._anonymize_address(anonymized['address'])
            
            # Anonymize financial identifiers
            if 'account_number' in anonymized:
                anonymized['account_number'] = self._anonymize_account_number(anonymized['account_number'])
            
            if 'upi_id' in anonymized:
                anonymized['upi_id'] = self._anonymize_upi_id(anonymized['upi_id'])
            
            if 'gstin' in anonymized:
                anonymized['gstin'] = self._anonymize_gstin(anonymized['gstin'])
            
            # Preserve numerical patterns but anonymize exact amounts
            if 'amount' in anonymized:
                anonymized['amount'] = self._anonymize_amount(anonymized['amount'])
            
            # Add anonymization metadata
            anonymized['_anonymized'] = True
            anonymized['_anonymization_timestamp'] = datetime.now().isoformat()
            
            return anonymized
        except Exception as e:
            logger.error(f"Data anonymization failed: {str(e)}")
            raise
    
    def _anonymize_name(self, name: str) -> str:
        """Anonymize personal/business names"""
        if not name:
            return name
        
        # Use consistent hash-based anonymization
        hash_value = hashlib.md5(f"{name}{self.salt}".encode()).hexdigest()
        index = int(hash_value[:8], 16) % len(self.fake_names)
        return self.fake_names[index]
    
    def _anonymize_company(self, company: str) -> str:
        """Anonymize company names"""
        if not company:
            return company
        
        hash_value = hashlib.md5(f"{company}{self.salt}".encode()).hexdigest()
        index = int(hash_value[:8], 16) % len(self.fake_companies)
        return self.fake_companies[index]
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone numbers while preserving format"""
        if not phone:
            return phone
        
        # Extract digits only
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 10:
            # Keep country code pattern, anonymize rest
            if len(digits) > 10:
                country_code = digits[:-10]
                anonymized_number = country_code + "XXXXXXXXXX"
            else:
                anonymized_number = "XXXXXXXXXX"
            
            # Preserve original format structure
            anonymized = phone
            for digit in digits:
                anonymized = anonymized.replace(digit, 'X', 1)
            return anonymized
        
        return "XXXXXXXXXX"
    
    def _anonymize_email(self, email: str) -> str:
        """Anonymize email addresses while preserving domain patterns"""
        if not email or '@' not in email:
            return "user@example.com"
        
        local, domain = email.split('@', 1)
        
        # Hash the local part consistently
        hash_value = hashlib.md5(f"{local}{self.salt}".encode()).hexdigest()
        anonymized_local = f"user{hash_value[:6]}"
        
        # Anonymize domain but preserve structure
        domain_parts = domain.split('.')
        if len(domain_parts) >= 2:
            anonymized_domain = f"example.{domain_parts[-1]}"
        else:
            anonymized_domain = "example.com"
        
        return f"{anonymized_local}@{anonymized_domain}"
    
    def _anonymize_address(self, address: str) -> str:
        """Anonymize addresses while preserving location patterns"""
        if not address:
            return address
        
        # Replace specific details with generic terms
        anonymized = address
        
        # Replace numbers with X
        anonymized = re.sub(r'\d+', 'XXX', anonymized)
        
        # Replace common address terms
        replacements = {
            r'\b\d+[a-zA-Z]*\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln)\b': 'XXX Street',
            r'\b(Apartment|Apt|Unit|Suite)\s+\d+[a-zA-Z]*\b': 'Unit XXX',
            r'\b\d{5,6}\b': 'XXXXXX',  # Postal codes
        }
        
        for pattern, replacement in replacements.items():
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    def _anonymize_account_number(self, account_number: str) -> str:
        """Anonymize account numbers while preserving length patterns"""
        if not account_number:
            return account_number
        
        # Preserve length and format, replace digits
        anonymized = ""
        for char in account_number:
            if char.isdigit():
                anonymized += "X"
            else:
                anonymized += char
        
        return anonymized
    
    def _anonymize_upi_id(self, upi_id: str) -> str:
        """Anonymize UPI IDs while preserving format"""
        if not upi_id or '@' not in upi_id:
            return "user@bank"
        
        local, provider = upi_id.split('@', 1)
        hash_value = hashlib.md5(f"{local}{self.salt}".encode()).hexdigest()
        anonymized_local = f"user{hash_value[:6]}"
        
        return f"{anonymized_local}@{provider}"
    
    def _anonymize_gstin(self, gstin: str) -> str:
        """Anonymize GSTIN while preserving format for compliance checking"""
        if not gstin or len(gstin) != 15:
            return "XXGSTXXXXXXX001"
        
        # Preserve state code (first 2 digits) and check digit patterns
        state_code = gstin[:2]
        anonymized = f"{state_code}XXXXXXXXX001"
        
        return anonymized
    
    def _anonymize_amount(self, amount: float) -> float:
        """Anonymize amounts while preserving magnitude patterns"""
        if not amount:
            return amount
        
        # Preserve order of magnitude but add noise
        magnitude = len(str(int(abs(amount))))
        
        # Add 10-20% random noise
        noise_factor = random.uniform(0.9, 1.2)
        anonymized_amount = round(amount * noise_factor, 2)
        
        return anonymized_amount
    
    def create_anonymization_report(self, original_data: Dict[str, Any], 
                                  anonymized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a report of what was anonymized"""
        report = {
            "anonymization_timestamp": datetime.now().isoformat(),
            "fields_anonymized": [],
            "data_preserved": [],
            "anonymization_method": "hash_based_consistent"
        }
        
        # Track what was anonymized
        anonymized_fields = [
            'customer_name', 'supplier_name', 'business_name', 'phone', 
            'email', 'address', 'account_number', 'upi_id', 'gstin'
        ]
        
        for field in anonymized_fields:
            if field in original_data and field in anonymized_data:
                if original_data[field] != anonymized_data[field]:
                    report["fields_anonymized"].append(field)
        
        # Track what was preserved
        preserved_fields = ['transaction_type', 'category', 'date', 'currency']
        for field in preserved_fields:
            if field in original_data:
                report["data_preserved"].append(field)
        
        return report

class PrivacyPreservingAnalyzer:
    """Performs analysis on anonymized data while preserving privacy"""
    
    def __init__(self):
        self.anonymizer = DataAnonymizer()
    
    def prepare_dataset_for_ml(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare entire dataset for ML training with anonymization"""
        try:
            anonymized_dataset = []
            
            for record in dataset:
                anonymized_record = self.anonymizer.anonymize_financial_data(record)
                anonymized_dataset.append(anonymized_record)
            
            logger.info(f"Anonymized {len(dataset)} records for ML processing")
            return anonymized_dataset
        except Exception as e:
            logger.error(f"Dataset anonymization failed: {str(e)}")
            raise
    
    def extract_patterns_safely(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analytical patterns without exposing sensitive data"""
        try:
            patterns = {
                "transaction_frequency": self._calculate_frequency_pattern(data),
                "amount_distribution": self._calculate_amount_pattern(data),
                "temporal_patterns": self._calculate_temporal_pattern(data),
                "category_distribution": self._calculate_category_pattern(data)
            }
            
            return patterns
        except Exception as e:
            logger.error(f"Pattern extraction failed: {str(e)}")
            raise
    
    def _calculate_frequency_pattern(self, data: Dict[str, Any]) -> str:
        """Calculate transaction frequency pattern"""
        # Implementation for frequency analysis
        return "regular"  # Placeholder
    
    def _calculate_amount_pattern(self, data: Dict[str, Any]) -> str:
        """Calculate amount distribution pattern"""
        # Implementation for amount pattern analysis
        return "moderate"  # Placeholder
    
    def _calculate_temporal_pattern(self, data: Dict[str, Any]) -> str:
        """Calculate temporal pattern"""
        # Implementation for temporal analysis
        return "business_hours"  # Placeholder
    
    def _calculate_category_pattern(self, data: Dict[str, Any]) -> str:
        """Calculate category distribution pattern"""
        # Implementation for category analysis
        return "diverse"  # Placeholder

# Global anonymization services
anonymizer = DataAnonymizer()
privacy_analyzer = PrivacyPreservingAnalyzer()