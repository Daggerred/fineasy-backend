"""
Data encryption utilities for AI processing
Provides encryption/decryption for sensitive financial data
"""
import os
import base64
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
import logging

logger = logging.getLogger(__name__)

class DataEncryption:
    """Handles encryption/decryption of sensitive data for AI processing"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize encryption with key from environment or provided key"""
        try:
            if encryption_key:
                # If key is provided, ensure it's properly formatted
                if len(encryption_key) == 44 and encryption_key.endswith('='):
                    self.key = encryption_key.encode()
                else:
                    # Generate proper key from provided string
                    self.key = self._generate_key_from_string(encryption_key)
            else:
                # Try to get key from environment
                env_key = os.environ.get('AI_ENCRYPTION_KEY', '')
                if env_key and len(env_key) == 44 and env_key.endswith('='):
                    self.key = env_key.encode()
                else:
                    # Generate a key from a password
                    password = os.environ.get('AI_ENCRYPTION_PASSWORD', 'default_ai_password_for_development').encode()
                    salt = os.environ.get('AI_ENCRYPTION_SALT', 'default_salt_for_development').encode()
                    self.key = self._generate_key_from_password(password, salt)
            
            self.cipher_suite = Fernet(self.key)
            self.enabled = True
            
        except Exception as e:
            logger.warning(f"Encryption initialization failed: {e}. Running without encryption.")
            self.cipher_suite = None
            self.enabled = False
    
    def _generate_key_from_password(self, password: bytes, salt: bytes) -> bytes:
        """Generate a proper Fernet key from password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _generate_key_from_string(self, key_string: str) -> bytes:
        """Generate a proper Fernet key from any string"""
        # Use SHA256 to create a consistent 32-byte key
        hash_digest = hashlib.sha256(key_string.encode()).digest()
        return base64.urlsafe_b64encode(hash_digest)
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data to base64 string"""
        if not self.enabled:
            logger.warning("Encryption not available, returning data as JSON")
            return json.dumps(data, default=str)
        
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self.cipher_suite.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt base64 string back to dictionary data"""
        if not self.enabled:
            logger.warning("Encryption not available, treating as JSON")
            try:
                return json.loads(encrypted_data)
            except:
                return {"data": encrypted_data}
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def encrypt_field(self, value: str) -> str:
        """Encrypt a single field value"""
        try:
            encrypted_value = self.cipher_suite.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_value).decode()
        except Exception as e:
            logger.error(f"Field encryption failed: {str(e)}")
            raise
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a single field value"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_value = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_value.decode()
        except Exception as e:
            logger.error(f"Field decryption failed: {str(e)}")
            raise
    
    def hash_data(self, data: str) -> str:
        """Create a hash of data for comparison without storing original"""
        return hashlib.sha256(data.encode()).hexdigest()

class SecureDataProcessor:
    """Processes sensitive data securely for AI analysis"""
    
    def __init__(self):
        self.encryption = DataEncryption()
    
    def prepare_for_ai_processing(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare financial data for AI processing with encryption"""
        try:
            # Encrypt sensitive fields
            sensitive_fields = ['amount', 'account_number', 'upi_id', 'phone', 'email']
            processed_data = financial_data.copy()
            
            for field in sensitive_fields:
                if field in processed_data and processed_data[field]:
                    processed_data[f"{field}_encrypted"] = self.encryption.encrypt_field(str(processed_data[field]))
                    # Replace original with hash for pattern analysis
                    processed_data[field] = self.encryption.hash_data(str(processed_data[field]))
            
            # Add processing metadata
            processed_data['_encrypted'] = True
            processed_data['_processing_timestamp'] = datetime.now().isoformat()
            
            return processed_data
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def restore_from_ai_processing(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore original data after AI processing"""
        try:
            if not processed_data.get('_encrypted'):
                return processed_data
            
            restored_data = processed_data.copy()
            sensitive_fields = ['amount', 'account_number', 'upi_id', 'phone', 'email']
            
            for field in sensitive_fields:
                encrypted_field = f"{field}_encrypted"
                if encrypted_field in restored_data:
                    restored_data[field] = self.encryption.decrypt_field(restored_data[encrypted_field])
                    del restored_data[encrypted_field]
            
            # Remove processing metadata
            restored_data.pop('_encrypted', None)
            restored_data.pop('_processing_timestamp', None)
            
            return restored_data
        except Exception as e:
            logger.error(f"Data restoration failed: {str(e)}")
            raise

# Global encryption instance
encryption_service = DataEncryption()
secure_processor = SecureDataProcessor()