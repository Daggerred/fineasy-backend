#!/usr/bin/env python3
"""
Test encryption service functionality
"""
import os
import sys
sys.path.append('app')

from app.utils.encryption import DataEncryption

def test_encryption():
    """Test encryption service"""
    print("Testing encryption service...")
    
    try:
        # Test with environment key
        encryption = DataEncryption()
        print(f"✅ Encryption service initialized successfully")
        print(f"✅ Encryption enabled: {encryption.enabled}")
        
        # Test data encryption
        test_data = {"amount": 1000, "account": "123456789"}
        encrypted = encryption.encrypt_data(test_data)
        print(f"✅ Data encrypted successfully")
        
        # Test data decryption
        decrypted = encryption.decrypt_data(encrypted)
        print(f"✅ Data decrypted successfully: {decrypted}")
        
        # Test field encryption
        field_encrypted = encryption.encrypt_field("sensitive_value")
        field_decrypted = encryption.decrypt_field(field_encrypted)
        print(f"✅ Field encryption/decryption successful: {field_decrypted}")
        
        print("🎉 All encryption tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_encryption()
    sys.exit(0 if success else 1)