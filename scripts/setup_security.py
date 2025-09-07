#!/usr/bin/env python3
"""
Security setup script for AI backend
Initializes encryption keys, security policies, and audit logging
"""
import os
import sys
import secrets
import base64
import hashlib
from pathlib import Path
from cryptography.fernet import Fernet
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.encryption import DataEncryption
from app.utils.audit_logger import AIAuditLogger
from app.utils.data_retention import DataRetentionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecuritySetup:
    """Security setup and configuration manager"""
    
    def __init__(self, env_file_path: str = ".env"):
        self.env_file_path = Path(env_file_path)
        self.backup_file_path = Path(f"{env_file_path}.backup")
    
    def generate_encryption_key(self) -> str:
        """Generate a new encryption key"""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a secure password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_salt(self, length: int = 16) -> str:
        """Generate a cryptographic salt"""
        return secrets.token_hex(length)
    
    def backup_env_file(self):
        """Create backup of existing .env file"""
        if self.env_file_path.exists():
            logger.info(f"Creating backup of {self.env_file_path}")
            with open(self.env_file_path, 'r') as src, open(self.backup_file_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Backup created at {self.backup_file_path}")
    
    def update_env_file(self, security_config: dict):
        """Update .env file with security configuration"""
        logger.info("Updating .env file with security configuration")
        
        # Read existing content
        existing_content = ""
        if self.env_file_path.exists():
            with open(self.env_file_path, 'r') as f:
                existing_content = f.read()
        
        # Check if security section already exists
        if "# Security and Privacy Configuration" in existing_content:
            logger.warning("Security configuration already exists in .env file")
            response = input("Do you want to regenerate security keys? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping security key generation")
                return
        
        # Update or add security configuration
        security_section = self._generate_security_section(security_config)
        
        if "# Security and Privacy Configuration" in existing_content:
            # Replace existing security section
            lines = existing_content.split('\n')
            new_lines = []
            in_security_section = False
            
            for line in lines:
                if line.strip() == "# Security and Privacy Configuration":
                    in_security_section = True
                    new_lines.extend(security_section.split('\n'))
                elif in_security_section and line.startswith('#') and "Configuration" in line:
                    in_security_section = False
                    new_lines.append(line)
                elif not in_security_section:
                    new_lines.append(line)
            
            updated_content = '\n'.join(new_lines)
        else:
            # Append security section
            updated_content = existing_content.rstrip() + '\n\n' + security_section
        
        # Write updated content
        with open(self.env_file_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Security configuration updated in {self.env_file_path}")
    
    def _generate_security_section(self, config: dict) -> str:
        """Generate security configuration section for .env file"""
        return f"""# -----------------------------------------------------------------------------
# Security and Privacy Configuration
# -----------------------------------------------------------------------------
# Data Encryption
AI_ENCRYPTION_KEY={config['encryption_key']}
AI_ENCRYPTION_PASSWORD={config['encryption_password']}
AI_ENCRYPTION_SALT={config['encryption_salt']}

# Rate Limiting
MAX_REQUESTS_PER_MINUTE={config.get('max_requests_per_minute', 60)}
MAX_REQUESTS_PER_HOUR={config.get('max_requests_per_hour', 1000)}
MAX_RECORDS_PER_REQUEST={config.get('max_records_per_request', 1000)}
MAX_PROCESSING_TIME_SECONDS={config.get('max_processing_time_seconds', 300)}

# Data Retention Policies
ENABLE_AUTO_PURGE={config.get('enable_auto_purge', 'true')}
DEFAULT_RETENTION_DAYS={config.get('default_retention_days', 90)}
AUDIT_LOG_RETENTION_DAYS={config.get('audit_log_retention_days', 2555)}
TEMP_DATA_RETENTION_HOURS={config.get('temp_data_retention_hours', 24)}
FRAUD_ALERT_RETENTION_DAYS={config.get('fraud_alert_retention_days', 2555)}
COMPLIANCE_REPORT_RETENTION_DAYS={config.get('compliance_report_retention_days', 2555)}

# Privacy Controls
REQUIRE_ENCRYPTION_FOR_SENSITIVE_DATA={config.get('require_encryption', 'true')}
REQUIRE_ANONYMIZATION_FOR_ML={config.get('require_anonymization', 'true')}
ENABLE_AUDIT_LOGGING={config.get('enable_audit_logging', 'true')}
ENABLE_DATA_ANONYMIZATION={config.get('enable_data_anonymization', 'true')}

# Security Monitoring
SUSPICIOUS_ACTIVITY_THRESHOLD={config.get('suspicious_activity_threshold', 10)}
FAILED_AUTH_THRESHOLD={config.get('failed_auth_threshold', 5)}
ENABLE_SECURITY_MONITORING={config.get('enable_security_monitoring', 'true')}
ENABLE_RATE_LIMITING={config.get('enable_rate_limiting', 'true')}

# Audit Configuration
AUDIT_LOG_DIRECTORY={config.get('audit_log_directory', 'logs/audit')}
ENABLE_FILE_AUDIT_LOGGING={config.get('enable_file_audit_logging', 'true')}
ENABLE_DATABASE_AUDIT_LOGGING={config.get('enable_database_audit_logging', 'true')}
AUDIT_LOG_ROTATION_DAYS={config.get('audit_log_rotation_days', 30)}

# Data Purging Configuration
ENABLE_AUTOMATIC_CLEANUP={config.get('enable_automatic_cleanup', 'true')}
CLEANUP_SCHEDULE_HOURS={config.get('cleanup_schedule_hours', 24)}
RETENTION_CLEANUP_BATCH_SIZE={config.get('retention_cleanup_batch_size', 1000)}"""
    
    def setup_audit_logging(self):
        """Set up audit logging directories and configuration"""
        logger.info("Setting up audit logging")
        
        # Create audit log directory
        audit_dir = Path("logs/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audit logger instance to initialize files
        audit_logger = AIAuditLogger(str(audit_dir))
        
        # Test audit logging
        test_audit_id = audit_logger.log_ai_operation(
            event_type="ai_processing_start",
            business_id="setup_test",
            operation_details={"operation": "security_setup_test"},
            severity="low"
        )
        
        logger.info(f"Audit logging setup complete. Test audit ID: {test_audit_id}")
    
    def setup_data_retention(self):
        """Set up data retention policies"""
        logger.info("Setting up data retention policies")
        
        retention_manager = DataRetentionManager()
        
        # Log retention policy setup
        for category in ["processed_ai_results", "anonymized_patterns", "ml_training_data", "temporary_processing"]:
            policy_info = retention_manager.get_retention_policy_info(category)
            logger.info(f"Retention policy for {category}: {policy_info['retention_days']} days, auto_purge: {policy_info['auto_purge']}")
    
    def test_encryption(self, encryption_key: str):
        """Test encryption functionality"""
        logger.info("Testing encryption functionality")
        
        try:
            # Test with generated key
            os.environ['AI_ENCRYPTION_KEY'] = encryption_key
            encryption = DataEncryption()
            
            # Test data
            test_data = {
                "customer_name": "Test Customer",
                "amount": 1000.50,
                "account_number": "1234567890"
            }
            
            # Encrypt and decrypt
            encrypted = encryption.encrypt_data(test_data)
            decrypted = encryption.decrypt_data(encrypted)
            
            assert decrypted == test_data
            logger.info("Encryption test passed")
            
        except Exception as e:
            logger.error(f"Encryption test failed: {str(e)}")
            raise
    
    def generate_security_report(self) -> dict:
        """Generate security configuration report"""
        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "encryption_configured": os.getenv('AI_ENCRYPTION_KEY') is not None,
            "audit_logging_configured": Path("logs/audit").exists(),
            "env_file_exists": self.env_file_path.exists(),
            "backup_created": self.backup_file_path.exists(),
            "security_features": {
                "data_encryption": True,
                "data_anonymization": True,
                "audit_logging": True,
                "data_retention": True,
                "rate_limiting": True,
                "security_monitoring": True
            }
        }
        
        return report
    
    def run_full_setup(self):
        """Run complete security setup"""
        logger.info("Starting comprehensive security setup")
        
        try:
            # 1. Backup existing .env file
            self.backup_env_file()
            
            # 2. Generate security configuration
            security_config = {
                'encryption_key': self.generate_encryption_key(),
                'encryption_password': self.generate_secure_password(),
                'encryption_salt': self.generate_salt(),
                'max_requests_per_minute': 60,
                'max_requests_per_hour': 1000,
                'max_records_per_request': 1000,
                'max_processing_time_seconds': 300,
                'enable_auto_purge': 'true',
                'default_retention_days': 90,
                'audit_log_retention_days': 2555,
                'temp_data_retention_hours': 24,
                'fraud_alert_retention_days': 2555,
                'compliance_report_retention_days': 2555,
                'require_encryption': 'true',
                'require_anonymization': 'true',
                'enable_audit_logging': 'true',
                'enable_data_anonymization': 'true',
                'suspicious_activity_threshold': 10,
                'failed_auth_threshold': 5,
                'enable_security_monitoring': 'true',
                'enable_rate_limiting': 'true',
                'audit_log_directory': 'logs/audit',
                'enable_file_audit_logging': 'true',
                'enable_database_audit_logging': 'true',
                'audit_log_rotation_days': 30,
                'enable_automatic_cleanup': 'true',
                'cleanup_schedule_hours': 24,
                'retention_cleanup_batch_size': 1000
            }
            
            # 3. Update .env file
            self.update_env_file(security_config)
            
            # 4. Test encryption
            self.test_encryption(security_config['encryption_key'])
            
            # 5. Setup audit logging
            self.setup_audit_logging()
            
            # 6. Setup data retention
            self.setup_data_retention()
            
            # 7. Generate report
            report = self.generate_security_report()
            
            logger.info("Security setup completed successfully")
            logger.info(f"Security report: {report}")
            
            # Display important information
            print("\n" + "="*60)
            print("SECURITY SETUP COMPLETED")
            print("="*60)
            print(f"✓ Encryption key generated and configured")
            print(f"✓ Audit logging setup complete")
            print(f"✓ Data retention policies configured")
            print(f"✓ Security monitoring enabled")
            print(f"✓ Rate limiting configured")
            print(f"✓ Environment backup created: {self.backup_file_path}")
            print("\nIMPORTANT SECURITY NOTES:")
            print("1. Keep your .env file secure and never commit it to version control")
            print("2. Regularly rotate encryption keys in production")
            print("3. Monitor audit logs for security events")
            print("4. Review data retention policies periodically")
            print("5. Test security features before deploying to production")
            print("="*60)
            
            return report
            
        except Exception as e:
            logger.error(f"Security setup failed: {str(e)}")
            raise


def main():
    """Main setup function"""
    print("AI Backend Security Setup")
    print("=" * 30)
    
    # Check if running in correct directory
    if not Path("app").exists():
        print("Error: Please run this script from the ai-backend directory")
        sys.exit(1)
    
    setup = SecuritySetup()
    
    try:
        # Interactive setup
        print("\nThis script will set up security features for the AI backend:")
        print("- Generate encryption keys")
        print("- Configure audit logging")
        print("- Set up data retention policies")
        print("- Enable security monitoring")
        
        response = input("\nProceed with security setup? (Y/n): ")
        if response.lower() in ['', 'y', 'yes']:
            report = setup.run_full_setup()
            
            # Save report
            report_file = Path("security_setup_report.json")
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nSecurity setup report saved to: {report_file}")
        else:
            print("Security setup cancelled")
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()