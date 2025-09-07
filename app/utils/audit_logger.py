"""
Audit logging system for AI operations
Tracks all AI processing activities for security and compliance
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid
import os
from pathlib import Path

class AuditEventType(Enum):
    """Types of audit events"""
    AI_PROCESSING_START = "ai_processing_start"
    AI_PROCESSING_END = "ai_processing_end"
    DATA_ENCRYPTION = "data_encryption"
    DATA_DECRYPTION = "data_decryption"
    DATA_ANONYMIZATION = "data_anonymization"
    FRAUD_DETECTION = "fraud_detection"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    NLP_PROCESSING = "nlp_processing"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    SECURITY_VIOLATION = "security_violation"
    PRIVACY_BREACH = "privacy_breach"
    ERROR_OCCURRED = "error_occurred"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AIAuditLogger:
    """Comprehensive audit logging for AI operations"""
    
    def __init__(self, log_directory: Optional[str] = None):
        """Initialize audit logger with optional custom log directory"""
        self.log_directory = Path(log_directory or "logs/audit")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up structured logging
        self.logger = logging.getLogger("ai_audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        audit_log_file = self.log_directory / "ai_audit.log"
        file_handler = logging.FileHandler(audit_log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        # Security events get separate high-priority log
        self.security_logger = logging.getLogger("ai_security")
        security_log_file = self.log_directory / "ai_security.log"
        security_handler = logging.FileHandler(security_log_file)
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(formatter)
        
        if not self.security_logger.handlers:
            self.security_logger.addHandler(security_handler)
    
    def log_ai_operation(self, 
                        event_type: AuditEventType,
                        business_id: str,
                        user_id: Optional[str] = None,
                        operation_details: Optional[Dict[str, Any]] = None,
                        severity: AuditSeverity = AuditSeverity.LOW,
                        session_id: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None) -> str:
        """Log an AI operation with full audit trail"""
        
        audit_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        audit_entry = {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "event_type": event_type.value,
            "severity": severity.value,
            "business_id": business_id,
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "operation_details": operation_details or {},
            "compliance_flags": self._check_compliance_requirements(event_type, operation_details),
            "data_classification": self._classify_data_sensitivity(operation_details)
        }
        
        # Log to appropriate logger based on severity
        log_message = json.dumps(audit_entry, default=str)
        
        if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
            self.security_logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Store in database for compliance reporting
        try:
            self._store_audit_record(audit_entry)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to store audit record: {e}")
        
        return audit_id
    
    def _check_compliance_requirements(self, event_type: AuditEventType, operation_details: Optional[Dict[str, Any]]) -> List[str]:
        """Check which compliance requirements apply to this operation"""
        flags = []
        
        if event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_EXPORT]:
            flags.append("GDPR_DATA_ACCESS")
        
        if event_type == AuditEventType.FRAUD_DETECTION:
            flags.append("PCI_DSS_FRAUD_MONITORING")
        
        if event_type == AuditEventType.COMPLIANCE_CHECK:
            flags.append("SOX_COMPLIANCE_AUDIT")
        
        return flags
    
    def _classify_data_sensitivity(self, operation_details: Optional[Dict[str, Any]]) -> str:
        """Classify the sensitivity level of data being processed"""
        if not operation_details:
            return "unknown"
        
        # Simple classification based on operation details
        if any(key in str(operation_details).lower() for key in ["ssn", "tax_id", "bank_account"]):
            return "highly_sensitive"
        elif any(key in str(operation_details).lower() for key in ["email", "phone", "address"]):
            return "sensitive"
        else:
            return "normal"
    
    def _store_audit_record(self, audit_entry: Dict[str, Any]):
        """Store audit record in database for compliance reporting"""
        try:
            # This would store in the database - simplified for now
            pass
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to store audit record in database: {e}")


# Global audit logger instance
audit_logger = AIAuditLogger()