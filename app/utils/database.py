"""
Database utility functions for AI operations
"""
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json
import logging
from datetime import datetime, timedelta
from ..database import DatabaseManager

logger = logging.getLogger(__name__)


class AIDataUtils:
    """Utility class for AI-specific database operations"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    @staticmethod
    def generate_data_hash(data: Dict[str, Any]) -> str:
        """Generate hash for data privacy and deduplication"""
        # Remove sensitive fields and sort for consistent hashing
        clean_data = {k: v for k, v in data.items() 
                     if k not in ['id', 'created_at', 'updated_at', 'user_id']}
        
        data_string = json.dumps(clean_data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    @staticmethod
    def anonymize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive financial data for AI processing"""
        anonymized = data.copy()
        
        # Remove or hash sensitive fields
        sensitive_fields = [
            'customer_name', 'supplier_name', 'email', 'phone', 
            'address', 'bank_account', 'upi_id', 'gstin'
        ]
        
        for field in sensitive_fields:
            if field in anonymized:
                if isinstance(anonymized[field], str) and anonymized[field]:
                    # Replace with hash
                    anonymized[field] = hashlib.md5(anonymized[field].encode()).hexdigest()[:8]
                else:
                    anonymized[field] = None
        
        return anonymized
    
    async def batch_save_analysis_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Save multiple analysis results in batch"""
        saved_ids = []
        
        for result in results:
            try:
                result_id = await self.db.save_analysis_result(result)
                if result_id:
                    saved_ids.append(result_id)
            except Exception as e:
                logger.error(f"Error saving analysis result: {e}")
                continue
        
        return saved_ids
    
    async def get_recent_analysis(self, business_id: str, 
                                analysis_type: str, 
                                hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent analysis results within specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            results = await self.db.get_analysis_results(
                business_id=business_id,
                analysis_type=analysis_type,
                limit=100
            )
            
            # Filter by time (assuming created_at is in the results)
            recent_results = []
            for result in results:
                created_at = result.get('created_at')
                if created_at:
                    # Parse timestamp and compare
                    if isinstance(created_at, str):
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_time = created_at
                    
                    if created_time >= cutoff_time:
                        recent_results.append(result)
            
            return recent_results
            
        except Exception as e:
            logger.error(f"Error fetching recent analysis: {e}")
            return []
    
    async def check_analysis_cache(self, business_id: str, 
                                 analysis_type: str,
                                 entity_id: str,
                                 cache_hours: int = 1) -> Optional[Dict[str, Any]]:
        """Check if analysis result exists in cache"""
        try:
            recent_results = await self.get_recent_analysis(
                business_id, analysis_type, cache_hours
            )
            
            # Find matching entity
            for result in recent_results:
                if result.get('entity_id') == entity_id:
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking analysis cache: {e}")
            return None
    
    async def get_business_data_summary(self, business_id: str) -> Dict[str, Any]:
        """Get comprehensive business data summary for AI analysis"""
        try:
            summary = {}
            
            # Get basic statistics
            stats = await self.db.get_business_statistics(business_id)
            summary['statistics'] = stats
            
            # Get recent transactions (last 90 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            transactions = await self.db.get_transactions(
                business_id=business_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=1000
            )
            summary['recent_transactions'] = len(transactions)
            
            # Get recent invoices
            invoices = await self.db.get_invoices(
                business_id=business_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=1000
            )
            summary['recent_invoices'] = len(invoices)
            
            # Get active alerts
            active_alerts = await self.db.get_fraud_alerts(
                business_id=business_id,
                status='active',
                limit=50
            )
            summary['active_alerts'] = len(active_alerts)
            
            # Get recent insights
            recent_insights = await self.db.get_business_insights(
                business_id=business_id,
                limit=20
            )
            summary['recent_insights'] = len(recent_insights)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting business data summary: {e}")
            return {}
    
    async def create_processing_log(self, business_id: str, 
                                  operation_type: str,
                                  operation_id: str,
                                  input_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create processing log entry"""
        try:
            log_data = {
                'business_id': business_id,
                'operation_type': operation_type,
                'operation_id': operation_id,
                'status': 'started',
                'created_at': datetime.now().isoformat()
            }
            
            if input_data:
                log_data['input_data_hash'] = self.generate_data_hash(input_data)
            
            return await self.db.log_ai_operation(log_data)
            
        except Exception as e:
            logger.error(f"Error creating processing log: {e}")
            return None
    
    async def update_processing_log(self, log_id: str,
                                  status: str,
                                  processing_time_ms: Optional[int] = None,
                                  memory_usage_mb: Optional[int] = None,
                                  error_message: Optional[str] = None,
                                  model_used: Optional[str] = None) -> bool:
        """Update processing log with results"""
        try:
            updates = {'status': status}
            
            if processing_time_ms is not None:
                updates['processing_time_ms'] = processing_time_ms
            if memory_usage_mb is not None:
                updates['memory_usage_mb'] = memory_usage_mb
            if error_message:
                updates['error_message'] = error_message
            if model_used:
                updates['model_used'] = model_used
            
            return await self.db.update_operation_log(log_id, updates)
            
        except Exception as e:
            logger.error(f"Error updating processing log: {e}")
            return False
    
    async def get_fraud_alert_statistics(self, business_id: str) -> Dict[str, Any]:
        """Get fraud alert statistics for a business"""
        try:
            # Get all alerts
            all_alerts = await self.db.get_fraud_alerts(business_id, limit=1000)
            
            stats = {
                'total_alerts': len(all_alerts),
                'active_alerts': len([a for a in all_alerts if a.get('status') == 'active']),
                'resolved_alerts': len([a for a in all_alerts if a.get('status') == 'resolved']),
                'dismissed_alerts': len([a for a in all_alerts if a.get('status') == 'dismissed']),
                'alert_types': {},
                'severity_distribution': {}
            }
            
            # Count by type and severity
            for alert in all_alerts:
                alert_type = alert.get('alert_type', 'unknown')
                severity = alert.get('severity', 'unknown')
                
                stats['alert_types'][alert_type] = stats['alert_types'].get(alert_type, 0) + 1
                stats['severity_distribution'][severity] = stats['severity_distribution'].get(severity, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting fraud alert statistics: {e}")
            return {}
    
    async def cleanup_old_data(self, business_id: str, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old AI data for a business"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            
            cleanup_stats = {
                'analysis_results_deleted': 0,
                'processing_logs_deleted': 0,
                'resolved_alerts_deleted': 0
            }
            
            # This would require custom SQL queries or RPC functions
            # For now, we'll call the database cleanup function
            deleted_count = await self.db.cleanup_expired_data()
            cleanup_stats['total_deleted'] = deleted_count
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}


# Global instance
ai_data_utils = AIDataUtils()