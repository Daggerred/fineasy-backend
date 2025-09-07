"""
Performance monitoring system for AI features with automatic rollback capabilities.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import time

from app.database import get_supabase_client
from app.utils.feature_flags import feature_flag_manager, FeatureFlagStatus

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time_ms"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput_per_minute"
    MEMORY_USAGE = "memory_usage_mb"
    CPU_USAGE = "cpu_usage_percent"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    duration_minutes: int = 5
    enabled: bool = True


@dataclass
class PerformanceAlert:
    """Performance alert data."""
    feature_name: str
    metric_type: MetricType
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False


class PerformanceMonitor:
    """Monitors AI feature performance and triggers rollbacks when needed."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.monitoring_active = True
        self.alert_history = []
        
        # Default performance thresholds
        self.thresholds = {
            "fraud_detection": [
                PerformanceThreshold(MetricType.RESPONSE_TIME, 1000, 3000),
                PerformanceThreshold(MetricType.ERROR_RATE, 0.05, 0.15),
                PerformanceThreshold(MetricType.SUCCESS_RATE, 0.95, 0.85)
            ],
            "predictive_insights": [
                PerformanceThreshold(MetricType.RESPONSE_TIME, 2000, 5000),
                PerformanceThreshold(MetricType.ERROR_RATE, 0.03, 0.10),
                PerformanceThreshold(MetricType.SUCCESS_RATE, 0.97, 0.90)
            ],
            "smart_notifications": [
                PerformanceThreshold(MetricType.RESPONSE_TIME, 500, 1500),
                PerformanceThreshold(MetricType.ERROR_RATE, 0.02, 0.08),
                PerformanceThreshold(MetricType.SUCCESS_RATE, 0.98, 0.92)
            ],
            "compliance_checking": [
                PerformanceThreshold(MetricType.RESPONSE_TIME, 1500, 4000),
                PerformanceThreshold(MetricType.ERROR_RATE, 0.04, 0.12),
                PerformanceThreshold(MetricType.SUCCESS_RATE, 0.96, 0.88)
            ],
            "nlp_invoice_generation": [
                PerformanceThreshold(MetricType.RESPONSE_TIME, 3000, 8000),
                PerformanceThreshold(MetricType.ERROR_RATE, 0.06, 0.20),
                PerformanceThreshold(MetricType.SUCCESS_RATE, 0.94, 0.80)
            ]
        }
    
    async def record_metric(self, feature_name: str, metric_type: MetricType, value: float, variant: str = None):
        """Record a performance metric."""
        try:
            await self.supabase.table("feature_performance").insert({
                "feature_name": feature_name,
                "metric_name": metric_type.value,
                "metric_value": value,
                "variant": variant,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            
            # Check if this metric triggers any alerts
            await self._check_thresholds(feature_name, metric_type, value)
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def _check_thresholds(self, feature_name: str, metric_type: MetricType, current_value: float):
        """Check if current metric value exceeds thresholds."""
        if feature_name not in self.thresholds:
            return
        
        feature_thresholds = [t for t in self.thresholds[feature_name] if t.metric_type == metric_type and t.enabled]
        
        for threshold in feature_thresholds:
            severity = None
            threshold_value = None
            
            # Determine severity based on threshold type
            if metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.MEMORY_USAGE, MetricType.CPU_USAGE]:
                # Higher values are worse
                if current_value >= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif current_value >= threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold
            else:
                # Lower values are worse (success rate, throughput)
                if current_value <= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif current_value <= threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold
            
            if severity:
                await self._trigger_alert(feature_name, metric_type, current_value, threshold_value, severity)
    
    async def _trigger_alert(self, feature_name: str, metric_type: MetricType, current_value: float, threshold_value: float, severity: AlertSeverity):
        """Trigger a performance alert."""
        alert = PerformanceAlert(
            feature_name=feature_name,
            metric_type=metric_type,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            message=f"{feature_name} {metric_type.value} is {current_value}, exceeding threshold of {threshold_value}",
            timestamp=datetime.utcnow()
        )
        
        self.alert_history.append(alert)
        
        # Log the alert
        logger.warning(f"Performance alert: {alert.message}")
        
        # Store alert in database
        try:
            await self.supabase.table("performance_alerts").insert({
                "feature_name": feature_name,
                "metric_type": metric_type.value,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "severity": severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
        
        # Check if automatic rollback is needed
        if severity == AlertSeverity.CRITICAL:
            await self._consider_automatic_rollback(feature_name, alert)
    
    async def _consider_automatic_rollback(self, feature_name: str, alert: PerformanceAlert):
        """Consider automatic rollback based on alert patterns."""
        # Get recent alerts for this feature
        recent_alerts = [
            a for a in self.alert_history 
            if a.feature_name == feature_name 
            and a.timestamp > datetime.utcnow() - timedelta(minutes=10)
            and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        ]
        
        # Rollback conditions
        should_rollback = False
        rollback_reason = ""
        
        # Multiple critical alerts in short time
        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) >= 3:
            should_rollback = True
            rollback_reason = f"Multiple critical alerts ({len(critical_alerts)}) in 10 minutes"
        
        # High error rate
        if alert.metric_type == MetricType.ERROR_RATE and alert.current_value > 0.20:
            should_rollback = True
            rollback_reason = f"Error rate too high: {alert.current_value:.2%}"
        
        # Extremely slow response times
        if alert.metric_type == MetricType.RESPONSE_TIME and alert.current_value > 10000:
            should_rollback = True
            rollback_reason = f"Response time too slow: {alert.current_value}ms"
        
        if should_rollback:
            await self._execute_automatic_rollback(feature_name, rollback_reason)
    
    async def _execute_automatic_rollback(self, feature_name: str, reason: str):
        """Execute automatic rollback of a feature."""
        try:
            # Get current feature flag
            flag = await feature_flag_manager.get_feature_flag(feature_name)
            if not flag:
                logger.error(f"Cannot rollback {feature_name}: feature flag not found")
                return
            
            previous_status = flag.status.value
            
            # Determine rollback strategy
            if flag.status == FeatureFlagStatus.ENABLED:
                # Reduce rollout to 50%
                new_status = FeatureFlagStatus.ROLLOUT.value
                rollout_percentage = 50.0
            elif flag.status == FeatureFlagStatus.ROLLOUT:
                # Reduce rollout percentage by half
                rollout_percentage = max(flag.rollout_percentage / 2, 10.0)
                new_status = FeatureFlagStatus.ROLLOUT.value
            else:
                # Disable feature
                new_status = FeatureFlagStatus.DISABLED.value
                rollout_percentage = 0.0
            
            # Update feature flag
            updates = {
                "status": new_status,
                "rollout_percentage": rollout_percentage,
                "metadata": {
                    **flag.metadata,
                    "automatic_rollback": True,
                    "rollback_reason": reason,
                    "rollback_timestamp": datetime.utcnow().isoformat(),
                    "previous_status": previous_status
                }
            }
            
            success = await feature_flag_manager.update_feature_flag(feature_name, updates)
            
            if success:
                # Record rollback event
                await self.supabase.table("feature_rollbacks").insert({
                    "feature_name": feature_name,
                    "previous_status": previous_status,
                    "new_status": new_status,
                    "reason": reason,
                    "rollback_type": "automatic",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"rollout_percentage": rollout_percentage}
                }).execute()
                
                logger.warning(f"Automatic rollback executed for {feature_name}: {reason}")
                
                # Send notification to administrators
                await self._notify_rollback(feature_name, reason, "automatic")
            
        except Exception as e:
            logger.error(f"Error executing automatic rollback: {e}")
    
    async def _notify_rollback(self, feature_name: str, reason: str, rollback_type: str):
        """Notify administrators about rollback."""
        try:
            # This would integrate with your notification system
            logger.critical(f"ROLLBACK: {feature_name} rolled back ({rollback_type}) - {reason}")
        except Exception as e:
            logger.error(f"Error sending rollback notification: {e}")
    
    async def get_feature_performance_summary(self, feature_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a feature."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Fetch performance data
            result = await self.supabase.table("feature_performance").select("*").eq("feature_name", feature_name).gte("timestamp", start_time.isoformat()).execute()
            
            if not result.data:
                return {"feature_name": feature_name, "no_data": True}
            
            # Group metrics by type
            metrics_by_type = {}
            for record in result.data:
                metric_type = record["metric_name"]
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(record["metric_value"])
            
            # Calculate statistics
            summary = {
                "feature_name": feature_name,
                "period_hours": hours,
                "metrics": {},
                "alerts_count": len([a for a in self.alert_history if a.feature_name == feature_name and a.timestamp > start_time]),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            for metric_type, values in metrics_by_type.items():
                if values:
                    summary["metrics"][metric_type] = {
                        "count": len(values),
                        "average": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1] if values else None
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def manual_rollback(self, feature_name: str, reason: str, target_status: str, admin_user_id: str) -> bool:
        """Execute manual rollback of a feature."""
        try:
            flag = await feature_flag_manager.get_feature_flag(feature_name)
            if not flag:
                return False
            
            previous_status = flag.status.value
            
            # Update feature flag
            updates = {
                "status": target_status,
                "metadata": {
                    **flag.metadata,
                    "manual_rollback": True,
                    "rollback_reason": reason,
                    "rollback_timestamp": datetime.utcnow().isoformat(),
                    "rollback_by": admin_user_id,
                    "previous_status": previous_status
                }
            }
            
            success = await feature_flag_manager.update_feature_flag(feature_name, updates)
            
            if success:
                # Record rollback event
                await self.supabase.table("feature_rollbacks").insert({
                    "feature_name": feature_name,
                    "previous_status": previous_status,
                    "new_status": target_status,
                    "reason": reason,
                    "triggered_by": admin_user_id,
                    "rollback_type": "manual",
                    "timestamp": datetime.utcnow().isoformat()
                }).execute()
                
                logger.info(f"Manual rollback executed for {feature_name} by {admin_user_id}: {reason}")
                
                # Send notification
                await self._notify_rollback(feature_name, reason, "manual")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing manual rollback: {e}")
            return False
    
    def start_monitoring(self):
        """Start the performance monitoring loop."""
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop the performance monitoring loop."""
        self.monitoring_active = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform periodic checks
                await self._check_all_features()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_all_features(self):
        """Check performance of all active features."""
        try:
            # This would check all active features
            # For now, just log that monitoring is active
            logger.debug("Performance monitoring check completed")
        except Exception as e:
            logger.error(f"Error checking all features: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()