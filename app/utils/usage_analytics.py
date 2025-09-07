"""
Usage analytics system for tracking AI feature performance and user behavior.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from app.database import get_supabase_client
from app.utils.performance_monitor import performance_monitor, MetricType

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Analytics event types."""
    FEATURE_VIEW = "feature_view"
    FEATURE_INTERACTION = "feature_interaction"
    FEATURE_CONVERSION = "feature_conversion"
    FEATURE_ERROR = "feature_error"
    PERFORMANCE_METRIC = "performance_metric"
    AB_TEST_ASSIGNMENT = "ab_test_assignment"


@dataclass
class AnalyticsEvent:
    """Analytics event data structure."""
    event_type: EventType
    feature_name: str
    user_id: str
    business_id: Optional[str] = None
    variant: Optional[str] = None
    properties: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class UsageAnalytics:
    """Tracks and analyzes AI feature usage patterns."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        self._start_background_flush()
    
    async def track_event(self, event: AnalyticsEvent):
        """Track an analytics event."""
        try:
            # Add to buffer
            self.event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.buffer_size:
                await self._flush_events()
                
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
    
    async def track_feature_view(self, feature_name: str, user_id: str, business_id: str = None, variant: str = None):
        """Track when a user views a feature."""
        event = AnalyticsEvent(
            event_type=EventType.FEATURE_VIEW,
            feature_name=feature_name,
            user_id=user_id,
            business_id=business_id,
            variant=variant
        )
        await self.track_event(event)
    
    async def track_feature_interaction(self, feature_name: str, user_id: str, interaction_type: str, properties: Dict[str, Any] = None):
        """Track user interaction with a feature."""
        event = AnalyticsEvent(
            event_type=EventType.FEATURE_INTERACTION,
            feature_name=feature_name,
            user_id=user_id,
            properties={
                "interaction_type": interaction_type,
                **(properties or {})
            }
        )
        await self.track_event(event)
    
    async def track_feature_conversion(self, feature_name: str, user_id: str, conversion_value: float = 1.0, properties: Dict[str, Any] = None):
        """Track feature conversion event."""
        event = AnalyticsEvent(
            event_type=EventType.FEATURE_CONVERSION,
            feature_name=feature_name,
            user_id=user_id,
            properties={
                "conversion_value": conversion_value,
                **(properties or {})
            }
        )
        await self.track_event(event)
    
    async def track_feature_error(self, feature_name: str, user_id: str, error_type: str, error_message: str):
        """Track feature error event."""
        event = AnalyticsEvent(
            event_type=EventType.FEATURE_ERROR,
            feature_name=feature_name,
            user_id=user_id,
            properties={
                "error_type": error_type,
                "error_message": error_message
            }
        )
        await self.track_event(event)
    
    async def track_performance_metric(self, feature_name: str, metric_type: str, value: float, variant: str = None):
        """Track performance metric."""
        event = AnalyticsEvent(
            event_type=EventType.PERFORMANCE_METRIC,
            feature_name=feature_name,
            user_id="system",
            variant=variant,
            properties={
                "metric_type": metric_type,
                "value": value
            }
        )
        await self.track_event(event)
        
        # Also record in performance monitor
        try:
            metric_enum = MetricType(metric_type)
            await performance_monitor.record_metric(feature_name, metric_enum, value, variant)
        except ValueError:
            # Unknown metric type, just log
            logger.debug(f"Unknown metric type: {metric_type}")
    
    async def _flush_events(self):
        """Flush events to database."""
        if not self.event_buffer:
            return
        
        try:
            # Prepare events for insertion
            events_to_insert = []
            for event in self.event_buffer:
                events_to_insert.append({
                    "event_type": event.event_type.value,
                    "feature_name": event.feature_name,
                    "user_id": event.user_id,
                    "business_id": event.business_id,
                    "variant": event.variant,
                    "properties": event.properties,
                    "timestamp": event.timestamp.isoformat()
                })
            
            # Insert into database
            await self.supabase.table("usage_analytics").insert(events_to_insert).execute()
            
            logger.debug(f"Flushed {len(events_to_insert)} analytics events")
            
            # Clear buffer
            self.event_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing events: {e}")
    
    def _start_background_flush(self):
        """Start background task to flush events periodically."""
        asyncio.create_task(self._background_flush_loop())
    
    async def _background_flush_loop(self):
        """Background loop to flush events periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except Exception as e:
                logger.error(f"Error in background flush loop: {e}")
    
    async def get_feature_usage_stats(self, feature_name: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a feature."""
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Get all events for the feature
            result = await self.supabase.table("usage_analytics").select("*").eq("feature_name", feature_name).gte("timestamp", start_date).execute()
            
            if not result.data:
                return {"feature_name": feature_name, "no_data": True}
            
            events = result.data
            
            # Calculate statistics
            stats = {
                "feature_name": feature_name,
                "period_days": days,
                "total_events": len(events),
                "unique_users": len(set(event["user_id"] for event in events if event["user_id"] != "system")),
                "event_breakdown": {},
                "daily_usage": {},
                "variant_breakdown": {},
                "error_rate": 0.0,
                "conversion_rate": 0.0,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Event type breakdown
            for event in events:
                event_type = event["event_type"]
                stats["event_breakdown"][event_type] = stats["event_breakdown"].get(event_type, 0) + 1
            
            # Daily usage
            for event in events:
                date = event["timestamp"][:10]  # Extract date part
                stats["daily_usage"][date] = stats["daily_usage"].get(date, 0) + 1
            
            # Variant breakdown
            for event in events:
                variant = event.get("variant", "none")
                if variant:
                    stats["variant_breakdown"][variant] = stats["variant_breakdown"].get(variant, 0) + 1
            
            # Calculate rates
            total_interactions = stats["event_breakdown"].get("feature_interaction", 0)
            total_conversions = stats["event_breakdown"].get("feature_conversion", 0)
            total_errors = stats["event_breakdown"].get("feature_error", 0)
            
            if total_interactions > 0:
                stats["conversion_rate"] = total_conversions / total_interactions
                stats["error_rate"] = total_errors / total_interactions
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}
    
    async def get_ab_test_results(self, feature_name: str, days: int = 30) -> Dict[str, Any]:
        """Get A/B test results for a feature."""
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Get events with variants
            result = await self.supabase.table("usage_analytics").select("*").eq("feature_name", feature_name).gte("timestamp", start_date).not_.is_("variant", "null").execute()
            
            if not result.data:
                return {"feature_name": feature_name, "no_ab_test_data": True}
            
            events = result.data
            
            # Group by variant
            variant_stats = {}
            
            for event in events:
                variant = event["variant"]
                if variant not in variant_stats:
                    variant_stats[variant] = {
                        "views": 0,
                        "interactions": 0,
                        "conversions": 0,
                        "errors": 0,
                        "unique_users": set()
                    }
                
                event_type = event["event_type"]
                variant_stats[variant]["unique_users"].add(event["user_id"])
                
                if event_type == "feature_view":
                    variant_stats[variant]["views"] += 1
                elif event_type == "feature_interaction":
                    variant_stats[variant]["interactions"] += 1
                elif event_type == "feature_conversion":
                    variant_stats[variant]["conversions"] += 1
                elif event_type == "feature_error":
                    variant_stats[variant]["errors"] += 1
            
            # Calculate metrics for each variant
            results = {
                "feature_name": feature_name,
                "period_days": days,
                "variants": {},
                "statistical_significance": None,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            for variant, stats in variant_stats.items():
                unique_users = len(stats["unique_users"])
                interactions = stats["interactions"]
                conversions = stats["conversions"]
                
                results["variants"][variant] = {
                    "unique_users": unique_users,
                    "views": stats["views"],
                    "interactions": interactions,
                    "conversions": conversions,
                    "errors": stats["errors"],
                    "conversion_rate": conversions / max(interactions, 1),
                    "error_rate": stats["errors"] / max(interactions, 1),
                    "engagement_rate": interactions / max(stats["views"], 1)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting A/B test results: {e}")
            return {"error": str(e)}
    
    async def get_user_journey(self, user_id: str, feature_name: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Get user journey for analytics."""
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            query = self.supabase.table("usage_analytics").select("*").eq("user_id", user_id).gte("timestamp", start_date).order("timestamp")
            
            if feature_name:
                query = query.eq("feature_name", feature_name)
            
            result = await query.execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting user journey: {e}")
            return []
    
    async def generate_insights_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive insights report."""
        try:
            features = [
                "fraud_detection",
                "predictive_insights",
                "compliance_checking",
                "nlp_invoice_generation",
                "smart_notifications",
                "ml_analytics_engine"
            ]
            
            report = {
                "period_days": days,
                "features": {},
                "overall_metrics": {
                    "total_users": 0,
                    "total_events": 0,
                    "most_popular_feature": None,
                    "highest_conversion_feature": None
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            all_users = set()
            total_events = 0
            feature_popularity = {}
            feature_conversions = {}
            
            for feature in features:
                stats = await self.get_feature_usage_stats(feature, days)
                if "no_data" not in stats:
                    report["features"][feature] = stats
                    
                    # Update overall metrics
                    if "unique_users" in stats:
                        feature_popularity[feature] = stats["unique_users"]
                    
                    if "conversion_rate" in stats:
                        feature_conversions[feature] = stats["conversion_rate"]
                    
                    total_events += stats.get("total_events", 0)
            
            # Calculate overall metrics
            report["overall_metrics"]["total_events"] = total_events
            
            if feature_popularity:
                report["overall_metrics"]["most_popular_feature"] = max(feature_popularity, key=feature_popularity.get)
            
            if feature_conversions:
                report["overall_metrics"]["highest_conversion_feature"] = max(feature_conversions, key=feature_conversions.get)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating insights report: {e}")
            return {"error": str(e)}


# Global usage analytics instance
usage_analytics = UsageAnalytics()