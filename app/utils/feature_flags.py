"""
Feature flag system for AI capabilities with A/B testing support.
"""
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import random

from app.database import get_supabase

logger = logging.getLogger(__name__)


class FeatureFlagStatus(Enum):
    """Feature flag status options."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TESTING = "testing"
    ROLLOUT = "rollout"


class ABTestVariant(Enum):
    """A/B test variant options."""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"


@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    status: FeatureFlagStatus
    description: str
    rollout_percentage: float = 0.0
    ab_test_enabled: bool = False
    ab_test_variants: Dict[str, float] = None
    target_users: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.ab_test_variants is None:
            self.ab_test_variants = {}
        if self.target_users is None:
            self.target_users = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class ABTestResult:
    """A/B test result tracking."""
    user_id: str
    feature_name: str
    variant: ABTestVariant
    assigned_at: datetime
    interactions: int = 0
    conversions: int = 0
    last_interaction: datetime = None


class FeatureFlagManager:
    """Manages feature flags and A/B testing."""
    
    def __init__(self):
        self.supabase = get_supabase()
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Initialize default feature flags
        self._initialize_default_flags()
    
    def _initialize_default_flags(self):
        """Initialize default AI feature flags."""
        default_flags = [
            FeatureFlag(
                name="fraud_detection",
                status=FeatureFlagStatus.ENABLED,
                description="AI-powered fraud detection system",
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="predictive_insights",
                status=FeatureFlagStatus.ROLLOUT,
                description="Predictive business insights dashboard",
                rollout_percentage=50.0
            ),
            FeatureFlag(
                name="compliance_checking",
                status=FeatureFlagStatus.TESTING,
                description="AI-driven compliance helper",
                rollout_percentage=25.0,
                ab_test_enabled=True,
                ab_test_variants={"control": 50.0, "variant_a": 50.0}
            ),
            FeatureFlag(
                name="nlp_invoice_generation",
                status=FeatureFlagStatus.DISABLED,
                description="Natural language invoice generation",
                rollout_percentage=0.0
            ),
            FeatureFlag(
                name="smart_notifications",
                status=FeatureFlagStatus.TESTING,
                description="Intelligent notification prioritization",
                rollout_percentage=30.0,
                ab_test_enabled=True,
                ab_test_variants={"control": 40.0, "variant_a": 30.0, "variant_b": 30.0}
            ),
            FeatureFlag(
                name="ml_analytics_engine",
                status=FeatureFlagStatus.ROLLOUT,
                description="Machine learning analytics engine",
                rollout_percentage=75.0
            )
        ]
        
        for flag in default_flags:
            self._ensure_flag_exists(flag)
    
    def _ensure_flag_exists(self, flag: FeatureFlag):
        """Ensure a feature flag exists in the database."""
        try:
            existing = self.supabase.table("feature_flags").select("*").eq("name", flag.name).execute()
            
            if not existing.data:
                self.supabase.table("feature_flags").insert(asdict(flag)).execute()
                logger.info(f"Created feature flag: {flag.name}")
        except Exception as e:
            logger.error(f"Error ensuring feature flag exists: {e}")
    
    def _get_user_hash(self, user_id: str, feature_name: str) -> float:
        """Generate consistent hash for user and feature combination."""
        hash_input = f"{user_id}:{feature_name}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_value[:8], 16) / 0xFFFFFFFF
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        return (key in self._cache and 
                key in self._cache_expiry and 
                datetime.utcnow() < self._cache_expiry[key])
    
    def _cache_flag(self, flag: FeatureFlag):
        """Cache feature flag data."""
        self._cache[flag.name] = flag
        self._cache_expiry[flag.name] = datetime.utcnow() + self.cache_ttl
    
    async def get_feature_flag(self, feature_name: str) -> Optional[FeatureFlag]:
        """Get feature flag configuration."""
        if self._is_cache_valid(feature_name):
            return self._cache[feature_name]
        
        try:
            result = self.supabase.table("feature_flags").select("*").eq("name", feature_name).execute()
            
            if result.data:
                flag_data = result.data[0]
                flag = FeatureFlag(
                    name=flag_data["name"],
                    status=FeatureFlagStatus(flag_data["status"]),
                    description=flag_data["description"],
                    rollout_percentage=flag_data.get("rollout_percentage", 0.0),
                    ab_test_enabled=flag_data.get("ab_test_enabled", False),
                    ab_test_variants=flag_data.get("ab_test_variants", {}),
                    target_users=flag_data.get("target_users", []),
                    created_at=datetime.fromisoformat(flag_data["created_at"]),
                    updated_at=datetime.fromisoformat(flag_data["updated_at"]),
                    metadata=flag_data.get("metadata", {})
                )
                self._cache_flag(flag)
                return flag
        except Exception as e:
            logger.error(f"Error fetching feature flag {feature_name}: {e}")
        
        return None
    
    async def is_feature_enabled(self, feature_name: str, user_id: str, business_id: str = None) -> bool:
        """Check if a feature is enabled for a specific user."""
        flag = await self.get_feature_flag(feature_name)
        if not flag:
            return False
        
        # Feature is completely disabled
        if flag.status == FeatureFlagStatus.DISABLED:
            return False
        
        # Feature is fully enabled
        if flag.status == FeatureFlagStatus.ENABLED:
            return True
        
        # Check if user is in target users list
        if user_id in flag.target_users:
            return True
        
        # Check rollout percentage
        user_hash = self._get_user_hash(user_id, feature_name)
        is_in_rollout = user_hash < (flag.rollout_percentage / 100.0)
        
        if flag.status == FeatureFlagStatus.ROLLOUT:
            return is_in_rollout
        
        if flag.status == FeatureFlagStatus.TESTING:
            return is_in_rollout
        
        return False
    
    async def get_ab_test_variant(self, feature_name: str, user_id: str) -> ABTestVariant:
        """Get A/B test variant for a user."""
        flag = await self.get_feature_flag(feature_name)
        if not flag or not flag.ab_test_enabled:
            return ABTestVariant.CONTROL
        
        # Check if user already has an assigned variant
        try:
            existing = self.supabase.table("ab_test_assignments").select("*").eq("user_id", user_id).eq("feature_name", feature_name).execute()
            
            if existing.data:
                return ABTestVariant(existing.data[0]["variant"])
        except Exception as e:
            logger.error(f"Error fetching existing A/B test assignment: {e}")
        
        # Assign new variant based on weights
        user_hash = self._get_user_hash(user_id, f"{feature_name}_ab")
        cumulative_weight = 0.0
        
        for variant, weight in flag.ab_test_variants.items():
            cumulative_weight += weight / 100.0
            if user_hash < cumulative_weight:
                assigned_variant = ABTestVariant(variant)
                
                # Store assignment
                try:
                    self.supabase.table("ab_test_assignments").insert({
                        "user_id": user_id,
                        "feature_name": feature_name,
                        "variant": assigned_variant.value,
                        "assigned_at": datetime.utcnow().isoformat()
                    }).execute()
                except Exception as e:
                    logger.error(f"Error storing A/B test assignment: {e}")
                
                return assigned_variant
        
        return ABTestVariant.CONTROL
    
    async def track_interaction(self, feature_name: str, user_id: str, interaction_type: str = "view"):
        """Track user interaction with a feature."""
        try:
            variant = await self.get_ab_test_variant(feature_name, user_id)
            
            # Update interaction count
            self.supabase.table("ab_test_results").upsert({
                "user_id": user_id,
                "feature_name": feature_name,
                "variant": variant.value,
                "interaction_type": interaction_type,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            
            # Update analytics
            await self._update_feature_analytics(feature_name, user_id, interaction_type)
            
        except Exception as e:
            logger.error(f"Error tracking interaction: {e}")
    
    async def track_conversion(self, feature_name: str, user_id: str, conversion_value: float = 1.0):
        """Track conversion event for A/B testing."""
        try:
            variant = await self.get_ab_test_variant(feature_name, user_id)
            
            self.supabase.table("ab_test_conversions").insert({
                "user_id": user_id,
                "feature_name": feature_name,
                "variant": variant.value,
                "conversion_value": conversion_value,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            
        except Exception as e:
            logger.error(f"Error tracking conversion: {e}")
    
    async def _update_feature_analytics(self, feature_name: str, user_id: str, interaction_type: str):
        """Update feature usage analytics."""
        try:
            self.supabase.table("feature_analytics").upsert({
                "feature_name": feature_name,
                "user_id": user_id,
                "interaction_type": interaction_type,
                "count": 1,
                "last_interaction": datetime.utcnow().isoformat()
            }, on_conflict="feature_name,user_id,interaction_type").execute()
            
        except Exception as e:
            logger.error(f"Error updating feature analytics: {e}")
    
    async def update_feature_flag(self, feature_name: str, updates: Dict[str, Any]) -> bool:
        """Update feature flag configuration."""
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("feature_flags").update(updates).eq("name", feature_name).execute()
            
            if result.data:
                # Clear cache
                if feature_name in self._cache:
                    del self._cache[feature_name]
                    del self._cache_expiry[feature_name]
                
                logger.info(f"Updated feature flag: {feature_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating feature flag {feature_name}: {e}")
        
        return False
    
    async def get_feature_analytics(self, feature_name: str, days: int = 30) -> Dict[str, Any]:
        """Get feature usage analytics."""
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Get interaction counts
            interactions = self.supabase.table("feature_analytics").select("*").eq("feature_name", feature_name).gte("last_interaction", start_date).execute()
            
            # Get A/B test results
            ab_results = self.supabase.table("ab_test_results").select("*").eq("feature_name", feature_name).gte("timestamp", start_date).execute()
            
            # Get conversion data
            conversions = self.supabase.table("ab_test_conversions").select("*").eq("feature_name", feature_name).gte("timestamp", start_date).execute()
            
            return {
                "feature_name": feature_name,
                "period_days": days,
                "total_interactions": len(interactions.data) if interactions.data else 0,
                "total_conversions": len(conversions.data) if conversions.data else 0,
                "ab_test_results": ab_results.data if ab_results.data else [],
                "conversion_rate": len(conversions.data) / max(len(interactions.data), 1) if interactions.data else 0,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting feature analytics: {e}")
            return {}
    
    async def emergency_disable_feature(self, feature_name: str, reason: str) -> bool:
        """Emergency disable a feature flag."""
        try:
            updates = {
                "status": FeatureFlagStatus.DISABLED.value,
                "metadata": {"emergency_disabled": True, "reason": reason, "disabled_at": datetime.utcnow().isoformat()}
            }
            
            result = await self.update_feature_flag(feature_name, updates)
            
            if result:
                logger.warning(f"Emergency disabled feature {feature_name}: {reason}")
                
                # Send alert to administrators
                await self._send_emergency_alert(feature_name, reason)
            
            return result
            
        except Exception as e:
            logger.error(f"Error emergency disabling feature {feature_name}: {e}")
            return False
    
    async def _send_emergency_alert(self, feature_name: str, reason: str):
        """Send emergency alert to administrators."""
        try:
            # This would integrate with your notification system
            logger.critical(f"EMERGENCY: Feature {feature_name} disabled - {reason}")
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")


# Global feature flag manager instance (lazy initialization)
feature_flag_manager = None

def get_feature_flag_manager():
    global feature_flag_manager
    if feature_flag_manager is None:
        feature_flag_manager = FeatureFlagManager()
    return feature_flag_manager