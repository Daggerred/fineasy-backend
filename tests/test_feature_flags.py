"""
Tests for feature flags and A/B testing system.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.utils.feature_flags import (
    FeatureFlagManager, 
    FeatureFlag, 
    FeatureFlagStatus, 
    ABTestVariant,
    feature_flag_manager
)
from app.utils.performance_monitor import PerformanceMonitor, MetricType, AlertSeverity
from app.utils.usage_analytics import UsageAnalytics, EventType, AnalyticsEvent


class TestFeatureFlagManager:
    """Test feature flag management functionality."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client."""
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{"updated": True}]
        mock_client.table.return_value.upsert.return_value.execute.return_value = Mock()
        return mock_client
    
    @pytest.fixture
    def flag_manager(self, mock_supabase):
        """Create feature flag manager with mocked dependencies."""
        with patch('app.utils.feature_flags.get_supabase_client', return_value=mock_supabase):
            manager = FeatureFlagManager()
            return manager
    
    @pytest.mark.asyncio
    async def test_feature_flag_creation(self, flag_manager):
        """Test creating a feature flag."""
        flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.ENABLED,
            description="Test feature",
            rollout_percentage=100.0
        )
        
        assert flag.name == "test_feature"
        assert flag.status == FeatureFlagStatus.ENABLED
        assert flag.rollout_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_is_feature_enabled_fully_enabled(self, flag_manager):
        """Test feature that is fully enabled."""
        # Mock the get_feature_flag method
        test_flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.ENABLED,
            description="Test feature",
            rollout_percentage=100.0
        )
        
        with patch.object(flag_manager, 'get_feature_flag', return_value=test_flag):
            result = await flag_manager.is_feature_enabled("test_feature", "user123")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_is_feature_enabled_disabled(self, flag_manager):
        """Test feature that is disabled."""
        test_flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.DISABLED,
            description="Test feature",
            rollout_percentage=0.0
        )
        
        with patch.object(flag_manager, 'get_feature_flag', return_value=test_flag):
            result = await flag_manager.is_feature_enabled("test_feature", "user123")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_is_feature_enabled_rollout(self, flag_manager):
        """Test feature with rollout percentage."""
        test_flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.ROLLOUT,
            description="Test feature",
            rollout_percentage=50.0
        )
        
        with patch.object(flag_manager, 'get_feature_flag', return_value=test_flag):
            # Mock user hash to return a value that should be included (0.3 < 0.5)
            with patch.object(flag_manager, '_get_user_hash', return_value=0.3):
                result = await flag_manager.is_feature_enabled("test_feature", "user123")
                assert result is True
            
            # Mock user hash to return a value that should be excluded (0.7 > 0.5)
            with patch.object(flag_manager, '_get_user_hash', return_value=0.7):
                result = await flag_manager.is_feature_enabled("test_feature", "user123")
                assert result is False
    
    @pytest.mark.asyncio
    async def test_is_feature_enabled_target_users(self, flag_manager):
        """Test feature with target users."""
        test_flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.ROLLOUT,
            description="Test feature",
            rollout_percentage=0.0,  # 0% rollout
            target_users=["user123"]
        )
        
        with patch.object(flag_manager, 'get_feature_flag', return_value=test_flag):
            # Target user should have access despite 0% rollout
            result = await flag_manager.is_feature_enabled("test_feature", "user123")
            assert result is True
            
            # Non-target user should not have access
            result = await flag_manager.is_feature_enabled("test_feature", "user456")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_ab_test_variant(self, flag_manager):
        """Test A/B test variant assignment."""
        test_flag = FeatureFlag(
            name="test_feature",
            status=FeatureFlagStatus.TESTING,
            description="Test feature",
            rollout_percentage=100.0,
            ab_test_enabled=True,
            ab_test_variants={"control": 50.0, "variant_a": 50.0}
        )
        
        # Mock no existing assignment
        flag_manager.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        with patch.object(flag_manager, 'get_feature_flag', return_value=test_flag):
            # Mock user hash to assign to control
            with patch.object(flag_manager, '_get_user_hash', return_value=0.3):
                variant = await flag_manager.get_ab_test_variant("test_feature", "user123")
                assert variant == ABTestVariant.CONTROL
            
            # Mock user hash to assign to variant_a
            with patch.object(flag_manager, '_get_user_hash', return_value=0.7):
                variant = await flag_manager.get_ab_test_variant("test_feature", "user123")
                assert variant == ABTestVariant.VARIANT_A
    
    @pytest.mark.asyncio
    async def test_track_interaction(self, flag_manager):
        """Test interaction tracking."""
        with patch.object(flag_manager, 'get_ab_test_variant', return_value=ABTestVariant.CONTROL):
            await flag_manager.track_interaction("test_feature", "user123", "click")
            
            # Verify upsert was called
            flag_manager.supabase.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_track_conversion(self, flag_manager):
        """Test conversion tracking."""
        with patch.object(flag_manager, 'get_ab_test_variant', return_value=ABTestVariant.VARIANT_A):
            await flag_manager.track_conversion("test_feature", "user123", 5.0)
            
            # Verify insert was called
            flag_manager.supabase.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_feature_flag(self, flag_manager):
        """Test updating feature flag."""
        updates = {
            "status": "enabled",
            "rollout_percentage": 75.0
        }
        
        result = await flag_manager.update_feature_flag("test_feature", updates)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_emergency_disable_feature(self, flag_manager):
        """Test emergency disable functionality."""
        with patch.object(flag_manager, 'update_feature_flag', return_value=True):
            with patch.object(flag_manager, '_send_emergency_alert') as mock_alert:
                result = await flag_manager.emergency_disable_feature("test_feature", "Critical error")
                
                assert result is True
                mock_alert.assert_called_once_with("test_feature", "Critical error")
    
    def test_user_hash_consistency(self, flag_manager):
        """Test that user hash is consistent for same user/feature combination."""
        hash1 = flag_manager._get_user_hash("user123", "test_feature")
        hash2 = flag_manager._get_user_hash("user123", "test_feature")
        
        assert hash1 == hash2
        assert 0 <= hash1 <= 1
    
    def test_user_hash_different_users(self, flag_manager):
        """Test that different users get different hashes."""
        hash1 = flag_manager._get_user_hash("user123", "test_feature")
        hash2 = flag_manager._get_user_hash("user456", "test_feature")
        
        assert hash1 != hash2


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client."""
        mock_client = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        return mock_client
    
    @pytest.fixture
    def performance_monitor(self, mock_supabase):
        """Create performance monitor with mocked dependencies."""
        with patch('app.utils.performance_monitor.get_supabase_client', return_value=mock_supabase):
            monitor = PerformanceMonitor()
            return monitor
    
    @pytest.mark.asyncio
    async def test_record_metric(self, performance_monitor):
        """Test recording performance metrics."""
        await performance_monitor.record_metric(
            "test_feature", 
            MetricType.RESPONSE_TIME, 
            150.5
        )
        
        # Verify metric was recorded
        performance_monitor.supabase.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_threshold_checking_warning(self, performance_monitor):
        """Test threshold checking triggers warning."""
        # Set up thresholds
        performance_monitor.thresholds["test_feature"] = [
            performance_monitor.PerformanceThreshold(
                MetricType.RESPONSE_TIME, 
                100.0,  # warning
                500.0   # critical
            )
        ]
        
        with patch.object(performance_monitor, '_trigger_alert') as mock_alert:
            await performance_monitor._check_thresholds(
                "test_feature", 
                MetricType.RESPONSE_TIME, 
                200.0  # Above warning threshold
            )
            
            mock_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_automatic_rollback_trigger(self, performance_monitor):
        """Test automatic rollback on critical alerts."""
        # Create multiple critical alerts
        for _ in range(3):
            performance_monitor.alert_history.append(
                performance_monitor.PerformanceAlert(
                    feature_name="test_feature",
                    metric_type=MetricType.ERROR_RATE,
                    current_value=0.25,
                    threshold_value=0.15,
                    severity=AlertSeverity.CRITICAL,
                    message="High error rate",
                    timestamp=datetime.utcnow()
                )
            )
        
        with patch.object(performance_monitor, '_execute_automatic_rollback') as mock_rollback:
            alert = performance_monitor.PerformanceAlert(
                feature_name="test_feature",
                metric_type=MetricType.ERROR_RATE,
                current_value=0.25,
                threshold_value=0.15,
                severity=AlertSeverity.CRITICAL,
                message="High error rate",
                timestamp=datetime.utcnow()
            )
            
            await performance_monitor._consider_automatic_rollback("test_feature", alert)
            mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_manual_rollback(self, performance_monitor):
        """Test manual rollback functionality."""
        mock_flag = Mock()
        mock_flag.status.value = "enabled"
        mock_flag.metadata = {}
        
        with patch('app.utils.performance_monitor.feature_flag_manager') as mock_manager:
            mock_manager.get_feature_flag.return_value = mock_flag
            mock_manager.update_feature_flag.return_value = True
            
            result = await performance_monitor.manual_rollback(
                "test_feature", 
                "Manual rollback test", 
                "disabled", 
                "admin123"
            )
            
            assert result is True
            mock_manager.update_feature_flag.assert_called_once()


class TestUsageAnalytics:
    """Test usage analytics functionality."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client."""
        mock_client = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value.data = []
        return mock_client
    
    @pytest.fixture
    def analytics(self, mock_supabase):
        """Create usage analytics with mocked dependencies."""
        with patch('app.utils.usage_analytics.get_supabase_client', return_value=mock_supabase):
            analytics = UsageAnalytics()
            return analytics
    
    @pytest.mark.asyncio
    async def test_track_event(self, analytics):
        """Test tracking analytics events."""
        event = AnalyticsEvent(
            event_type=EventType.FEATURE_VIEW,
            feature_name="test_feature",
            user_id="user123"
        )
        
        await analytics.track_event(event)
        
        # Event should be in buffer
        assert len(analytics.event_buffer) == 1
        assert analytics.event_buffer[0] == event
    
    @pytest.mark.asyncio
    async def test_track_feature_view(self, analytics):
        """Test tracking feature views."""
        await analytics.track_feature_view("test_feature", "user123", "business456")
        
        assert len(analytics.event_buffer) == 1
        event = analytics.event_buffer[0]
        assert event.event_type == EventType.FEATURE_VIEW
        assert event.feature_name == "test_feature"
        assert event.user_id == "user123"
        assert event.business_id == "business456"
    
    @pytest.mark.asyncio
    async def test_track_feature_interaction(self, analytics):
        """Test tracking feature interactions."""
        await analytics.track_feature_interaction(
            "test_feature", 
            "user123", 
            "click",
            {"button": "submit"}
        )
        
        assert len(analytics.event_buffer) == 1
        event = analytics.event_buffer[0]
        assert event.event_type == EventType.FEATURE_INTERACTION
        assert event.properties["interaction_type"] == "click"
        assert event.properties["button"] == "submit"
    
    @pytest.mark.asyncio
    async def test_track_feature_conversion(self, analytics):
        """Test tracking feature conversions."""
        await analytics.track_feature_conversion(
            "test_feature", 
            "user123", 
            5.0,
            {"conversion_type": "purchase"}
        )
        
        assert len(analytics.event_buffer) == 1
        event = analytics.event_buffer[0]
        assert event.event_type == EventType.FEATURE_CONVERSION
        assert event.properties["conversion_value"] == 5.0
        assert event.properties["conversion_type"] == "purchase"
    
    @pytest.mark.asyncio
    async def test_flush_events(self, analytics):
        """Test flushing events to database."""
        # Add events to buffer
        for i in range(5):
            event = AnalyticsEvent(
                event_type=EventType.FEATURE_VIEW,
                feature_name="test_feature",
                user_id=f"user{i}"
            )
            analytics.event_buffer.append(event)
        
        await analytics._flush_events()
        
        # Buffer should be empty after flush
        assert len(analytics.event_buffer) == 0
        
        # Verify database insert was called
        analytics.supabase.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_buffer_auto_flush(self, analytics):
        """Test automatic buffer flush when full."""
        # Set small buffer size for testing
        analytics.buffer_size = 3
        
        with patch.object(analytics, '_flush_events') as mock_flush:
            # Add events to trigger auto-flush
            for i in range(3):
                event = AnalyticsEvent(
                    event_type=EventType.FEATURE_VIEW,
                    feature_name="test_feature",
                    user_id=f"user{i}"
                )
                await analytics.track_event(event)
            
            # Flush should have been called
            mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_feature_usage_stats_no_data(self, analytics):
        """Test getting usage stats when no data exists."""
        # Mock empty result
        analytics.supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value.data = []
        
        stats = await analytics.get_feature_usage_stats("test_feature", 30)
        
        assert stats["feature_name"] == "test_feature"
        assert stats["no_data"] is True
    
    @pytest.mark.asyncio
    async def test_get_feature_usage_stats_with_data(self, analytics):
        """Test getting usage stats with data."""
        # Mock data
        mock_data = [
            {
                "event_type": "feature_view",
                "user_id": "user1",
                "timestamp": "2023-01-01T10:00:00"
            },
            {
                "event_type": "feature_interaction",
                "user_id": "user1",
                "timestamp": "2023-01-01T10:05:00"
            },
            {
                "event_type": "feature_conversion",
                "user_id": "user1",
                "timestamp": "2023-01-01T10:10:00"
            }
        ]
        
        analytics.supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value.data = mock_data
        
        stats = await analytics.get_feature_usage_stats("test_feature", 30)
        
        assert stats["feature_name"] == "test_feature"
        assert stats["total_events"] == 3
        assert stats["unique_users"] == 1
        assert stats["event_breakdown"]["feature_view"] == 1
        assert stats["event_breakdown"]["feature_interaction"] == 1
        assert stats["event_breakdown"]["feature_conversion"] == 1
        assert stats["conversion_rate"] == 1.0  # 1 conversion / 1 interaction


@pytest.mark.integration
class TestFeatureFlagIntegration:
    """Integration tests for feature flag system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_feature_flag_flow(self):
        """Test complete feature flag flow from creation to analytics."""
        # This would be a more comprehensive integration test
        # that tests the entire flow in a real environment
        pass
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration between feature flags and performance monitoring."""
        # This would test the integration between feature flags
        # and performance monitoring systems
        pass


if __name__ == "__main__":
    pytest.main([__file__])