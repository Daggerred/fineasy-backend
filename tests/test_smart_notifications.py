"""
Tests for Smart Notification System

Tests notification prioritization, scheduling, consolidation,
and preference learning functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from app.services.smart_notifications import (
    SmartNotificationService,
    SmartNotification,
    NotificationPreference,
    NotificationType,
    NotificationPriority,
    NotificationChannel,
    NotificationScheduler
)


class TestSmartNotificationService:
    """Test cases for SmartNotificationService"""
    
    @pytest.fixture
    def notification_service(self):
        """Create a notification service instance for testing"""
        with patch('app.services.smart_notifications.get_supabase_client'), \
             patch('app.services.smart_notifications.get_redis_client'):
            service = SmartNotificationService()
            service.supabase = Mock()
            service.redis = AsyncMock()
            return service
    
    @pytest.fixture
    def sample_notification(self):
        """Create a sample notification for testing"""
        return SmartNotification(
            id="test_notif_123",
            user_id="user_123",
            business_id="business_123",
            type=NotificationType.FRAUD_ALERT,
            priority=NotificationPriority.HIGH,
            title="Test Fraud Alert",
            message="Suspicious activity detected",
            data={"transaction_id": "txn_123"},
            channels=[NotificationChannel.PUSH, NotificationChannel.IN_APP],
            scheduled_for=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self, notification_service):
        """Test successful notification sending"""
        # Mock dependencies
        notification_service._determine_optimal_channels = AsyncMock(return_value=[NotificationChannel.PUSH])
        notification_service._calculate_optimal_send_time = AsyncMock(return_value=datetime.utcnow())
        notification_service._is_user_fatigued = AsyncMock(return_value=False)
        notification_service._store_notification = AsyncMock()
        notification_service._schedule_notification = AsyncMock()
        
        # Test notification sending
        result = await notification_service.send_notification(
            user_id="user_123",
            business_id="business_123",
            notification_type=NotificationType.FRAUD_ALERT,
            title="Test Alert",
            message="Test message",
            priority=NotificationPriority.HIGH
        )
        
        assert result is True
        notification_service._store_notification.assert_called_once()
        notification_service._schedule_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_notification_consolidation(self, notification_service):
        """Test notification consolidation functionality"""
        # Mock Redis to return existing notifications
        existing_notifications = [
            {
                "id": "notif_1",
                "user_id": "user_123",
                "business_id": "business_123",
                "type": "fraud_alert",
                "title": "Fraud Alert 1",
                "message": "First alert"
            }
        ]
        notification_service.redis.get.return_value = json.dumps(existing_notifications)
        notification_service.redis.setex = AsyncMock(return_value=True)
        notification_service._schedule_notification = AsyncMock()
        
        # Create test notification with consolidation key
        notification = SmartNotification(
            id="notif_2",
            user_id="user_123",
            business_id="business_123",
            type=NotificationType.FRAUD_ALERT,
            priority=NotificationPriority.HIGH,
            title="Fraud Alert 2",
            message="Second alert",
            data={},
            channels=[NotificationChannel.PUSH],
            scheduled_for=datetime.utcnow(),
            consolidation_key="fraud_business_123"
        )
        
        # Test consolidation
        result = await notification_service._try_consolidate_notification(notification)
        
        assert result is True
        notification_service.redis.setex.assert_called()
        notification_service._schedule_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fatigue_prevention(self, notification_service):
        """Test notification fatigue prevention"""
        # Mock Redis to return high notification count
        notification_service.redis.get.return_value = "15"  # Above threshold
        notification_service._schedule_for_later = AsyncMock()
        
        # Test fatigue check
        is_fatigued = await notification_service._is_user_fatigued("user_123")
        assert is_fatigued is True
        
        # Test low priority notification gets delayed
        notification_service._determine_optimal_channels = AsyncMock(return_value=[NotificationChannel.PUSH])
        notification_service._calculate_optimal_send_time = AsyncMock(return_value=datetime.utcnow())
        notification_service._store_notification = AsyncMock()
        
        result = await notification_service.send_notification(
            user_id="user_123",
            business_id="business_123",
            notification_type=NotificationType.BUSINESS_INSIGHT,
            title="Business Insight",
            message="New insight available",
            priority=NotificationPriority.LOW
        )
        
        assert result is True
        notification_service._schedule_for_later.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_critical_notifications_bypass_fatigue(self, notification_service):
        """Test that critical notifications bypass fatigue prevention"""
        # Mock Redis to return high notification count
        notification_service.redis.get.return_value = "15"  # Above threshold
        notification_service._determine_optimal_channels = AsyncMock(return_value=[NotificationChannel.PUSH])
        notification_service._calculate_optimal_send_time = AsyncMock(return_value=datetime.utcnow())
        notification_service._store_notification = AsyncMock()
        notification_service._schedule_notification = AsyncMock()
        
        # Test critical notification bypasses fatigue
        result = await notification_service.send_notification(
            user_id="user_123",
            business_id="business_123",
            notification_type=NotificationType.FRAUD_ALERT,
            title="Critical Fraud Alert",
            message="Urgent security issue",
            priority=NotificationPriority.CRITICAL
        )
        
        assert result is True
        notification_service._schedule_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimal_channel_determination(self, notification_service):
        """Test optimal channel determination based on user preferences"""
        # Mock user preferences
        preferences = [
            NotificationPreference(
                user_id="user_123",
                channel=NotificationChannel.PUSH,
                notification_type=NotificationType.FRAUD_ALERT,
                enabled=True
            ),
            NotificationPreference(
                user_id="user_123",
                channel=NotificationChannel.EMAIL,
                notification_type=NotificationType.FRAUD_ALERT,
                enabled=False
            )
        ]
        notification_service._get_user_preferences = AsyncMock(return_value=preferences)
        
        # Test channel determination
        channels = await notification_service._determine_optimal_channels(
            user_id="user_123",
            notification_type=NotificationType.FRAUD_ALERT,
            priority=NotificationPriority.HIGH
        )
        
        assert NotificationChannel.PUSH in channels
        assert NotificationChannel.EMAIL not in channels
        assert NotificationChannel.IN_APP in channels  # Default for high priority
    
    @pytest.mark.asyncio
    async def test_optimal_send_time_calculation(self, notification_service):
        """Test optimal send time calculation based on user activity"""
        # Mock user active hours
        active_hours = {
            'start_hour': 9,
            'end_hour': 18,
            'peak_hours': [9, 10, 14, 15, 16, 17]
        }
        notification_service._get_user_active_hours = AsyncMock(return_value=active_hours)
        
        # Test during active hours
        with patch('app.services.smart_notifications.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 14, 0, 0)  # 2 PM
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            send_time = await notification_service._calculate_optimal_send_time(
                user_id="user_123",
                priority=NotificationPriority.MEDIUM
            )
            
            # Should be immediate during active hours
            assert send_time.hour == 14
    
    @pytest.mark.asyncio
    async def test_user_preference_learning(self, notification_service):
        """Test user preference learning from interactions"""
        # Mock database response
        notification_data = {
            'id': 'notif_123',
            'type': 'fraud_alert',
            'priority': 'high',
            'sent_at': datetime.utcnow().isoformat()
        }
        notification_service.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [notification_data]
        notification_service.supabase.table.return_value.insert.return_value.execute = Mock()
        
        # Test learning from acknowledgment
        await notification_service._learn_from_acknowledgment("notif_123", "user_123")
        
        # Verify interaction was recorded
        notification_service.supabase.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_notification_analytics(self, notification_service):
        """Test notification analytics generation"""
        # Mock database responses
        metrics_data = [
            {'delivered_at': datetime.utcnow().isoformat()},
            {'delivered_at': datetime.utcnow().isoformat()}
        ]
        interactions_data = [
            {'response_time_seconds': 120},
            {'response_time_seconds': 180}
        ]
        
        notification_service.supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value.data = metrics_data
        
        # Mock second call for interactions
        def side_effect(*args, **kwargs):
            if 'user_interactions' in str(args):
                return Mock(data=interactions_data)
            return Mock(data=metrics_data)
        
        notification_service.supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.side_effect = side_effect
        
        # Test analytics generation
        analytics = await notification_service.get_notification_analytics("user_123")
        
        assert 'total_notifications_sent' in analytics
        assert 'acknowledgment_rate' in analytics
        assert 'average_response_time_minutes' in analytics


class TestNotificationScheduler:
    """Test cases for NotificationScheduler"""
    
    @pytest.fixture
    def scheduler(self):
        """Create a notification scheduler for testing"""
        with patch('app.services.smart_notifications.get_redis_client'):
            scheduler = NotificationScheduler()
            scheduler.redis = AsyncMock()
            scheduler.notification_service = Mock()
            scheduler.notification_service._deliver_notification = AsyncMock()
            return scheduler
    
    @pytest.mark.asyncio
    async def test_scheduled_notification_processing(self, scheduler):
        """Test processing of scheduled notifications"""
        # Mock Redis keys and data
        current_time = datetime.utcnow()
        past_time = current_time - timedelta(minutes=5)
        
        scheduler.redis.keys.return_value = [f"scheduled_notification:{past_time.timestamp()}"]
        
        notification_data = {
            'id': 'notif_123',
            'user_id': 'user_123',
            'business_id': 'business_123',
            'type': 'fraud_alert',
            'priority': 'high',
            'title': 'Test Alert',
            'message': 'Test message',
            'data': {},
            'channels': ['push'],
            'scheduled_for': past_time.isoformat(),
            'created_at': current_time.isoformat()
        }
        
        scheduler.redis.get.return_value = json.dumps(notification_data)
        scheduler.redis.delete = AsyncMock()
        
        # Test processing
        await scheduler._process_scheduled_notifications()
        
        # Verify notification was delivered and removed from queue
        scheduler.notification_service._deliver_notification.assert_called_once()
        scheduler.redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scheduler_error_handling(self, scheduler):
        """Test scheduler error handling"""
        # Mock Redis to raise an exception
        scheduler.redis.keys.side_effect = Exception("Redis connection error")
        
        # Test that scheduler handles errors gracefully
        try:
            await scheduler._process_scheduled_notifications()
        except Exception:
            pytest.fail("Scheduler should handle errors gracefully")


class TestNotificationModels:
    """Test cases for notification data models"""
    
    def test_smart_notification_creation(self):
        """Test SmartNotification model creation"""
        notification = SmartNotification(
            id="test_123",
            user_id="user_123",
            business_id="business_123",
            type=NotificationType.FRAUD_ALERT,
            priority=NotificationPriority.HIGH,
            title="Test Alert",
            message="Test message",
            data={"key": "value"},
            channels=[NotificationChannel.PUSH],
            scheduled_for=datetime.utcnow()
        )
        
        assert notification.id == "test_123"
        assert notification.type == NotificationType.FRAUD_ALERT
        assert notification.priority == NotificationPriority.HIGH
        assert notification.created_at is not None
    
    def test_notification_preference_creation(self):
        """Test NotificationPreference model creation"""
        preference = NotificationPreference(
            user_id="user_123",
            channel=NotificationChannel.EMAIL,
            notification_type=NotificationType.BUSINESS_INSIGHT,
            enabled=True,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
            frequency_limit=5
        )
        
        assert preference.user_id == "user_123"
        assert preference.channel == NotificationChannel.EMAIL
        assert preference.enabled is True
        assert preference.frequency_limit == 5


class TestNotificationIntegration:
    """Integration tests for notification system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_notification_flow(self):
        """Test complete notification flow from creation to delivery"""
        with patch('app.services.smart_notifications.get_supabase_client'), \
             patch('app.services.smart_notifications.get_redis_client'):
            
            service = SmartNotificationService()
            service.supabase = Mock()
            service.redis = AsyncMock()
            
            # Mock all dependencies
            service._determine_optimal_channels = AsyncMock(return_value=[NotificationChannel.PUSH])
            service._calculate_optimal_send_time = AsyncMock(return_value=datetime.utcnow())
            service._is_user_fatigued = AsyncMock(return_value=False)
            service._store_notification = AsyncMock()
            service._deliver_notification = AsyncMock()
            
            # Test complete flow
            result = await service.send_notification(
                user_id="user_123",
                business_id="business_123",
                notification_type=NotificationType.FRAUD_ALERT,
                title="Integration Test Alert",
                message="This is an integration test",
                priority=NotificationPriority.HIGH
            )
            
            assert result is True
            service._store_notification.assert_called_once()
            service._deliver_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_notification_preference_update_flow(self):
        """Test notification preference update flow"""
        with patch('app.services.smart_notifications.get_supabase_client'):
            service = SmartNotificationService()
            service.supabase = Mock()
            service.supabase.table.return_value.delete.return_value.eq.return_value.execute = Mock()
            service.supabase.table.return_value.insert.return_value.execute = Mock()
            
            preferences = [
                NotificationPreference(
                    user_id="user_123",
                    channel=NotificationChannel.PUSH,
                    notification_type=NotificationType.FRAUD_ALERT,
                    enabled=True
                )
            ]
            
            result = await service.update_user_preferences("user_123", preferences)
            
            assert result is True
            # Verify database operations were called
            service.supabase.table.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])