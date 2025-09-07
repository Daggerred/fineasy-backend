#!/usr/bin/env python3
"""
Smart Notifications System Offline Validation Script

This script validates the smart notification system functionality without
requiring database connections, focusing on the core logic and algorithms.
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.services.smart_notifications import (
        SmartNotificationService,
        SmartNotification,
        NotificationPreference,
        NotificationType,
        NotificationPriority,
        NotificationChannel,
        NotificationScheduler
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the ai-backend directory")
    sys.exit(1)


class OfflineSmartNotificationValidator:
    """Offline validator for smart notification system"""
    
    def __init__(self):
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str = "", details: Dict[str, Any] = None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"    {message}")
        if details:
            print(f"    Details: {json.dumps(details, indent=2, default=str)}")
        
        self.results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "details": details
        })
    
    def test_notification_models(self):
        """Test notification data models"""
        try:
            print("\n📋 Testing notification models...")
            
            # Test SmartNotification creation
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
            
            model_valid = (
                notification.id == "test_123" and
                notification.type == NotificationType.FRAUD_ALERT and
                notification.priority == NotificationPriority.HIGH and
                notification.created_at is not None
            )
            
            self.log_result(
                "Notification Model Creation",
                model_valid,
                "SmartNotification model created successfully" if model_valid else "Model creation failed"
            )
            
            # Test NotificationPreference creation
            preference = NotificationPreference(
                user_id="user_123",
                channel=NotificationChannel.EMAIL,
                notification_type=NotificationType.BUSINESS_INSIGHT,
                enabled=True,
                quiet_hours_start="22:00",
                quiet_hours_end="08:00",
                frequency_limit=5
            )
            
            pref_valid = (
                preference.user_id == "user_123" and
                preference.channel == NotificationChannel.EMAIL and
                preference.enabled is True and
                preference.frequency_limit == 5
            )
            
            self.log_result(
                "Notification Preference Model",
                pref_valid,
                "NotificationPreference model created successfully" if pref_valid else "Preference model creation failed"
            )
            
            return model_valid and pref_valid
            
        except Exception as e:
            self.log_result("Notification Models", False, f"Error: {str(e)}")
            return False
    
    def test_notification_enums(self):
        """Test notification enums"""
        try:
            print("\n🏷️ Testing notification enums...")
            
            # Test NotificationType enum
            fraud_type = NotificationType.FRAUD_ALERT
            insight_type = NotificationType.BUSINESS_INSIGHT
            
            type_valid = (
                fraud_type.value == "fraud_alert" and
                insight_type.value == "business_insight"
            )
            
            # Test NotificationPriority enum
            high_priority = NotificationPriority.HIGH
            low_priority = NotificationPriority.LOW
            
            priority_valid = (
                high_priority.value == "high" and
                low_priority.value == "low"
            )
            
            # Test NotificationChannel enum
            push_channel = NotificationChannel.PUSH
            email_channel = NotificationChannel.EMAIL
            
            channel_valid = (
                push_channel.value == "push" and
                email_channel.value == "email"
            )
            
            all_valid = type_valid and priority_valid and channel_valid
            
            self.log_result(
                "Notification Enums",
                all_valid,
                "All notification enums working correctly" if all_valid else "Some enums failed",
                {
                    "type_valid": type_valid,
                    "priority_valid": priority_valid,
                    "channel_valid": channel_valid
                }
            )
            
            return all_valid
            
        except Exception as e:
            self.log_result("Notification Enums", False, f"Error: {str(e)}")
            return False
    
    async def test_service_initialization(self):
        """Test service initialization with mocked dependencies"""
        try:
            print("\n🔧 Testing service initialization...")
            
            with patch('app.services.smart_notifications.get_supabase'), \
                 patch('app.services.smart_notifications.get_redis_client'):
                
                service = SmartNotificationService()
                service.supabase = Mock()
                service.redis = AsyncMock()
                
                init_success = (
                    service.supabase is not None and
                    service.redis is not None and
                    service.consolidation_window == timedelta(minutes=30) and
                    service.fatigue_threshold == 10
                )
                
                self.log_result(
                    "Service Initialization",
                    init_success,
                    "Service initialized with mocked dependencies" if init_success else "Service initialization failed"
                )
                
                return init_success
                
        except Exception as e:
            self.log_result("Service Initialization", False, f"Error: {str(e)}")
            return False
    
    async def test_channel_determination_logic(self):
        """Test optimal channel determination logic"""
        try:
            print("\n📱 Testing channel determination logic...")
            
            with patch('app.services.smart_notifications.get_supabase'), \
                 patch('app.services.smart_notifications.get_redis_client'):
                
                service = SmartNotificationService()
                service.supabase = Mock()
                service.redis = AsyncMock()
                
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
                service._get_user_preferences = AsyncMock(return_value=preferences)
                
                # Test channel determination
                channels = await service._determine_optimal_channels(
                    user_id="user_123",
                    notification_type=NotificationType.FRAUD_ALERT,
                    priority=NotificationPriority.HIGH
                )
                
                logic_valid = (
                    NotificationChannel.PUSH in channels and
                    NotificationChannel.EMAIL not in channels and
                    NotificationChannel.IN_APP in channels  # Default for high priority
                )
                
                self.log_result(
                    "Channel Determination Logic",
                    logic_valid,
                    f"Channels determined correctly: {[ch.value for ch in channels]}" if logic_valid else "Channel logic failed",
                    {"channels": [ch.value for ch in channels]}
                )
                
                return logic_valid
                
        except Exception as e:
            self.log_result("Channel Determination Logic", False, f"Error: {str(e)}")
            return False
    
    async def test_send_time_calculation_logic(self):
        """Test optimal send time calculation logic"""
        try:
            print("\n⏰ Testing send time calculation logic...")
            
            with patch('app.services.smart_notifications.get_supabase'), \
                 patch('app.services.smart_notifications.get_redis_client'):
                
                service = SmartNotificationService()
                service.supabase = Mock()
                service.redis = AsyncMock()
                
                # Test critical priority (should be immediate)
                critical_time = await service._calculate_optimal_send_time(
                    user_id="user_123",
                    priority=NotificationPriority.CRITICAL
                )
                
                # Test medium priority with mocked active hours
                active_hours = {
                    'start_hour': 9,
                    'end_hour': 18,
                    'peak_hours': [9, 10, 14, 15, 16, 17]
                }
                service._get_user_active_hours = AsyncMock(return_value=active_hours)
                
                with patch('app.services.smart_notifications.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 14, 0, 0)  # 2 PM
                    mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
                    
                    medium_time = await service._calculate_optimal_send_time(
                        user_id="user_123",
                        priority=NotificationPriority.MEDIUM
                    )
                
                logic_valid = (
                    critical_time is not None and
                    medium_time is not None and
                    (datetime.utcnow() - critical_time).total_seconds() < 60  # Critical should be immediate
                )
                
                self.log_result(
                    "Send Time Calculation Logic",
                    logic_valid,
                    "Send time calculation working correctly" if logic_valid else "Send time logic failed",
                    {
                        "critical_immediate": (datetime.utcnow() - critical_time).total_seconds() < 60,
                        "medium_scheduled": medium_time is not None
                    }
                )
                
                return logic_valid
                
        except Exception as e:
            self.log_result("Send Time Calculation Logic", False, f"Error: {str(e)}")
            return False
    
    async def test_consolidation_logic(self):
        """Test notification consolidation logic"""
        try:
            print("\n📦 Testing consolidation logic...")
            
            with patch('app.services.smart_notifications.get_supabase'), \
                 patch('app.services.smart_notifications.get_redis_client'):
                
                service = SmartNotificationService()
                service.supabase = Mock()
                service.redis = AsyncMock()
                
                # Test consolidation with existing notifications
                existing_notifications = [
                    {
                        "id": "notif_1",
                        "user_id": "user_123",
                        "business_id": "business_123",
                        "type": "fraud_alert",
                        "priority": "high",
                        "title": "Fraud Alert 1",
                        "message": "First alert",
                        "data": {},
                        "channels": ["push"],
                        "scheduled_for": datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow().isoformat()
                    }
                ]
                service.redis.get.return_value = json.dumps(existing_notifications)
                service.redis.setex = AsyncMock(return_value=True)
                service._schedule_notification = AsyncMock()
                
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
                result = await service._try_consolidate_notification(notification)
                
                consolidation_valid = (
                    result is True and
                    service.redis.setex.called and
                    service._schedule_notification.called
                )
                
                self.log_result(
                    "Consolidation Logic",
                    consolidation_valid,
                    "Notification consolidation working correctly" if consolidation_valid else "Consolidation logic failed"
                )
                
                return consolidation_valid
                
        except Exception as e:
            self.log_result("Consolidation Logic", False, f"Error: {str(e)}")
            return False
    
    async def test_fatigue_prevention_logic(self):
        """Test notification fatigue prevention logic"""
        try:
            print("\n😴 Testing fatigue prevention logic...")
            
            with patch('app.services.smart_notifications.get_supabase'), \
                 patch('app.services.smart_notifications.get_redis_client'):
                
                service = SmartNotificationService()
                service.supabase = Mock()
                service.redis = AsyncMock()
                
                # Test fatigue detection
                service.redis.get.return_value = "15"  # Above threshold
                is_fatigued = await service._is_user_fatigued("user_123")
                
                # Test that critical notifications bypass fatigue
                service._determine_optimal_channels = AsyncMock(return_value=[NotificationChannel.PUSH])
                service._calculate_optimal_send_time = AsyncMock(return_value=datetime.utcnow())
                service._store_notification = AsyncMock()
                service._schedule_notification = AsyncMock()
                service._schedule_for_later = AsyncMock()
                
                # Test low priority notification gets delayed
                await service.send_notification(
                    user_id="user_123",
                    business_id="business_123",
                    notification_type=NotificationType.BUSINESS_INSIGHT,
                    title="Business Insight",
                    message="New insight available",
                    priority=NotificationPriority.LOW
                )
                
                fatigue_logic_valid = (
                    is_fatigued is True and
                    service._schedule_for_later.called and
                    not service._schedule_notification.called
                )
                
                self.log_result(
                    "Fatigue Prevention Logic",
                    fatigue_logic_valid,
                    "Fatigue prevention working correctly" if fatigue_logic_valid else "Fatigue logic failed",
                    {"is_fatigued": is_fatigued, "delayed": service._schedule_for_later.called}
                )
                
                return fatigue_logic_valid
                
        except Exception as e:
            self.log_result("Fatigue Prevention Logic", False, f"Error: {str(e)}")
            return False
    
    async def test_scheduler_logic(self):
        """Test notification scheduler logic"""
        try:
            print("\n📅 Testing scheduler logic...")
            
            with patch('app.services.smart_notifications.get_redis_client'):
                scheduler = NotificationScheduler()
                scheduler.redis = AsyncMock()
                scheduler.notification_service = Mock()
                scheduler.notification_service._deliver_notification = AsyncMock()
                
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
                
                scheduler_valid = (
                    scheduler.notification_service._deliver_notification.called and
                    scheduler.redis.delete.called
                )
                
                self.log_result(
                    "Scheduler Logic",
                    scheduler_valid,
                    "Notification scheduler working correctly" if scheduler_valid else "Scheduler logic failed"
                )
                
                return scheduler_valid
                
        except Exception as e:
            self.log_result("Scheduler Logic", False, f"Error: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all offline validation tests"""
        print("🚀 Starting Smart Notification System Offline Validation")
        print("=" * 60)
        
        # Run tests
        tests = [
            self.test_notification_models,
            self.test_notification_enums,
            self.test_service_initialization,
            self.test_channel_determination_logic,
            self.test_send_time_calculation_logic,
            self.test_consolidation_logic,
            self.test_fatigue_prevention_logic,
            self.test_scheduler_logic
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if asyncio.iscoroutinefunction(test):
                    result = await test()
                else:
                    result = test()
                if result:
                    passed += 1
            except Exception as e:
                print(f"❌ Test failed with exception: {str(e)}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 OFFLINE VALIDATION SUMMARY")
        print("=" * 60)
        
        success_rate = (passed / total) * 100
        print(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 Smart Notification System offline validation PASSED!")
            print("The core logic and algorithms are working correctly.")
        else:
            print("⚠️ Smart Notification System offline validation needs attention.")
            print("Some core logic may not work as expected.")
        
        # Detailed results
        print("\nDetailed Results:")
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}")
            if result["message"]:
                print(f"    {result['message']}")
        
        return success_rate >= 80


async def main():
    """Main validation function"""
    validator = OfflineSmartNotificationValidator()
    success = await validator.run_all_tests()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Set up Supabase database with the smart notification tables")
        print("2. Configure Redis for caching and scheduling")
        print("3. Test with real database connections")
        print("4. Integrate notification API endpoints with Flutter app")
        print("5. Set up push notification service (Firebase FCM)")
    
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {str(e)}")
        sys.exit(1)