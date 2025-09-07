#!/usr/bin/env python3
"""
Emergency rollback script for AI features.
This script can be used to quickly disable or rollback features in production.
"""
import asyncio
import argparse
import sys
import logging
from datetime import datetime
from typing import Optional

# Add the app directory to the path
sys.path.append('/app')

from app.utils.feature_flags import feature_flag_manager, FeatureFlagStatus
from app.utils.performance_monitor import performance_monitor
from app.database import init_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def rollback_feature(
    feature_name: str,
    target_status: str,
    reason: str,
    admin_user_id: str = "system",
    rollout_percentage: Optional[float] = None
):
    """
    Rollback a feature to a specific status.
    
    Args:
        feature_name: Name of the feature to rollback
        target_status: Target status (disabled, rollout, testing)
        reason: Reason for rollback
        admin_user_id: ID of admin performing rollback
        rollout_percentage: New rollout percentage (for rollout status)
    """
    try:
        logger.info(f"Starting rollback for feature: {feature_name}")
        
        # Initialize database connection
        await init_database()
        
        # Get current feature flag
        current_flag = await feature_flag_manager.get_feature_flag(feature_name)
        if not current_flag:
            logger.error(f"Feature flag '{feature_name}' not found")
            return False
        
        logger.info(f"Current status: {current_flag.status.value}")
        logger.info(f"Current rollout: {current_flag.rollout_percentage}%")
        
        # Prepare updates
        updates = {
            "status": target_status,
            "metadata": {
                **current_flag.metadata,
                "rollback_performed": True,
                "rollback_reason": reason,
                "rollback_timestamp": datetime.utcnow().isoformat(),
                "rollback_by": admin_user_id,
                "previous_status": current_flag.status.value,
                "previous_rollout_percentage": current_flag.rollout_percentage
            }
        }
        
        # Set rollout percentage if provided
        if rollout_percentage is not None:
            updates["rollout_percentage"] = rollout_percentage
        elif target_status == "disabled":
            updates["rollout_percentage"] = 0.0
        
        # Perform rollback
        success = await feature_flag_manager.update_feature_flag(feature_name, updates)
        
        if success:
            logger.info(f"✅ Successfully rolled back {feature_name}")
            logger.info(f"   Status: {current_flag.status.value} → {target_status}")
            if rollout_percentage is not None:
                logger.info(f"   Rollout: {current_flag.rollout_percentage}% → {rollout_percentage}%")
            logger.info(f"   Reason: {reason}")
            
            # Use performance monitor to record manual rollback
            await performance_monitor.manual_rollback(
                feature_name, reason, target_status, admin_user_id
            )
            
            return True
        else:
            logger.error(f"❌ Failed to rollback {feature_name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during rollback: {e}")
        return False


async def emergency_disable(feature_name: str, reason: str):
    """
    Emergency disable a feature completely.
    
    Args:
        feature_name: Name of the feature to disable
        reason: Reason for emergency disable
    """
    logger.warning(f"🚨 EMERGENCY DISABLE: {feature_name}")
    logger.warning(f"Reason: {reason}")
    
    success = await feature_flag_manager.emergency_disable_feature(feature_name, reason)
    
    if success:
        logger.info(f"✅ Emergency disable successful for {feature_name}")
    else:
        logger.error(f"❌ Emergency disable failed for {feature_name}")
    
    return success


async def list_features():
    """List all feature flags and their current status."""
    try:
        await init_database()
        
        features = [
            "fraud_detection",
            "predictive_insights", 
            "compliance_checking",
            "nlp_invoice_generation",
            "smart_notifications",
            "ml_analytics_engine"
        ]
        
        logger.info("Current feature flag status:")
        logger.info("=" * 50)
        
        for feature in features:
            flag = await feature_flag_manager.get_feature_flag(feature)
            if flag:
                status_emoji = {
                    "enabled": "🟢",
                    "disabled": "🔴", 
                    "testing": "🟡",
                    "rollout": "🟠"
                }.get(flag.status.value, "⚪")
                
                logger.info(f"{status_emoji} {feature}")
                logger.info(f"   Status: {flag.status.value}")
                logger.info(f"   Rollout: {flag.rollout_percentage}%")
                if flag.ab_test_enabled:
                    logger.info(f"   A/B Test: {flag.ab_test_variants}")
                if flag.metadata.get("rollback_performed"):
                    logger.info(f"   Last Rollback: {flag.metadata.get('rollback_timestamp')}")
                    logger.info(f"   Rollback Reason: {flag.metadata.get('rollback_reason')}")
                logger.info("")
            else:
                logger.info(f"⚪ {feature} - Not found")
                
    except Exception as e:
        logger.error(f"Error listing features: {e}")


async def gradual_rollout(feature_name: str, target_percentage: float, reason: str):
    """
    Gradually adjust rollout percentage for a feature.
    
    Args:
        feature_name: Name of the feature
        target_percentage: Target rollout percentage
        reason: Reason for adjustment
    """
    try:
        await init_database()
        
        current_flag = await feature_flag_manager.get_feature_flag(feature_name)
        if not current_flag:
            logger.error(f"Feature flag '{feature_name}' not found")
            return False
        
        logger.info(f"Adjusting rollout for {feature_name}")
        logger.info(f"Current: {current_flag.rollout_percentage}% → Target: {target_percentage}%")
        
        updates = {
            "rollout_percentage": target_percentage,
            "status": "rollout" if target_percentage > 0 else "disabled",
            "metadata": {
                **current_flag.metadata,
                "rollout_adjustment": True,
                "adjustment_reason": reason,
                "adjustment_timestamp": datetime.utcnow().isoformat(),
                "previous_rollout_percentage": current_flag.rollout_percentage
            }
        }
        
        success = await feature_flag_manager.update_feature_flag(feature_name, updates)
        
        if success:
            logger.info(f"✅ Rollout adjustment successful")
        else:
            logger.error(f"❌ Rollout adjustment failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error adjusting rollout: {e}")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Feature flag rollback utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a feature")
    rollback_parser.add_argument("feature", help="Feature name to rollback")
    rollback_parser.add_argument("status", choices=["disabled", "rollout", "testing"], 
                                help="Target status")
    rollback_parser.add_argument("reason", help="Reason for rollback")
    rollback_parser.add_argument("--rollout-percentage", type=float, 
                                help="Rollout percentage (for rollout status)")
    rollback_parser.add_argument("--admin-id", default="system", 
                                help="Admin user ID performing rollback")
    
    # Emergency disable command
    emergency_parser = subparsers.add_parser("emergency", help="Emergency disable a feature")
    emergency_parser.add_argument("feature", help="Feature name to disable")
    emergency_parser.add_argument("reason", help="Reason for emergency disable")
    
    # List command
    subparsers.add_parser("list", help="List all features and their status")
    
    # Gradual rollout command
    gradual_parser = subparsers.add_parser("gradual", help="Adjust rollout percentage")
    gradual_parser.add_argument("feature", help="Feature name")
    gradual_parser.add_argument("percentage", type=float, help="Target rollout percentage")
    gradual_parser.add_argument("reason", help="Reason for adjustment")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == "rollback":
        asyncio.run(rollback_feature(
            args.feature, 
            args.status, 
            args.reason,
            args.admin_id,
            args.rollout_percentage
        ))
    elif args.command == "emergency":
        asyncio.run(emergency_disable(args.feature, args.reason))
    elif args.command == "list":
        asyncio.run(list_features())
    elif args.command == "gradual":
        asyncio.run(gradual_rollout(args.feature, args.percentage, args.reason))


if __name__ == "__main__":
    main()