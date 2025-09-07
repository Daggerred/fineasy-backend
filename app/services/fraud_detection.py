"""
Fraud Detection Service - AI-powered fraud and error detection
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import hashlib
import re
from collections import defaultdict, Counter
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from ..models.base import FraudAlert, FraudType
from ..models.responses import FraudAnalysisResponse
from ..database import DatabaseManager

logger = logging.getLogger(__name__)


class FraudDetector:
    """AI-powered fraud detection service"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.duplicate_threshold = 0.85
        self.pattern_threshold = 0.75
        self.mismatch_threshold = 0.95
        
        # Initialize ML models for pattern detection
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    async def analyze_fraud(self, business_id: str) -> FraudAnalysisResponse:
        """Comprehensive fraud analysis for a business"""
        logger.info(f"Starting fraud analysis for business: {business_id}")
        
        try:
            all_alerts = []
            
            # Run all fraud detection algorithms
            duplicate_alerts = await self.detect_duplicates(business_id)
            mismatch_alerts = await self.detect_payment_mismatches(business_id)
            pattern_alerts = await self.analyze_transaction_patterns(business_id)
            
            all_alerts.extend(duplicate_alerts)
            all_alerts.extend(mismatch_alerts)
            all_alerts.extend(pattern_alerts)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(all_alerts)
            
            # Save alerts to database
            for alert in all_alerts:
                await self._save_fraud_alert(alert)
            
            # Log analysis completion
            await self._log_fraud_analysis(business_id, len(all_alerts), risk_score)
            
            logger.info(f"Fraud analysis completed for business {business_id}: {len(all_alerts)} alerts, risk score: {risk_score}")
            
            return FraudAnalysisResponse(
                business_id=business_id,
                alerts=all_alerts,
                risk_score=risk_score,
                analysis_metadata={
                    "duplicate_alerts": len(duplicate_alerts),
                    "mismatch_alerts": len(mismatch_alerts),
                    "pattern_alerts": len(pattern_alerts),
                    "total_alerts": len(all_alerts),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")
            return FraudAnalysisResponse(
                success=False,
                message=f"Fraud analysis failed: {str(e)}",
                business_id=business_id,
                alerts=[],
                risk_score=0.0
            )
    
    async def detect_duplicates(self, business_id: str) -> List[FraudAlert]:
        """Detect duplicate invoices and transactions using fuzzy matching"""
        logger.info(f"Starting duplicate detection for business: {business_id}")
        alerts = []
        
        try:
            # Get recent invoices and transactions
            invoices = await self.db.get_invoices(business_id, limit=1000)
            transactions = await self.db.get_transactions(business_id, limit=1000)
            
            # Detect duplicate invoices
            invoice_alerts = await self._detect_duplicate_invoices(invoices, business_id)
            alerts.extend(invoice_alerts)
            
            # Detect duplicate transactions
            transaction_alerts = await self._detect_duplicate_transactions(transactions, business_id)
            alerts.extend(transaction_alerts)
            
            # Detect supplier duplicate billing
            supplier_alerts = await self._detect_supplier_duplicates(invoices, business_id)
            alerts.extend(supplier_alerts)
            
            logger.info(f"Duplicate detection completed: {len(alerts)} alerts found")
            return alerts
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            return []
    
    async def detect_payment_mismatches(self, business_id: str) -> List[FraudAlert]:
        """Detect payment mismatches between records"""
        logger.info(f"Starting payment mismatch detection for business: {business_id}")
        alerts = []
        
        try:
            # Get invoices and payments
            invoices = await self.db.get_invoices(business_id, limit=1000)
            transactions = await self.db.get_transactions(business_id, limit=1000)
            
            # Create payment tracking
            payment_map = self._create_payment_map(transactions)
            
            # Check each invoice for payment mismatches
            for invoice in invoices:
                mismatch_alert = await self._check_invoice_payment_mismatch(
                    invoice, payment_map, business_id
                )
                if mismatch_alert:
                    alerts.append(mismatch_alert)
            
            # Check for orphaned payments
            orphaned_alerts = await self._detect_orphaned_payments(
                payment_map, invoices, business_id
            )
            alerts.extend(orphaned_alerts)
            
            logger.info(f"Payment mismatch detection completed: {len(alerts)} alerts found")
            return alerts
            
        except Exception as e:
            logger.error(f"Error in payment mismatch detection: {e}")
            return []
    
    async def analyze_transaction_patterns(self, business_id: str) -> List[FraudAlert]:
        """Analyze suspicious transaction patterns using ML"""
        logger.info(f"Starting transaction pattern analysis for business: {business_id}")
        alerts = []
        
        try:
            # Get transaction data
            transactions = await self.db.get_transactions(business_id, limit=2000)
            
            if len(transactions) < 10:
                logger.info("Insufficient transaction data for pattern analysis")
                return []
            
            # Prepare features for ML analysis
            features = self._extract_transaction_features(transactions)
            
            if len(features) == 0:
                return []
            
            # Detect anomalies using Isolation Forest
            anomaly_alerts = await self._detect_anomalous_transactions(
                transactions, features, business_id
            )
            alerts.extend(anomaly_alerts)
            
            # Detect suspicious patterns
            pattern_alerts = await self._detect_suspicious_patterns(transactions, business_id)
            alerts.extend(pattern_alerts)
            
            # Detect velocity anomalies
            velocity_alerts = await self._detect_velocity_anomalies(transactions, business_id)
            alerts.extend(velocity_alerts)
            
            logger.info(f"Transaction pattern analysis completed: {len(alerts)} alerts found")
            return alerts
            
        except Exception as e:
            logger.error(f"Error in transaction pattern analysis: {e}")
            return []
    
    # =============================================================================
    # Private Helper Methods - Duplicate Detection
    # =============================================================================
    
    async def _detect_duplicate_invoices(self, invoices: List[Dict], business_id: str) -> List[FraudAlert]:
        """Detect duplicate invoices using fuzzy matching"""
        alerts = []
        
        # Group invoices by customer for more efficient comparison
        customer_invoices = defaultdict(list)
        for invoice in invoices:
            customer_id = invoice.get('customer_id', 'unknown')
            customer_invoices[customer_id].append(invoice)
        
        # Check for duplicates within each customer group
        for customer_id, customer_invoice_list in customer_invoices.items():
            for i, invoice1 in enumerate(customer_invoice_list):
                for j, invoice2 in enumerate(customer_invoice_list[i+1:], i+1):
                    similarity = self._calculate_invoice_similarity(invoice1, invoice2)
                    
                    if similarity >= self.duplicate_threshold:
                        alert = FraudAlert(
                            type=FraudType.DUPLICATE_INVOICE,
                            message=f"Potential duplicate invoice detected: Invoice #{invoice1.get('invoice_number', 'N/A')} and #{invoice2.get('invoice_number', 'N/A')} are {similarity:.1%} similar",
                            confidence_score=similarity,
                            evidence={
                                "original_invoice_id": invoice1.get('id'),
                                "duplicate_invoice_id": invoice2.get('id'),
                                "original_invoice_number": invoice1.get('invoice_number'),
                                "duplicate_invoice_number": invoice2.get('invoice_number'),
                                "similarity_score": similarity,
                                "customer_id": customer_id,
                                "amount_difference": abs(float(invoice1.get('total_amount', 0)) - float(invoice2.get('total_amount', 0))),
                                "date_difference_days": self._calculate_date_difference(
                                    invoice1.get('created_at'), invoice2.get('created_at')
                                )
                            },
                            business_id=business_id,
                            entity_id=invoice2.get('id')
                        )
                        alerts.append(alert)
        
        return alerts
    
    async def _detect_duplicate_transactions(self, transactions: List[Dict], business_id: str) -> List[FraudAlert]:
        """Detect duplicate transactions"""
        alerts = []
        
        # Sort transactions by date for efficient comparison
        sorted_transactions = sorted(transactions, key=lambda x: x.get('created_at', ''))
        
        for i, trans1 in enumerate(sorted_transactions):
            for j, trans2 in enumerate(sorted_transactions[i+1:], i+1):
                # Only compare transactions within 48 hours of each other
                date_diff = self._calculate_date_difference(
                    trans1.get('created_at'), trans2.get('created_at')
                )
                
                if date_diff > 2:  # More than 2 days apart
                    break
                
                similarity = self._calculate_transaction_similarity(trans1, trans2)
                
                if similarity >= self.duplicate_threshold:
                    alert = FraudAlert(
                        type=FraudType.DUPLICATE_INVOICE,
                        message=f"Potential duplicate transaction detected: {similarity:.1%} similarity",
                        confidence_score=similarity,
                        evidence={
                            "original_transaction_id": trans1.get('id'),
                            "duplicate_transaction_id": trans2.get('id'),
                            "similarity_score": similarity,
                            "amount": trans1.get('amount'),
                            "description": trans1.get('description'),
                            "date_difference_hours": date_diff * 24
                        },
                        business_id=business_id,
                        entity_id=trans2.get('id')
                    )
                    alerts.append(alert)
        
        return alerts
    
    async def _detect_supplier_duplicates(self, invoices: List[Dict], business_id: str) -> List[FraudAlert]:
        """Detect when suppliers bill for the same purchase twice"""
        alerts = []
        
        # Get supplier information
        suppliers = await self.db.get_suppliers(business_id)
        supplier_map = {s['id']: s['name'] for s in suppliers}
        
        # Group invoices by supplier
        supplier_invoices = defaultdict(list)
        for invoice in invoices:
            supplier_id = invoice.get('supplier_id')
            if supplier_id:
                supplier_invoices[supplier_id].append(invoice)
        
        # Check for duplicate billing within each supplier
        for supplier_id, supplier_invoice_list in supplier_invoices.items():
            supplier_name = supplier_map.get(supplier_id, 'Unknown Supplier')
            
            for i, invoice1 in enumerate(supplier_invoice_list):
                for j, invoice2 in enumerate(supplier_invoice_list[i+1:], i+1):
                    similarity = self._calculate_invoice_similarity(invoice1, invoice2)
                    
                    if similarity >= self.duplicate_threshold:
                        alert = FraudAlert(
                            type=FraudType.SUPPLIER_DUPLICATE,
                            message=f"Supplier {supplier_name} billed you twice for the same purchase",
                            confidence_score=similarity,
                            evidence={
                                "supplier_id": supplier_id,
                                "supplier_name": supplier_name,
                                "original_invoice_id": invoice1.get('id'),
                                "duplicate_invoice_id": invoice2.get('id'),
                                "similarity_score": similarity,
                                "amount": invoice1.get('total_amount'),
                                "description": invoice1.get('description', '')
                            },
                            business_id=business_id,
                            entity_id=invoice2.get('id')
                        )
                        alerts.append(alert)
        
        return alerts
    
    # =============================================================================
    # Private Helper Methods - Payment Mismatch Detection
    # =============================================================================
    
    def _create_payment_map(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """Create a map of payments by invoice reference"""
        payment_map = defaultdict(list)
        
        for transaction in transactions:
            if transaction.get('type') == 'income':  # Payment received
                # Try to extract invoice reference from description
                invoice_ref = self._extract_invoice_reference(transaction.get('description', ''))
                if invoice_ref:
                    payment_map[invoice_ref].append(transaction)
        
        return payment_map
    
    async def _check_invoice_payment_mismatch(self, invoice: Dict, payment_map: Dict, 
                                            business_id: str) -> Optional[FraudAlert]:
        """Check if an invoice has payment mismatches"""
        invoice_number = invoice.get('invoice_number', '')
        invoice_amount = float(invoice.get('total_amount', 0))
        
        # Find payments for this invoice
        payments = payment_map.get(invoice_number, [])
        
        if not payments:
            return None  # No payments found, not necessarily a mismatch
        
        # Calculate total payments
        total_payments = sum(float(p.get('amount', 0)) for p in payments)
        
        # Check for significant mismatch
        amount_difference = abs(invoice_amount - total_payments)
        mismatch_percentage = amount_difference / max(invoice_amount, 0.01)
        
        if mismatch_percentage > 0.05:  # More than 5% difference
            return FraudAlert(
                type=FraudType.PAYMENT_MISMATCH,
                message=f"Payment mismatch detected: Invoice amount ₹{invoice_amount:.2f} vs payments ₹{total_payments:.2f}",
                confidence_score=min(mismatch_percentage, 1.0),
                evidence={
                    "invoice_id": invoice.get('id'),
                    "invoice_number": invoice_number,
                    "invoice_amount": invoice_amount,
                    "total_payments": total_payments,
                    "amount_difference": amount_difference,
                    "mismatch_percentage": mismatch_percentage,
                    "payment_count": len(payments),
                    "payment_ids": [p.get('id') for p in payments]
                },
                business_id=business_id,
                entity_id=invoice.get('id')
            )
        
        return None
    
    async def _detect_orphaned_payments(self, payment_map: Dict, invoices: List[Dict], 
                                      business_id: str) -> List[FraudAlert]:
        """Detect payments that don't match any invoice"""
        alerts = []
        
        # Create set of valid invoice numbers
        valid_invoice_numbers = {inv.get('invoice_number', '') for inv in invoices}
        
        # Check for orphaned payments
        for invoice_ref, payments in payment_map.items():
            if invoice_ref not in valid_invoice_numbers:
                total_amount = sum(float(p.get('amount', 0)) for p in payments)
                
                alert = FraudAlert(
                    type=FraudType.PAYMENT_MISMATCH,
                    message=f"Orphaned payment found: ₹{total_amount:.2f} for non-existent invoice {invoice_ref}",
                    confidence_score=0.8,
                    evidence={
                        "invoice_reference": invoice_ref,
                        "total_amount": total_amount,
                        "payment_count": len(payments),
                        "payment_ids": [p.get('id') for p in payments]
                    },
                    business_id=business_id
                )
                alerts.append(alert)
        
        return alerts
    
    # =============================================================================
    # Private Helper Methods - Pattern Analysis
    # =============================================================================
    
    def _extract_transaction_features(self, transactions: List[Dict]) -> np.ndarray:
        """Extract numerical features from transactions for ML analysis"""
        features = []
        
        for transaction in transactions:
            try:
                amount = float(transaction.get('amount', 0))
                
                # Extract time-based features
                created_at = transaction.get('created_at', '')
                hour_of_day = self._extract_hour_from_timestamp(created_at)
                day_of_week = self._extract_day_of_week_from_timestamp(created_at)
                
                # Extract description-based features
                description = transaction.get('description', '')
                desc_length = len(description)
                desc_word_count = len(description.split())
                
                # Extract type-based features
                trans_type = transaction.get('type', 'unknown')
                type_numeric = 1 if trans_type == 'income' else 0
                
                feature_vector = [
                    amount,
                    hour_of_day,
                    day_of_week,
                    desc_length,
                    desc_word_count,
                    type_numeric
                ]
                
                features.append(feature_vector)
                
            except (ValueError, TypeError):
                continue
        
        return np.array(features) if features else np.array([])
    
    async def _detect_anomalous_transactions(self, transactions: List[Dict], 
                                           features: np.ndarray, business_id: str) -> List[FraudAlert]:
        """Detect anomalous transactions using Isolation Forest"""
        alerts = []
        
        if len(features) < 10:
            return alerts
        
        try:
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_scores = self.isolation_forest.fit_predict(features_scaled)
            anomaly_probabilities = self.isolation_forest.score_samples(features_scaled)
            
            # Create alerts for anomalies
            for i, (transaction, is_anomaly, anomaly_score) in enumerate(
                zip(transactions, anomaly_scores, anomaly_probabilities)
            ):
                if is_anomaly == -1:  # Anomaly detected
                    confidence = abs(anomaly_score)
                    
                    alert = FraudAlert(
                        type=FraudType.SUSPICIOUS_PATTERN,
                        message=f"Suspicious transaction pattern detected: Unusual amount or timing",
                        confidence_score=min(confidence * 2, 1.0),  # Scale confidence
                        evidence={
                            "transaction_id": transaction.get('id'),
                            "amount": transaction.get('amount'),
                            "description": transaction.get('description', ''),
                            "anomaly_score": float(anomaly_score),
                            "detection_method": "isolation_forest"
                        },
                        business_id=business_id,
                        entity_id=transaction.get('id')
                    )
                    alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return alerts
    
    async def _detect_suspicious_patterns(self, transactions: List[Dict], 
                                        business_id: str) -> List[FraudAlert]:
        """Detect suspicious patterns in transactions"""
        alerts = []
        
        # Detect round number patterns (potential fraud indicator)
        round_number_alerts = self._detect_round_number_pattern(transactions, business_id)
        alerts.extend(round_number_alerts)
        
        # Detect unusual frequency patterns
        frequency_alerts = self._detect_frequency_anomalies(transactions, business_id)
        alerts.extend(frequency_alerts)
        
        return alerts
    
    def _detect_round_number_pattern(self, transactions: List[Dict], 
                                   business_id: str) -> List[FraudAlert]:
        """Detect suspicious round number patterns"""
        alerts = []
        round_amounts = []
        
        for transaction in transactions:
            try:
                amount = float(transaction.get('amount', 0))
                if amount > 0 and amount % 1000 == 0:  # Round thousands
                    round_amounts.append(transaction)
            except (ValueError, TypeError):
                continue
        
        # If more than 30% of transactions are round numbers, flag as suspicious
        if len(transactions) > 10:
            round_percentage = len(round_amounts) / len(transactions)
            
            if round_percentage > 0.3:
                alert = FraudAlert(
                    type=FraudType.SUSPICIOUS_PATTERN,
                    message=f"Suspicious round number pattern: {round_percentage:.1%} of transactions are round amounts",
                    confidence_score=min(round_percentage, 1.0),
                    evidence={
                        "round_transaction_count": len(round_amounts),
                        "total_transaction_count": len(transactions),
                        "round_percentage": round_percentage,
                        "detection_method": "round_number_analysis"
                    },
                    business_id=business_id
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_frequency_anomalies(self, transactions: List[Dict], 
                                  business_id: str) -> List[FraudAlert]:
        """Detect unusual transaction frequency patterns"""
        alerts = []
        
        # Group transactions by day
        daily_counts = defaultdict(int)
        for transaction in transactions:
            date_str = transaction.get('created_at', '')[:10]  # Extract date part
            daily_counts[date_str] += 1
        
        if len(daily_counts) < 7:  # Need at least a week of data
            return alerts
        
        # Calculate statistics
        counts = list(daily_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Detect days with unusually high activity
        for date, count in daily_counts.items():
            if count > mean_count + 3 * std_count:  # 3 standard deviations above mean
                alert = FraudAlert(
                    type=FraudType.SUSPICIOUS_PATTERN,
                    message=f"Unusual transaction frequency: {count} transactions on {date} (average: {mean_count:.1f})",
                    confidence_score=min((count - mean_count) / (mean_count + 1), 1.0),
                    evidence={
                        "date": date,
                        "transaction_count": count,
                        "average_count": mean_count,
                        "standard_deviation": std_count,
                        "detection_method": "frequency_analysis"
                    },
                    business_id=business_id
                )
                alerts.append(alert)
        
        return alerts
    
    async def _detect_velocity_anomalies(self, transactions: List[Dict], 
                                       business_id: str) -> List[FraudAlert]:
        """Detect velocity anomalies (too many transactions in short time)"""
        alerts = []
        
        # Sort transactions by timestamp
        sorted_transactions = sorted(
            transactions, 
            key=lambda x: x.get('created_at', '')
        )
        
        # Check for velocity anomalies (more than 10 transactions in 1 hour)
        velocity_window = []
        
        for transaction in sorted_transactions:
            timestamp = transaction.get('created_at', '')
            
            # Remove transactions older than 1 hour from window
            velocity_window = [
                t for t in velocity_window 
                if self._calculate_time_difference_minutes(t.get('created_at', ''), timestamp) <= 60
            ]
            
            velocity_window.append(transaction)
            
            # Check if velocity is too high
            if len(velocity_window) > 10:
                alert = FraudAlert(
                    type=FraudType.SUSPICIOUS_PATTERN,
                    message=f"High transaction velocity: {len(velocity_window)} transactions in 1 hour",
                    confidence_score=min(len(velocity_window) / 20, 1.0),
                    evidence={
                        "transaction_count": len(velocity_window),
                        "time_window_minutes": 60,
                        "latest_transaction_id": transaction.get('id'),
                        "detection_method": "velocity_analysis"
                    },
                    business_id=business_id,
                    entity_id=transaction.get('id')
                )
                alerts.append(alert)
        
        return alerts
    
    # =============================================================================
    # Private Helper Methods - Similarity Calculations
    # =============================================================================
    
    def _calculate_invoice_similarity(self, invoice1: Dict, invoice2: Dict) -> float:
        """Calculate similarity between two invoices"""
        similarity_scores = []
        
        # Amount similarity
        amount1 = float(invoice1.get('total_amount', 0))
        amount2 = float(invoice2.get('total_amount', 0))
        if max(amount1, amount2) > 0:
            amount_similarity = 1 - abs(amount1 - amount2) / max(amount1, amount2)
            similarity_scores.append(amount_similarity * 0.4)  # 40% weight
        
        # Description similarity
        desc1 = invoice1.get('description', '').lower()
        desc2 = invoice2.get('description', '').lower()
        if desc1 and desc2:
            desc_similarity = fuzz.ratio(desc1, desc2) / 100
            similarity_scores.append(desc_similarity * 0.3)  # 30% weight
        
        # Customer similarity
        if invoice1.get('customer_id') == invoice2.get('customer_id'):
            similarity_scores.append(0.2)  # 20% weight
        
        # Date proximity (closer dates = higher similarity for duplicates)
        date_diff = self._calculate_date_difference(
            invoice1.get('created_at'), invoice2.get('created_at')
        )
        if date_diff <= 7:  # Within a week
            date_similarity = max(0, 1 - date_diff / 7) * 0.1  # 10% weight
            similarity_scores.append(date_similarity)
        
        return sum(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_transaction_similarity(self, trans1: Dict, trans2: Dict) -> float:
        """Calculate similarity between two transactions"""
        similarity_scores = []
        
        # Amount similarity (exact match gets high score)
        amount1 = float(trans1.get('amount', 0))
        amount2 = float(trans2.get('amount', 0))
        if amount1 == amount2 and amount1 > 0:
            similarity_scores.append(0.5)  # 50% weight for exact amount match
        elif max(amount1, amount2) > 0:
            amount_similarity = 1 - abs(amount1 - amount2) / max(amount1, amount2)
            similarity_scores.append(amount_similarity * 0.5)
        
        # Description similarity
        desc1 = trans1.get('description', '').lower()
        desc2 = trans2.get('description', '').lower()
        if desc1 and desc2:
            desc_similarity = fuzz.ratio(desc1, desc2) / 100
            similarity_scores.append(desc_similarity * 0.3)  # 30% weight
        
        # Type similarity
        if trans1.get('type') == trans2.get('type'):
            similarity_scores.append(0.2)  # 20% weight
        
        return sum(similarity_scores) if similarity_scores else 0.0
    
    # =============================================================================
    # Private Helper Methods - Utilities
    # =============================================================================
    
    def _calculate_risk_score(self, alerts: List[FraudAlert]) -> float:
        """Calculate overall risk score based on alerts"""
        if not alerts:
            return 0.0
        
        # Weight different alert types
        type_weights = {
            FraudType.DUPLICATE_INVOICE: 0.8,
            FraudType.PAYMENT_MISMATCH: 0.9,
            FraudType.SUSPICIOUS_PATTERN: 0.6,
            FraudType.SUPPLIER_DUPLICATE: 0.85
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for alert in alerts:
            weight = type_weights.get(alert.type, 0.5)
            total_weighted_score += alert.confidence_score * weight
            total_weight += weight
        
        return min(total_weighted_score / max(total_weight, 1), 1.0)
    
    def _calculate_date_difference(self, date1: str, date2: str) -> int:
        """Calculate difference in days between two date strings"""
        try:
            from dateutil.parser import parse
            d1 = parse(date1).date()
            d2 = parse(date2).date()
            return abs((d1 - d2).days)
        except:
            return 0
    
    def _calculate_time_difference_minutes(self, time1: str, time2: str) -> int:
        """Calculate difference in minutes between two timestamps"""
        try:
            from dateutil.parser import parse
            t1 = parse(time1)
            t2 = parse(time2)
            return abs(int((t1 - t2).total_seconds() / 60))
        except:
            return 0
    
    def _extract_hour_from_timestamp(self, timestamp: str) -> int:
        """Extract hour of day from timestamp"""
        try:
            from dateutil.parser import parse
            return parse(timestamp).hour
        except:
            return 12  # Default to noon
    
    def _extract_day_of_week_from_timestamp(self, timestamp: str) -> int:
        """Extract day of week from timestamp (0=Monday, 6=Sunday)"""
        try:
            from dateutil.parser import parse
            return parse(timestamp).weekday()
        except:
            return 0  # Default to Monday
    
    def _extract_invoice_reference(self, description: str) -> Optional[str]:
        """Extract invoice reference from transaction description"""
        # Look for patterns like "INV-001", "Invoice 123", etc.
        patterns = [
            r'INV[-_]?(\d+)',
            r'Invoice\s+(\d+)',
            r'Bill\s+(\d+)',
            r'#(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    async def _save_fraud_alert(self, alert: FraudAlert) -> Optional[str]:
        """Save fraud alert to database"""
        try:
            alert_data = {
                "business_id": alert.business_id,
                "alert_type": alert.type.value,
                "message": alert.message,
                "evidence": alert.evidence,
                "status": "active",
                "confidence_score": alert.confidence_score,
                "entity_id": alert.entity_id
            }
            
            return await self.db.save_fraud_alert(alert_data)
        except Exception as e:
            logger.error(f"Error saving fraud alert: {e}")
            return None
    
    async def _log_fraud_analysis(self, business_id: str, alert_count: int, risk_score: float):
        """Log fraud analysis completion"""
        try:
            log_data = {
                "business_id": business_id,
                "operation_type": "fraud_analysis",
                "status": "completed",
                "results": {
                    "alert_count": alert_count,
                    "risk_score": risk_score
                },
                "processing_time_ms": 0,  # Would be calculated in real implementation
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.db.log_ai_operation(log_data)
        except Exception as e:
            logger.error(f"Error logging fraud analysis: {e}")    

    # =============================================================================
    # Additional API Support Methods
    # =============================================================================
    
    async def get_fraud_alerts(self, business_id: str, include_resolved: bool = False,
                              alert_type: Optional[str] = None, limit: int = 50,
                              offset: int = 0) -> List[FraudAlert]:
        """Get fraud alerts from database with filtering and pagination"""
        try:
            filters = {
                "business_id": business_id,
                "limit": limit,
                "offset": offset
            }
            
            if not include_resolved:
                filters["status"] = "active"
            
            if alert_type:
                filters["alert_type"] = alert_type
            
            alert_records = await self.db.get_fraud_alerts(filters)
            
            # Convert database records to FraudAlert objects
            alerts = []
            for record in alert_records:
                try:
                    alert = FraudAlert(
                        id=record.get('id'),
                        type=FraudType(record.get('alert_type')),
                        message=record.get('message'),
                        confidence_score=record.get('confidence_score', 0.0),
                        evidence=record.get('evidence', {}),
                        detected_at=record.get('created_at'),
                        business_id=record.get('business_id'),
                        entity_id=record.get('entity_id')
                    )
                    alerts.append(alert)
                except Exception as e:
                    logger.error(f"Error converting alert record to FraudAlert: {e}")
                    continue
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error fetching fraud alerts: {e}")
            return []
    
    async def update_fraud_alert(self, alert_id: str, status: str,
                                resolution_notes: Optional[str] = None,
                                updated_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update fraud alert status and resolution"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if resolution_notes:
                update_data["resolution_notes"] = resolution_notes
            
            if updated_by:
                update_data["updated_by"] = updated_by
            
            if status in ["resolved", "false_positive"]:
                update_data["resolved_at"] = datetime.utcnow().isoformat()
            
            updated_alert = await self.db.update_fraud_alert(alert_id, update_data)
            return updated_alert
            
        except Exception as e:
            logger.error(f"Error updating fraud alert {alert_id}: {e}")
            return None
    
    async def get_fraud_statistics(self, business_id: str, days: int = 30) -> Dict[str, Any]:
        """Get fraud detection statistics for a business"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get alerts within date range
            alerts = await self.db.get_fraud_alerts({
                "business_id": business_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            # Calculate statistics
            total_alerts = len(alerts)
            resolved_alerts = len([a for a in alerts if a.get('status') == 'resolved'])
            false_positives = len([a for a in alerts if a.get('status') == 'false_positive'])
            active_alerts = len([a for a in alerts if a.get('status') == 'active'])
            
            # Alert type distribution
            alert_type_counts = {}
            for alert in alerts:
                alert_type = alert.get('alert_type', 'unknown')
                alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
            
            # Risk level distribution
            high_risk = len([a for a in alerts if a.get('confidence_score', 0) >= 0.8])
            medium_risk = len([a for a in alerts if 0.5 <= a.get('confidence_score', 0) < 0.8])
            low_risk = len([a for a in alerts if a.get('confidence_score', 0) < 0.5])
            
            # Calculate trends (daily alert counts)
            daily_counts = {}
            for alert in alerts:
                created_at = alert.get('created_at', '')
                if created_at:
                    date_key = created_at[:10]  # Extract date part
                    daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            
            # Calculate average risk score
            risk_scores = [a.get('confidence_score', 0) for a in alerts if a.get('confidence_score')]
            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "alert_counts": {
                    "total": total_alerts,
                    "active": active_alerts,
                    "resolved": resolved_alerts,
                    "false_positives": false_positives
                },
                "alert_types": alert_type_counts,
                "risk_levels": {
                    "high": high_risk,
                    "medium": medium_risk,
                    "low": low_risk
                },
                "metrics": {
                    "average_risk_score": round(avg_risk_score, 2),
                    "resolution_rate": round(resolved_alerts / max(total_alerts, 1) * 100, 1),
                    "false_positive_rate": round(false_positives / max(total_alerts, 1) * 100, 1)
                },
                "trends": {
                    "daily_alert_counts": daily_counts
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating fraud statistics: {e}")
            return {
                "error": str(e),
                "period": {"days": days},
                "alert_counts": {"total": 0, "active": 0, "resolved": 0, "false_positives": 0},
                "alert_types": {},
                "risk_levels": {"high": 0, "medium": 0, "low": 0},
                "metrics": {"average_risk_score": 0.0, "resolution_rate": 0.0, "false_positive_rate": 0.0},
                "trends": {"daily_alert_counts": {}}
            }